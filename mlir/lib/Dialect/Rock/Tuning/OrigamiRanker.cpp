//===- OrigamiRanker.cpp - Rank quick-tune configs with Origami ----------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/OrigamiRanker.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include "origami/origami.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>

#define DEBUG_TYPE "rock-origami-ranker"

using namespace mlir;
using namespace mlir::rock;

namespace {

using ArchEnum = origami::hardware_t::architecture_t;

/// The physical hardware facts Origami needs but cannot supply itself: its
/// per-arch table holds only fitted model coefficients, and upstream expects
/// everything here to come from a live device via `hipDeviceProp_t`. This
/// build is HIP-free and cross-targets chips it is not running on, so the
/// values are tabulated instead.
///
/// Taken verbatim from Origami's own offline test fixture,
/// external/origami/python/tests/helpers.py, which is what upstream generates
/// its ranking regression baselines against. Architectures absent from that
/// fixture are deliberately absent here too: Origami models gfx1150-gfx1153
/// and gfx1250, but supplying invented constants for them would silently skew
/// the ranking rather than fail, so those chips keep the unranked order.
/// `l2Bytes` is the whole chip's L2, not a per-XCD slice.
struct OrigamiArchInfo {
  ArchEnum arch;
  int64_t l2Bytes;
  int computeClockKHz;
};

std::optional<OrigamiArchInfo> lookupArch(StringRef arch) {
  constexpr int64_t kKiB = 1024;
  constexpr int64_t kMiB = 1024 * kKiB;
  auto [_, chip] = getArch(arch);

  return llvm::StringSwitch<std::optional<OrigamiArchInfo>>(chip)
      .Case("gfx90a", OrigamiArchInfo{ArchEnum::gfx90a, 8 * kMiB, 1700000})
      .Case("gfx942", OrigamiArchInfo{ArchEnum::gfx942, 24 * kMiB, 1700000})
      .Case("gfx950", OrigamiArchInfo{ArchEnum::gfx950, 32 * kMiB, 2100000})
      .Case("gfx1100", OrigamiArchInfo{ArchEnum::gfx1100, 6 * kMiB, 2500000})
      .Case("gfx1101", OrigamiArchInfo{ArchEnum::gfx1101, 4 * kMiB, 2276000})
      .Case("gfx1200", OrigamiArchInfo{ArchEnum::gfx1200, 4 * kMiB, 2700000})
      .Case("gfx1201", OrigamiArchInfo{ArchEnum::gfx1201, 6 * kMiB, 2500000})
      .Default(std::nullopt);
}

/// Register file bytes per CU. Origami's fixture uses 512 KiB for every
/// architecture it covers. Only the attention model reads it, to decide
/// whether a macro tile's Q/K/V/O/P working set fits in registers.
constexpr int64_t kRegisterFileBytes = 512 * 1024;

origami::data_type_t toOrigamiDataType(Type type) {
  Type elem = getElementTypeOrSelf(type);
  unsigned width = elem.getIntOrFloatBitWidth();

  if (auto floatType = dyn_cast<FloatType>(elem)) {
    switch (width) {
    case 4:
      return origami::data_type_t::Float4;
    case 6:
      return origami::data_type_t::Float6;
    case 8:
      // Origami separates the OCP and FNUZ 8-bit encodings; they differ in
      // exponent bias, not in the element size or instruction shape the model
      // keys on, but pick the matching one anyway.
      if (isa<Float8E4M3FNUZType>(elem))
        return origami::data_type_t::Float8_fnuz;
      if (isa<Float8E5M2FNUZType>(elem))
        return origami::data_type_t::BFloat8_fnuz;
      if (elem.isF8E5M2())
        return origami::data_type_t::BFloat8;
      return origami::data_type_t::Float8;
    case 16:
      return floatType.isBF16() ? origami::data_type_t::BFloat16
                                : origami::data_type_t::Half;
    case 32:
      return origami::data_type_t::Float;
    case 64:
      return origami::data_type_t::Double;
    default:
      return origami::data_type_t::None;
    }
  }

  if (elem.isInteger()) {
    switch (width) {
    case 4:
      return origami::data_type_t::Int4;
    case 8:
      return origami::data_type_t::Int8;
    case 32:
      return origami::data_type_t::Int32;
    case 64:
      return origami::data_type_t::Int64;
    default:
      return origami::data_type_t::None;
    }
  }
  return origami::data_type_t::None;
}

/// Whether Origami has a measured latency for `instr`.
///
/// It does not reject an instruction it has never seen: `get_mi_latency` falls
/// back to a flat 32 cycles, which is a plausible-looking number that would
/// rank the config on its memory terms alone. Screen those configs out here so
/// they keep their original position instead of being scored against a
/// made-up compute cost.
bool isModelledInstr(const origami::hardware_t &hardware,
                     origami::data_type_t miDataType, origami::dim3_t instr) {
  return llvm::is_contained(hardware.get_valid_matrix_instructions(miDataType),
                            instr);
}

/// The matrix-instruction shape a config will lower to, which Origami needs to
/// count instructions per macro tile. Fails when the kernel has no matrix
/// instruction, or none that covers these operand types.
///
/// Which instruction gets issued depends on the config's `kPerBlock` and not
/// just on its `matrixInstrNonkdim`: where an arch offers several K extents at
/// the same tile, Triton takes the widest one the block's K can feed. Asking
/// for the narrowest instead would model more, shorter instructions per tile
/// and overstate the compute time.
FailureOr<origami::dim3_t>
getMatrixInstrShape(StringRef arch, MatrixAccelKind accelKind, uint32_t nonKDim,
                    uint32_t kPerBlock, Type aType, Type bType) {
  // `matrixInstrNonkdim` is an MFMA-only knob. WMMA and the non-accelerated
  // FMA path both spell it 0, so it cannot distinguish them -- the accel kind
  // is what separates "instruction is always 16x16" from "no instruction".
  bool isMfma = accelKind == MatrixAccelKind::MFMA ||
                accelKind == MatrixAccelKind::ScaledMFMA;
  uint32_t instrMN = isMfma && nonKDim != 0 ? nonKDim : 16;

  FailureOr<int64_t> instrK =
      getAccelInstrKDim(arch, aType, bType, instrMN, kPerBlock);
  if (failed(instrK) || *instrK <= 0)
    return failure();

  return origami::dim3_t{instrMN, instrMN, static_cast<size_t>(*instrK)};
}

/// Escape hatch for A/B-ing the ranking: with this set the quick-tune list
/// keeps the order the tuning tables give it, so a tuning run can be compared
/// against the same build with ranking on.
bool rankingDisabled() {
  return std::getenv("ROCMLIR_DISABLE_ORIGAMI_RANKING") != nullptr;
}

/// How many configs to keep once the list is in best-first order, read from
/// `ROCMLIR_ORIGAMI_TOP_N`. Unset defaults to 30; zero or unparseable keeps all
/// of them.
///
/// This trades tuning time against the risk of cropping away the config that
/// would actually have won, so it only ever applies to a list Origami really
/// ranked: every path that bails out early leaves the candidates untouched,
/// and an unranked list is in no particular order to crop.
std::optional<size_t> rankedListLimit() {
  const char *env = std::getenv("ROCMLIR_ORIGAMI_TOP_N");
  if (!env)
    return 30;

  size_t limit = 0;
  if (StringRef(env).trim().getAsInteger(10, limit) || limit == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Ignoring ROCMLIR_ORIGAMI_TOP_N=\"" << env
                            << "\": expected a positive count\n");
    return std::nullopt;
  }
  return limit;
}

/// The innermost (fastest-varying) entry of a Rock conv layout attribute, e.g.
/// the `"ci"` of `input_layout = ["ni", "gi", "0i", "1i", "ci"]`.
FailureOr<StringRef> innermostDim(Operation *op, StringRef layoutAttrName) {
  auto layout = op->getAttrOfType<ArrayAttr>(layoutAttrName);
  if (!layout || layout.empty())
    return failure();
  auto innermost = dyn_cast<StringAttr>(layout.getValue().back());
  if (!innermost)
    return failure();
  return innermost.getValue();
}

/// Which axis of A and of B is the contiguous one, in Origami's spelling.
///
/// `transpose_t` is not a BLAS op flag. The model branches on it to pick the
/// macro-tile extent that forms the contiguous run it rounds up to cache lines
/// (`a_contig`/`b_contig` in origami's gemm.cpp): `T` on A means the reduction
/// axis is contiguous, `T` on B means the N axis is. Rock's `aTransposed`
/// names the opposite condition, which is why these invert.
///
/// Fails when the operand layout cannot be established, so that the caller
/// skips ranking instead of feeding the model a guess.
FailureOr<std::pair<origami::transpose_t, origami::transpose_t>>
getOrigamiTransposes(RockGemmWrapperInterface gemmOp, KernelType kernelType) {
  using origami::transpose_t;
  Operation *op = gemmOp.getOperation();

  switch (kernelType) {
  case KernelType::Gemm: {
    auto gemm = dyn_cast<GemmOp>(op);
    if (!gemm)
      return failure();
    // Untransposed A is [g, m, k] (k contiguous) and B is [g, k, n] (n
    // contiguous); each transpose attribute swaps its matrix's last two dims.
    return std::make_pair(
        gemm.getATransposed() ? transpose_t::N : transpose_t::T,
        gemm.getBTransposed() ? transpose_t::N : transpose_t::T);
  }
  case KernelType::Conv: {
    // Implicit GEMM: A is the filter seen as [g, m=k, gemmK=c*y*x] and B is
    // the input seen as [g, gemmK=c*y*x, n=n*ho*wo].
    FailureOr<StringRef> filterDim = innermostDim(op, "filter_layout");
    FailureOr<StringRef> inputDim = innermostDim(op, "input_layout");
    if (failed(filterDim) || failed(inputDim))
      return failure();
    return std::make_pair(*filterDim == "k" ? transpose_t::N : transpose_t::T,
                          *inputDim == "ci" ? transpose_t::N : transpose_t::T);
  }
  case KernelType::ConvBwdData: {
    // A is the filter seen as [g, m=c, gemmK=k*...] and B is the output
    // gradient seen as [g, gemmK=k*..., n=n*...].
    FailureOr<StringRef> filterDim = innermostDim(op, "filter_layout");
    FailureOr<StringRef> outputDim = innermostDim(op, "output_layout");
    if (failed(filterDim) || failed(outputDim))
      return failure();
    return std::make_pair(*filterDim == "c" ? transpose_t::N : transpose_t::T,
                          *outputDim == "ko" ? transpose_t::N : transpose_t::T);
  }
  default:
    return failure();
  }
}

/// Rebuild `params` best-first from Origami's `ranked` verdict.
///
/// rank_configs drops the configs it rejects instead of ranking them last, so
/// walk the results first and then sweep up everything they did not mention:
/// a config Origami would not score is still a config the tuner may pick, and
/// it only loses its place in the list rather than its place in the space. The
/// list is then cropped to the configured limit, which defaults to 30.
template <typename ParamsAttrT>
void reorderByRanking(const std::vector<origami::prediction_result_t> &ranked,
                      std::vector<ParamsAttrT> &params) {
  std::vector<ParamsAttrT> reordered;
  reordered.reserve(params.size());
  std::vector<bool> taken(params.size(), false);
  for (const origami::prediction_result_t &result : ranked) {
    size_t idx = result.config.index;
    if (idx >= params.size() || taken[idx])
      continue;
    taken[idx] = true;
    LLVM_DEBUG(llvm::dbgs() << "  " << llvm::format("%12.1f", result.latency)
                            << " cycles  " << params[idx] << "\n");
    reordered.push_back(params[idx]);
  }
  for (auto [idx, param] : llvm::enumerate(params))
    if (!taken[idx]) {
      LLVM_DEBUG(llvm::dbgs() << "     (rejected)  " << param << "\n");
      reordered.push_back(param);
    }

  assert(reordered.size() == params.size() &&
         "Origami ranking must preserve every candidate config");

  if (std::optional<size_t> limit = rankedListLimit();
      limit && *limit < reordered.size()) {
    LLVM_DEBUG(llvm::dbgs() << "  cropping to the top " << *limit << " of "
                            << reordered.size() << " configs\n");
    reordered.resize(*limit);
  }

  params = std::move(reordered);
}

} // namespace

bool mlir::rock::origamiSupportsArch(StringRef arch) {
  return lookupArch(arch).has_value();
}

void mlir::rock::rankGemmParamsByOrigami(RockGemmWrapperInterface gemmOp,
                                         const PopulateParamsInfo &info,
                                         std::vector<GemmParamsAttr> &params) {
  // Origami throws on an empty candidate list, and this TU is built with
  // -fno-exceptions, so never hand it one. A single config is already ordered.
  if (params.size() < 2 || rankingDisabled())
    return;

  StringRef arch = info.arch;
  std::optional<OrigamiArchInfo> archInfo = lookupArch(arch);
  if (!archInfo) {
    LLVM_DEBUG(llvm::dbgs() << "No Origami hardware constants for " << arch
                            << "; leaving the quick-tune order alone\n");
    return;
  }

  // The FMA path has no matrix instruction, and Origami's config_t requires a
  // non-degenerate one (is_valid() demands mi.m/n/k > 0).
  MatrixAccelKind accelKind = getMatrixAccelKind(arch, gemmOp);
  if (accelKind == MatrixAccelKind::None) {
    LLVM_DEBUG(llvm::dbgs()
               << "Non-accelerated GEMM has no matrix instruction to model; "
                  "leaving the quick-tune order alone\n");
    return;
  }

  FailureOr<std::pair<origami::transpose_t, origami::transpose_t>> transposes =
      getOrigamiTransposes(gemmOp, info.kernelType);
  if (failed(transposes)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot establish operand layouts for this "
                               "kernel; leaving the quick-tune order alone\n");
    return;
  }

  origami::problem_t problem;
  const GemmSize &size = info.gemmSize;
  problem.size =
      origami::dim3_t{static_cast<size_t>(size.m), static_cast<size_t>(size.n),
                      static_cast<size_t>(size.k)};
  problem.batch = static_cast<size_t>(std::max<int64_t>(size.g, 1));
  problem.a_transpose = transposes->first;
  problem.b_transpose = transposes->second;
  problem.a_dtype = toOrigamiDataType(info.gemmAType);
  problem.b_dtype = toOrigamiDataType(info.gemmBType);
  problem.d_dtype = toOrigamiDataType(gemmOp.getCType());
  problem.c_dtype = problem.d_dtype;
  // Keys the instruction-latency lookup, so it tracks the operands feeding the
  // matrix instruction rather than the accumulator.
  problem.mi_dtype = problem.a_dtype;
  if (info.quantBlockSize) {
    problem.a_mx_block_size = static_cast<size_t>(*info.quantBlockSize);
    problem.b_mx_block_size = static_cast<size_t>(*info.quantBlockSize);
  }

  origami::hardware_t hardware = origami::hardware_t::get_hardware_for_arch(
      archInfo->arch, static_cast<size_t>(getNumCUValue(gemmOp)),
      static_cast<size_t>(getLDSSize(arch)), kRegisterFileBytes,
      static_cast<size_t>(archInfo->l2Bytes), archInfo->computeClockKHz);

  // `index` is never read by Origami, so it survives ranking and is how a
  // result maps back to the GemmParamsAttr it came from.
  std::vector<origami::config_t> configs;
  configs.reserve(params.size());
  for (auto [idx, param] : llvm::enumerate(params)) {
    FailureOr<origami::dim3_t> instrShape = getMatrixInstrShape(
        arch, accelKind, param.getMatrixInstrNonkdim(), param.getKPerBlock(),
        info.gemmAType, info.gemmBType);
    if (failed(instrShape))
      continue;
    if (!isModelledInstr(hardware, problem.mi_dtype, *instrShape)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  skipping " << instrShape->m << "x" << instrShape->n
                 << "x" << instrShape->k << " "
                 << origami::datatype_to_string(problem.mi_dtype)
                 << ": no measured latency for this instruction\n");
      continue;
    }

    origami::config_t config;
    config.mt = origami::dim3_t{static_cast<size_t>(param.getMPerBlock()),
                                static_cast<size_t>(param.getNPerBlock()),
                                static_cast<size_t>(param.getKPerBlock())};
    config.mi = *instrShape;
    // Origami means waves resident per CU and clamps to at least 1. The
    // perf-config's 0 means "unset", not "none".
    config.occupancy = param.getWavesPerEU() > 0 ? param.getWavesPerEU() : 1;
    // Rock emits a plain data-parallel grid, never a stream-K one, and carries
    // the K split as a tuning knob instead of deriving it from a grid size.
    // Left at its default, stream_k would have Origami invent a split of its
    // own for a kernel Rock will not generate.
    config.stream_k = 0;
    config.split_k = std::max<int64_t>(param.getSplitKFactor(), 1);
    // Inert today -- no Origami model code branches on `target`, and its
    // fitted coefficients carry no backend dimension -- but these kernels are
    // Triton-generated rather than Tensile, so say so and pick up a
    // Triton-specific path should upstream grow one.
    config.target = origami::target_t::triton;
    config.index = idx;
    configs.push_back(config);
  }

  if (configs.size() < 2)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Origami ranking " << configs.size() << " of "
                          << params.size() << " GEMM configs for " << arch
                          << " g=" << size.g << " m=" << size.m
                          << " n=" << size.n << " k=" << size.k << "\n");
  reorderByRanking(origami::rank_configs(problem, hardware, configs), params);
}

void mlir::rock::rankAttentionParamsByOrigami(
    RockGemmGemmWrapperInterface gemmGemmOp,
    std::vector<GemmGemmParamsAttr> &params) {
  // Origami throws on an empty candidate list, and this TU is built with
  // -fno-exceptions, so never hand it one. A single config is already ordered.
  if (params.size() < 2 || rankingDisabled())
    return;

  StringRef arch = getArchValue(gemmGemmOp).getValue();
  std::optional<OrigamiArchInfo> archInfo = lookupArch(arch);
  if (!archInfo) {
    LLVM_DEBUG(llvm::dbgs() << "No Origami hardware constants for " << arch
                            << "; leaving the quick-tune order alone\n");
    return;
  }

  // Checking the first gemm is sufficient: the two gemms in attention always
  // share the same accel kind on every currently supported arch.
  MatrixAccelKind accelKind = getMatrixAccelKind(arch, gemmGemmOp).first;
  if (accelKind == MatrixAccelKind::None) {
    LLVM_DEBUG(llvm::dbgs()
               << "Non-accelerated attention has no matrix instruction to "
                  "model; leaving the quick-tune order alone\n");
    return;
  }

  Type aType = gemmGemmOp.getAType();
  Type bType = gemmGemmOp.getBType();

  origami::problem_t problem;
  GemmGemmSize size = gemmGemmOp.getGemmGemmSize();
  // Origami's attention model reads `size` as (Q_SEQ, K_SEQ, H_DIM), which is
  // exactly the first GEMM's (m, n, k): S = Q * K^T.
  problem.size =
      origami::dim3_t{static_cast<size_t>(size.m), static_cast<size_t>(size.n),
                      static_cast<size_t>(size.k)};
  // Rock folds the heads into the group dim, so there is no separate head
  // count to hand over. Origami only ever uses batch and q_heads multiplied
  // together -- the tile total is batch * q_heads * grid -- so carrying the
  // whole product in `batch` is exact. Setting q_heads is not optional: it
  // defaults to 32, which would inflate the launch by 32x.
  problem.batch = static_cast<size_t>(std::max<int64_t>(size.g, 1));
  problem.q_heads = 1;
  problem.a_dtype = toOrigamiDataType(aType);
  problem.b_dtype = toOrigamiDataType(bType);
  problem.c_dtype = toOrigamiDataType(gemmGemmOp.getCType());
  problem.d_dtype = toOrigamiDataType(gemmGemmOp.getOutType());
  // Keys the instruction-latency lookup, so it tracks the operands feeding the
  // matrix instruction rather than the accumulator.
  problem.mi_dtype = problem.a_dtype;

  origami::hardware_t hardware = origami::hardware_t::get_hardware_for_arch(
      archInfo->arch, static_cast<size_t>(getNumCUValue(gemmGemmOp)),
      static_cast<size_t>(getLDSSize(arch)), kRegisterFileBytes,
      static_cast<size_t>(archInfo->l2Bytes), archInfo->computeClockKHz);

  // `index` is never read by Origami, so it survives ranking and is how a
  // result maps back to the GemmGemmParamsAttr it came from.
  std::vector<origami::config_t> configs;
  configs.reserve(params.size());
  for (auto [idx, param] : llvm::enumerate(params)) {
    FailureOr<origami::dim3_t> instrShape =
        getMatrixInstrShape(arch, accelKind, param.getMatrixInstrNonkdim(),
                            param.getKPerBlock(), aType, bType);
    if (failed(instrShape) ||
        !isModelledInstr(hardware, problem.mi_dtype, *instrShape))
      continue;

    origami::config_t config;
    // The attention macro tile is (Q tile, K-sequence tile, head-dim tile).
    // Origami models one head dim shared by both gemms, using MT_K as the
    // first gemm's K and the second gemm's N, so `nPerBlockG1` has nowhere to
    // go: configs that differ only in how they tile the output head dim look
    // identical to the model.
    config.mt = origami::dim3_t{static_cast<size_t>(param.getMPerBlockG0()),
                                static_cast<size_t>(param.getNPerBlockG0()),
                                static_cast<size_t>(param.getKPerBlock())};
    config.mi = *instrShape;
    // Origami means waves resident per CU and clamps to at least 1. The
    // perf-config's 0 means "unset", not "none".
    config.occupancy = param.getWavesPerEU() > 0 ? param.getWavesPerEU() : 1;
    // Inert today; see rankGemmParamsByOrigami.
    config.target = origami::target_t::triton;
    config.index = idx;
    configs.push_back(config);
  }

  if (configs.size() < 2)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Origami ranking " << configs.size() << " of "
                          << params.size() << " attention configs for " << arch
                          << " g=" << size.g << " qSeq=" << size.m << " kSeq="
                          << size.n << " headDim=" << size.k << "\n");
  reorderByRanking(origami::rank_configs(problem, hardware, configs,
                                         origami::model_t::attention),
                   params);
}
