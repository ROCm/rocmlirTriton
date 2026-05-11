//===- fusionUtils.cpp - Rock utility for fusion -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::rock;
using namespace arith;

static bool validOperationGemmOut(Operation *op) {
  return isa<MulFOp, DivFOp, AddFOp, SubFOp, SIToFPOp, UIToFPOp, NegFOp,
             ExtUIOp, ExtSIOp, ExtFOp, TruncFOp, TruncIOp>(op);
}

LogicalResult mlir::rock::checkValidOutputFusion(
    Value gemmResult,
    SmallVector<std::tuple<Operation *, int>> &adds) {
  /* We can only fuse:
  - add/sub gemmResult, otherTensor (which will be converted to add gemmResult,
  otherTensor/splitKFactor)
  - add/sub gemmResult, gemmResult
  - mul/div gemmResult, otherTensor
  - neg
  - type conversion functions
  Where gemmResult != otherTensor for all cases.
  */
  auto fusionInfo = rock::collectFusionInfo(gemmResult);
  for (Operation *fusionOp : fusionInfo.fusionOps) {
    // check if any operand is derived from the GEMM result
    int numGemmResults = 0;
    for (Value operand : fusionOp->getOperands()) {
      if (fusionInfo.chainValues.contains(operand))
        numGemmResults++;
    }
    if (numGemmResults > 0) {
      // check it's a valid operation
      if (!validOperationGemmOut(fusionOp)) {
        return failure();
      }

      if (isa<MulFOp, DivFOp>(fusionOp) && numGemmResults > 1) {
        // gemmOut^2 is not allowed
        return failure();
      }

      // save add and sub ops to modify them: divide by splitKFactor
      // if both operands come from gemmOut, no need to modify anything
      if (isa<AddFOp, SubFOp>(fusionOp) && numGemmResults == 1) {
        int index =
            fusionInfo.chainValues.contains(fusionOp->getOperand(0)) ? 0 : 1;
        adds.push_back(std::make_tuple(fusionOp, index));
      }
    }
  }
  return success();
}

bool mlir::rock::gemmGemmHasPreSecondGemmFusion(
    RockGemmGemmWrapperInterface gemmGemmOp) {
  Region &region = gemmGemmOp.getPreSecondGemmRegion();
  if (region.empty())
    return false;
  return !region.front().without_terminator().empty();
}

LogicalResult mlir::rock::testFusionLegalitySplitK(func::FuncOp func) {
  // can't fuse reduce_max with split-k
  WalkResult reduceMaxRes = func.walk([](ReduceOp reduceOp) -> WalkResult {
    if (reduceOp.getReduceMethod() == ReduceMethod::Max)
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  WalkResult gemmWalkResult =
      func.walk([&](rock::RockGemmWrapperInterface gemmOp) -> WalkResult {
        // Use the result directly if there's no output argument (e.g., GemmOp)
        Value gemmResult = gemmOp->getResult(0);

        auto maybeBlockArgs = traceRootOutputToArgs(gemmResult, func);
        if (failed(maybeBlockArgs))
          return WalkResult::interrupt();

        // Verify hardware compatibility (split-k) for kernel output.
        // Checks if atomic_add operations are supported by the target hardware.
        auto blockArgs = maybeBlockArgs.value();
        for (auto blockArg : blockArgs) {
          auto outElementType =
              cast<ShapedType>(blockArg.getType()).getElementType();
          if (!isFastAtomicAddSupported(rock::getArchValue(gemmOp),
                                        outElementType))
            return WalkResult::interrupt();
        }

        SmallVector<std::tuple<Operation *, int>> adds;
        if (failed(checkValidOutputFusion(gemmResult, adds)))
          return WalkResult::interrupt();

        return WalkResult::advance();
      });

  WalkResult gemmGemmWalkResult = func.walk(
      [&](rock::RockGemmGemmWrapperInterface gemmGemmOp) -> WalkResult {
        // We have two results for attention (output + LSE)
        // But we only support split-k for gemm+gemm, so there's a single result
        // here
        auto gemmGemmResult = gemmGemmOp->getResult(0);

        auto maybeBlockArgs = traceRootOutputToArgs(gemmGemmResult, func);
        if (failed(maybeBlockArgs))
          return WalkResult::interrupt();

        // Verify hardware compatibility (split-k) for kernel output.
        // Checks if atomic_add operations are supported by the target hardware.
        auto blockArgs = maybeBlockArgs.value();
        for (auto blockArg : blockArgs) {
          auto outElementType =
              cast<ShapedType>(blockArg.getType()).getElementType();
          if (!isFastAtomicAddSupported(rock::getArchValue(gemmGemmOp),
                                        outElementType))
            return WalkResult::interrupt();
        }

        // no fusions allowed for now
        auto fusionInfo = rock::collectFusionInfo(gemmGemmResult);
        if (!fusionInfo.fusionOps.empty())
          return WalkResult::interrupt();

        // fusions between gemm0 and gemm1 are not allowed
        bool fusionsFound = gemmGemmHasPreSecondGemmFusion(gemmGemmOp);
        if (fusionsFound)
          return WalkResult::interrupt();

        return WalkResult::advance();
      });

  return success(!gemmWalkResult.wasInterrupted() &&
                 !gemmGemmWalkResult.wasInterrupted() &&
                 !reduceMaxRes.wasInterrupted());
}

LogicalResult mlir::rock::testFusionLegalitySplitK(ModuleOp mod) {
  auto funcs = mod.getOps<func::FuncOp>();
  assert(std::distance(funcs.begin(), funcs.end()) &&
         "expected ModuleOp containing a single func::FuncOp");
  func::FuncOp func = *(funcs.begin());
  return testFusionLegalitySplitK(func);
}

LogicalResult mlir::rock::testFusionLegalityReduce(func::FuncOp func) {
  WalkResult walkResult = func.walk([&](rock::ReduceOp reduceOp) -> WalkResult {
    auto outElemType = reduceOp.getResult().getType().getElementType();
    if (reduceOp.getReduceMethod() == ReduceMethod::Max) {
      if (!isFastAtomicMaxSupported(rock::getArchValue(reduceOp), outElemType))
        return WalkResult::interrupt();
    } else {
      if (!isFastAtomicAddSupported(rock::getArchValue(reduceOp), outElemType))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return success(!walkResult.wasInterrupted());
}

LogicalResult mlir::rock::testFusionLegalityReduce(ModuleOp mod) {
  auto funcs = mod.getOps<func::FuncOp>();
  assert(std::distance(funcs.begin(), funcs.end()) &&
         "expected ModuleOp containing a single func::FuncOp");
  func::FuncOp func = *(funcs.begin());
  return testFusionLegalityReduce(func);
}

LogicalResult mlir::rock::testFusionLegalityBwdDataConv(func::FuncOp func) {
  // For right now, no BwdDataConv ops are fusible
  WalkResult walkResult = func.walk([&](Operation *op) -> WalkResult {
    if (auto bwdData = dyn_cast<rock::ConvBwdDataOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  return success(!walkResult.wasInterrupted());
}

LogicalResult mlir::rock::testFusionLegalityBwdDataConv(ModuleOp mod) {
  auto funcs = mod.getOps<func::FuncOp>();
  bool isFusible = true;
  for (auto f : funcs) {
    isFusible &= succeeded(testFusionLegalityBwdDataConv(f));
  }

  return success(isFusible);
}

// Conservative peak-LDS estimate for a fused gemm-gemm op under a chosen
// perfConfig. Mirrors how `getGemm1Params` derives `gemm1NPerBlock` and the
// operand-staging strategy in `GridwiseAttnToBlockwise.cpp`. Returns the
// upper-bound peak LDS in bytes for the fused kernel.
static int64_t estimateGemmGemmPeakLDSBytes(RockGemmGemmWrapperInterface op,
                                            GemmGemmParamsAttr params) {
  int64_t mPerBlockG0 = params.getMPerBlockG0();
  int64_t nPerBlockG0 = params.getNPerBlockG0();
  int64_t kPerBlock = params.getKPerBlock();
  int64_t numStages = std::max<int64_t>(1, params.getNumStages());

  // gemm1KPerBlock comes from PopulateParamsGemmGemm (pinned to nPerBlockG0
  // because gemm0's per-block N tile becomes gemm1's per-block K tile), and
  // gemm1NPerBlock is pinned to PowerOf2Ceil(N1) by `getGemm1Params`.
  int64_t gemm1KPerBlock = PopulateParamsGemmGemm::getGemm1KPerBlock(params);
  int64_t gemm1NPerBlock =
      llvm::PowerOf2Ceil(PopulateParamsGemmGemm::getGemm1N(op));

  // Per-tile element widths. Each LDS-resident tile is sized in its own
  // element type, since a fused gemm-gemm/attention can mix precisions:
  //   - A/B (gemm0 inputs) share a width: enforced by `AllElementTypesMatch`
  //     on `rock.gemm_elementwise_gemm` and by Q/K being constrained to the
  //     same element type on `rock.attention`.
  //   - The P tile (gemm0 output staged for gemm1's A operand) is cast to
  //     the op's `softmaxType` when set (e.g. f32 for mixed-precision
  //     attention, see `GridwiseAttnToBlockwise.cpp` `elemTypeSoftmax`),
  //     otherwise to the V/C element type. `rock.gemm_elementwise_gemm`
  //     never sets `softmaxType`, so for the GEG path this resolves to C.
  //   - The V tile (gemm1's B operand) is in the C element type.
  int64_t abBits =
      cast<ShapedType>(op.getAType()).getElementType().getIntOrFloatBitWidth();
  int64_t cBits =
      cast<ShapedType>(op.getCType()).getElementType().getIntOrFloatBitWidth();
  int64_t pBits = cBits;
  if (auto softmaxAttr = op->getAttrOfType<TypeAttr>("softmaxType"))
    pBits = softmaxAttr.getValue().getIntOrFloatBitWidth();

  // Phase A (gemm0): A tile (M x K) + B tile (K x N) live in LDS.
  int64_t phaseABits =
      abBits * (mPerBlockG0 * kPerBlock + nPerBlockG0 * kPerBlock);
  // Phase B (gemm1): P tile (M x K1) + V tile (K1 x N1) live in LDS.
  int64_t phaseBBits = pBits * (mPerBlockG0 * gemm1KPerBlock) +
                       cBits * (gemm1KPerBlock * gemm1NPerBlock);

  // The Triton allocator does liveness-based reuse across the phase boundary,
  // so peak LDS is roughly the larger of the two phases. We then multiply
  // each phase by `numStages` as an over-approximation of software
  // pipelining: in reality only the globally-loaded operand tiles need
  // `numStages` slots (A and B in phase A; V in phase B), while tiles
  // produced in LDS by an earlier phase (the P tile passed from gemm0 to
  // gemm1) do not. Applying the uniform multiplier may reject some configs
  // that would actually fit. Compute the footprint in bits (so sub-byte
  // element types like i4 aren't rounded up to 1 byte/elem) and convert to
  // bytes at the end. We do not account for kpack alignment padding or any
  // auxiliary persistent buffers (e.g. softmax accumulators), so this is a
  // best-effort estimate but it catches the
  // very bad cases (`PowerOf2Ceil(N1)` blowing up `gemm1NPerBlock`)
  // that the original LDS overflow at ResolveKernelLaunchPass exposed.
  int64_t peakBits = numStages * std::max(phaseABits, phaseBBits);
  return llvm::divideCeil(peakBits, int64_t{8});
}

LogicalResult mlir::rock::testFusionLegalityGemmGemmLDS(func::FuncOp func,
                                                        StringRef perfConfig) {
  // If the caller passed a gemm-gemm perfConfig, use it for every gemm-gemm
  // op in the function. Otherwise (empty string, or a non-gemm-gemm config
  // such as `gemm:v1:...` for a module that only contains `rock.gemm`), fall
  // back per op to whatever `obtainTuningParameters` would pick — which
  // honors any `perf_config` attribute on the op and otherwise uses the
  // hardcoded default. This keeps the budget check honest in the
  // no-perfConfig case (e.g., when MIGraphX hasn't tuned yet and the
  // lowering will pick the default tile).
  GemmGemmParamsAttr explicitParams;
  if (!perfConfig.empty()) {
    auto perfConfigAttr = StringAttr::get(func.getContext(), perfConfig);
    // Note: `GemmGemmParamsAttr::get` returns a null attribute for any
    // string that doesn't start with `attn:` (e.g., a `gemm:v1:...`
    // solution string for a rock.gemm kernel). That's intentional as
    // when the caller's perfConfig isn't a gemm-gemm config, `explicitParams`
    // stays null and the walk below falls back to `obtainTuningParameters`
    // per op.
    explicitParams = GemmGemmParamsAttr::get(perfConfigAttr);
  }

  OpBuilder builder(func.getContext());
  // Walk only the GEG ops; if the func has none (e.g., a `rock.gemm`-only
  // module funnelled through `isModuleFusible`, or a host helper sitting
  // next to a kernel) we return success immediately without touching
  // `rock.arch` or the LDS DB.
  WalkResult walkResult =
      func.walk([&](RockGemmGemmWrapperInterface gemmGemmOp) -> WalkResult {
        // The arch is a module-level attribute so it's the same for every
        // GEG op in `func`, but resolving it here keeps the no-GEG path
        // arch-free and matches the `llvm_unreachable` contract -- if we
        // reach a GEG op, MIGraphX has set `rock.arch` on the module.
        auto maybeArch = getArch(gemmGemmOp);
        if (failed(maybeArch))
          llvm_unreachable(
              "rock.arch missing on a kernel containing a "
              "RockGemmGemmWrapperInterface op; callers (e.g. MIGraphX) "
              "must set it before invoking isModuleFusible");
        int64_t maxLDS = getLDSSize(maybeArch->getValue());
        if (maxLDS <= 0)
          llvm_unreachable("getLDSSize returned non-positive for an arch "
                           "that getArch accepted; this should have failed "
                           "during target selection");

        GemmGemmParamsAttr params = explicitParams;
        if (!params) {
          auto maybeParams = PopulateParamsGemmGemm::obtainTuningParameters(
              builder, gemmGemmOp);
          if (failed(maybeParams))
            return WalkResult::interrupt();
          params = maybeParams.value();
        }

        int64_t peakLDS = estimateGemmGemmPeakLDSBytes(gemmGemmOp, params);
        if (peakLDS > maxLDS)
          return WalkResult::interrupt();
        return WalkResult::advance();
      });

  return success(!walkResult.wasInterrupted());
}

LogicalResult mlir::rock::testFusionLegalityGemmGemmLDS(ModuleOp mod,
                                                        StringRef perfConfig) {
  auto funcs = mod.getOps<func::FuncOp>();
  bool isFusible = true;
  for (auto f : funcs) {
    isFusible &= succeeded(testFusionLegalityGemmGemmLDS(f, perfConfig));
  }
  return success(isFusible);
}
