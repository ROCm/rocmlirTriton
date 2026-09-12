//===- AmdArchDb.cpp - Database of AMD GPU features ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/tritonUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

// Include Triton AMD APIs for intrinsic selection
#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonAMDGPUTransforms/WmmaGroup.h"

// triton::AMD::TargetInfo
#include "lib/TritonAMDGPUToLLVM/TargetInfo.h"

#include <cassert>

#define DEBUG_TYPE "rock-amd-arch-db"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::triton::amdgpu;

std::tuple<StringRef, unsigned> mlir::rock::parseArchString(StringRef arch) {
  std::tuple<StringRef, unsigned> ret("", 0);

  StringRef firstPart, remainingParts;
  std::tie(firstPart, remainingParts) = arch.split(':');
  auto chipPos = firstPart.find("gfx");
  if (chipPos != StringRef::npos) {
    firstPart = firstPart.substr(chipPos);
  } else {
    std::tie(firstPart, remainingParts) = remainingParts.split(':');
  }
  std::get<0>(ret) = firstPart;

  return ret;
}

std::tuple<ISAFamily, StringRef> mlir::rock::getArch(StringRef arch) {
  auto [chip, _] = parseArchString(arch);
  ISAFamily isaFamily = TargetFeatures(chip).getISAFamily();
  if (isaFamily == ISAFamily::Unknown) {
    llvm_unreachable("Unknown chip");
  }
  return std::make_tuple(isaFamily, chip);
}

bool mlir::rock::isCDNA(StringRef arch) {
  auto [isaFamily, _] = getArch(arch);
  return triton::amdgpu::isCDNA(isaFamily);
}

bool mlir::rock::isRDNA(StringRef arch) {
  auto [isaFamily, _] = getArch(arch);
  return triton::amdgpu::isRDNA(isaFamily);
}

//===----------------------------------------------------------------------===//
// Matrix Acceleration Support Detection (using Triton APIs)
//===----------------------------------------------------------------------===//

/// The M and N extent of every WMMA intrinsic, across RDNA3/RDNA4 and both the
/// plain and scaled flavours: only kDim varies (16, 32, 64, 128).
/// Mirror of the `wmmaMap` entries in
/// `external/triton/third_party/amd/lib/TritonAMDGPUTransforms/WmmaGroup.cpp`
static constexpr unsigned kWmmaNonKDim = 16;

/// Check if MFMA is supported for the given types on the specified version.
/// Triton's MfmaIntrinsic::selectFor() requires tile dimensions, but we only
/// want to know if ANY MFMA intrinsic exists for these types. We try all known
/// tile sizes that MFMA intrinsics use across AMD CDNA architectures.
///
/// Note: If AMD adds new MFMA tile sizes in future architectures, this list
/// would need to be updated. Consider proposing an "hasTypeSupport" API to
/// upstream Triton to eliminate this maintenance burden.
///
/// \param withScale If true, check for scaled MFMA support (CDNA4 with
///                  F8/F6/F4 types). Scaled MFMA only supports 16x16 and 32x32.
static bool hasMfmaSupport(Location loc, int mfmaVersion, Type elemA,
                           Type elemB, bool withScale) {
  // Scaled MFMA (with withScale=true in selectFor) is only available on CDNA4
  // (mfmaVersion 4). For gfx1250, scaled operations use WMMA, not MFMA.
  if (withScale && mfmaVersion < 4)
    return false;

  // All known MFMA tile sizes across CDNA1/CDNA2/CDNA3/CDNA4:
  // - 16x16: mfma_f32_16x16x*, mfma_i32_16x16x*,
  // mfma_scale_f32_16x16x128_f8f6f4
  // - 32x32: mfma_f32_32x32x*, mfma_i32_32x32x*, mfma_scale_f32_32x32x64_f8f6f4
  // - 4x64/64x4: specialized shapes for certain types (not for scaled)
  // For scaled MFMA, only 16x16 and 32x32 are available.
  static constexpr std::pair<unsigned, unsigned> regularMfmaTileSizes[] = {
      {16, 16}, {32, 32}, {4, 64}, {64, 4}};
  static constexpr std::pair<unsigned, unsigned> scaledMfmaTileSizes[] = {
      {16, 16}, {32, 32}};

  auto tileSizes = withScale ? llvm::ArrayRef(scaledMfmaTileSizes)
                             : llvm::ArrayRef(regularMfmaTileSizes);

  // Use a large K value (larger than any kDim in the database) so selectFor()
  // always matches if the type combination exists, regardless of actual K.
  constexpr unsigned largeK = 512;

  for (auto [mDim, nDim] : tileSizes) {
    auto result = MfmaIntrinsic::selectFor(loc, mfmaVersion, mDim, nDim, largeK,
                                           elemA, elemB, withScale,
                                           /*useTF32=*/false);
    if (succeeded(result))
      return true;
  }
  return false;
}

/// Check if scaled WMMA is supported for the given types.
/// gfx1250 (WMMA version 3) supports scaled operations for F8/F4 types using
/// WMMA intrinsics with kDim=128. Unlike scaled MFMA, there's no separate
/// "withScale" flag - the fp8/fp4 WMMA intrinsics ARE the scaled instructions.
static bool hasScaledWmmaSupport(int wmmaVersion, Type elemA, Type elemB,
                                 Type elemOut) {
  // Scaled WMMA is only available on gfx1250 (wmmaVersion 3)
  if (wmmaVersion < 3)
    return false;

  // Use large K to match any intrinsic - gfx1250 fp8 WMMA uses kDim=128
  constexpr unsigned largeK = 512;

  auto result = WmmaIntrinsic::selectFor(
      wmmaVersion, kWmmaNonKDim, kWmmaNonKDim, largeK, elemA, elemB, elemOut);
  return succeeded(result);
}

/// Check if WMMA is supported for the given types on the specified version.
static bool hasWmmaSupport(int wmmaVersion, Type elemA, Type elemB,
                           Type elemOut) {
  // Use a large K so selectFor() matches whichever kDim the table offers.
  constexpr unsigned largeK = 512;

  auto result = WmmaIntrinsic::selectFor(
      wmmaVersion, kWmmaNonKDim, kWmmaNonKDim, largeK, elemA, elemB, elemOut);
  return succeeded(result);
}

/// Check if a type is valid for scaled WMMA on gfx1250.
/// gfx1250 supports only E2M1, E4M3, E5M2 (NOT E3M2, E2M3).
/// See ScaledBlockedToScaledWMMAF8F6F4 in AccelerateAMDMatmul.cpp
static bool isScaledWmmaType(Type type) {
  return isa<Float8E4M3FNType, Float8E5M2Type, Float4E2M1FNType>(type);
}

MatrixAccelKind mlir::rock::getMatrixAccelKind(StringRef arch, Type inputTypeA,
                                               Type inputTypeB, Type scaleAType,
                                               Type scaleBType) {
  auto [isaFamily, _] = getArch(arch);

  // Get element types if these are shaped types
  Type elemA = getElementTypeOrSelf(inputTypeA);
  Type elemB = getElementTypeOrSelf(inputTypeB);

  // We always use f32 or i32 for output element
  Type elemOut = rock::getAccType(elemA, elemB);

  // We need an MLIRContext for creating a dummy location
  MLIRContext *ctx = elemA.getContext();
  Location loc = UnknownLoc::get(ctx);

  // Determine if scales are provided
  bool hasScales =
      static_cast<bool>(scaleAType) || static_cast<bool>(scaleBType);

  // Check MFMA support (CDNA architectures)
  int mfmaVersion = rock::getMfmaVersion(isaFamily);
  if (mfmaVersion > 0) {
    // Scaled MFMA requires: CDNA4 (version 4) + F8/F6/F4 input types
    // This mirrors AccelerateAMDMatmul.cpp:
    bool canUseScaledMfma =
        hasScales && mfmaVersion == 4 && isF8F6F4(elemA) && isF8F6F4(elemB);

    if (canUseScaledMfma &&
        hasMfmaSupport(loc, mfmaVersion, elemA, elemB, /*withScale=*/true)) {
      return MatrixAccelKind::ScaledMFMA;
    }
    // Fallback to regular MFMA (either no scales, or scaled failed)
    if (hasMfmaSupport(loc, mfmaVersion, elemA, elemB, /*withScale=*/false)) {
      return MatrixAccelKind::MFMA;
    }
  }

  // Check WMMA support (RDNA architectures)
  int wmmaVersion = rock::getWmmaVersion(isaFamily);
  if (wmmaVersion > 0) {
    // Scaled WMMA requires: gfx1250 (version 3) + specific types (E4M3, E5M2,
    // E2M1) Note: gfx1250 does NOT support E3M2 or E2M3 for scaled ops. See
    // supportsTypes() in ScaledBlockedToScaledWMMAF8F6F4
    bool canUseScaledWmma = hasScales && wmmaVersion >= 3 &&
                            isScaledWmmaType(elemA) && isScaledWmmaType(elemB);

    if (canUseScaledWmma &&
        hasScaledWmmaSupport(wmmaVersion, elemA, elemB, elemOut)) {
      return MatrixAccelKind::ScaledWMMA;
    }
    // Fallback to regular WMMA
    if (hasWmmaSupport(wmmaVersion, elemA, elemB, elemOut)) {
      return MatrixAccelKind::WMMA;
    }
  }

  return MatrixAccelKind::None;
}

MatrixAccelKind
mlir::rock::getMatrixAccelKind(StringRef arch,
                               RockGemmWrapperInterface gemmOp) {
  Type aType = gemmOp.getAType();
  Type bType = gemmOp.getBType();
  Type scaleAType = gemmOp.getScaleAType();
  Type scaleBType = gemmOp.getScaleBType();
  return getMatrixAccelKind(arch, aType, bType, scaleAType, scaleBType);
}

bool mlir::rock::hasAccel(StringRef arch,
                          RockGemmGemmWrapperInterface gemmGemmOp) {
  auto [firstGemm, secondGemm] = getMatrixAccelKind(arch, gemmGemmOp);
  return firstGemm != MatrixAccelKind::None &&
         secondGemm != MatrixAccelKind::None;
}

std::pair<MatrixAccelKind, MatrixAccelKind>
mlir::rock::getMatrixAccelKind(StringRef arch,
                               RockGemmGemmWrapperInterface gemmGemmOp) {
  Type aType = gemmGemmOp.getAType();
  Type bType = gemmGemmOp.getBType();
  Type cType = gemmGemmOp.getCType();

  // TODO: no scaled gemms for attention yet
  Type scaleAType = nullptr;
  Type scaleBType = nullptr;
  auto kindFirstGemm =
      getMatrixAccelKind(arch, aType, bType, scaleAType, scaleBType);
  // we convert the output of A*B to cType
  auto kindSecondGemm =
      getMatrixAccelKind(arch, cType, cType, scaleAType, scaleBType);

  return std::make_pair(kindFirstGemm, kindSecondGemm);
}

bool mlir::rock::hasAccel(StringRef arch, RockGemmWrapperInterface gemmOp) {
  return getMatrixAccelKind(arch, gemmOp) != MatrixAccelKind::None;
}

FailureOr<int64_t>
mlir::rock::getAccelInstrMinKDim(StringRef arch, Type inputTypeA,
                                 Type inputTypeB, uint32_t instrNonKDim,
                                 Type scaleAType, Type scaleBType) {
  MatrixAccelKind accelKind =
      getMatrixAccelKind(arch, inputTypeA, inputTypeB, scaleAType, scaleBType);
  if (accelKind == MatrixAccelKind::None)
    return failure();

  Type elemA = getElementTypeOrSelf(inputTypeA);
  Type elemB = getElementTypeOrSelf(inputTypeB);
  Type elemOut = rock::getAccType(elemA, elemB);
  auto [isaFamily, _] = getArch(arch);

  // Both selectFor()s walk their candidates widest-K first and then fall
  // through to "the only / smallest-K intrinsic", so an input K of zero skips
  // every candidate and lands on exactly the one we are asking for.
  constexpr unsigned narrowest = 0;

  if (accelKind == MatrixAccelKind::WMMA ||
      accelKind == MatrixAccelKind::ScaledWMMA) {
    auto instr = WmmaIntrinsic::selectFor(rock::getWmmaVersion(isaFamily),
                                          kWmmaNonKDim, kWmmaNonKDim, narrowest,
                                          elemA, elemB, elemOut);

    assert(succeeded(instr) && "WMMA arch has no intrinsic for its own types");
    if (failed(instr))
      return failure();
    return instr->kDim;
  }

  MLIRContext *ctx = elemA.getContext();
  auto instr = MfmaIntrinsic::selectFor(
      UnknownLoc::get(ctx), rock::getMfmaVersion(isaFamily), instrNonKDim,
      instrNonKDim, narrowest, elemA, elemB,
      /*withScale=*/accelKind == MatrixAccelKind::ScaledMFMA,
      /*useTF32=*/false);
  if (failed(instr))
    return failure();
  return instr->kDim;
}

FailureOr<int64_t> mlir::rock::getAccelInstrMinKDim(
    StringRef arch, RockGemmWrapperInterface gemmOp, uint32_t instrNonKDim) {
  return getAccelInstrMinKDim(arch, gemmOp.getAType(), gemmOp.getBType(),
                              instrNonKDim, gemmOp.getScaleAType(),
                              gemmOp.getScaleBType());
}

bool mlir::rock::archSupportsAccelFp8(StringRef arch) {
  // Hardware-capability check via the underlying MFMA / WMMA version tables.
  // We deliberately do NOT probe through getMatrixAccelKind here, because
  // Triton's composeMfmaKeyFor silently rewrites OCP FP8 (E4M3FN / E5M2)
  // inputs to f16 on any MFMA v<=3 (see MfmaGroup.cpp), which would make
  // such a probe report MFMA success on CDNA1 / CDNA2 / CDNA3 even though
  // those archs have no native FP8 intrinsics. The version cut-offs below
  // mirror the Triton MFMA / WMMA databases:
  //   - MFMA v3 (CDNA3 / gfx942) : FNUZ FP8 native.
  //   - MFMA v4 (CDNA4 / gfx950) : FNUZ + OCP FP8 native.
  //   - WMMA v2 (RDNA4 / gfx12*) : OCP FP8 native.
  //   - WMMA v3 (gfx1250)        : OCP FP8 native.
  auto [isaFamily, _] = getArch(arch);
  return rock::getMfmaVersion(isaFamily) >= 3 ||
         rock::getWmmaVersion(isaFamily) >= 2;
}

bool mlir::rock::archSupportsScaledGemm(StringRef arch) {
  // Probe both FP8 variants via getMatrixAccelKind with a non-null scale type
  // so this stays in sync with the actual ScaledMFMA / ScaledWMMA dispatch:
  // CDNA4 (gfx950) handles scaled MFMA over F8/F6/F4; GFX1250 handles scaled
  // WMMA over F8E4M3 / F8E5M2 / F4E2M1.
  MLIRContext ctx;
  Builder b(&ctx);
  Type scale = b.getF8E5M2Type();
  for (Type fp8 : {Type(Float8E4M3FNUZType::get(&ctx)),
                   Type(Float8E4M3FNType::get(&ctx))}) {
    MatrixAccelKind k = getMatrixAccelKind(arch, fp8, fp8, scale, scale);
    if (k == MatrixAccelKind::ScaledMFMA || k == MatrixAccelKind::ScaledWMMA)
      return true;
  }
  return false;
}

bool mlir::rock::archSupportsNonKPackedScaledInput(StringRef arch) {
  // Only CDNA4 (gfx950) scaled MFMA implements that path (via
  // ds_load_tr_b4). GFX1250 scaled WMMA supports this in hardware,
  // but Triton does not currently support it.
  //
  // NOTE 1: the "K-pack" here is different from the `kpack` in the perf-config
  // This kpack is the per-operand bool `matrixA/BKPack` (-> `lhs/rhs_k_pack` on
  // tt.dot_scaled): for a sub-byte (fp4) operand it says whether the two 4-bit
  // values packed into an i8 are adjacent along K (true) or along M/N (false).
  //
  // NOTE 2: Triton supports emulating non-K-packed scaled input in software via
  // DecomposeScaledBlocked, but this is currently broken on gfx1250. It crashes
  // with: error: 'ttg.convert_layout' op requires the same shape for all
  // operands and results
  //
  // TODO: In the future, whenever DecomposeScaledBlocked is fixed on gfx1250,
  // or the native path for gfx1250 is implemented, we should add gfx1250 here.
  auto [isaFamily, _] = getArch(arch);
  return isaFamily == ISAFamily::CDNA4;
}

int64_t mlir::rock::getMaxNumChiplets(StringRef arch) {
  auto [isaFamily, _] = getArch(arch);

  switch (isaFamily) {
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
  case ISAFamily::GFX1250:
    return 8;
  default:
    return 1;
  }
  return 1;
}

int64_t mlir::rock::inferNumChiplets(StringRef arch, int64_t numCUs) {
  auto [isaFamily, _] = getArch(arch);
  switch (isaFamily) {
  case ISAFamily::CDNA3:
    if (numCUs == 304)
      return 8;
    if (numCUs == 228) // MI300A: 6 XCDs of 38 CUs
      return 6;
    if (numCUs == 80)
      return 4;
    return 1;
  case ISAFamily::CDNA4:
    if (numCUs == 256)
      return 8;
    if (numCUs == 128)
      return 4;
    if (numCUs == 64)
      return 2;
    return 1;
  case ISAFamily::GFX1250:
    if (numCUs == 256)
      return 8;
    return 1;
  case ISAFamily::Unknown:
  case ISAFamily::GCN5_1:
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::RDNA1:
  case ISAFamily::RDNA2:
  case ISAFamily::RDNA3:
  case ISAFamily::GFX1170:
  case ISAFamily::RDNA4:
    return 1;
  }
  llvm_unreachable("unhandled ISAFamily in inferNumChiplets");
}

int64_t mlir::rock::getMinNumCU(StringRef arch) {
  auto [isaFamily, _] = getArch(arch);

  switch (isaFamily) {
  case ISAFamily::GCN5_1:
    return 10;
  case ISAFamily::CDNA1:
    return 120;
  case ISAFamily::CDNA2:
    return 104;
  case ISAFamily::CDNA3:
    return 20;
  case ISAFamily::CDNA4:
    // CPX, the narrowest compute-partition mode, exposes one XCD of 32 CUs.
    return 32;
  case ISAFamily::RDNA1:
    return 20;
  case ISAFamily::RDNA2:
    return 2;
  case ISAFamily::RDNA3:
  case ISAFamily::GFX1170:
    return 2;
  case ISAFamily::RDNA4:
    return 12;
  case ISAFamily::GFX1250:
    return 256;
  case ISAFamily::Unknown:
    return 1;
  }
  llvm_unreachable("unhandled ISAFamily in getMinNumCU");
}

int64_t mlir::rock::getDefaultNumCU(StringRef arch) {
  auto [isaFamily, _] = getArch(arch);

  // SPX, the unpartitioned mode, exposes all 8 XCDs.
  if (isaFamily == ISAFamily::CDNA4)
    return 256;

  // Every other family still hands back what used to double as the validation
  // floor, so a family that spans APUs (CDNA3, RDNA2, RDNA3) assumes an
  // implausibly small device. Raising those changes the tuning heuristics for
  // everyone who does not pass a CU count, so it wants its own change with
  // perf numbers behind it. getNativeNumCU supplies the real count whenever a
  // matching device is visible, so these values only apply when no device can
  // be queried at all.
  return getMinNumCU(arch);
}

int64_t mlir::rock::getWaveSize(StringRef arch) {
  auto [_, chip] = getArch(arch);
  triton::AMD::TargetInfo targetInfo(chip.str());
  return targetInfo.getWarpSize();
}

int64_t mlir::rock::getLDSSize(StringRef arch) {
  auto [_, chip] = getArch(arch);
  triton::AMD::TargetInfo targetInfo(chip.str());
  return targetInfo.getSharedMemorySize();
}

int64_t mlir::rock::getLastLevelCacheSize(StringRef arch) {
  auto [isaFamily, chip] = getArch(arch);

  constexpr int64_t kKiB = 1024;
  constexpr int64_t kMiB = 1024 * kKiB;

  // The gfx10xx / gfx11xx / gfx12xx ISA families each span both discrete parts
  // with a memory-attached last level cache (Infinity Cache / MALL) and APUs,
  // most of which stop at an L2 of 2 MiB or less - though gfx1151 shows an APU
  // can carry a MALL too. Either way the answer is not a property of the
  // family, whose maximum overshoots a MALL-less APU by up to 512x. Chips whose
  // configuration is published are therefore listed individually. Where a chip
  // ships in several cache configurations (gfx1100 at 96/80/64 MiB, gfx1201 at
  // 64/48 MiB, ...) the largest is used, so that a cache-flush buffer sized
  // from this always covers the live device.
  //
  // gfx11xx / gfx12xx sizes from
  // https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html, which
  // does not cover gfx10xx; those come from the per-chip tables amdgpu reports
  // to userspace in drivers/gpu/drm/amd/amdkfd/kfd_crat.c (cache_level 3 where
  // a chip has a MALL, otherwise cache_level 2). Both sources index by product
  // or codename rather than by gfx number, so looking a figure up again means
  // mapping the target back to its chip first.
  int64_t perChipSize =
      llvm::StringSwitch<int64_t>(chip)
          // RDNA1: no MALL anywhere in the family, L2 is the last level.
          .Case("gfx1010", 4 * kMiB)
          .Case("gfx1011", 4 * kMiB)
          .Case("gfx1012", 2 * kMiB)
          .Case("gfx1013", 4 * kMiB)
          // RDNA2 discrete: Infinity Cache.
          .Case("gfx1030", 128 * kMiB)
          .Case("gfx1031", 96 * kMiB)
          .Case("gfx1032", 32 * kMiB)
          .Case("gfx1034", 16 * kMiB)
          // RDNA2 APUs: no MALL. gfx1036 is a 2 CU part and its L2 is sized to
          // match, so it is the smallest last level cache of any target here.
          .Case("gfx1033", 1 * kMiB)
          .Case("gfx1035", 2 * kMiB)
          .Case("gfx1036", 256 * kKiB)
          // RDNA3 discrete: Infinity Cache.
          .Case("gfx1100", 96 * kMiB)
          .Case("gfx1101", 64 * kMiB)
          .Case("gfx1102", 32 * kMiB)
          // RDNA3 / RDNA3.5 APUs. An APU is not automatically MALL-less: of
          // these, only gfx1151 has one and the rest stop at their L2.
          .Case("gfx1103", 2 * kMiB)
          .Case("gfx1150", 2 * kMiB)
          .Case("gfx1151", 32 * kMiB) // MALL over a 2 MiB L2
          .Case("gfx1152", 1 * kMiB)
          // TODO(gfx1153): guess, AMD has not published this chip's cache. It
          // is the smallest RDNA3.5 APU, which bounds its L2 above by gfx1152's
          // 1 MiB, and gfx1151 is the only RDNA3.5 APU *known* to carry a MALL
          // - a MALL here would make this value far too small. Replace it once
          // real numbers exist.
          .Case("gfx1153", 1 * kMiB)
          // RDNA4 discrete: Infinity Cache.
          .Case("gfx1200", 32 * kMiB)
          .Case("gfx1201", 64 * kMiB)
          .Default(0);
  if (perChipSize != 0)
    return perChipSize;

  // Chips absent from the table above fall back to their ISA family. This is
  // exact for the single-chip CDNA / GCN5 families and an upper bound
  // elsewhere.
  //
  // TODO(gfx1170, gfx1171, gfx1172): AMD has not published the caches of the
  // gfx117x parts, and being APUs settles nothing - gfx1151 is an APU with a
  // 32 MiB MALL. The family value below is a guess at a bare APU L2, so it
  // under-reports any of these that turns out to carry a MALL, which would
  // under-size a cache-flush buffer rather than merely blunt a load hint. Give
  // each one a per-chip entry as soon as real numbers exist.
  switch (isaFamily) {
  // No Infinity Cache: L2 is the last level (largest L2 in the family).
  case ISAFamily::GCN5_1:
  case ISAFamily::RDNA1:
    return 4 * kMiB;
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2: // per-GCD
    return 8 * kMiB;
  // Infinity Cache.
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return 256 * kMiB;
  case ISAFamily::GFX1250:
    return 192 * kMiB;
  case ISAFamily::RDNA2:
    return 128 * kMiB;
  case ISAFamily::RDNA3:
    return 96 * kMiB;
  case ISAFamily::GFX1170: // guess, see the TODO above
    return 1 * kMiB;
  case ISAFamily::RDNA4:
    return 64 * kMiB;
  case ISAFamily::Unknown: // Unknown arch: assume Infinity-Cache-class LLC.
    return 256 * kMiB;
  }
  llvm_unreachable("unhandled ISAFamily in getLastLevelCacheSize");
}

int64_t mlir::rock::getMaxWavesPerEU(StringRef arch) {
  auto [isaFamily, _] = getArch(arch);

  switch (isaFamily) {
  case ISAFamily::GCN5_1:
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return 8;
    break;
  case ISAFamily::RDNA1:
  case ISAFamily::RDNA2:
  case ISAFamily::RDNA3:
  case ISAFamily::GFX1170:
  case ISAFamily::RDNA4:
  case ISAFamily::GFX1250:
    return 16;
    break;
  default:
    return 1;
  }
  return 1;
}

int64_t mlir::rock::getVGPRsPerEU(StringRef arch) {
  auto [isaFamily, chip] = getArch(arch);

  switch (isaFamily) {
  case ISAFamily::GCN5_1:
  case ISAFamily::CDNA1:
    return 256;
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return 512;
  case ISAFamily::RDNA1:
  case ISAFamily::RDNA2:
    return 1024;
  case ISAFamily::RDNA3:
    // Match LLVM's coverage: getTotalNumVGPRs() in
    // llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp and the
    // Feature1536VGPRs definition / FeatureISAVersion11_* lists in
    // llvm/lib/Target/AMDGPU/AMDGPU.td.
    if (chip == "gfx1100" || chip == "gfx1101" || chip == "gfx1151")
      return 1536;
    return 1024;
  case ISAFamily::GFX1170:
    return 1024;
  case ISAFamily::RDNA4:
    return 1536;
  case ISAFamily::GFX1250:
    return 1024;
  case ISAFamily::Unknown:
    return 512;
  }
  llvm_unreachable("unhandled ISAFamily in getVGPRsPerEU");
}

bool mlir::rock::supportsMultiCTALaunch(StringRef arch) {
  auto [_, chip] = getArch(arch);
  triton::AMD::TargetInfo targetInfo(chip.str());
  return targetInfo.supportsMultiCTALaunch();
}

int64_t mlir::rock::getMaxNumCTAs(StringRef arch) {
  if (!supportsMultiCTALaunch(arch))
    return 1;
  return 16;
}

// Triton's HIPOptions currently normalizes kpack to 1 only for gfx950 in
// compiler.py. Instead of updating Pipelines.cpp, we make this function
// available here since rock uses this helper earlier for tuning and perf-config
// validation, and applies the same unsupported policy forward to gfx950 and
// future archs instead of generating configs that the backend should not use.
int64_t mlir::rock::getMaxKpack(StringRef arch) {
  // kpack != 1 is unsupported on gfx950 and gfx1250 (and any newer arch);
  // older archs (gfx9 < gfx950, all of gfx10/gfx11, gfx12 < gfx1250) still
  // accept kpack in {1, 2}.
  auto [chip, _] = parseArchString(arch);
  // consume_front does double duty: it checks for the "gfx" prefix and
  // strips it from `chip` in-place when present
  if (!chip.consume_front("gfx"))
    return 1; // not a gfx target -> safest

  // Parse the stripped digits as hex (e.g. "950" -> n = 0x950).
  unsigned n = 0;
  if (chip.getAsInteger(/*radix=*/16, n))
    return 1; // malformed id (e.g. "gfx" alone) -> safest

  if (n < 0x950) // gfx9 pre-CDNA4 (e.g., gfx908/90a)
    return 2;

  if (0x1000 <= n && n < 0x1250) // all of gfx10/gfx11 + gfx12 < gfx1250
    return 2;

  return 1; // gfx950+, gfx1250+, gfx13xx+, anything else
}

// A non-power-of-two kPerBlock makes rock-gridwise-gemm-to-blockwise peel the
// K loop into several power-of-two segments. That peeled form currently
// miscompiles on gfx950 because of an unfixed LLVM backend bug, so we neither
// tune nor accept such a kPerBlock there.
// TODO: Enable this on gfx950 too once the LLVM bug is fixed.
bool mlir::rock::supportsNonPow2KPerBlock(StringRef arch) {
  auto [chip, _] = parseArchString(arch);
  return chip != "gfx950";
}

// Decomposing an f32 dot into three bf16 products trades the f32 MFMA
// for the higher throughput bf16 ops. CDNA4 was measured to perform better
// overall with this decomposition.
bool mlir::rock::preferBf16x3ForF32Dot(StringRef arch) {
  auto [isaFamily, _] = getArch(arch);
  return isaFamily == ISAFamily::CDNA4;
}

bool mlir::rock::supportsTDM(StringRef arch) {
  auto [_, chip] = getArch(arch);
  triton::AMD::TargetInfo targetInfo(chip.str());
  return targetInfo.supportsTDM();
}
