//===- AmdArchDb.cpp - Database of AMD GPU features ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

// Include Triton AMD APIs for intrinsic selection
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonAMDGPUTransforms/WmmaGroup.h"

// triton::AMD::TargetInfo
#include "lib/TritonAMDGPUToLLVM/TargetInfo.h"

#define DEBUG_TYPE "rock-amd-arch-db"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::triton::AMD;

static std::tuple<StringRef, unsigned> parseArchString(StringRef arch) {
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

static std::tuple<ISAFamily, StringRef> getArch(StringRef arch) {
  auto [chip, _] = parseArchString(arch);
  ISAFamily isaFamily = triton::AMD::deduceISAFamily(chip);
  if (isaFamily == ISAFamily::Unknown) {
    llvm_unreachable("Unknown chip");
  }
  return std::make_tuple(isaFamily, chip);
}

//===----------------------------------------------------------------------===//
// Matrix Acceleration Support Detection (using Triton APIs)
//===----------------------------------------------------------------------===//

namespace {

// getMfmaVersion and getWmmaVersion are internal triton functions in
// AccelerateAMDMatmul.cpp keep them in sync.

/// Get MFMA version from ISA family
int getMfmaVersion(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::CDNA1:
    return 1;
  case ISAFamily::CDNA2:
    return 2;
  case ISAFamily::CDNA3:
    return 3;
  case ISAFamily::CDNA4:
    return 4;
  default:
    return 0;
  }
}

/// Get WMMA version from architecture string
int getWmmaVersion(StringRef arch) {
  if (arch.starts_with("gfx11"))
    return 1; // RDNA3
  if (arch.starts_with("gfx12") && !arch.ends_with("50"))
    return 2; // RDNA4
  if (arch == "gfx1250")
    return 3; // GFX1250
  return 0;
}
} // anonymous namespace

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
  // - 16x16: mfma_f32_16x16x*, mfma_i32_16x16x*, mfma_scale_f32_16x16x128_f8f6f4
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

  auto result = WmmaIntrinsic::selectFor(wmmaVersion, /*mDim=*/16, /*nDim=*/16,
                                         largeK, elemA, elemB, elemOut);
  return succeeded(result);
}

/// Check if WMMA is supported for the given types on the specified version.
/// All WMMA intrinsics across RDNA3/RDNA4 use 16x16 tiles (only kDim varies).
static bool hasWmmaSupport(int wmmaVersion, Type elemA, Type elemB,
                           Type elemOut) {
  // WMMA has consistently used 16x16 tiles across all RDNA architectures.
  // The kDim varies (16, 32, 64, 128) but selectFor handles that internally.
  constexpr unsigned largeK = 512;

  auto result = WmmaIntrinsic::selectFor(wmmaVersion, /*mDim=*/16, /*nDim=*/16,
                                         largeK, elemA, elemB, elemOut);
  return succeeded(result);
}

/// Check if a type is valid for scaled WMMA on gfx1250.
/// gfx1250 supports only E2M1, E4M3, E5M2 (NOT E3M2, E2M3).
/// See ScaledBlockedToScaledWMMAF8F6F4 in AccelerateAMDMatmul.cpp
static bool isScaledWmmaType(Type type) {
  return isa<Float8E4M3FNType, Float8E5M2Type, Float4E2M1FNType>(type);
}

MatrixAccelKind mlir::rock::getMatrixAccelKind(StringRef arch, Type inputTypeA,
                                               Type inputTypeB, Type outputType,
                                               Type scaleAType,
                                               Type scaleBType) {
  auto [isaFamily, chip] = getArch(arch);

  // Get element types if these are shaped types
  Type elemA = getElementTypeOrSelf(inputTypeA);
  Type elemB = getElementTypeOrSelf(inputTypeB);
  Type elemOut = getElementTypeOrSelf(outputType);

  // We need an MLIRContext for creating a dummy location
  MLIRContext *ctx = elemA.getContext();
  Location loc = UnknownLoc::get(ctx);

  // Determine if scales are provided
  bool hasScales = static_cast<bool>(scaleAType) || static_cast<bool>(scaleBType);

  // Check MFMA support (CDNA architectures)
  int mfmaVersion = getMfmaVersion(isaFamily);
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
  int wmmaVersion = getWmmaVersion(chip);
  if (wmmaVersion > 0) {
    // Scaled WMMA requires: gfx1250 (version 3) + specific types (E4M3, E5M2, E2M1)
    // Note: gfx1250 does NOT support E3M2 or E2M3 for scaled ops.
    // See supportsTypes() in ScaledBlockedToScaledWMMAF8F6F4
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

MatrixAccelKind mlir::rock::getMatrixAccelKind(StringRef arch,
                                               RockGemmWrapperInterface gemmOp) {
  Type aType = gemmOp.getAType();
  Type bType = gemmOp.getBType();
  Type cType = gemmOp.getCType();
  Type scaleAType = gemmOp.getScaleAType();
  Type scaleBType = gemmOp.getScaleBType();
  return getMatrixAccelKind(arch, aType, bType, cType, scaleAType, scaleBType);
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
  Type outputType = gemmGemmOp.getOutType();

  // TODO: no scaled gemms for attention yet
  Type scaleAType = nullptr;
  Type scaleBType = nullptr;
  auto kindFirstGemm = getMatrixAccelKind(arch, aType, bType, outputType,
                                          scaleAType, scaleBType);
  // we convert the output of A*B to cType
  auto kindSecondGemm = getMatrixAccelKind(arch, cType, cType, outputType,
                                           scaleAType, scaleBType);

  return std::make_pair(kindFirstGemm, kindSecondGemm);
}

bool mlir::rock::hasAccel(StringRef arch, RockGemmWrapperInterface gemmOp) {
  return getMatrixAccelKind(arch, gemmOp) != MatrixAccelKind::None;
}

bool mlir::rock::isFastAtomicAddSupported(StringRef arch, Type type) {
  auto [isaFamily, _] = getArch(arch);

  Type elem = getElementTypeOrSelf(type);
  if (elem.isF32()) {
    switch (isaFamily) {
    case ISAFamily::CDNA1:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA3:
    case ISAFamily::CDNA4:
    case ISAFamily::RDNA1:
    case ISAFamily::RDNA2:
    case ISAFamily::RDNA3:
    case ISAFamily::RDNA4:
    case ISAFamily::GFX1250:
      return true;
    default:
      return false;
    }
  } else if (elem.isF16()) {
    switch (isaFamily) {
    case ISAFamily::CDNA1:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA3:
    case ISAFamily::CDNA4:
    case ISAFamily::RDNA4:
    case ISAFamily::GFX1250:
      return true;
    default:
      return false;
    }
  } else if (elem.isBF16()) {
    switch (isaFamily) {
    case ISAFamily::CDNA4:
    case ISAFamily::RDNA4:
    case ISAFamily::GFX1250:
      return true;
    default:
      return false;
    }
  }
  return false;
}

bool mlir::rock::isFastAtomicMaxSupported(StringRef arch, Type type) {
  auto [isaFamily, _] = getArch(arch);

  Type elem = getElementTypeOrSelf(type);
  if (elem.isF32()) {
    switch (isaFamily) {
    case ISAFamily::RDNA1:
    case ISAFamily::RDNA2:
    case ISAFamily::RDNA3:
    case ISAFamily::RDNA4:
    case ISAFamily::GFX1250:
      return true;
    default:
      return false;
    }
  }
  return false;
}

int64_t mlir::rock::getMinNumCU(StringRef arch) {
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

int64_t mlir::rock::getMaxNumChiplets(StringRef arch) {
  auto [isaFamily, _] = getArch(arch);

  switch (isaFamily) {
  case ISAFamily::CDNA1:
    return 120;
    break;
  case ISAFamily::CDNA2:
    return 104;
    break;
  case ISAFamily::CDNA3:
    return 20;
    break;
  case ISAFamily::CDNA4:
    return 256;
    break;
  case ISAFamily::RDNA1:
  case ISAFamily::RDNA2:
    return 30;
    break;
  case ISAFamily::RDNA3:
    return 2;
    break;
  case ISAFamily::RDNA4:
    return 12;
    break;
  case ISAFamily::GFX1250:
    return 256;
    break;
  default:
    return 1;
  }
  return 1;
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

int64_t mlir::rock::getMaxWavesPerEU(StringRef arch) {
  auto [isaFamily, _] = getArch(arch);

  switch (isaFamily) {
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return 8;
    break;
  case ISAFamily::RDNA1:
  case ISAFamily::RDNA2:
  case ISAFamily::RDNA3:
  case ISAFamily::RDNA4:
  case ISAFamily::GFX1250:
    return 16;
    break;
  default:
    return 1;
  }
  return 1;
}
