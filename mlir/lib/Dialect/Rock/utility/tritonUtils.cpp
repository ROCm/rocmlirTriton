//===- tritonUtils.cpp - Triton-dependent utilities for Rock --------------===//
//
// Centralizes C++ replicas of Triton-internal functions that must be kept in
// sync on every Triton version bump.  See tritonUtils.h for upstream sources.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/tritonUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonAMDGPUTransforms/WmmaGroup.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::triton::amdgpu;

namespace mlir {
namespace rock {

// Keep in sync with AccelerateAMDMatmul.cpp::getMfmaVersion()
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

// Keep in sync with AccelerateAMDMatmul.cpp::getWmmaVersion()
int getWmmaVersion(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::RDNA3:
    return 1;
  // gfx1170 uses the gfx12/RDNA4-style 128-bit WMMA layout
  // (FeatureWMMA128bInsts + FeatureSWMMACGfx1200Insts in LLVM), so it selects
  // WMMA v2 like RDNA4. Keep in sync with AccelerateAMDMatmul.cpp.
  case ISAFamily::GFX1170:
  case ISAFamily::RDNA4:
    return 2;
  case ISAFamily::GFX1250:
    return 3;
  default:
    break;
  }
  return 0;
}

// Keep in sync with TT_Float in TritonTypes.td.
bool isTTFloat(Type t) {
  return isa<Float8E4M3FNType, Float8E4M3FNUZType, Float8E5M2Type,
             Float8E5M2FNUZType, Float16Type, BFloat16Type, Float32Type,
             Float64Type>(t);
}

// Keep in sync with AccelerateAMDMatmul.cpp::mlirTypeToScaledElemType()
// Extended with BF16/FP16 coverage.
FailureOr<triton::ScaleDotElemType> mlirTypeToScaleDotElemType(Type type) {
  return llvm::TypeSwitch<Type, FailureOr<triton::ScaleDotElemType>>(type)
      .Case<Float8E4M3FNType>(
          [](Type) { return triton::ScaleDotElemType::E4M3; })
      .Case<Float8E5M2Type>([](Type) { return triton::ScaleDotElemType::E5M2; })
      .Case<Float6E2M3FNType>(
          [](Type) { return triton::ScaleDotElemType::E2M3; })
      .Case<Float6E3M2FNType>(
          [](Type) { return triton::ScaleDotElemType::E3M2; })
      .Case<Float4E2M1FNType>(
          [](Type) { return triton::ScaleDotElemType::E2M1; })
      .Case<BFloat16Type>([](Type) { return triton::ScaleDotElemType::BF16; })
      .Case<Float16Type>([](Type) { return triton::ScaleDotElemType::FP16; })
      .Default([](Type) { return failure(); });
}

/// Whether any matrix-core intrinsic on `isaFamily` can serve a dot with these
/// element types and this contraction length.
///
/// Keep in sync with AccelerateAMDMatmul.cpp::chooseMfmaInstruction(),
/// chooseWmmaInstruction() and the BlockedToWMMA pattern. Unlike
/// `chooseMfmaInstruction`, which derives a single mDim/nDim from the result
/// shape (or from the forced `matrixInstructionSize`), every tile shape is tried
/// here so that a forced non-K dimension cannot make this return false for an
/// accelerable dot. The tile list matches AmdArchDb.cpp::hasMfmaSupport().
static bool hasMatrixCoreForK(ISAFamily isaFamily, Type aElemTy, Type bElemTy,
                              Type cElemTy, int64_t kDim) {
  if (kDim <= 0 || kDim > std::numeric_limits<unsigned>::max())
    return false;
  auto inputKDim = static_cast<unsigned>(kDim);

  MLIRContext *ctx = aElemTy.getContext();
  // The selection helpers below emit performance remarks (fp8 emulation, and
  // the rejections these probes expect to hit) that belong to the real
  // lowering, not to a speculative query, so drop whatever they report.
  ScopedDiagnosticHandler discardProbeDiagnostics(
      ctx, [](Diagnostic &) { return success(); });

  if (int mfmaVersion = getMfmaVersion(isaFamily); mfmaVersion > 0) {
    static constexpr std::pair<unsigned, unsigned> mfmaTileSizes[] = {
        {16, 16}, {32, 32}, {4, 64}, {64, 4}};
    Location loc = UnknownLoc::get(ctx);
    for (auto [mDim, nDim] : mfmaTileSizes) {
      FailureOr<MfmaIntrinsic> intrinsic =
          MfmaIntrinsic::selectFor(loc, mfmaVersion, mDim, nDim, inputKDim,
                                   aElemTy, bElemTy, /*withScale=*/false,
                                   /*useTF32=*/false);
      // `selectFor` falls back to the smallest-K intrinsic of a matching type
      // pair, so it also succeeds for a K shorter than every candidate. It is
      // `chooseMfmaInstruction` that then rejects a kDim not dividing K,
      // because selecting it would duplicate data.
      if (succeeded(intrinsic) && intrinsic->kDim != 0 &&
          inputKDim % intrinsic->kDim == 0)
        return true;
    }
  }

  // WMMA has consistently used 16x16 tiles across all RDNA architectures, and
  // only its kDim varies. `chooseWmmaInstruction` deliberately has no
  // counterpart to the MFMA divisibility rejection above, so any K that
  // BlockedToWMMA does not turn away for being 1 is accelerable here.
  if (int wmmaVersion = getWmmaVersion(isaFamily); wmmaVersion > 0) {
    if (inputKDim > 1 &&
        succeeded(WmmaIntrinsic::selectFor(wmmaVersion, /*mDim=*/16,
                                           /*nDim=*/16, inputKDim, aElemTy,
                                           bElemTy, cElemTy)))
      return true;
  }
  return false;
}

/// The packed-`v_dot` half of AccelerateAMDMatmul.cpp::isLegalFMAForm(): the
/// type/K combinations that FMA.cpp::chooseIntrinsic() turns into a `v_dot4` or
/// `v_dot2`. The f16 x f16 -> f16 case upstream guards with `if (false)` is
/// omitted for the same reason it is disabled there.
static bool isPackedVDotForm(ISAFamily isaFamily, Type aElemTy, Type bElemTy,
                             Type cElemTy, int64_t kDim) {
  if (aElemTy.isF16() && bElemTy.isF16() && cElemTy.isF32() && kDim % 2 == 0)
    return true;
  // CDNA4 has bf16 v_dot2.
  if (isaFamily == ISAFamily::CDNA4 && aElemTy.isBF16() && bElemTy.isBF16() &&
      cElemTy.isF32() && kDim % 2 == 0)
    return true;
  if (aElemTy.isInteger(8) && bElemTy.isInteger(8) && cElemTy.isInteger(32) &&
      kDim % 4 == 0)
    return true;
  return false;
}

/// The trailing loop of AccelerateAMDMatmul.cpp::isLegalFMAForm(): every
/// operand already shares one element type and that type has a scalar
/// `llvm.fmuladd`, so `tryLegalizeFMA` leaves the dot alone.
static bool isScalarFMAForm(Type aElemTy, Type bElemTy, Type cElemTy) {
  if (aElemTy != bElemTy || aElemTy != cElemTy)
    return false;
  return aElemTy.isF16() || aElemTy.isF32() || aElemTy.isF64();
}

DotLowering classifyDotLowering(ISAFamily isaFamily, Type aElemTy, Type bElemTy,
                                Type cElemTy, int64_t kDim) {
  if (hasMatrixCoreForK(isaFamily, aElemTy, bElemTy, cElemTy, kDim))
    return DotLowering::MatrixCore;
  if (isPackedVDotForm(isaFamily, aElemTy, bElemTy, cElemTy, kDim))
    return DotLowering::PackedVDot;
  if (isScalarFMAForm(aElemTy, bElemTy, cElemTy))
    return DotLowering::ScalarFMA;
  return DotLowering::UpcastedFMA;
}

int64_t getRequiredDotKMultiple(ISAFamily isaFamily, Type aElemTy, Type bElemTy,
                                Type cElemTy) {
  // Mirrors the K conditions in isPackedVDotForm above.
  if (aElemTy.isInteger(8) && bElemTy.isInteger(8) && cElemTy.isInteger(32))
    return 4;
  if (aElemTy.isF16() && bElemTy.isF16() && cElemTy.isF32())
    return 2;
  if (isaFamily == ISAFamily::CDNA4 && aElemTy.isBF16() && bElemTy.isBF16() &&
      cElemTy.isF32())
    return 2;
  return 1;
}

} // namespace rock
} // namespace mlir
