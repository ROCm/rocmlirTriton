//===- tritonUtils.cpp - Triton-dependent utilities for Rock --------------===//
//
// Centralizes C++ replicas of Triton-internal functions that must be kept in
// sync on every Triton version bump.  See tritonUtils.h for upstream sources.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/tritonUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

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

} // namespace rock
} // namespace mlir
