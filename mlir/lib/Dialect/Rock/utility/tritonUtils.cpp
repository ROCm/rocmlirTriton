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

// Mirrors _launch() from external/triton/third_party/amd/backend/driver.c
// (lines 603-646). Simplified: gridY/gridZ always 1, blockSize pre-computed,
// launch_cooperative_grid always 0. Returns LogicalResult instead of void.
// Note: hipEventRecord is handled by callers, not by this function.
LogicalResult launchKernel(hipFunction_t function, uint32_t gridX,
                           uint32_t blockSize, uint32_t shared_memory,
                           uint32_t num_ctas, hipStream_t stream,
                           void **params) {
  if (gridX == 0)
    return success();
  if (num_ctas > 1) {
    // Note: driver.c checks hipSymbolTable.hipDrvLaunchKernelEx here because
    // it loads HIP symbols via dlsym. We link directly, so no check needed.
    hipLaunchAttribute attributes[2];
    // Attribute0: Cluster dimensions
    attributes[0].id = static_cast<hipLaunchAttributeID>(4);
    int *cluster_dims = (int *)attributes[0].val.pad;
    cluster_dims[0] = num_ctas;
    cluster_dims[1] = 1;
    cluster_dims[2] = 1;
    // Attribute1: Cooperative launch
    attributes[1].id = hipLaunchAttributeCooperative;
    attributes[1].val.cooperative = 0;

    HIP_LAUNCH_CONFIG config = {
        gridX * num_ctas, 1,      1,            // Grid size
        blockSize,        1,      1,            // Block size
        shared_memory,    stream, attributes, 2 // Number of attributes
    };
    hipError_t status = hipDrvLaunchKernelEx(&config, function, params, 0);
    if (status != hipSuccess)
      return failure();
  } else {
    hipError_t status =
        hipModuleLaunchKernel(function, gridX, 1, 1, blockSize, 1, 1,
                              shared_memory, stream, params, nullptr);
    if (status != hipSuccess)
      return failure();
  }
  return success();
}

} // namespace rock
} // namespace mlir
