//===- tritonUtils.cpp - Triton-dependent utilities for Rock --------------===//
//
// Centralizes C++ replicas of Triton-internal functions that must be kept in
// sync on every Triton version bump.  See tritonUtils.h for upstream sources.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/tritonUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton::AMD;

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
int getWmmaVersion(StringRef arch) {
  if (arch.starts_with("gfx11"))
    return 1; // RDNA3
  if (arch.starts_with("gfx12") && !arch.ends_with("50"))
    return 2; // RDNA4
  if (arch == "gfx1250")
    return 3; // GFX1250
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

// Mirrors python/triton/language/semantic.py::atomic_max. See header for the
// per-lane sign-bit-split rationale. The op verifier for
// `rock.blockwise_store_ptr` (RockOps.td) constrains its source operand to
// `NativeMemoryOpTypes`, which excludes F64, so the f64 arm of upstream
// `semantic.py::atomic_max` is unreachable here. F16/BF16 are in
// `NativeMemoryOpTypes` and so can reach this guard via hand-written IR,
// but the production emitter, `rock.reduce max`, gated by
// `rock::isFastAtomicMaxSupported` only emits f32, so this diagnostic is
// primarily a safety net for test IR and any future direct emitter of
// `StoreMethod::AtomicMax`.
LogicalResult emitFloatAtomicMax(PatternRewriter &rewriter, Operation *op,
                                 Value value, Value ptrTensor, Value mask,
                                 triton::MemSemantic sem,
                                 triton::MemSyncScope scope) {
  Location loc = op->getLoc();
  auto valueType = cast<RankedTensorType>(value.getType());
  auto ptrTensorType = cast<RankedTensorType>(ptrTensor.getType());
  auto maskTensorType = cast<RankedTensorType>(mask.getType());
  auto fpType = cast<FloatType>(valueType.getElementType());

  if (!fpType.isF32()) {
    return op->emitError("atomic_max on floating point requires f32; got ")
           << fpType;
  }
  unsigned bw = fpType.getWidth();

  Type intElemType = rewriter.getIntegerType(bw);
  auto intTensorType = RankedTensorType::get(valueType.getShape(), intElemType,
                                             valueType.getEncoding());
  auto intPtrType = triton::PointerType::get(intElemType, 1);
  auto intPtrTensorType = RankedTensorType::get(
      ptrTensorType.getShape(), intPtrType, ptrTensorType.getEncoding());

  // tt.bitcast value and pointer to signless integer. Upstream emits two
  // bitcasts (one to the signed Triton-language int type, one to the
  // unsigned), but both lower to the same signless MLIR op so we only emit
  // one of each here; CSE would otherwise dedupe them. The signed-vs-
  // unsigned distinction lives in the RMWOp attribute on the atomic, not
  // in the operand type.
  Value intVal = triton::BitcastOp::create(rewriter, loc, intTensorType, value);
  Value intPtr =
      triton::BitcastOp::create(rewriter, loc, intPtrTensorType, ptrTensor);

  // _signbit(val) mirrors semantic.py:1285-1290 exactly:
  //   ix      = bitcast(val, uint{bw})    # == intVal above
  //   shifted = lshr(ix, bw - 1)          # arith.shrui  -- not ashr
  //   neg     = cast(shifted, i1)
  // The final cast to i1 takes the int->bool branch of cast() at
  // semantic.py:872-875, which lowers to not_equal(shifted, 0), i.e.
  // arith.cmpi ne -- NOT an arith.trunci. Using shrui (logical, not
  // arithmetic) matters: ashr would smear the sign bit and produce
  // non-{0,1} bytes that the cmpi-ne would still flatten correctly, but
  // the IR would no longer match upstream.
  Value shiftAmt = arith::ConstantOp::create(
      rewriter, loc, intTensorType,
      DenseElementsAttr::get(intTensorType,
                             rewriter.getIntegerAttr(intElemType, bw - 1)));
  Value shifted = arith::ShRUIOp::create(rewriter, loc, intVal, shiftAmt);
  Value zeroInt = arith::ConstantOp::create(
      rewriter, loc, intTensorType, rewriter.getZeroAttr(intTensorType));
  Value neg = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                    shifted, zeroInt);

  // pos = not_(neg) -- semantic.py invert() xors against an all-ones value
  // of the operand type; for i1 that's a true splat (semantic.py:454-457,
  // 487-492).
  Value trueSplat = arith::ConstantOp::create(
      rewriter, loc, maskTensorType,
      DenseElementsAttr::get(maskTensorType, rewriter.getBoolAttr(true)));
  Value pos = arith::XOrIOp::create(rewriter, loc, neg, trueSplat);

  // Disjoint per-lane masks: every lane participates in exactly one of the
  // two atomics, so the trick stays per-lane atomic.
  Value posMask = arith::AndIOp::create(rewriter, loc, mask, pos);
  Value negMask = arith::AndIOp::create(rewriter, loc, mask, neg);

  triton::AtomicRMWOp::create(rewriter, loc, intTensorType, triton::RMWOp::MAX,
                              intPtr, intVal, posMask, sem, scope);
  triton::AtomicRMWOp::create(rewriter, loc, intTensorType, triton::RMWOp::UMIN,
                              intPtr, intVal, negMask, sem, scope);
  return success();
}

} // namespace rock
} // namespace mlir
