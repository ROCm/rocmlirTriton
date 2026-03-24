//===- LegalizeFloatTypes.cpp - non-TT_Float -> integer legalization ------===//
//
// Copyright 2026 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Triton's tt.load only supports TT_Float types. Float types outside that set
// (e.g. f8E8M0FNU) are bitcast to integer types of the same bit width so that
// downstream TTIR lowering works correctly.
//
// The original float type is saved on BlockwiseGemmOp for RockToTTIR to set on
// tt.dot_scaled.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Rock/utility/tritonUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/WalkResult.h"

#include "llvm/Support/Debug.h"
#include <optional>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLEGALIZEFLOATTYPESPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-legalize-float-types"

using namespace mlir;
using namespace mlir::rock;

namespace {

static bool isNonTTFloat(Type t) {
  // TODO: implement f4 support in this pass
  if (isa<FloatType>(t) && !rock::isTTFloat(t))
    assert(t.getIntOrFloatBitWidth() == 8 && "No support for f4 yet");
  return isa<FloatType>(t) && !rock::isTTFloat(t) &&
         (t.getIntOrFloatBitWidth() == 8 || t.getIntOrFloatBitWidth() == 4);
}

//===----------------------------------------------------------------------===//
// Type conversion: unsupported float -> integer of same bit width
//===----------------------------------------------------------------------===//

/// Bitcast non-TT_Float types to integer types of the same bit width
/// (e.g. f8E8M0FNU -> i8).
static Type convertElementType(Type elemType, MLIRContext *ctx) {
  if (!isNonTTFloat(elemType))
    return elemType;
  return IntegerType::get(ctx, elemType.getIntOrFloatBitWidth());
}

static Type convertType(Type type, MLIRContext *ctx) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Type newElem = convertElementType(tensorType.getElementType(), ctx);
    if (newElem == tensorType.getElementType())
      return type;
    return RankedTensorType::get(tensorType.getShape(), newElem,
                                 tensorType.getEncoding());
  }
  return convertElementType(type, ctx);
}

/// Record original element types on every BlockwiseGemmOp before the type
/// conversion rewrites them.
static void recordOrigTypesOnGemm(func::FuncOp funcOp) {
  funcOp.walk([](BlockwiseGemmOp gemmOp) {
    Type aElem =
        cast<ShapedType>(gemmOp.getMatrixA().getType()).getElementType();
    Type bElem =
        cast<ShapedType>(gemmOp.getMatrixB().getType()).getElementType();
    if (isNonTTFloat(aElem))
      gemmOp.setMatrixAOrigElemTypeAttr(TypeAttr::get(aElem));
    if (isNonTTFloat(bElem))
      gemmOp.setMatrixBOrigElemTypeAttr(TypeAttr::get(bElem));
  });
}

/// Legalize all non-TT_Float types to integer types of the same bit width
/// throughout the kernel function (no shape changes).
static void convertKernel(func::FuncOp funcOp, MLIRContext *ctx) {
  // Save original types on BlockwiseGemmOp BEFORE converting.
  recordOrigTypesOnGemm(funcOp);

  // Step 1: Simple type swap (f8->i8, f4->i4) with no shape changes.
  SmallVector<Type> newArgTypes;
  for (auto arg : funcOp.getArguments()) {
    Type newType = convertType(arg.getType(), ctx);
    if (newType != arg.getType())
      arg.setType(newType);
    newArgTypes.push_back(arg.getType());
  }

  funcOp.walk([&](Operation *op) {
    for (OpResult result : op->getResults()) {
      Type newType = convertType(result.getType(), ctx);
      if (newType != result.getType())
        result.setType(newType);
    }
  });

  SmallVector<Type> newResultTypes;
  for (Type t : funcOp.getResultTypes())
    newResultTypes.push_back(convertType(t, ctx));
  funcOp.setFunctionType(FunctionType::get(ctx, newArgTypes, newResultTypes));
}

//===----------------------------------------------------------------------===//
// GPU wrapper conversion
//===----------------------------------------------------------------------===//

/// Legalize a memref type with a non-TT_Float element type to its i8
/// equivalent: f8E8M0FNU -> i8 (same shape).
static std::optional<MemRefType> convertMemRefType(MemRefType memrefType,
                                                   Type i8Ty) {
  if (!isNonTTFloat(memrefType.getElementType()))
    return std::nullopt;
  SmallVector<int64_t> newShape(memrefType.getShape());
  return MemRefType::get(newShape, i8Ty, memrefType.getLayout(),
                         memrefType.getMemorySpace());
}

/// For the GPU wrapper: legalize all memref-typed values with non-TT_Float
/// element types to i8.  This covers gpu.alloc, gpu.memcpy, gpu.dealloc, and
/// bufferization.to_tensor so that the LLVM lowering computes correct byte
/// counts for sub-byte types.
static void convertWrapper(func::FuncOp funcOp, MLIRContext *ctx) {
  OpBuilder builder(ctx);
  Type i8Ty = IntegerType::get(ctx, 8);

  // Step 1: Reinterpret each function arg with a non-TT_Float memref type
  // and replace all interior uses.  The function signature stays unchanged
  // so the caller (main) doesn't need updating.
  builder.setInsertionPointToStart(&funcOp.getBody().front());
  for (auto arg : funcOp.getArguments()) {
    auto memrefType = dyn_cast<MemRefType>(arg.getType());
    if (!memrefType)
      continue;
    auto newType = convertMemRefType(memrefType, i8Ty);
    if (!newType)
      continue;
    auto castOp = UnrealizedConversionCastOp::create(builder, funcOp.getLoc(),
                                                     *newType, arg);
    arg.replaceAllUsesExcept(castOp.getResult(0), castOp);
  }

  // Step 2: Legalize result types of all ops (gpu.alloc, etc.) that still
  // carry a non-TT_Float memref element type.
  funcOp.walk([&](Operation *op) {
    if (isa<UnrealizedConversionCastOp>(op))
      return;
    for (OpResult result : op->getResults()) {
      auto memrefType = dyn_cast<MemRefType>(result.getType());
      if (!memrefType)
        continue;
      if (auto newType = convertMemRefType(memrefType, i8Ty))
        result.setType(*newType);
    }
  });

  // Step 3: Fix bufferization.to_tensor result types to match their
  // (now converted) buffer operand types.
  funcOp.walk([&](bufferization::ToTensorOp toTensorOp) {
    auto bufType = cast<MemRefType>(toTensorOp.getBuffer().getType());
    auto resultType = cast<RankedTensorType>(toTensorOp.getResult().getType());
    if (bufType.getElementType() != resultType.getElementType()) {
      toTensorOp.getResult().setType(cast<bufferization::TensorLikeType>(
          RankedTensorType::get(bufType.getShape(), bufType.getElementType())));
    }
  });
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

/// Return true if funcOp contains a call to a function with rock.kernel.
static bool callsKernel(func::FuncOp funcOp) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return false;
  auto res = funcOp.walk([&](func::CallOp callOp) -> WalkResult {
    auto callee = moduleOp.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (callee && callee->hasAttr(rock::KernelAttr::getMnemonic())) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return res.wasInterrupted();
}

struct RockLegalizeFloatTypesPass
    : public rock::impl::RockLegalizeFloatTypesPassBase<
          RockLegalizeFloatTypesPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockLegalizeFloatTypesPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *ctx = &getContext();

  if (funcOp->hasAttr(rock::KernelAttr::getMnemonic())) {
    convertKernel(funcOp, ctx);
  } else if (callsKernel(funcOp)) {
    convertWrapper(funcOp, ctx);
  }
}
