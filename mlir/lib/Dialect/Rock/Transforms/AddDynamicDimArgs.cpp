//===- AddDynamicDimArgs.cpp - Pass runtime extents into dynamic kernels --===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A kernel whose shapes are not fully known at compile time needs the unknown
// extents as runtime values: the grid size, the block id to tile mapping and
// the out-of-bounds masks are all derived from them.
//
// This pass appends four `i32` arguments to such a kernel, carrying the gemm
// dimensions G, M, N and K in that order, and rewrites the callers to compute
// them. The order is a convention rather than something recorded in the IR, so
// the pass only handles kernels with a single gemm; anything else is rejected
// rather than numbered ambiguously.
//
// Dimensions that are static in the kernel's types keep their compile-time
// constants everywhere; their argument is passed for uniformity and is simply
// unused. Only the dynamic ones are meant to be read from these arguments.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKADDDYNAMICDIMARGSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-add-dynamic-dim-args"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// How a caller obtains one gemm dimension. A static dimension is a literal; a
/// dynamic one is recovered by dividing the element count of the buffer the
/// gemm operand is a view of by the product of the operand's other extents.
struct ExtentRecipe {
  /// Set unless the dimension is dynamic.
  int64_t staticSize = ShapedType::kDynamic;
  /// Kernel argument the gemm operand is a view of.
  unsigned bufferArgIndex = 0;
  /// Product of the operand's remaining, static, extents.
  int64_t divisor = 1;
};

struct RockAddDynamicDimArgsPass
    : public rock::impl::RockAddDynamicDimArgsPassBase<
          RockAddDynamicDimArgsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

/// Whether `transforms` only reshuffles coordinates, so that the view and the
/// buffer underneath it hold the same number of elements. That equality is what
/// lets a dynamic extent be recovered from the buffer's element count.
static bool preservesElementCount(ArrayRef<TransformMapAttr> transforms) {
  for (TransformMapAttr map : transforms) {
    ArrayRef<int64_t> upperBounds = map.getUpperBounds().asArrayRef();
    for (TransformAttr transform : map.getOps()) {
      switch (transform.getType()) {
      case TransformType::PassThrough:
      case TransformType::Unmerge:
      case TransformType::Merge:
        break;
      case TransformType::AddDim:
        // A unit dimension mapped to nothing leaves the count alone; a wider
        // one repeats the data underneath it.
        for (uint32_t upperDim : transform.getUpperDims())
          if (upperBounds[upperDim] != 1)
            return false;
        break;
      case TransformType::Pad:
      case TransformType::Slice:
      case TransformType::Embed:
      case TransformType::Broadcast:
      case TransformType::ConstDim:
        return false;
      }
    }
  }
  return true;
}

/// Work out how a caller can obtain dimension `dimIndex` of gemm operand
/// `operand`.
static FailureOr<ExtentRecipe> buildExtentRecipe(Operation *gemmOp,
                                                 StringRef dimName,
                                                 Value operand,
                                                 unsigned dimIndex) {
  ArrayRef<int64_t> shape = cast<ShapedType>(operand.getType()).getShape();
  if (!ShapedType::isDynamic(shape[dimIndex]))
    return ExtentRecipe{shape[dimIndex], 0, 1};

  // With two unknowns in one view, the buffer's element count is one equation
  // short of pinning either of them down.
  if (llvm::count_if(
          shape, [](int64_t dim) { return ShapedType::isDynamic(dim); }) != 1)
    return gemmOp->emitOpError()
           << "cannot recover the runtime value of dimension " << dimName
           << " because more than one extent of that operand is dynamic";

  int64_t divisor = 1;
  for (auto [index, dim] : llvm::enumerate(shape))
    if (index != dimIndex)
      divisor *= dim;

  SmallVector<TransformMapAttr> transforms;
  auto [root, unused] = untransform(operand, transforms);
  if (!preservesElementCount(transforms))
    return gemmOp->emitOpError()
           << "cannot recover the runtime value of dimension " << dimName
           << " because the operand is not an element-count-preserving view of "
              "a kernel argument";

  auto blockArg = dyn_cast<BlockArgument>(root);
  if (!blockArg)
    return gemmOp->emitOpError()
           << "cannot recover the runtime value of dimension " << dimName
           << " because the operand does not trace back to a kernel argument";

  return ExtentRecipe{ShapedType::kDynamic, blockArg.getArgNumber(), divisor};
}

/// Collect a recipe for each of G, M, N and K of the single gemm in `funcOp`.
static FailureOr<SmallVector<ExtentRecipe>>
buildGemmExtentRecipes(func::FuncOp funcOp) {
  SmallVector<GemmOp> gemmOps;
  funcOp.walk([&](GemmOp gemmOp) { gemmOps.push_back(gemmOp); });

  if (gemmOps.size() != 1)
    return funcOp.emitOpError()
           << "dynamic shapes are only supported for kernels with exactly one "
              "rock.gemm, found "
           << gemmOps.size()
           << "; the appended G, M, N, K arguments would be ambiguous";

  GemmOp gemmOp = gemmOps.front();
  Value a = gemmOp.getA(), b = gemmOp.getB();

  // A is G x M x K and B is G x K x N once the gemm has been normalized, in
  // RuntimeGemmDim order.
  const std::pair<Value, unsigned> sources[kNumRuntimeGemmDims] = {
      {a, 0}, {a, 1}, {b, 2}, {a, 2}};

  SmallVector<ExtentRecipe> recipes;
  for (auto [index, source] : llvm::enumerate(sources)) {
    StringRef dimName =
        getRuntimeGemmDimName(static_cast<RuntimeGemmDim>(index));
    FailureOr<ExtentRecipe> recipe =
        buildExtentRecipe(gemmOp, dimName, source.first, source.second);
    if (failed(recipe))
      return failure();
    recipes.push_back(*recipe);
  }
  return recipes;
}

/// Emit the number of elements of `buffer` at the builder's insertion point.
static Value emitNumElements(OpBuilder &builder, Location loc, Value buffer) {
  ArrayRef<int64_t> shape = cast<ShapedType>(buffer.getType()).getShape();
  Value numElements;
  int64_t staticFactor = 1;
  for (auto [dimIndex, dim] : llvm::enumerate(shape)) {
    if (!ShapedType::isDynamic(dim)) {
      staticFactor *= dim;
      continue;
    }
    Value extent = tensor::DimOp::create(builder, loc, buffer,
                                         static_cast<int64_t>(dimIndex));
    numElements = numElements
                      ? arith::MulIOp::create(builder, loc, numElements, extent)
                      : extent;
  }
  if (staticFactor == 1 && numElements)
    return numElements;

  Value staticExtent =
      arith::ConstantIndexOp::create(builder, loc, staticFactor);
  if (!numElements)
    return staticExtent;
  return arith::MulIOp::create(builder, loc, numElements, staticExtent);
}

/// Rewrite every call to `funcOp` so it passes the gemm dimensions described by
/// `recipes`. The callee signature must already have been widened.
static LogicalResult appendExtentsToCallers(func::FuncOp funcOp,
                                            ArrayRef<ExtentRecipe> recipes,
                                            ModuleOp moduleOp) {
  std::optional<SymbolTable::UseRange> uses =
      SymbolTable::getSymbolUses(funcOp, moduleOp);
  if (!uses)
    return funcOp.emitOpError(
        "has uses that cannot be enumerated, so its callers cannot be given "
        "the runtime gemm dimensions");

  for (SymbolTable::SymbolUse use : *uses) {
    auto callOp = dyn_cast<func::CallOp>(use.getUser());
    if (!callOp)
      return use.getUser()->emitOpError(
          "unexpected non-call use of a kernel with dynamic dimensions");

    Location loc = callOp.getLoc();
    OpBuilder builder(callOp);
    Type extentType = builder.getI32Type();
    SmallVector<Value> operands(callOp.getOperands());
    for (const ExtentRecipe &recipe : recipes) {
      if (!ShapedType::isDynamic(recipe.staticSize)) {
        operands.push_back(arith::ConstantIntOp::create(
            builder, loc, extentType, recipe.staticSize));
        continue;
      }

      Value buffer = callOp.getOperand(recipe.bufferArgIndex);
      if (!isa<RankedTensorType>(buffer.getType()))
        return callOp.emitOpError()
               << "operand " << recipe.bufferArgIndex
               << " must be a tensor to take its element count, got "
               << buffer.getType();

      Value extent = emitNumElements(builder, loc, buffer);
      if (recipe.divisor != 1) {
        Value divisor =
            arith::ConstantIndexOp::create(builder, loc, recipe.divisor);
        extent = arith::DivUIOp::create(builder, loc, extent, divisor);
      }
      operands.push_back(
          arith::IndexCastOp::create(builder, loc, extentType, extent));
    }

    auto newCallOp = func::CallOp::create(builder, loc, funcOp, operands);
    callOp.replaceAllUsesWith(newCallOp.getResults());
    callOp.erase();
  }
  return success();
}

/// Whether any argument of `funcOp` has an extent that is only known at
/// runtime.
static bool hasDynamicArgument(func::FuncOp funcOp) {
  return llvm::any_of(funcOp.getArgumentTypes(), [](Type argType) {
    auto shapedType = dyn_cast<ShapedType>(argType);
    return shapedType && !shapedType.hasStaticShape();
  });
}

void RockAddDynamicDimArgsPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  OpBuilder builder(&getContext());
  Type extentType = builder.getI32Type();

  SmallVector<func::FuncOp> kernels;
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (funcOp.isExternal())
      continue;
    if (funcOp->hasAttr(rock::KernelAttr::getMnemonic()) &&
        hasDynamicArgument(funcOp))
      kernels.push_back(funcOp);
  }

  for (func::FuncOp funcOp : kernels) {
    FailureOr<SmallVector<ExtentRecipe>> recipes =
        buildGemmExtentRecipes(funcOp);
    if (failed(recipes))
      return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs()
               << "appending G, M, N, K args to " << funcOp.getName() << "\n");

    unsigned firstExtentArg = funcOp.getNumArguments();
    for (unsigned offset = 0; offset < kNumRuntimeGemmDims; ++offset) {
      if (failed(funcOp.insertArgument(firstExtentArg + offset, extentType,
                                       /*argAttrs=*/nullptr, funcOp.getLoc())))
        return signalPassFailure();
    }

    if (failed(appendExtentsToCallers(funcOp, *recipes, moduleOp)))
      return signalPassFailure();
  }
}
