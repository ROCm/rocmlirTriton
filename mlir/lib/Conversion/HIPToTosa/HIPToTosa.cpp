//===- HIPToTosa.cpp - Lowering HIP to Tosa Dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the HIP to the Tosa dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/HIPToTosa/HIPToTosa.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/tosaUtils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "hip/Dialect/IR/HipDialect.h"

using namespace mlir;

#define DEBUG_TYPE "hip-to-tosa"

namespace {

/// Collapse the leading batch dimensions of `shape` into one, so that an N-D
/// operand can be fed to the strictly 3-D tosa.matmul. Rank 2 gains a unit
/// batch dimension.
static SmallVector<int64_t, 3> to3DShape(ArrayRef<int64_t> shape) {
  int64_t batch = 1;
  for (int64_t dim : shape.drop_back(2))
    batch *= dim;
  return {batch, shape[shape.size() - 2], shape.back()};
}

static Value reshapeTo(PatternRewriter &rewriter, Location loc, Value value,
                       ArrayRef<int64_t> shape) {
  auto type = cast<RankedTensorType>(value.getType());
  if (type.getShape() == shape)
    return value;
  return tosa::ReshapeOp::create(
      rewriter, loc, RankedTensorType::get(shape, type.getElementType()), value,
      tosa::getTosaConstShape(rewriter, loc, shape));
}

/// Lower `hip.matmul` to `tosa.matmul`, which TosaToRock in turn lowers to
/// `rock.gemm`.
///
/// The HIP op is destination-passing style and carries a `!hip.context`; both
/// the context and the `outs` buffer are dropped here, since the result type
/// already encodes the destination shape. Only static shapes are supported so
/// far.
struct MatmulConverter final : public OpRewritePattern<hip::MatmulOp> {
  using OpRewritePattern<hip::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hip::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (op.getTransA() != 0 || op.getTransB() != 0)
      return rewriter.notifyMatchFailure(op, "transA/transB not handled yet");

    auto aType = dyn_cast<RankedTensorType>(op.getA().getType());
    auto bType = dyn_cast<RankedTensorType>(op.getB().getType());
    if (!aType || !bType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor operands");
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected a tensor-mode matmul");
    auto resultType = cast<RankedTensorType>(op.getResult(0).getType());

    if (!aType.hasStaticShape() || !bType.hasStaticShape() ||
        !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not handled yet");
    if (aType.getRank() < 2 || bType.getRank() != aType.getRank())
      return rewriter.notifyMatchFailure(op, "expected matching operand ranks");

    SmallVector<int64_t, 3> aShape = to3DShape(aType.getShape());
    SmallVector<int64_t, 3> bShape = to3DShape(bType.getShape());
    SmallVector<int64_t, 3> outShape = to3DShape(resultType.getShape());
    if (aShape[0] != bShape[0])
      return rewriter.notifyMatchFailure(op, "broadcast batch not handled yet");

    Value a = reshapeTo(rewriter, loc, op.getA(), aShape);
    Value b = reshapeTo(rewriter, loc, op.getB(), bShape);

    std::optional<Value> aZp =
        tosa::createZeroPointTensor(rewriter, loc, a.getType(), 0);
    std::optional<Value> bZp =
        tosa::createZeroPointTensor(rewriter, loc, b.getType(), 0);
    if (!aZp || !bZp)
      return rewriter.notifyMatchFailure(op, "could not build zero points");

    auto matmul = tosa::MatMulOp::create(
        rewriter, loc,
        RankedTensorType::get(outShape, resultType.getElementType()), a, b,
        *aZp, *bZp);
    // TosaToRock reads acc_type to pick the rock.gemm accumulator.
    matmul->setAttr("acc_type",
                    TypeAttr::get(rock::getAccType(aType.getElementType(),
                                                   bType.getElementType())));

    rewriter.replaceOp(op, reshapeTo(rewriter, loc, matmul.getResult(),
                                     resultType.getShape()));
    return success();
  }
};

/// Lower `hip.transpose` to `tosa.transpose`. Both spell the permutation the
/// same way -- output dim `i` comes from input dim `perm[i]` -- so `perm`
/// carries over unchanged. As with matmul, the `!hip.context` and the `outs`
/// buffer are dropped.
struct TransposeConverter final : public OpRewritePattern<hip::TransposeOp> {
  using OpRewritePattern<hip::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hip::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "expected a ranked tensor input");
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "expected a tensor-mode transpose");

    SmallVector<int32_t, 4> permutation;
    permutation.reserve(op.getPerm().size());
    for (auto permElem : op.getPerm().getAsRange<IntegerAttr>())
      permutation.push_back(permElem.getInt());

    rewriter.replaceOp(op,
                       rock::tosa::getTransposeOp(rewriter, op.getLoc(),
                                                  op.getInput(), permutation));
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
//
//   TODO: THIS SHOULD PROBABLY GO SOMEWHERE ELSE!
//
//   Stamping `rock.kernel` / `rock.arch` onto the function is not part of
//   converting HIP ops to TOSA -- it only lives here because TosaToRock
//   refuses to run without them and nothing else on the HIP path supplies
//   them yet. On the MIGraphX path these come from the driver pipeline
//   (rocmlir-driver knows the target and sets `rock.arch` on the module),
//   not from MIGraphXToTosa.
//
//   Once HIP ingestion has a real pipeline, this belongs there -- alongside
//   whatever decides which functions are kernels at all, since right now we
//   assume every function we see is one.
//
//===----------------------------------------------------------------------===//
void mlir::hip::annotateAsRockKernel(func::FuncOp func, StringRef arch) {
  OpBuilder builder(func.getContext());
  // Presence is all that is checked; rock::KernelAttr is parameterless and
  // every consumer tests it with hasAttr, so a unit attribute is the canonical
  // form (see ConvGenerator.cpp).
  func->setAttr(rock::KernelAttr::getMnemonic(), builder.getUnitAttr());
  if (!arch.empty())
    func->setAttr(rock::ArchAttr::getMnemonic(), builder.getStringAttr(arch));
}
//===----------------------------------------------------------------------===//
//   END OF THE BIT THAT SHOULD PROBABLY GO SOMEWHERE ELSE
//===----------------------------------------------------------------------===//

void mlir::hip::populateHIPToTosaConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MatmulConverter, TransposeConverter>(patterns.getContext());
}
