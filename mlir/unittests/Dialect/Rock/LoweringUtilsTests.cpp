//===- LoweringUtilsTests.cpp - Tests for Rock lowering utilities ---------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct LoweringUtilsTest : public testing::Test {
  MLIRContext context;
  OpBuilder builder{&context};
  ModuleOp module;

  LoweringUtilsTest() {
    DialectRegistry registry;
    registry.insert<arith::ArithDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());
  }
};

TEST_F(LoweringUtilsTest, IdentifiesOnlyDenseNonSplatTensorConstants) {
  Location loc = builder.getUnknownLoc();
  auto tensorType = RankedTensorType::get({4}, builder.getF32Type());
  SmallVector<Attribute> values{
      builder.getF32FloatAttr(1.0), builder.getF32FloatAttr(2.0),
      builder.getF32FloatAttr(3.0), builder.getF32FloatAttr(4.0)};
  Value denseNonSplat = arith::ConstantOp::create(
      builder, loc, DenseElementsAttr::get(tensorType, values));
  Value denseSplat = arith::ConstantOp::create(
      builder, loc,
      DenseElementsAttr::get(tensorType, builder.getF32FloatAttr(1.0)));
  Value scalar =
      arith::ConstantOp::create(builder, loc, builder.getF32FloatAttr(1.0));
  auto vectorType = VectorType::get({4}, builder.getF32Type());
  Value vector = arith::ConstantOp::create(
      builder, loc, DenseElementsAttr::get(vectorType, values));

  Block block;
  Value blockArgument = block.addArgument(tensorType, loc);

  EXPECT_EQ(getDenseTensorConstantAttr(denseNonSplat),
            cast<DenseElementsAttr>(
                denseNonSplat.getDefiningOp<arith::ConstantOp>().getValue()));
  EXPECT_TRUE(isDenseNonSplatConstant(denseNonSplat));
  EXPECT_FALSE(isDenseNonSplatConstant(denseSplat));
  EXPECT_FALSE(isDenseNonSplatConstant(scalar));
  EXPECT_FALSE(getDenseTensorConstantAttr(vector));
  EXPECT_FALSE(isDenseNonSplatConstant(vector));
  EXPECT_FALSE(isDenseNonSplatConstant(blockArgument));
}

TEST_F(LoweringUtilsTest, RejectsSparseTensorConstants) {
  Location loc = builder.getUnknownLoc();
  auto tensorType = RankedTensorType::get({4}, builder.getF32Type());
  auto indicesType = RankedTensorType::get({2, 1}, builder.getI64Type());
  auto valuesType = RankedTensorType::get({2}, builder.getF32Type());
  SmallVector<int64_t> indexValues{0, 3};
  auto indices = DenseIntElementsAttr::get(indicesType, indexValues);
  auto sparseValues = DenseElementsAttr::get(
      valuesType, ArrayRef<Attribute>{builder.getF32FloatAttr(1.0),
                                      builder.getF32FloatAttr(4.0)});
  Value sparse = arith::ConstantOp::create(
      builder, loc, SparseElementsAttr::get(tensorType, indices, sparseValues));

  EXPECT_FALSE(isDenseNonSplatConstant(sparse));
}

} // namespace
