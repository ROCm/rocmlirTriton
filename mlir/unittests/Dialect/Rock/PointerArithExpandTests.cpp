//===- PointerArithExpandTests.cpp - Tests for PointerArithExpand --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PointerArithExpand.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {

// Builds the ops the helpers emit (arith / triton) directly into a module body.
// The IR is never verified, so a hosting func is unnecessary.
struct TestEnv {
  MLIRContext ctx;
  OpBuilder builder;
  Location loc;
  OwningOpRef<ModuleOp> module;

  TestEnv() : builder(&ctx), loc(builder.getUnknownLoc()) {
    DialectRegistry reg;
    reg.insert<arith::ArithDialect, RockDialect, triton::TritonDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module->getBody());
  }

  // A scalar i32 SSA value.
  Value scalar() {
    return arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0));
  }

  // A tensor<...xi32> SSA value of the requested shape.
  Value tensor(ArrayRef<int64_t> shape) {
    auto ty = RankedTensorType::get(shape, builder.getI32Type());
    auto attr = cast<TypedAttr>(builder.getZeroAttr(ty));
    return arith::ConstantOp::create(builder, loc, attr);
  }
};

// A scalar is splat to the full target shape.
TEST(BroadcastToShape, ScalarIsSplat) {
  TestEnv env;
  Value res = broadcastToShape(env.builder, env.loc, env.scalar(), {4, 8});
  ASSERT_TRUE(res);
  auto tt = dyn_cast<RankedTensorType>(res.getType());
  ASSERT_TRUE(tt);
  EXPECT_EQ(tt.getShape(), ArrayRef<int64_t>({4, 8}));
  EXPECT_TRUE(isa<triton::SplatOp>(res.getDefiningOp()));
}

// A tensor already matching the target shape is returned unchanged.
TEST(BroadcastToShape, MatchingShapeIsIdentity) {
  TestEnv env;
  Value in = env.tensor({4, 8});
  Value res = broadcastToShape(env.builder, env.loc, in, {4, 8});
  EXPECT_EQ(res, in);
}

// A unit dim is broadcast up to the peer size.
TEST(BroadcastToShape, UnitDimIsBroadcast) {
  TestEnv env;
  Value res = broadcastToShape(env.builder, env.loc, env.tensor({1, 8}), {4, 8});
  ASSERT_TRUE(res);
  auto tt = dyn_cast<RankedTensorType>(res.getType());
  ASSERT_TRUE(tt);
  EXPECT_EQ(tt.getShape(), ArrayRef<int64_t>({4, 8}));
  EXPECT_TRUE(isa<triton::BroadcastOp>(res.getDefiningOp()));
}

// A rank mismatch is not broadcastable: a null Value is returned.
TEST(BroadcastToShape, RankMismatchReturnsNull) {
  TestEnv env;
  Value res = broadcastToShape(env.builder, env.loc, env.tensor({8}), {4, 8});
  EXPECT_FALSE(res);
}

// A non-unit dim that would have to change size is not broadcastable.
TEST(BroadcastToShape, NonUnitDimMismatchReturnsNull) {
  TestEnv env;
  Value res = broadcastToShape(env.builder, env.loc, env.tensor({2, 8}), {4, 8});
  EXPECT_FALSE(res);
}

// makeRange restores rank with unit dims around the non-unit dim and keeps i32.
TEST(MakeRange, RestoresRankAtI32) {
  TestEnv env;
  Value res = makeRange(env.builder, env.loc, /*start=*/0, /*end=*/8,
                        /*numDims=*/2, /*nonUnitDim=*/1, env.builder.getI32Type());
  ASSERT_TRUE(res);
  auto tt = dyn_cast<RankedTensorType>(res.getType());
  ASSERT_TRUE(tt);
  EXPECT_EQ(tt.getShape(), ArrayRef<int64_t>({1, 8}));
  EXPECT_TRUE(tt.getElementType().isInteger(32));
}

// A wider index type sign-extends the i32 range.
TEST(MakeRange, WidensToI64) {
  TestEnv env;
  Value res = makeRange(env.builder, env.loc, /*start=*/0, /*end=*/8,
                        /*numDims=*/1, /*nonUnitDim=*/0, env.builder.getI64Type());
  ASSERT_TRUE(res);
  auto tt = dyn_cast<RankedTensorType>(res.getType());
  ASSERT_TRUE(tt);
  EXPECT_EQ(tt.getShape(), ArrayRef<int64_t>({8}));
  EXPECT_TRUE(tt.getElementType().isInteger(64));
  EXPECT_TRUE(isa<arith::ExtSIOp>(res.getDefiningOp()));
}

// When the offset can't be broadcast to the output tile shape,
// expandCoordsToOffsetAndMask fails and emits a descriptive diagnostic. Empty
// transforms leave the offset equal to the single start coordinate, so a
// rank-1 coordinate against a rank-2 tile shape is non-broadcastable.
TEST(ExpandCoordsToOffsetAndMask, NonBroadcastableOffsetEmitsError) {
  TestEnv env;
  Value coord = env.tensor({8});

  std::string diag;
  ScopedDiagnosticHandler handler(&env.ctx, [&](Diagnostic &d) {
    diag += d.str();
    return success();
  });

  FailureOr<OffsetAndMask> res = expandCoordsToOffsetAndMask(
      env.builder, env.loc, /*transforms=*/{}, /*startCoords=*/{coord},
      /*outShape=*/{4, 8}, env.builder.getI32Type(), /*computeOffset=*/true);

  EXPECT_TRUE(failed(res));
  EXPECT_NE(diag.find("cannot broadcast offset"), std::string::npos);
}

} // namespace
