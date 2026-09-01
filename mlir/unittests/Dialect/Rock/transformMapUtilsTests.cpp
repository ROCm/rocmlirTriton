//===- transformMapUtilsTests.cpp - Tests for Rock transformMapUtils ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct TestEnv {
  MLIRContext ctx;
  OpBuilder builder;
  ModuleOp module;

  TestEnv() : builder(&ctx) {
    DialectRegistry reg;
    reg.insert<arith::ArithDialect, RockDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());
  }
};

// Helper to create a transformed tensor with a simple identity transformation
Value createTransformedTensor(OpBuilder &b, Location loc,
                              ArrayRef<int64_t> shape, Type elemType) {
  // Create base tensor
  auto tensorType = RankedTensorType::get(shape, elemType);
  Value base = arith::ConstantOp::create(
      b, loc, DenseElementsAttr::get(tensorType, b.getZeroAttr(elemType)));

  // Create an identity transform to get a transformed value
  BottomUpTMBuilder builder(b, shape, loc);
  SmallVector<StringRef> names;
  builder.getStartNames(names);
  if (!names.empty())
    builder.passThrough(names);
  TransformMapAttr transform = builder.get();

  // Apply transform
  return TransformOp::create(b, loc, base, transform);
}

//===----------------------------------------------------------------------===//
// addPassThroughIndices Tests
//===----------------------------------------------------------------------===//

// Test: Add extra indices at position 0 (beginning)
TEST(AddPassThroughIndicesTest, AddAtPositionZero) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a simple 1D tensor<16xf16>
  Value transformed = createTransformedTensor(b, loc, {16}, b.getF16Type());

  // Add 2 dimensions at position 0 with sizes [2, 8]
  FailureOr<Value> result = addPassThroughIndices(b, transformed, {2, 8}, 0);

  ASSERT_TRUE(succeeded(result));
  auto resultType = cast<ShapedType>(result.value().getType());

  // Result should have shape [2, 8, 16] (new dims prepended)
  EXPECT_EQ(resultType.getRank(), 3);
  EXPECT_EQ(resultType.getShape()[0], 2);
  EXPECT_EQ(resultType.getShape()[1], 8);
  EXPECT_EQ(resultType.getShape()[2], 16);
  EXPECT_TRUE(resultType.getElementType().isF16());
}

// Test: Add extra indices at position 1 (middle)
TEST(AddPassThroughIndicesTest, AddAtPositionMiddle) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a 2D tensor<8x2xf16>
  Value transformed = createTransformedTensor(b, loc, {8, 2}, b.getF16Type());

  // Add 1 dimension at position 1 with size [4]
  FailureOr<Value> result = addPassThroughIndices(b, transformed, {4}, 1);

  ASSERT_TRUE(succeeded(result));
  auto resultType = cast<ShapedType>(result.value().getType());

  // Result should have shape [8, 4, 2] (new dim inserted in middle)
  EXPECT_EQ(resultType.getRank(), 3);
  EXPECT_EQ(resultType.getShape()[0], 8);
  EXPECT_EQ(resultType.getShape()[1], 4);
  EXPECT_EQ(resultType.getShape()[2], 2);
}

// Test: Add extra indices at end position
TEST(AddPassThroughIndicesTest, AddAtPositionEnd) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a 2D tensor<8x2xf16>
  Value transformed = createTransformedTensor(b, loc, {8, 2}, b.getF16Type());

  // Add 2 dimensions at position 2 (end) with sizes [3, 4]
  FailureOr<Value> result = addPassThroughIndices(b, transformed, {3, 4}, 2);

  ASSERT_TRUE(succeeded(result));
  auto resultType = cast<ShapedType>(result.value().getType());

  // Result should have shape [8, 2, 3, 4] (new dims appended)
  EXPECT_EQ(resultType.getRank(), 4);
  EXPECT_EQ(resultType.getShape()[0], 8);
  EXPECT_EQ(resultType.getShape()[1], 2);
  EXPECT_EQ(resultType.getShape()[2], 3);
  EXPECT_EQ(resultType.getShape()[3], 4);
}

// Test: Add no extra indices (empty array)
TEST(AddPassThroughIndicesTest, AddEmptyIndices) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a simple 1D tensor<16xf16>
  Value transformed = createTransformedTensor(b, loc, {16}, b.getF16Type());

  // Add 0 dimensions - should return the original value
  FailureOr<Value> result = addPassThroughIndices(b, transformed, {}, 0);

  ASSERT_TRUE(succeeded(result));
  // When no indices are added, it should return the original transformed value
  EXPECT_EQ(result.value(), transformed);
}

// Test: Multi-buffer case - add indices at position 0 for a 2D tensor
TEST(AddPassThroughIndicesTest, MultiBufferAtPositionZero) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a 2D tensor<2x16xf16> (multi-buffer case)
  Value transformed = createTransformedTensor(b, loc, {2, 16}, b.getF16Type());

  // Add extra indices at position 0 with sizes [2]
  FailureOr<Value> result = addPassThroughIndices(b, transformed, {2}, 0);

  ASSERT_TRUE(succeeded(result));
  auto resultType = cast<ShapedType>(result.value().getType());

  // Result should have shape [2, 2, 16]
  EXPECT_EQ(resultType.getRank(), 3);
  EXPECT_EQ(resultType.getShape()[0], 2);
  EXPECT_EQ(resultType.getShape()[1], 2);
  EXPECT_EQ(resultType.getShape()[2], 16);
}

// Test: Complex case with multiple dimensions at position 0
TEST(AddPassThroughIndicesTest, MultipleDimensionsAtPositionZero) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a 3D tensor<8x2x4xf32>
  Value transformed =
      createTransformedTensor(b, loc, {8, 2, 4}, b.getF32Type());

  // Add 3 dimensions at position 0 with sizes [1, 2, 3]
  FailureOr<Value> result = addPassThroughIndices(b, transformed, {1, 2, 3}, 0);

  ASSERT_TRUE(succeeded(result));
  auto resultType = cast<ShapedType>(result.value().getType());

  // Result should have shape [1, 2, 3, 8, 2, 4]
  EXPECT_EQ(resultType.getRank(), 6);
  EXPECT_EQ(resultType.getShape()[0], 1);
  EXPECT_EQ(resultType.getShape()[1], 2);
  EXPECT_EQ(resultType.getShape()[2], 3);
  EXPECT_EQ(resultType.getShape()[3], 8);
  EXPECT_EQ(resultType.getShape()[4], 2);
  EXPECT_EQ(resultType.getShape()[5], 4);
}

// Test: Single dimension addition at position 0
TEST(AddPassThroughIndicesTest, SingleDimensionAtPositionZero) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value transformed = createTransformedTensor(b, loc, {16}, b.getF16Type());

  // Add just 1 dimension at position 0
  FailureOr<Value> result = addPassThroughIndices(b, transformed, {5}, 0);

  ASSERT_TRUE(succeeded(result));
  auto resultType = cast<ShapedType>(result.value().getType());

  EXPECT_EQ(resultType.getRank(), 2);
  EXPECT_EQ(resultType.getShape()[0], 5);
  EXPECT_EQ(resultType.getShape()[1], 16);
}

// Test: Verify transform stack is properly constructed
// This test checks that the resulting value has a valid transform stack
TEST(AddPassThroughIndicesTest, ValidTransformStack) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value transformed = createTransformedTensor(b, loc, {16}, b.getF16Type());

  FailureOr<Value> result = addPassThroughIndices(b, transformed, {2, 8}, 0);

  ASSERT_TRUE(succeeded(result));

  // The result should be a TransformOp
  auto transformOp = result.value().getDefiningOp<TransformOp>();
  ASSERT_TRUE(transformOp);

  // The transform should have proper bounds
  auto transform = transformOp.getTransform();
  auto upperBounds = transform.getUpperBounds();
  auto lowerBounds = transform.getLowerBounds();

  // Upper bounds should be [2, 8, 16]
  EXPECT_EQ(upperBounds.size(), 3U);
  EXPECT_EQ(upperBounds[0], 2);
  EXPECT_EQ(upperBounds[1], 8);
  EXPECT_EQ(upperBounds[2], 16);

  // The addPassThroughIndices function creates a transform that widens
  // the existing transform stack, so lower bounds include the added dimensions
  EXPECT_EQ(lowerBounds.size(), 3U);
  EXPECT_EQ(lowerBounds[0], 2);
  EXPECT_EQ(lowerBounds[1], 8);
  EXPECT_EQ(lowerBounds[2], 16);
}

// Test: Invalid position - out of bounds
TEST(AddPassThroughIndicesTest, InvalidPositionOutOfBounds) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Create a 2D tensor<8x2xf16>
  Value transformed = createTransformedTensor(b, loc, {8, 2}, b.getF16Type());

  // Try to add dimensions at position 5, which is out of bounds (rank is 2)
  FailureOr<Value> result = addPassThroughIndices(b, transformed, {3, 4}, 5);

  ASSERT_TRUE(failed(result));
}

//===----------------------------------------------------------------------===//
// isIdentityOnShape Tests
//===----------------------------------------------------------------------===//

// Test: Plain identity map matches any shape of equal rank.
TEST(IsIdentityOnShapeTest, PlainIdentity) {
  TestEnv env;
  MLIRContext &ctx = env.ctx;

  AffineMap map = AffineMap::getMultiDimIdentityMap(3, &ctx);

  EXPECT_TRUE(isIdentityOnShape(map, {1, 124, 664}));
  EXPECT_TRUE(isIdentityOnShape(map, {4, 8, 16}));
}

// Test: Constant 0 in a unit-bound position is treated as identity.
// Mirrors the documented example `(d0,d1,d2) -> (0,d1,d2)` with shape
// `[1, 124, 664]`.
TEST(IsIdentityOnShapeTest, BroadcastingOnUnitDim) {
  TestEnv env;
  MLIRContext &ctx = env.ctx;

  SmallVector<AffineExpr> exprs = {getAffineConstantExpr(0, &ctx),
                                   getAffineDimExpr(1, &ctx),
                                   getAffineDimExpr(2, &ctx)};
  AffineMap map =
      AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, exprs, &ctx);

  EXPECT_TRUE(isIdentityOnShape(map, {1, 124, 664}));
}

// Test: Constant 0 in a non-unit-bound position is not identity, since the
// dimension actually takes nonzero values at runtime.
TEST(IsIdentityOnShapeTest, BroadcastingOnNonUnitDim) {
  TestEnv env;
  MLIRContext &ctx = env.ctx;

  SmallVector<AffineExpr> exprs = {getAffineConstantExpr(0, &ctx),
                                   getAffineDimExpr(1, &ctx),
                                   getAffineDimExpr(2, &ctx)};
  AffineMap map =
      AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, exprs, &ctx);

  EXPECT_FALSE(isIdentityOnShape(map, {2, 124, 664}));
}

// Test: A transposition is not identity, even on shapes whose unit dims would
// otherwise allow broadcasting. Mirrors `(d0,d1,d2) -> (d0,d2,d1)`.
TEST(IsIdentityOnShapeTest, TransposeIsNotIdentity) {
  TestEnv env;
  MLIRContext &ctx = env.ctx;

  SmallVector<AffineExpr> exprs = {getAffineDimExpr(0, &ctx),
                                   getAffineDimExpr(2, &ctx),
                                   getAffineDimExpr(1, &ctx)};
  AffineMap map =
      AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, exprs, &ctx);

  EXPECT_FALSE(isIdentityOnShape(map, {1, 124, 664}));
  EXPECT_FALSE(isIdentityOnShape(map, {4, 4, 4}));
}

// Test: A nonzero constant in a unit-bound position is not identity.
// Mirrors the documented example `(d0,d1,d2) -> (1,d1,d2)`.
TEST(IsIdentityOnShapeTest, WrongConstantOnUnitDim) {
  TestEnv env;
  MLIRContext &ctx = env.ctx;

  SmallVector<AffineExpr> exprs = {getAffineConstantExpr(1, &ctx),
                                   getAffineDimExpr(1, &ctx),
                                   getAffineDimExpr(2, &ctx)};
  AffineMap map =
      AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, exprs, &ctx);

  EXPECT_FALSE(isIdentityOnShape(map, {1, 124, 664}));
}

// Test: A null map is rejected rather than crashing.
TEST(IsIdentityOnShapeTest, NullMap) {
  EXPECT_FALSE(isIdentityOnShape(AffineMap(), {1, 124, 664}));
}

// Test: Maps whose result count does not match the shape rank are rejected.
TEST(IsIdentityOnShapeTest, ShapeRankMismatch) {
  TestEnv env;
  MLIRContext &ctx = env.ctx;

  AffineMap map = AffineMap::getMultiDimIdentityMap(3, &ctx);

  EXPECT_FALSE(isIdentityOnShape(map, {1, 124}));
  EXPECT_FALSE(isIdentityOnShape(map, {1, 124, 664, 2}));
}

// Test: Maps with a different number of input dims than results are not
// considered identity (the function requires a square map).
TEST(IsIdentityOnShapeTest, NonSquareMap) {
  TestEnv env;
  MLIRContext &ctx = env.ctx;

  SmallVector<AffineExpr> exprs = {getAffineDimExpr(0, &ctx),
                                   getAffineDimExpr(1, &ctx)};
  AffineMap map =
      AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, exprs, &ctx);

  EXPECT_FALSE(isIdentityOnShape(map, {4, 8}));
}

// Test: The empty (rank-0) identity map matches the empty shape.
TEST(IsIdentityOnShapeTest, EmptyShape) {
  TestEnv env;
  MLIRContext &ctx = env.ctx;

  AffineMap map = AffineMap::getMultiDimIdentityMap(0, &ctx);

  EXPECT_TRUE(isIdentityOnShape(map, {}));
}

//===----------------------------------------------------------------------===//
// getLowerSubDimensions Tests
//===----------------------------------------------------------------------===//

// Callers use getLowerSubDimensions to decide whether an upper dimension can
// move the underlying address at all, which is the case exactly when some lower
// sub-dimension spans more than one value.
static bool movesTheAddress(
    const llvm::SmallDenseMap<int64_t, SmallVector<SubDimInfo>> &subDims) {
  for (const auto &[lowerDim, infos] : subDims)
    for (const SubDimInfo &info : infos)
      if (info.size > 1)
        return true;
  return false;
}

// A dimension that a Broadcast{1} collapses reaches a single address, and
// splitting it into block and iteration halves first does not change that.
TEST(GetLowerSubDimensionsTest, BroadcastedTileAxisReachesOneAddress) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  BottomUpTMBuilder addDim(b, {"m"}, {4}, loc);
  addDim.passThrough("m");
  addDim.addDim("n", 1, 1);
  TransformMapAttr addDimAttr = addDim.get();

  BottomUpTMBuilder bcast = BottomUpTMBuilder::above(addDim, addDimAttr);
  bcast.passThrough("m");
  bcast.broadcast({1}, {16});
  TransformMapAttr bcastAttr = bcast.get();

  BottomUpTMBuilder tile = BottomUpTMBuilder::above(bcast, bcastAttr);
  tile.passThrough("m");
  tile.unmerge({"n_block", "n_iter"}, {1, 2}, "n", {2, 8});
  TransformMapAttr tileAttr = tile.get();

  // getLowerSubDimensions walks top to bottom, the order untransform returns.
  ArrayAttr transforms = b.getArrayAttr({tileAttr, bcastAttr, addDimAttr});

  auto nIter = getLowerSubDimensions(b, transforms, /*dim=*/2);
  ASSERT_TRUE(succeeded(nIter));
  EXPECT_FALSE(movesTheAddress(*nIter));

  auto m = getLowerSubDimensions(b, transforms, /*dim=*/0);
  ASSERT_TRUE(succeeded(m));
  EXPECT_TRUE(movesTheAddress(*m));
}

// Slice only shifts a coordinate by a constant, so it is tracked like a
// PassThrough. This matters because rock-decompose-nonpow2-tiles introduces
// slices above the tiling views.
TEST(GetLowerSubDimensionsTest, SliceKeepsTrackingTheDimension) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  BottomUpTMBuilder addDim(b, {"m"}, {4}, loc);
  addDim.passThrough("m");
  addDim.addDim("n", 1, 1);
  TransformMapAttr addDimAttr = addDim.get();

  BottomUpTMBuilder bcast = BottomUpTMBuilder::above(addDim, addDimAttr);
  bcast.passThrough("m");
  bcast.broadcast({1}, {16});
  TransformMapAttr bcastAttr = bcast.get();

  BottomUpTMBuilder slice = BottomUpTMBuilder::above(bcast, bcastAttr);
  slice.passThrough("m");
  slice.slice({"n_slice"}, {"n"}, {0}, {8});
  TransformMapAttr sliceAttr = slice.get();

  ArrayAttr transforms = b.getArrayAttr({sliceAttr, bcastAttr, addDimAttr});

  auto nSlice = getLowerSubDimensions(b, transforms, /*dim=*/1);
  ASSERT_TRUE(succeeded(nSlice));
  EXPECT_FALSE(movesTheAddress(*nSlice));

  auto m = getLowerSubDimensions(b, transforms, /*dim=*/0);
  ASSERT_TRUE(succeeded(m));
  EXPECT_TRUE(movesTheAddress(*m));
}

// One Broadcast attribute can cover several dimensions. Each has its own
// modulus, so all of them must be tracked; skipping any would make a dimension
// that does move the address look invariant.
TEST(GetLowerSubDimensionsTest, MultiDimBroadcastTracksEveryDimension) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  BottomUpTMBuilder bcast(b, {"m", "n"}, {1, 4}, loc);
  bcast.broadcast({0, 1}, {8, 256});
  TransformMapAttr bcastAttr = bcast.get();

  ArrayAttr transforms = b.getArrayAttr({bcastAttr});

  // "m" is replicated from a single element, so it reaches one address.
  auto m = getLowerSubDimensions(b, transforms, /*dim=*/0);
  ASSERT_TRUE(succeeded(m));
  EXPECT_FALSE(movesTheAddress(*m));

  // "n" cycles through four distinct elements, despite sharing the attribute.
  auto n = getLowerSubDimensions(b, transforms, /*dim=*/1);
  ASSERT_TRUE(succeeded(n));
  EXPECT_TRUE(movesTheAddress(*n));
}

// Repartitioning a contiguous 16-element tile axis into groups of 10 crosses a
// group boundary. Broadcasting away the within-group coordinate must not also
// erase that group dependency.
TEST(GetLowerSubDimensionsTest, MisalignedMergeKeepsAddressDependency) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  BottomUpTMBuilder bcast(b, {"group", "channel"}, {32, 1}, loc);
  bcast.passThrough("group");
  bcast.broadcast({1}, {10});
  TransformMapAttr bcastAttr = bcast.get();

  BottomUpTMBuilder merge = BottomUpTMBuilder::above(bcast, bcastAttr);
  merge.merge("flat", 0, {"group", "channel"});
  TransformMapAttr mergeAttr = merge.get();

  BottomUpTMBuilder tile = BottomUpTMBuilder::above(merge, mergeAttr);
  tile.unmerge({"row", "column"}, {0, 1}, "flat", {20, 16});
  TransformMapAttr tileAttr = tile.get();

  ArrayAttr transforms = b.getArrayAttr({tileAttr, mergeAttr, bcastAttr});
  auto column = getLowerSubDimensions(b, transforms, /*dim=*/1);
  ASSERT_TRUE(succeeded(column));
  EXPECT_TRUE(movesTheAddress(*column));
}

// Pad is not modelled, so the analysis reports failure instead of a wrong
// answer, and callers proving invariance must read that as "cannot prove".
TEST(GetLowerSubDimensionsTest, PadIsUnsupported) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  BottomUpTMBuilder pad(b, {"m", "n"}, {64, 200}, loc);
  pad.passThrough("m");
  pad.pad("nPad", "n", 0, 56);
  TransformMapAttr padAttr = pad.get();

  ArrayAttr transforms = b.getArrayAttr({padAttr});
  EXPECT_TRUE(failed(getLowerSubDimensions(b, transforms, /*dim=*/1)));
}

//===----------------------------------------------------------------------===//
// validityDependsOnAnyDim Tests
//===----------------------------------------------------------------------===//

TEST(ValidityDependsOnAnyDimTest, FindsDependenciesInTransformRange) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  BottomUpTMBuilder lower(b, {"m", "k"}, {8, 16}, loc);
  lower.passThrough("m");
  lower.pad("kPad", "k", 1, 1);
  TransformMapAttr lowerAttr = lower.get();

  BottomUpTMBuilder upper = BottomUpTMBuilder::above(lower, lowerAttr);
  upper.pad("mPad", "m", 1, 1);
  upper.passThrough("kPad");
  TransformMapAttr upperAttr = upper.get();

  SmallVector<TransformMapAttr> transforms{upperAttr, lowerAttr};
  SmallVector<unsigned> mDim{0};
  SmallVector<unsigned> kDim{1};
  SmallVector<unsigned> bothDims{0, 1};
  SmallVector<unsigned> noDims;
  EXPECT_TRUE(validityDependsOnAnyDim(transforms, bothDims));
  EXPECT_FALSE(validityDependsOnAnyDim(transforms, noDims));
  ArrayRef<TransformMapAttr> lowerTransforms =
      ArrayRef<TransformMapAttr>(transforms).drop_front();
  EXPECT_FALSE(validityDependsOnAnyDim(lowerTransforms, mDim));
  EXPECT_TRUE(validityDependsOnAnyDim(lowerTransforms, kDim));
}

// Create a zero constant tensor of the given shape/element type.
static Value makeConstant(OpBuilder &b, Location loc, ArrayRef<int64_t> shape,
                          Type elemType) {
  auto tensorType = RankedTensorType::get(shape, elemType);
  return arith::ConstantOp::create(
      b, loc, DenseElementsAttr::get(tensorType, b.getZeroAttr(elemType)));
}

// Wrap `src` (whose type matches `transform`'s lower bounds) in a
// rock.transform.
static Value applyTransform(OpBuilder &b, Location loc, Value src,
                            TransformMapAttr transform) {
  return TransformOp::create(b, loc, src, transform);
}

//===----------------------------------------------------------------------===//
// collectInputFusionPaths Tests
//===----------------------------------------------------------------------===//

TEST(CollectInputFusionPathsTest, PreservesTransformsPerLeaf) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value baseA = makeConstant(b, loc, {16}, b.getF32Type());
  BottomUpTMBuilder tm(b, {"a"}, {16}, loc);
  tm.passThrough("a");
  TransformMapAttr transform = tm.get();
  Value transformedA = applyTransform(b, loc, baseA, transform);

  Value baseB = makeConstant(b, loc, {16}, b.getF32Type());
  Value fused = arith::AddFOp::create(b, loc, transformedA, baseB).getResult();

  FailureOr<SmallVector<InputFusionPath>> paths =
      collectInputFusionPaths(fused);
  ASSERT_TRUE(succeeded(paths));
  ASSERT_EQ(paths->size(), 2U);
  EXPECT_EQ((*paths)[0].leaf, baseA);
  ASSERT_EQ((*paths)[0].transforms.size(), 1U);
  EXPECT_EQ((*paths)[0].transforms[0], transform);
  EXPECT_EQ((*paths)[1].leaf, baseB);
  EXPECT_TRUE((*paths)[1].transforms.empty());
}

// A value feeding two fusion ops whose results merge again is reachable by two
// routes. Both reach its leaves under the same transforms, so each leaf is
// collected once.
TEST(CollectInputFusionPathsTest, CollectsSharedOperandOnce) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value baseA = makeConstant(b, loc, {16}, b.getF32Type());
  Value baseB = makeConstant(b, loc, {16}, b.getF32Type());
  Value shared = arith::AddFOp::create(b, loc, baseA, baseB).getResult();
  Value left = arith::MulFOp::create(b, loc, shared, baseA).getResult();
  Value right = arith::SubFOp::create(b, loc, shared, baseB).getResult();
  Value merged = arith::AddFOp::create(b, loc, left, right).getResult();

  FailureOr<SmallVector<InputFusionPath>> paths =
      collectInputFusionPaths(merged);
  ASSERT_TRUE(succeeded(paths));
  ASSERT_EQ(paths->size(), 2U);
  EXPECT_EQ((*paths)[0].leaf, baseA);
  EXPECT_EQ((*paths)[1].leaf, baseB);
}

// The same leaf reached under different transforms describes different loads,
// so both routes are kept.
TEST(CollectInputFusionPathsTest, KeepsSharedLeafUnderDifferentTransforms) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value base = makeConstant(b, loc, {16}, b.getF32Type());
  BottomUpTMBuilder tm(b, {"a"}, {16}, loc);
  tm.passThrough("a");
  TransformMapAttr transform = tm.get();
  Value transformed = applyTransform(b, loc, base, transform);
  Value fused = arith::AddFOp::create(b, loc, transformed, base).getResult();

  FailureOr<SmallVector<InputFusionPath>> paths =
      collectInputFusionPaths(fused);
  ASSERT_TRUE(succeeded(paths));
  ASSERT_EQ(paths->size(), 2U);
  EXPECT_EQ((*paths)[0].leaf, base);
  ASSERT_EQ((*paths)[0].transforms.size(), 1U);
  EXPECT_EQ((*paths)[0].transforms[0], transform);
  EXPECT_EQ((*paths)[1].leaf, base);
  EXPECT_TRUE((*paths)[1].transforms.empty());
}

//===----------------------------------------------------------------------===//
// isInputNonInjective Tests
//===----------------------------------------------------------------------===//

// PassThrough is injective: indexing distinct upper coords reads distinct
// elements.
TEST(IsInputNonInjectiveTest, PassThroughIsInjective) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value base = makeConstant(b, loc, {16}, b.getF16Type());
  BottomUpTMBuilder tm(b, {"a"}, {16}, loc);
  tm.passThrough("a");
  Value v = applyTransform(b, loc, base, tm.get());

  FailureOr<bool> result = isInputNonInjective(v);
  ASSERT_TRUE(succeeded(result));
  EXPECT_FALSE(result.value());
}

// Unmerge (a reshape) is injective.
TEST(IsInputNonInjectiveTest, UnmergeIsInjective) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value base = makeConstant(b, loc, {24}, b.getF16Type());
  BottomUpTMBuilder tm(b, {"a"}, {24}, loc);
  tm.unmerge({"x", "y"}, {0, 1}, "a", {4, 6});
  Value v = applyTransform(b, loc, base, tm.get());

  FailureOr<bool> result = isInputNonInjective(v);
  ASSERT_TRUE(succeeded(result));
  EXPECT_FALSE(result.value());
}

// Embed (e.g. overlapping conv im2col) is treated as non-injective.
TEST(IsInputNonInjectiveTest, EmbedIsNonInjective) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value base = makeConstant(b, loc, {24}, b.getF16Type());
  BottomUpTMBuilder tm(b, {"a"}, {24}, loc);
  tm.embed({"x", "y", "z"}, {0, 1, 2}, {2, 3, 4}, "a", {6, 2, 1});
  Value v = applyTransform(b, loc, base, tm.get());

  FailureOr<bool> result = isInputNonInjective(v);
  ASSERT_TRUE(succeeded(result));
  EXPECT_TRUE(result.value());
}

// AddDim with size > 1 re-reads the same element across the added axis.
TEST(IsInputNonInjectiveTest, AddDimNonUnitIsNonInjective) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value base = makeConstant(b, loc, {16}, b.getF16Type());
  BottomUpTMBuilder tm(b, {"a"}, {16}, loc);
  tm.addDim("d", 1, 4);
  tm.passThrough(ArrayRef<uint32_t>{0}, ArrayRef<uint32_t>{0});
  Value v = applyTransform(b, loc, base, tm.get());

  FailureOr<bool> result = isInputNonInjective(v);
  ASSERT_TRUE(succeeded(result));
  EXPECT_TRUE(result.value());
}

// AddDim of size 1 is just a unit axis and does not cause reloads.
TEST(IsInputNonInjectiveTest, AddDimUnitIsInjective) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value base = makeConstant(b, loc, {16}, b.getF16Type());
  BottomUpTMBuilder tm(b, {"a"}, {16}, loc);
  tm.addDim("d", 1, 1);
  tm.passThrough(ArrayRef<uint32_t>{0}, ArrayRef<uint32_t>{0});
  Value v = applyTransform(b, loc, base, tm.get());

  FailureOr<bool> result = isInputNonInjective(v);
  ASSERT_TRUE(succeeded(result));
  EXPECT_FALSE(result.value());
}

// Broadcast that replicates a size-1 dim to many is non-injective.
TEST(IsInputNonInjectiveTest, BroadcastExpandingIsNonInjective) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value base = makeConstant(b, loc, {1}, b.getF16Type());
  BottomUpTMBuilder tm(b, {"a"}, {1}, loc);
  tm.broadcast({0}, {8});
  Value v = applyTransform(b, loc, base, tm.get());

  FailureOr<bool> result = isInputNonInjective(v);
  ASSERT_TRUE(succeeded(result));
  EXPECT_TRUE(result.value());
}

// A degenerate broadcast that does not actually expand (upper size == modulus)
// is injective.
TEST(IsInputNonInjectiveTest, BroadcastNonExpandingIsInjective) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value base = makeConstant(b, loc, {1}, b.getF16Type());
  BottomUpTMBuilder tm(b, {"a"}, {1}, loc);
  tm.broadcast({0}, {1});
  Value v = applyTransform(b, loc, base, tm.get());

  FailureOr<bool> result = isInputNonInjective(v);
  ASSERT_TRUE(succeeded(result));
  EXPECT_FALSE(result.value());
}

// A chain of injective transforms stays injective.
TEST(IsInputNonInjectiveTest, InjectiveChainIsInjective) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  Value base = makeConstant(b, loc, {24}, b.getF16Type());
  BottomUpTMBuilder tm0(b, {"a"}, {24}, loc);
  tm0.unmerge({"x", "y"}, {0, 1}, "a", {4, 6});
  TransformMapAttr attr0 = tm0.get();
  Value v0 = applyTransform(b, loc, base, attr0);

  BottomUpTMBuilder tm1 = BottomUpTMBuilder::above(tm0, attr0);
  tm1.passThrough({"x", "y"});
  Value v1 = applyTransform(b, loc, v0, tm1.get());

  FailureOr<bool> result = isInputNonInjective(v1);
  ASSERT_TRUE(succeeded(result));
  EXPECT_FALSE(result.value());
}

// A fusion op (arith.addf) with one reloading operand is non-injective: the
// walk must follow every operand and OR the results together.
TEST(IsInputNonInjectiveTest, FusionWithOneReloadingOperand) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  // Injective operand: passthrough over a [2,3,4] tensor.
  Value baseA = makeConstant(b, loc, {2, 3, 4}, b.getF32Type());
  BottomUpTMBuilder tmA(b, {"x", "y", "z"}, {2, 3, 4}, loc);
  tmA.passThrough({"x", "y", "z"});
  Value a = applyTransform(b, loc, baseA, tmA.get());

  // Reloading operand: embed producing the same [2,3,4] shape.
  Value baseB = makeConstant(b, loc, {24}, b.getF32Type());
  BottomUpTMBuilder tmB(b, {"a"}, {24}, loc);
  tmB.embed({"x", "y", "z"}, {0, 1, 2}, {2, 3, 4}, "a", {6, 2, 1});
  Value bVal = applyTransform(b, loc, baseB, tmB.get());

  auto fused = arith::AddFOp::create(b, loc, a, bVal);

  FailureOr<bool> result = isInputNonInjective(fused.getResult());
  ASSERT_TRUE(succeeded(result));
  EXPECT_TRUE(result.value());
}

// An unexpected/unsupported op in the chain yields a failure.
TEST(IsInputNonInjectiveTest, UnexpectedOpFails) {
  TestEnv env;
  OpBuilder &b = env.builder;
  Location loc = b.getUnknownLoc();

  auto tensorType = RankedTensorType::get({16}, b.getF16Type());
  auto cast = UnrealizedConversionCastOp::create(b, loc, TypeRange{tensorType},
                                                 ValueRange{});

  FailureOr<bool> result = isInputNonInjective(cast.getResult(0));
  EXPECT_TRUE(failed(result));
}

} // end anonymous namespace
