//===- BuilderUtilsTests.cpp - Tests for Builder Utils --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/builderUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct BuilderTestEnv {
  MLIRContext ctx;
  OpBuilder builder;
  ModuleOp module;
  func::FuncOp func;

  BuilderTestEnv(ArrayRef<Type> argTypes = {}) : builder(&ctx) {
    DialectRegistry reg;
    reg.insert<arith::ArithDialect>();
    reg.insert<func::FuncDialect>();
    reg.insert<bufferization::BufferizationDialect>();
    reg.insert<memref::MemRefDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());

    auto funcType = builder.getFunctionType(argTypes, {});
    func = func::FuncOp::create(builder, builder.getUnknownLoc(), "test",
                                funcType);
    func.addEntryBlock();
    builder.setInsertionPointToStart(&func.front());
  }

  Location loc() { return builder.getUnknownLoc(); }
};
} // namespace

// --- createAPFloat ---

TEST(BuilderUtilsTest, CreateAPFloatF32Exact) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  auto f32 = b.getF32Type();

  auto [apVal, status] = createAPFloat(f32, 3.14f);
  EXPECT_EQ(status, APFloat::opOK);
  EXPECT_TRUE(apVal.isExactlyValue(3.14f));
}

TEST(BuilderUtilsTest, CreateAPFloatF16Inexact) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  auto f16 = b.getF16Type();

  auto [apVal, status] = createAPFloat(f16, 0.1f);
  // 0.1 is not exactly representable in f16
  EXPECT_NE(status, APFloat::opOK);
  EXPECT_TRUE(&apVal.getSemantics() == &APFloat::IEEEhalf());
}

TEST(BuilderUtilsTest, CreateAPFloatF16Exact) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  auto f16 = b.getF16Type();

  auto [apVal, status] = createAPFloat(f16, 1.0f);
  EXPECT_EQ(status, APFloat::opOK);
  EXPECT_TRUE(apVal.isExactlyValue(1.0));
}

TEST(BuilderUtilsTest, CreateAPFloatZero) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  auto bf16 = b.getBF16Type();

  auto [apVal, status] = createAPFloat(bf16, 0.0f);
  EXPECT_EQ(status, APFloat::opOK);
  EXPECT_TRUE(apVal.isZero());
}

// --- createConstantIntOp ---

TEST(BuilderUtilsTest, CreateConstantIntOpScalar) {
  BuilderTestEnv env;
  auto i32 = env.builder.getI32Type();

  Value v = createConstantIntOp(env.builder, env.loc(), i32, i32, 42);
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(cst);
  auto intAttr = dyn_cast<IntegerAttr>(cst.getValue());
  ASSERT_TRUE(intAttr);
  EXPECT_EQ(intAttr.getInt(), 42);
}

TEST(BuilderUtilsTest, CreateConstantIntOpSplat) {
  BuilderTestEnv env;
  auto i16 = env.builder.getIntegerType(16);
  auto tensorType = RankedTensorType::get({2, 3}, i16);

  Value v = createConstantIntOp(env.builder, env.loc(), tensorType, i16, 7);
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(cst);
  auto splatAttr = dyn_cast<SplatElementsAttr>(cst.getValue());
  ASSERT_TRUE(splatAttr);
  EXPECT_EQ(splatAttr.getSplatValue<IntegerAttr>().getInt(), 7);
}

TEST(BuilderUtilsTest, CreateConstantIntOpNegative) {
  BuilderTestEnv env;
  auto i32 = env.builder.getI32Type();

  Value v = createConstantIntOp(env.builder, env.loc(), i32, i32, -5);
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(cst);
  EXPECT_EQ(dyn_cast<IntegerAttr>(cst.getValue()).getInt(), -5);
}

// --- createConstantFloatOp ---

TEST(BuilderUtilsTest, CreateConstantFloatOpScalar) {
  BuilderTestEnv env;
  auto f32 = env.builder.getF32Type();

  Value v = createConstantFloatOp(env.builder, env.loc(), f32, f32, 2.5f);
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(cst);
  auto floatAttr = dyn_cast<FloatAttr>(cst.getValue());
  ASSERT_TRUE(floatAttr);
  EXPECT_DOUBLE_EQ(floatAttr.getValueAsDouble(), 2.5);
}

TEST(BuilderUtilsTest, CreateConstantFloatOpSplat) {
  BuilderTestEnv env;
  auto f16 = env.builder.getF16Type();
  auto tensorType = RankedTensorType::get({4}, f16);

  Value v =
      createConstantFloatOp(env.builder, env.loc(), tensorType, f16, 1.0f);
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(cst);
  auto splatAttr = dyn_cast<SplatElementsAttr>(cst.getValue());
  ASSERT_TRUE(splatAttr);
  EXPECT_TRUE(
      splatAttr.getSplatValue<FloatAttr>().getValue().isExactlyValue(1.0));
}

// --- createZeroConstantOp ---

TEST(BuilderUtilsTest, CreateZeroConstantOpInt) {
  BuilderTestEnv env;
  auto i32 = env.builder.getI32Type();

  Value v = createZeroConstantOp(env.builder, env.loc(), i32);
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(cst);
  EXPECT_EQ(dyn_cast<IntegerAttr>(cst.getValue()).getInt(), 0);
}

TEST(BuilderUtilsTest, CreateZeroConstantOpFloat) {
  BuilderTestEnv env;
  auto f32 = env.builder.getF32Type();

  Value v = createZeroConstantOp(env.builder, env.loc(), f32);
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(cst);
  EXPECT_TRUE(dyn_cast<FloatAttr>(cst.getValue()).getValue().isZero());
}

TEST(BuilderUtilsTest, CreateZeroConstantOpTensor) {
  BuilderTestEnv env;
  auto f32 = env.builder.getF32Type();
  auto tensorType = RankedTensorType::get({3, 3}, f32);

  Value v = createZeroConstantOp(env.builder, env.loc(), tensorType);
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(cst);
  auto splatAttr = dyn_cast<SplatElementsAttr>(cst.getValue());
  ASSERT_TRUE(splatAttr);
  EXPECT_TRUE(splatAttr.getSplatValue<FloatAttr>().getValue().isZero());
}

// --- createTypeConversionOp ---

TEST(BuilderUtilsTest, TypeConversionSameType) {
  MLIRContext ctx;
  DialectRegistry reg;
  reg.insert<arith::ArithDialect, func::FuncDialect>();
  ctx.appendDialectRegistry(reg);
  ctx.loadAllAvailableDialects();

  OpBuilder b(&ctx);
  auto module = ModuleOp::create(b.getUnknownLoc());
  b.setInsertionPointToEnd(module.getBody());

  auto i32 = b.getI32Type();
  auto funcType = b.getFunctionType({i32}, {});
  auto func = func::FuncOp::create(b, b.getUnknownLoc(), "same", funcType);
  func.addEntryBlock();
  b.setInsertionPointToStart(&func.front());

  Value arg = func.getArgument(0);
  Value result = createTypeConversionOp(b, b.getUnknownLoc(), arg, i32);
  EXPECT_EQ(result, arg);
}

TEST(BuilderUtilsTest, TypeConversionIntWiden) {
  MLIRContext ctx;
  DialectRegistry reg;
  reg.insert<arith::ArithDialect, func::FuncDialect>();
  ctx.appendDialectRegistry(reg);
  ctx.loadAllAvailableDialects();

  OpBuilder b(&ctx);
  auto module = ModuleOp::create(b.getUnknownLoc());
  b.setInsertionPointToEnd(module.getBody());

  auto i16 = b.getIntegerType(16);
  auto i32 = b.getI32Type();
  auto funcType = b.getFunctionType({i16}, {});
  auto func = func::FuncOp::create(b, b.getUnknownLoc(), "widen", funcType);
  func.addEntryBlock();
  b.setInsertionPointToStart(&func.front());

  Value arg = func.getArgument(0);
  Value result = createTypeConversionOp(b, b.getUnknownLoc(), arg, i32);
  ASSERT_NE(result, arg);
  EXPECT_TRUE(isa<arith::ExtSIOp>(result.getDefiningOp()));
  EXPECT_EQ(result.getType(), i32);
}

TEST(BuilderUtilsTest, TypeConversionIntNarrow) {
  MLIRContext ctx;
  DialectRegistry reg;
  reg.insert<arith::ArithDialect, func::FuncDialect>();
  ctx.appendDialectRegistry(reg);
  ctx.loadAllAvailableDialects();

  OpBuilder b(&ctx);
  auto module = ModuleOp::create(b.getUnknownLoc());
  b.setInsertionPointToEnd(module.getBody());

  auto i32 = b.getI32Type();
  auto i8 = b.getIntegerType(8);
  auto funcType = b.getFunctionType({i32}, {});
  auto func = func::FuncOp::create(b, b.getUnknownLoc(), "narrow", funcType);
  func.addEntryBlock();
  b.setInsertionPointToStart(&func.front());

  Value arg = func.getArgument(0);
  Value result = createTypeConversionOp(b, b.getUnknownLoc(), arg, i8);
  ASSERT_NE(result, arg);
  EXPECT_TRUE(isa<arith::TruncIOp>(result.getDefiningOp()));
  EXPECT_EQ(result.getType(), i8);
}

TEST(BuilderUtilsTest, TypeConversionFloatWiden) {
  MLIRContext ctx;
  DialectRegistry reg;
  reg.insert<arith::ArithDialect, func::FuncDialect>();
  ctx.appendDialectRegistry(reg);
  ctx.loadAllAvailableDialects();

  OpBuilder b(&ctx);
  auto module = ModuleOp::create(b.getUnknownLoc());
  b.setInsertionPointToEnd(module.getBody());

  auto f16 = b.getF16Type();
  auto f32 = b.getF32Type();
  auto funcType = b.getFunctionType({f16}, {});
  auto func = func::FuncOp::create(b, b.getUnknownLoc(), "fwiden", funcType);
  func.addEntryBlock();
  b.setInsertionPointToStart(&func.front());

  Value arg = func.getArgument(0);
  Value result = createTypeConversionOp(b, b.getUnknownLoc(), arg, f32);
  ASSERT_NE(result, arg);
  EXPECT_TRUE(isa<arith::ExtFOp>(result.getDefiningOp()));
  EXPECT_EQ(result.getType(), f32);
}

TEST(BuilderUtilsTest, TypeConversionFloatNarrow) {
  MLIRContext ctx;
  DialectRegistry reg;
  reg.insert<arith::ArithDialect, func::FuncDialect>();
  ctx.appendDialectRegistry(reg);
  ctx.loadAllAvailableDialects();

  OpBuilder b(&ctx);
  auto module = ModuleOp::create(b.getUnknownLoc());
  b.setInsertionPointToEnd(module.getBody());

  auto f32 = b.getF32Type();
  auto f16 = b.getF16Type();
  auto funcType = b.getFunctionType({f32}, {});
  auto func = func::FuncOp::create(b, b.getUnknownLoc(), "fnarrow", funcType);
  func.addEntryBlock();
  b.setInsertionPointToStart(&func.front());

  Value arg = func.getArgument(0);
  Value result = createTypeConversionOp(b, b.getUnknownLoc(), arg, f16);
  ASSERT_NE(result, arg);
  EXPECT_TRUE(isa<arith::TruncFOp>(result.getDefiningOp()));
  EXPECT_EQ(result.getType(), f16);
}

TEST(BuilderUtilsTest, TypeConversionFloatToInt) {
  MLIRContext ctx;
  DialectRegistry reg;
  reg.insert<arith::ArithDialect, func::FuncDialect>();
  ctx.appendDialectRegistry(reg);
  ctx.loadAllAvailableDialects();

  OpBuilder b(&ctx);
  auto module = ModuleOp::create(b.getUnknownLoc());
  b.setInsertionPointToEnd(module.getBody());

  auto f32 = b.getF32Type();
  auto i32 = b.getI32Type();
  auto funcType = b.getFunctionType({f32}, {});
  auto func = func::FuncOp::create(b, b.getUnknownLoc(), "f2i", funcType);
  func.addEntryBlock();
  b.setInsertionPointToStart(&func.front());

  Value arg = func.getArgument(0);
  Value result = createTypeConversionOp(b, b.getUnknownLoc(), arg, i32);
  ASSERT_NE(result, arg);
  EXPECT_TRUE(isa<arith::FPToSIOp>(result.getDefiningOp()));
  EXPECT_EQ(result.getType(), i32);
}

TEST(BuilderUtilsTest, TypeConversionIntToFloat) {
  MLIRContext ctx;
  DialectRegistry reg;
  reg.insert<arith::ArithDialect, func::FuncDialect>();
  ctx.appendDialectRegistry(reg);
  ctx.loadAllAvailableDialects();

  OpBuilder b(&ctx);
  auto module = ModuleOp::create(b.getUnknownLoc());
  b.setInsertionPointToEnd(module.getBody());

  auto i32 = b.getI32Type();
  auto f32 = b.getF32Type();
  auto funcType = b.getFunctionType({i32}, {});
  auto func = func::FuncOp::create(b, b.getUnknownLoc(), "i2f", funcType);
  func.addEntryBlock();
  b.setInsertionPointToStart(&func.front());

  Value arg = func.getArgument(0);
  Value result = createTypeConversionOp(b, b.getUnknownLoc(), arg, f32);
  ASSERT_NE(result, arg);
  EXPECT_TRUE(isa<arith::SIToFPOp>(result.getDefiningOp()));
  EXPECT_EQ(result.getType(), f32);
}

// --- getFlattenedType ---

TEST(BuilderUtilsTest, GetFlattenedTypeTensor) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  auto tensorType = RankedTensorType::get({2, 3, 4}, b.getF32Type());

  Type flat = getFlattenedType(tensorType);
  auto flatTensor = dyn_cast<RankedTensorType>(flat);
  ASSERT_TRUE(flatTensor);
  EXPECT_EQ(flatTensor.getRank(), 1);
  EXPECT_EQ(flatTensor.getDimSize(0), 24);
  EXPECT_EQ(flatTensor.getElementType(), b.getF32Type());
}

TEST(BuilderUtilsTest, GetFlattenedTypeHighRank) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  auto tensorType = RankedTensorType::get({2, 3, 4, 5}, b.getI8Type());

  Type flat = getFlattenedType(tensorType);
  auto flatTensor = dyn_cast<RankedTensorType>(flat);
  ASSERT_TRUE(flatTensor);
  EXPECT_EQ(flatTensor.getRank(), 1);
  EXPECT_EQ(flatTensor.getDimSize(0), 120);
  EXPECT_EQ(flatTensor.getElementType(), b.getI8Type());
}

TEST(BuilderUtilsTest, GetFlattenedTypeAlreadyFlat) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  auto tensorType = RankedTensorType::get({10}, b.getF16Type());

  Type flat = getFlattenedType(tensorType);
  auto flatTensor = dyn_cast<RankedTensorType>(flat);
  ASSERT_TRUE(flatTensor);
  EXPECT_EQ(flatTensor.getRank(), 1);
  EXPECT_EQ(flatTensor.getDimSize(0), 10);
}

// --- getAsTensor ---

TEST(BuilderUtilsTest, GetAsTensorBasic) {
  MLIRContext ctx;
  DialectRegistry reg;
  reg.insert<arith::ArithDialect, func::FuncDialect,
             bufferization::BufferizationDialect, memref::MemRefDialect>();
  ctx.appendDialectRegistry(reg);
  ctx.loadAllAvailableDialects();

  OpBuilder b(&ctx);
  auto module = ModuleOp::create(b.getUnknownLoc());
  b.setInsertionPointToEnd(module.getBody());

  auto memrefType = MemRefType::get({8, 8}, b.getF32Type());
  auto funcType = b.getFunctionType({memrefType}, {});
  auto func =
      func::FuncOp::create(b, b.getUnknownLoc(), "tensor_test", funcType);
  func.addEntryBlock();
  b.setInsertionPointToStart(&func.front());

  Value memrefArg = func.getArgument(0);
  Value tensor = getAsTensor(b, b.getUnknownLoc(), memrefArg);

  ASSERT_TRUE(isa<RankedTensorType>(tensor.getType()));
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  EXPECT_EQ(tensorType.getShape(), ArrayRef<int64_t>({8, 8}));
  EXPECT_EQ(tensorType.getElementType(), b.getF32Type());

  auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(tensor.getDefiningOp());
  ASSERT_TRUE(toTensorOp);
  EXPECT_TRUE(toTensorOp.getRestrict());
  EXPECT_FALSE(toTensorOp.getWritable());
}

TEST(BuilderUtilsTest, GetAsTensorWritable) {
  MLIRContext ctx;
  DialectRegistry reg;
  reg.insert<arith::ArithDialect, func::FuncDialect,
             bufferization::BufferizationDialect, memref::MemRefDialect>();
  ctx.appendDialectRegistry(reg);
  ctx.loadAllAvailableDialects();

  OpBuilder b(&ctx);
  auto module = ModuleOp::create(b.getUnknownLoc());
  b.setInsertionPointToEnd(module.getBody());

  auto memrefType = MemRefType::get({4}, b.getI32Type());
  auto funcType = b.getFunctionType({memrefType}, {});
  auto func = func::FuncOp::create(b, b.getUnknownLoc(), "writable", funcType);
  func.addEntryBlock();
  b.setInsertionPointToStart(&func.front());

  Value memrefArg = func.getArgument(0);
  Value tensor =
      getAsTensor(b, b.getUnknownLoc(), memrefArg, /*isWritable=*/true);

  auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(tensor.getDefiningOp());
  ASSERT_TRUE(toTensorOp);
  EXPECT_TRUE(toTensorOp.getRestrict());
  EXPECT_TRUE(toTensorOp.getWritable());
}
