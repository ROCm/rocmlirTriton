//===- CompileUtilsTests.cpp - Tests for Rock compile utilities -----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/compileUtils.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using mlir::rock::estimatePeakLiveValues;

namespace {

TEST(CompileUtilsTest, EstimatesOverlappingLiveValues) {
  llvm::LLVMContext context;
  llvm::Module module("overlapping", context);
  llvm::IRBuilder<> builder(context);
  llvm::Type *i32 = builder.getInt32Ty();
  auto *functionType = llvm::FunctionType::get(i32, {i32}, false);
  auto *function = llvm::Function::Create(
      functionType, llvm::GlobalValue::ExternalLinkage, "overlapping", module);
  llvm::BasicBlock *block =
      llvm::BasicBlock::Create(context, "entry", function);
  builder.SetInsertPoint(block);

  llvm::Value *argument = function->getArg(0);
  llvm::Value *first = builder.CreateAdd(argument, builder.getInt32(1));
  llvm::Value *second = builder.CreateAdd(argument, builder.getInt32(2));
  llvm::Value *third = builder.CreateAdd(argument, builder.getInt32(3));
  llvm::Value *partial = builder.CreateAdd(first, second);
  llvm::Value *result = builder.CreateAdd(partial, third);
  builder.CreateRet(result);

  // At `third`, the argument and all three delayed results are live.
  EXPECT_EQ(estimatePeakLiveValues(*block), 4u);
}

TEST(CompileUtilsTest, CountsValuesRatherThanValueWidth) {
  llvm::LLVMContext context;
  llvm::Module module("wide-value", context);
  llvm::IRBuilder<> builder(context);
  llvm::Type *i32 = builder.getInt32Ty();
  llvm::Type *wideType = llvm::FixedVectorType::get(i32, 4096);
  auto *functionType = llvm::FunctionType::get(i32, {wideType}, false);
  auto *function = llvm::Function::Create(
      functionType, llvm::GlobalValue::ExternalLinkage, "wide_value", module);
  llvm::BasicBlock *block =
      llvm::BasicBlock::Create(context, "entry", function);
  builder.SetInsertPoint(block);

  llvm::Value *element =
      builder.CreateExtractElement(function->getArg(0), builder.getInt32(0));
  builder.CreateRet(element);

  EXPECT_EQ(estimatePeakLiveValues(*block), 2u);
}

TEST(CompileUtilsTest, KeepsSelfPhiInputsLiveThroughBlockEnd) {
  llvm::LLVMContext context;
  llvm::Module module("self-phi", context);
  llvm::IRBuilder<> builder(context);
  auto *functionType = llvm::FunctionType::get(builder.getVoidTy(), false);
  auto *function = llvm::Function::Create(
      functionType, llvm::GlobalValue::ExternalLinkage, "self_phi", module);
  llvm::BasicBlock *entry =
      llvm::BasicBlock::Create(context, "entry", function);
  llvm::BasicBlock *loop = llvm::BasicBlock::Create(context, "loop", function);

  builder.SetInsertPoint(entry);
  builder.CreateBr(loop);

  builder.SetInsertPoint(loop);
  llvm::PHINode *firstPhi = builder.CreatePHI(builder.getInt32Ty(), 2);
  llvm::PHINode *secondPhi = builder.CreatePHI(builder.getInt32Ty(), 2);
  llvm::Value *firstNext = builder.CreateAdd(firstPhi, builder.getInt32(1));
  llvm::Value *secondNext = builder.CreateAdd(secondPhi, builder.getInt32(1));
  builder.CreateBr(loop);
  firstPhi->addIncoming(builder.getInt32(0), entry);
  firstPhi->addIncoming(firstNext, loop);
  secondPhi->addIncoming(builder.getInt32(0), entry);
  secondPhi->addIncoming(secondNext, loop);

  // Both PHIs and one backedge value overlap in the middle of the block.
  EXPECT_EQ(estimatePeakLiveValues(*loop), 3u);
}

} // namespace
