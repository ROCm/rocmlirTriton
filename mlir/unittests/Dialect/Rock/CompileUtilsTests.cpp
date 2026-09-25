//===- CompileUtilsTests.cpp - Tests for Rock compile utilities -----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/compileUtils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "gtest/gtest.h"

using mlir::rock::estimatePeakLocalLiveValues;

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
  EXPECT_EQ(estimatePeakLocalLiveValues(*block), 4u);
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

  EXPECT_EQ(estimatePeakLocalLiveValues(*block), 2u);
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
  EXPECT_EQ(estimatePeakLocalLiveValues(*loop), 3u);
}

TEST(CompileUtilsTest, HandlesLoopCarriedValuesAcrossSeparateLatch) {
  llvm::LLVMContext context;
  llvm::Module module("separate-latch", context);
  llvm::IRBuilder<> builder(context);
  llvm::Type *i32 = builder.getInt32Ty();
  auto *functionType = llvm::FunctionType::get(i32, {i32}, false);
  auto *function =
      llvm::Function::Create(functionType, llvm::GlobalValue::ExternalLinkage,
                             "separate_latch", module);
  llvm::BasicBlock *entry =
      llvm::BasicBlock::Create(context, "entry", function);
  llvm::BasicBlock *header =
      llvm::BasicBlock::Create(context, "header", function);
  llvm::BasicBlock *latch =
      llvm::BasicBlock::Create(context, "latch", function);
  llvm::BasicBlock *exit = llvm::BasicBlock::Create(context, "exit", function);

  builder.SetInsertPoint(entry);
  builder.CreateBr(header);

  builder.SetInsertPoint(header);
  llvm::SmallVector<llvm::PHINode *, 4> phis;
  for (unsigned i = 0; i < 4; ++i)
    phis.push_back(builder.CreatePHI(i32, 2));
  llvm::SmallVector<llvm::Value *, 4> nextValues;
  for (unsigned i = 0; i < 4; ++i)
    nextValues.push_back(builder.CreateAdd(phis[i], builder.getInt32(i + 1)));
  llvm::Value *condition =
      builder.CreateICmpSLT(nextValues.front(), function->getArg(0));
  builder.CreateCondBr(condition, latch, exit);

  builder.SetInsertPoint(latch);
  builder.CreateBr(header);

  builder.SetInsertPoint(exit);
  llvm::Value *sum = nextValues.front();
  for (unsigned i = 1; i < nextValues.size(); ++i)
    sum = builder.CreateAdd(sum, nextValues[i]);
  builder.CreateRet(sum);

  for (unsigned i = 0; i < phis.size(); ++i) {
    phis[i]->addIncoming(builder.getInt32(0), entry);
    phis[i]->addIncoming(nextValues[i], latch);
  }

  ASSERT_FALSE(llvm::verifyFunction(*function));
  // Foreign PHI operands must not create backwards intervals that cancel the
  // four PHI-result ranges.
  EXPECT_EQ(estimatePeakLocalLiveValues(*header), 6u);
}

TEST(CompileUtilsTest, DoesNotCombineMutuallyExclusivePhiInputs) {
  llvm::LLVMContext context;
  llvm::Module module("join", context);
  llvm::IRBuilder<> builder(context);
  llvm::Type *i32 = builder.getInt32Ty();
  auto *functionType = llvm::FunctionType::get(i32, {i32}, false);
  auto *function = llvm::Function::Create(
      functionType, llvm::GlobalValue::ExternalLinkage, "join", module);
  llvm::BasicBlock *entry =
      llvm::BasicBlock::Create(context, "entry", function);
  llvm::BasicBlock *thenBlock =
      llvm::BasicBlock::Create(context, "then", function);
  llvm::BasicBlock *elseBlock =
      llvm::BasicBlock::Create(context, "else", function);
  llvm::BasicBlock *join = llvm::BasicBlock::Create(context, "join", function);
  llvm::Value *argument = function->getArg(0);

  builder.SetInsertPoint(entry);
  builder.CreateCondBr(builder.CreateICmpSLT(argument, builder.getInt32(0)),
                       thenBlock, elseBlock);

  llvm::SmallVector<llvm::Value *, 4> thenValues;
  builder.SetInsertPoint(thenBlock);
  for (unsigned i = 0; i < 4; ++i)
    thenValues.push_back(builder.CreateAdd(argument, builder.getInt32(i + 1)));
  builder.CreateBr(join);

  llvm::SmallVector<llvm::Value *, 4> elseValues;
  builder.SetInsertPoint(elseBlock);
  for (unsigned i = 0; i < 4; ++i)
    elseValues.push_back(builder.CreateMul(argument, builder.getInt32(i + 2)));
  builder.CreateBr(join);

  builder.SetInsertPoint(join);
  llvm::SmallVector<llvm::PHINode *, 4> phis;
  for (unsigned i = 0; i < 4; ++i) {
    llvm::PHINode *phi = builder.CreatePHI(i32, 2);
    phi->addIncoming(thenValues[i], thenBlock);
    phi->addIncoming(elseValues[i], elseBlock);
    phis.push_back(phi);
  }
  llvm::Value *sum = phis.front();
  for (unsigned i = 1; i < phis.size(); ++i)
    sum = builder.CreateAdd(sum, phis[i]);
  builder.CreateRet(sum);

  ASSERT_FALSE(llvm::verifyFunction(*function));
  // Inputs from the two mutually exclusive predecessor edges are not live in
  // the join block. Only the PHI results and reduction temporaries count.
  EXPECT_EQ(estimatePeakLocalLiveValues(*join), 5u);
}

} // namespace
