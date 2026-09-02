//===- GetRockInfoTests.cpp - Tests for GetRockInfo helpers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/GetRockInfo.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct GetRockInfoTestEnv {
  MLIRContext ctx;
  OpBuilder b;
  Location loc;
  OwningOpRef<ModuleOp> module;

  GetRockInfoTestEnv() : b(&ctx), loc(b.getUnknownLoc()) {
    DialectRegistry reg;
    reg.insert<rock::RockDialect>();
    reg.insert<func::FuncDialect>();
    reg.insert<arith::ArithDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    module = ModuleOp::create(loc);
  }

  // Append a `func.func @kernel() { ... }` to the module and return its
  // entry block.
  Block *addFunc() {
    b.setInsertionPointToEnd(module->getBody());
    auto funcType = b.getFunctionType({}, {});
    auto func = func::FuncOp::create(b, loc, "kernel", funcType);
    return func.addEntryBlock();
  }

  // Create a placeholder op in `block` to act as the "rock op" the lookup
  // walks up from.  The lookup is op-type agnostic, so any op suffices.
  arith::ConstantOp makeOp(Block *block) {
    b.setInsertionPointToEnd(block);
    return arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(0));
  }
};

} // namespace

// `rock::getArch` should walk past the function (which doesn't carry the
// attribute) and find `rock.arch` on the enclosing module.
TEST(GetRockInfoTest, GetArchValueFindsModuleAttrFromNestedOp) {
  GetRockInfoTestEnv e;
  (*e.module)->setAttr(rock::ArchAttr::getMnemonic(),
                       e.b.getStringAttr("amdgcn-amd-amdhsa:gfx950"));
  Block *body = e.addFunc();
  auto op = e.makeOp(body);

  auto arch = rock::getArch(op);
  ASSERT_TRUE(succeeded(arch));
  EXPECT_EQ(arch->getValue(), "amdgcn-amd-amdhsa:gfx950");
  EXPECT_EQ(rock::getArchValue(op).getValue(), "amdgcn-amd-amdhsa:gfx950");
}

// Rock kernel attributes are no longer expected on individual rock ops --
// the lookup must ignore them and only consult the function/module.
TEST(GetRockInfoTest, GetArchIgnoresAttrOnTheOpItself) {
  GetRockInfoTestEnv e;
  // Deliberately leave rock.arch off the module and the function.
  Block *body = e.addFunc();
  auto op = e.makeOp(body);
  op->setAttr(rock::ArchAttr::getMnemonic(),
              e.b.getStringAttr("amdgcn-amd-amdhsa:gfx950"));

  EXPECT_TRUE(failed(rock::getArch(op)));
}

// The legacy unprefixed `"num_cu"` name (used by older rocMLIR consumers) is
// no longer accepted -- only the canonical `rock.num_cu` mnemonic is.
TEST(GetRockInfoTest, GetNumCURejectsLegacyUnprefixedName) {
  GetRockInfoTestEnv e;
  Operation *moduleOp = *e.module;
  moduleOp->setAttr(rock::ArchAttr::getMnemonic(),
                    e.b.getStringAttr("amdgcn-amd-amdhsa:gfx950"));
  moduleOp->setAttr("num_cu", e.b.getI64IntegerAttr(256));
  Block *body = e.addFunc();
  auto op = e.makeOp(body);

  EXPECT_TRUE(failed(rock::getNumCU(op)));

  // Sanity check: replacing with the canonical mnemonic succeeds.
  moduleOp->removeAttr("num_cu");
  moduleOp->setAttr(rock::NumCUAttr::getMnemonic(), e.b.getI64IntegerAttr(256));
  auto maybeNumCU = rock::getNumCU(op);
  ASSERT_TRUE(succeeded(maybeNumCU));
  EXPECT_EQ(*maybeNumCU, 256);
}

TEST(GetRockInfoTest, DefaultNumChipletsFollowDefaultNumCU) {
  GetRockInfoTestEnv e;
  (*e.module)->setAttr(rock::ArchAttr::getMnemonic(),
                       e.b.getStringAttr("amdgcn-amd-amdhsa:gfx950"));
  Block *body = e.addFunc();
  auto op = e.makeOp(body);
  auto func = cast<func::FuncOp>(body->getParentOp());

  // With no rock.num_cu the flagship part is assumed, which on CDNA4 is the
  // unpartitioned 256 CUs over all 8 XCDs.
  EXPECT_EQ(rock::getNumCUValue(op), 256);
  EXPECT_EQ(rock::getNumChipletsValueOnFunc(func), 8);
  EXPECT_EQ(rock::getNumChipletsValue(op), 8);
}

// A partitioned device reports fewer CUs than the family's flagship. That is
// valid input and must not be rejected, and the chiplet count has to follow it
// down rather than staying at the flagship's XCD count.
TEST(GetRockInfoTest, PartitionedNumCUIsAcceptedAndDrivesChiplets) {
  GetRockInfoTestEnv e;
  Operation *moduleOp = *e.module;
  moduleOp->setAttr(rock::ArchAttr::getMnemonic(),
                    e.b.getStringAttr("amdgcn-amd-amdhsa:gfx950"));
  Block *body = e.addFunc();
  auto op = e.makeOp(body);
  auto func = cast<func::FuncOp>(body->getParentOp());

  for (auto [numCU, expectedChiplets] :
       {std::pair<int64_t, int64_t>{128, 4}, {64, 2}, {32, 1}}) {
    moduleOp->setAttr(rock::NumCUAttr::getMnemonic(),
                      e.b.getI64IntegerAttr(numCU));
    auto maybeNumCU = rock::getNumCU(op);
    ASSERT_TRUE(succeeded(maybeNumCU)) << numCU;
    EXPECT_EQ(*maybeNumCU, numCU);
    EXPECT_EQ(rock::getNumChipletsValueOnFunc(func), expectedChiplets) << numCU;
  }
}

// An op with no enclosing function should yield a clean failure rather than
// crash.  The module's `rock.arch` is intentionally set to make sure the
// lookup doesn't accidentally fall through to it without going via a func.
TEST(GetRockInfoTest, GetArchReturnsFailureForOpWithoutEnclosingFunc) {
  GetRockInfoTestEnv e;
  (*e.module)->setAttr(rock::ArchAttr::getMnemonic(),
                       e.b.getStringAttr("amdgcn-amd-amdhsa:gfx950"));
  e.b.setInsertionPointToEnd(e.module->getBody());
  auto op = arith::ConstantOp::create(e.b, e.loc, e.b.getI32IntegerAttr(0));

  EXPECT_TRUE(failed(rock::getArch(op)));
}
