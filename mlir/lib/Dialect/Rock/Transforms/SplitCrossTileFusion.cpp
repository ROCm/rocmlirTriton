//===- SplitCrossTileFusion.cpp - split cross-tile output fusions --------===//
//
// Copyright 2026 Advanced Micro Devices.
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
// ============================================================
//
// Splits a fusion that combines two different slices of the same gemm output
// (e.g. addf(gemm[:,0:12,:], gemm[:,12:24,:])). This cannot be a single fused
// gemm-writeback kernel because the two slices live in different workgroup
// tiles; rock-regularize-output miscompiles it by collapsing both slices to the
// same gemm value (addf(%g, %g) = 2*gemm).
//
// Running before rock-regularize-output, this pass splits the kernel into two
// kernels joined by an intermediate global buffer:
//
//   gemm kernel:        gemm inputs   -> rock.store %g to %intermediate
//   elementwise kernel: %intermediate -> slice + slice + add -> rock.store
//
// The elementwise kernel has no FusionRoot, so it flows through the
// pure-elementwise path (rock-elementwise-to-gridwise /
// rock-gridwise-elementwise-to-blockwise) reading a fully-materialized buffer.
// The host driver is built later by rock-link-split-kernels.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/splitLink.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSPLITCROSSTILEFUSIONPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-split-cross-tile-fusion"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockSplitCrossTileFusionPass
    : public rock::impl::RockSplitCrossTileFusionPassBase<
          RockSplitCrossTileFusionPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// Forward flood-fill from `root`, recording the accumulated rock.transform
// stack reaching each value. Detect whether some fusion op is reached from the
// root through two operands with *different* transform stacks (e.g. two Slices
// with different offsets) — the cross-tile pattern this pass splits.
static bool hasCrossTileFusion(Value root) {
  DenseMap<Value, SmallVector<Attribute>> stackOf;
  SmallVector<Value> work{root};
  stackOf[root] = {};

  bool crossTile = false;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    SmallVector<Attribute> stack = stackOf[v];

    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();
      if (auto tOp = dyn_cast<TransformOp>(owner)) {
        SmallVector<Attribute> newStack(stack);
        newStack.push_back(tOp.getTransform());
        Value res = tOp.getResult();
        if (!stackOf.count(res)) {
          stackOf[res] = newStack;
          work.push_back(res);
        }
      } else if (isFusionOp(owner)) {
        // Inspect operands that trace back to the root.
        const SmallVector<Attribute> *first = nullptr;
        for (Value operand : owner->getOperands()) {
          auto it = stackOf.find(operand);
          if (it == stackOf.end())
            continue;
          if (!first)
            first = &it->second;
          else if (ArrayRef<Attribute>(*first) != ArrayRef<Attribute>(it->second))
            crossTile = true;
        }
        for (Value res : owner->getResults()) {
          if (!stackOf.count(res)) {
            stackOf[res] = stack;
            work.push_back(res);
          }
        }
      }
    }
  }
  return crossTile;
}

// Collect, in program order, the ops that transitively produce `rootOp`'s
// operands (the gemm-input chain), including `rootOp` itself.
static SmallVector<Operation *> collectHeadOps(Operation *rootOp) {
  DenseSet<Operation *> seen;
  SmallVector<Operation *> stack{rootOp};
  while (!stack.empty()) {
    Operation *op = stack.pop_back_val();
    if (!seen.insert(op).second)
      continue;
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        stack.push_back(def);
  }
  SmallVector<Operation *> ordered(seen.begin(), seen.end());
  llvm::sort(ordered,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return ordered;
}

// Collect, in program order, the ops forward-reachable from `rootResult`
// (the slice/add/merge/store tail), excluding the function terminator.
static SmallVector<Operation *> collectTailOps(Value rootResult) {
  DenseSet<Operation *> seen;
  SmallVector<Operation *> stack;
  for (Operation *user : rootResult.getUsers())
    stack.push_back(user);
  while (!stack.empty()) {
    Operation *op = stack.pop_back_val();
    if (isa<func::ReturnOp>(op))
      continue;
    if (!seen.insert(op).second)
      continue;
    for (Value res : op->getResults())
      for (Operation *user : res.getUsers())
        stack.push_back(user);
  }
  SmallVector<Operation *> ordered(seen.begin(), seen.end());
  llvm::sort(ordered,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return ordered;
}

// Block arguments of `funcOp` used (as operands) by any op in `ops`, in
// ascending argument-number order.
static SmallVector<BlockArgument>
usedBlockArgs(func::FuncOp funcOp, ArrayRef<Operation *> ops) {
  DenseSet<Operation *> opSet(ops.begin(), ops.end());
  SmallVector<bool> used(funcOp.getNumArguments(), false);
  for (Operation *op : ops)
    for (Value operand : op->getOperands())
      if (auto barg = dyn_cast<BlockArgument>(operand))
        if (barg.getOwner() == &funcOp.getBody().front())
          used[barg.getArgNumber()] = true;
  SmallVector<BlockArgument> result;
  for (auto [idx, isUsed] : llvm::enumerate(used))
    if (isUsed)
      result.push_back(funcOp.getArgument(idx));
  return result;
}

// Carry over the rock kernel marker + device attributes onto a split kernel.
static void copyKernelAttrs(func::FuncOp from, func::FuncOp to) {
  for (StringRef name :
       {rock::KernelAttr::getMnemonic(), rock::ArchAttr::getMnemonic(),
        rock::NumCUAttr::getMnemonic(), rock::NumChipletsAttr::getMnemonic()}) {
    if (Attribute attr = from->getAttr(name))
      to->setAttr(name, attr);
  }
}

// Split `funcOp` (which contains a single gemm-like FusionRoot feeding a
// cross-tile output fusion) into a gemm kernel + an elementwise kernel joined
// by an intermediate buffer of the root's full shape.
static LogicalResult splitKernel(func::FuncOp funcOp, Operation *rootOp,
                                 StoreOp storeOp) {
  Value rootResult = rootOp->getResult(0);
  auto bufType = cast<RankedTensorType>(rootResult.getType());
  // Rock global buffers are 1-D linearized tensors at the function boundary, so
  // the intermediate buffer uses the flattened type: the gemm kernel flattens
  // its result on the way out and the elementwise kernel expands it back.
  auto flatBufType = cast<RankedTensorType>(getFlattenedType(bufType));

  // Generic per-dimension names for the flatten/expand views.
  SmallVector<std::string> dimNameStorage;
  for (int64_t i = 0; i < bufType.getRank(); ++i)
    dimNameStorage.push_back(("dim" + Twine(i)).str());
  SmallVector<StringRef> dimNames(dimNameStorage.begin(), dimNameStorage.end());

  SmallVector<Operation *> headOps = collectHeadOps(rootOp);
  SmallVector<Operation *> tailOps = collectTailOps(rootResult);

  // Drop the original store: the elementwise kernel returns the pre-store value
  // and carries no rock.store / output arg, so rock-elementwise-to-gridwise can
  // add its own (it bails if a store already exists).
  SmallVector<Operation *> tailOpsNoStore;
  for (Operation *op : tailOps)
    if (op != storeOp.getOperation())
      tailOpsNoStore.push_back(op);

  SmallVector<BlockArgument> headArgs = usedBlockArgs(funcOp, headOps);
  SmallVector<BlockArgument> tailArgs = usedBlockArgs(funcOp, tailOpsNoStore);

  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  OpBuilder b(funcOp);
  Location loc = funcOp.getLoc();
  std::string baseName = funcOp.getName().str();

  // Argument-source maps recorded on the rock.split_link op (see below).
  SmallVector<int64_t> gemmArgSrc, elemArgSrc, elemOutSrc;

  // --- kernel A: gemm inputs -> store gemm result to the intermediate buf ---
  {
    SmallVector<Type> argTypes;
    for (BlockArgument arg : headArgs)
      argTypes.push_back(arg.getType());
    argTypes.push_back(flatBufType); // intermediate output buffer (1-D)

    auto funcType = b.getFunctionType(argTypes, {flatBufType});
    auto gemmFunc =
        func::FuncOp::create(b, loc, baseName + "_gemm", funcType);
    copyKernelAttrs(funcOp, gemmFunc);

    for (BlockArgument arg : headArgs)
      gemmArgSrc.push_back(arg.getArgNumber());
    gemmArgSrc.push_back(SplitIntermediateArg);

    Block *body = gemmFunc.addEntryBlock();
    OpBuilder bodyB(body, body->end());
    IRMapping map;
    for (auto [i, arg] : llvm::enumerate(headArgs))
      map.map(Value(arg), body->getArgument(i));
    for (Operation *op : headOps)
      bodyB.clone(*op, map);

    Value bufArg = body->getArgument(headArgs.size());
    Value clonedRoot = map.lookup(rootResult);
    // Flatten the logical N-D gemm result to the 1-D buffer layout.
    Value flatResult = flattenOutput(bodyB, loc, clonedRoot, dimNames);
    auto storeMethodAttr = bodyB.getAttr<StoreMethodAttr>(StoreMethod::Set);
    auto store = StoreOp::create(bodyB, loc, /*result=*/flatBufType,
                                 /*source=*/flatResult, /*dest=*/bufArg,
                                 storeMethodAttr);
    func::ReturnOp::create(bodyB, loc, store.getResult());
  }

  // --- kernel B: intermediate buf -> slice + slice + add -> store output ---
  {
    SmallVector<Type> argTypes;
    argTypes.push_back(flatBufType); // intermediate input buffer (1-D)
    for (BlockArgument arg : tailArgs)
      argTypes.push_back(arg.getType());

    // Returns the pre-store value; output arg + store added later.
    Type retType = storeOp.getSource().getType();
    auto funcType = b.getFunctionType(argTypes, {retType});
    auto pwFunc =
        func::FuncOp::create(b, loc, baseName + "_elementwise", funcType);
    copyKernelAttrs(funcOp, pwFunc);

    elemArgSrc.push_back(SplitIntermediateArg);
    for (BlockArgument arg : tailArgs)
      elemArgSrc.push_back(arg.getArgNumber());
    if (auto outArg = dyn_cast<BlockArgument>(storeOp.getDest()))
      elemOutSrc.push_back(outArg.getArgNumber());

    Block *body = pwFunc.addEntryBlock();
    OpBuilder bodyB(body, body->begin());
    // Expand the flat intermediate argument back to the logical N-D shape so
    // the cloned slice views see the same type they did in the fused kernel.
    SmallVector<SmallVector<StringRef>> argNames{dimNames};
    SmallVector<Value> expanded;
    expandFlatFunctionArguments(bodyB, pwFunc, argNames, TypeRange{bufType},
                                expanded);

    IRMapping map;
    map.map(rootResult, expanded.front());
    for (auto [i, arg] : llvm::enumerate(tailArgs))
      map.map(Value(arg), body->getArgument(i + 1));
    for (Operation *op : tailOpsNoStore)
      bodyB.clone(*op, map);
    func::ReturnOp::create(bodyB, loc, map.lookupOrDefault(storeOp.getSource()));
  }

  // Leave a non-kernel declaration with the original name and signature in place
  // of the erased kernel, so the host's call to `@orig` stays resolvable (and
  // module verification stays happy) until rock-link-split-kernels fills in its
  // body once both kernels' signatures are final.
  b.setInsertionPoint(funcOp);
  auto stub = func::FuncOp::create(b, loc, baseName, funcOp.getFunctionType());
  // A bodyless declaration must be private (calls to it are intra-module).
  stub.setPrivate();

  // Record the linkage in a typed op for rock-link-split-kernels. The symbol
  // references also keep both halves + the stub alive across the pipeline.
  MLIRContext *ctx = b.getContext();
  auto ref = [&](const Twine &n) {
    return FlatSymbolRefAttr::get(ctx, n.str());
  };
  b.setInsertionPointToEnd(moduleOp.getBody());
  SplitLinkOp::create(b, loc, ref(baseName), ref(baseName + "_gemm"),
                      ref(baseName + "_elementwise"),
                      b.getDenseI64ArrayAttr(gemmArgSrc),
                      b.getDenseI64ArrayAttr(elemArgSrc),
                      b.getDenseI64ArrayAttr(elemOutSrc));

  funcOp.erase();
  return success();
}

void RockSplitCrossTileFusionPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  SmallVector<func::FuncOp> kernelFuncs;
  moduleOp.walk([&](func::FuncOp funcOp) {
    if (funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
      kernelFuncs.push_back(funcOp);
  });

  for (func::FuncOp funcOp : kernelFuncs) {
    // Find FusionRoots and stores in this kernel.
    SmallVector<Operation *> fusionRoots;
    SmallVector<StoreOp> stores;
    funcOp.walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::rock::FusionRoot>())
        fusionRoots.push_back(op);
      if (auto st = dyn_cast<StoreOp>(op))
        stores.push_back(st);
    });

    // Only handle the simple single-root, single-store, single-result case.
    if (fusionRoots.size() != 1 || stores.size() != 1)
      continue;
    Operation *rootOp = fusionRoots.front();
    if (rootOp->getNumResults() != 1)
      continue;

    if (!hasCrossTileFusion(rootOp->getResult(0)))
      continue;

    LLVM_DEBUG(llvm::dbgs()
               << "Splitting cross-tile fusion kernel: " << funcOp.getName()
               << "\n");
    if (failed(splitKernel(funcOp, rootOp, stores.front())))
      return signalPassFailure();
  }
}
