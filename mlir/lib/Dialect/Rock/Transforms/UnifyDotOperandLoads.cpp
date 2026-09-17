//===- UnifyDotOperandLoads.cpp - One layout per dot operand chain -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs right after Triton's tritongpu-coalesce, which picks a layout per
// memory op from that op's own pointer contiguity. An input fusion puts more
// than one load in a tt.dot operand's chain, those loads can land in different
// layouts, and the elementwise op combining them then needs a convert_layout
// to bring them together. That conversion is scratch LDS the unfused kernel
// never allocated.
//
// Broadcasts and transposes live in the rock.transform index math here, so
// every load in a fused chain has the same shape, and one common layout is
// always legal: a layout says which thread holds which logical element, the
// indexing says which address it reads. So the conversion can be removed
// rather than made cheaper.
//
// The layout comes from the chain's largest load, per rock.load_tensor_bytes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKUNIFYDOTOPERANDLOADSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-unify-dot-operand-loads"

using namespace mlir;
using namespace mlir::rock;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {
struct RockUnifyDotOperandLoadsPass
    : public rock::impl::RockUnifyDotOperandLoadsPassBase<
          RockUnifyDotOperandLoadsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// The tensor size Rock recorded for `loadOp`, or nothing when the load was not
// tagged. An untagged load is one this pipeline did not create, so there is
// nothing to say about which tensor is behind it and it is left out of the
// running rather than guessed at.
static std::optional<int64_t> getLoadTensorBytes(tt::LoadOp loadOp) {
  auto attr = dyn_cast_or_null<IntegerAttr>(
      loadOp->getDiscardableAttr(rock::LoadTensorBytesAttr::getMnemonic()));
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

// Collect the loads feeding `root` through ops that preserve the layout of
// their operands elementwise. The walk stops at anything else, since a shape
// change means the layouts were never comparable in the first place.
//
// Only same-shape ops are crossed: a broadcast or a reshape between a load and
// the dot would make "one common layout" meaningless, and in this pipeline
// those live in the index math rather than as Triton ops.
static void collectChainLoads(Value root, llvm::SetVector<tt::LoadOp> &loads) {
  auto rootType = dyn_cast<RankedTensorType>(root.getType());
  if (!rootType)
    return;

  SmallVector<Value> worklist{root};
  llvm::SmallPtrSet<Value, 16> seen;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;

    Operation *def = value.getDefiningOp();
    if (!def)
      continue;

    if (auto loadOp = dyn_cast<tt::LoadOp>(def)) {
      loads.insert(loadOp);
      continue;
    }

    // A conversion is what this pass exists to remove, so look through it to
    // the load on the far side.
    if (auto cvtOp = dyn_cast<ttg::ConvertLayoutOp>(def)) {
      worklist.push_back(cvtOp.getSrc());
      continue;
    }

    if (!def->hasTrait<OpTrait::Elementwise>() || !isMemoryEffectFree(def))
      continue;

    for (Value operand : def->getOperands()) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (operandType && operandType.getShape() == rootType.getShape())
        worklist.push_back(operand);
    }
  }
}

// Rewrite every load in `loads` that does not already produce `encoding` so
// that it does. convertDistributedOpEncoding relayouts the pointer and mask
// operands along with the load and converts the result back, leaving a
// conversion that remove-layout-conversions folds into the consumers.
//
// Returns whether anything changed.
static bool unifyLoadEncodings(ArrayRef<tt::LoadOp> loads, Attribute encoding) {
  bool changed = false;
  for (tt::LoadOp loadOp : loads) {
    auto type = dyn_cast<RankedTensorType>(loadOp.getType());
    if (!type || type.getEncoding() == encoding)
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "relayouting " << loadOp << " to " << encoding << "\n");
    convertDistributedOpEncoding(encoding, loadOp);
    changed = true;
  }
  return changed;
}

void RockUnifyDotOperandLoadsPass::runOnOperation() {
  // Gather the work before doing any of it: convertDistributedOpEncoding
  // erases the op it rewrites, which would invalidate a walk in progress.
  SmallVector<std::pair<SmallVector<tt::LoadOp>, Attribute>> work;

  getOperation()->walk([&](Operation *dotOp) {
    if (!isa<tt::DotOp, tt::DotScaledOp>(dotOp))
      return;

    for (Value operand : dotOp->getOperands()) {
      llvm::SetVector<tt::LoadOp> loads;
      collectChainLoads(operand, loads);
      if (loads.size() < 2)
        continue;

      // The largest tagged load sets the layout. Without at least one tag
      // there is no basis for choosing, so the chain is left alone.
      tt::LoadOp leader;
      int64_t leaderBytes = 0;
      for (tt::LoadOp loadOp : loads) {
        std::optional<int64_t> bytes = getLoadTensorBytes(loadOp);
        if (!bytes || *bytes <= leaderBytes)
          continue;
        leader = loadOp;
        leaderBytes = *bytes;
      }
      if (!leader)
        continue;

      auto leaderType = dyn_cast<RankedTensorType>(leader.getType());
      if (!leaderType || !leaderType.getEncoding())
        continue;

      work.emplace_back(SmallVector<tt::LoadOp>(loads.begin(), loads.end()),
                        leaderType.getEncoding());
    }
  });

  for (auto &[loads, encoding] : work)
    unifyLoadEncodings(loads, encoding);
}
