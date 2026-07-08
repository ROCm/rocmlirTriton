//===- BridgeVectorizationHints.cpp - Preserve vec hints across canon -----===//
//
// Copyright 2026 The MLIR Authors.
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
//===----------------------------------------------------------------------===//
//
// TransformsToPointerArith attaches im2col vectorization hints
// (`tt.contiguity` / `tt.divisibility`) onto the `tt.addptr` that feeds a
// global load, because Triton's `AxisInfoAnalysis` cannot recover the
// contiguous run length once the im2col coordinate chain is flattened into
// `divui`/`remui`. The AMD `tritonamdgpu-canonicalize-pointers` pass rebuilds
// the pointer arithmetic into fresh ops and drops those discardable attrs, so
// by the time `tritonamdgpu-convert-buffer-ops` runs its `AxisInfoAnalysis` the
// hint is gone and the load is scalarized to `buffer_load_dword`.
//
// This pass bridges that gap entirely on the Rock side, so no patch to the
// Triton submodule is needed. It runs in two phases around
// canonicalize-pointers:
//
//   - `stash` (before): copy the hint from each hinted `tt.addptr` onto the
//     consuming memory op (`tt.load`/`tt.store`) under Rock-private attr names.
//     Memory ops survive canonicalize-pointers with their discardable attrs
//     intact (MaterializeFatPointer recreates them with `op->getAttrs()`).
//   - `apply` (after): re-stamp `tt.contiguity`/`tt.divisibility` onto the
//     defining `tt.addptr` of each marked memory op's pointer operand -- the
//     same op `convert-buffer-ops` resolves via `ptr.getDefiningOp<AddPtrOp>()`
//     -- then erase the Rock-private markers.
//
// The async-copy path does not need this: `coalesce-async-copy` consumes the
// addptr hint before canonicalize-pointers runs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKBRIDGEVECTORIZATIONHINTSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-bridge-vectorization-hints"

using namespace mlir;
using namespace mlir::rock;

namespace {

// Triton attrs consumed by AxisInfoAnalysis; must live on the pointer op.
static constexpr StringLiteral kContiguity = "tt.contiguity";
static constexpr StringLiteral kDivisibility = "tt.divisibility";
// Rock-private carriers, parked on the memory op across canonicalize-pointers.
static constexpr StringLiteral kMarkContiguity = "rock.vec_contiguity";
static constexpr StringLiteral kMarkDivisibility = "rock.vec_divisibility";

/// Return the pointer operand of a global memory op we care about, or a null
/// Value for ops we do not handle.
static Value getMemoryOpPtr(Operation *op) {
  if (auto load = dyn_cast<triton::LoadOp>(op))
    return load.getPtr();
  if (auto store = dyn_cast<triton::StoreOp>(op))
    return store.getPtr();
  return Value();
}

struct RockBridgeVectorizationHintsPass
    : public rock::impl::RockBridgeVectorizationHintsPassBase<
          RockBridgeVectorizationHintsPass> {
  using rock::impl::RockBridgeVectorizationHintsPassBase<
      RockBridgeVectorizationHintsPass>::RockBridgeVectorizationHintsPassBase;

  void runOnOperation() override {
    bool isStash = (phase == "stash");
    if (!isStash && phase != "apply") {
      getOperation()->emitError("rock-bridge-vectorization-hints: unknown "
                                "phase '")
          << phase << "' (expected 'stash' or 'apply')";
      return signalPassFailure();
    }

    getOperation()->walk([&](Operation *op) {
      Value ptr = getMemoryOpPtr(op);
      if (!ptr)
        return;
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      if (!addPtrOp)
        return;

      if (isStash) {
        // addptr (hinted by TransformsToPointerArith) -> memory op marker.
        if (Attribute c = addPtrOp->getDiscardableAttr(kContiguity))
          op->setAttr(kMarkContiguity, c);
        if (Attribute d = addPtrOp->getDiscardableAttr(kDivisibility))
          op->setAttr(kMarkDivisibility, d);
      } else {
        // memory op marker -> rebuilt addptr, then drop the markers.
        if (Attribute c = op->getDiscardableAttr(kMarkContiguity)) {
          addPtrOp->setAttr(kContiguity, c);
          op->removeDiscardableAttr(kMarkContiguity);
        }
        if (Attribute d = op->getDiscardableAttr(kMarkDivisibility)) {
          addPtrOp->setAttr(kDivisibility, d);
          op->removeDiscardableAttr(kMarkDivisibility);
        }
      }
    });
  }
};

} // namespace
