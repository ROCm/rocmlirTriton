//===- CollapseContiguousMerges.cpp - Collapse contiguous merges --------===//
//
// Copyright Advanced Micro Devices, Inc.
// Copyright 2026 The MLIR Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// This pass runs BEFORE RockTransformsToPointerArithPass. It simplifies the
// rock.transform chains feeding TransformsToPtrOp by collapsing contiguous
// merges: a dimension split by a Merge whose members are genuinely contiguous
// in memory, so the decompose/recompose round-trip can be fused into a single
// wider dimension. Keeping the contiguous extent intact yields a composed
// affine map whose offset along that dimension stays stride-1, which lets
// Triton's AxisInfoAnalysis recover the real pointer contiguity and vectorize
// the loads.
//
// The collapse is propagated from the Merge down to the Unmerge that recombines
// the group, traveling only through size-preserving transforms (PassThrough and
// zero Pad). Any other transform along the way (Embed, nested Merge, Slice,
// Broadcast, non-zero Pad, ...) aborts the collapse for that group, so only
// Unmerge-recombined groups are handled today.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKCOLLAPSECONTIGUOUSMERGESPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-collapse-contiguous-merges"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockCollapseContiguousMergesPass
    : public rock::impl::RockCollapseContiguousMergesPassBase<
          RockCollapseContiguousMergesPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void RockCollapseContiguousMergesPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  OpBuilder b(func.getContext());
  func.walk([&](TransformsToPtrOp op) {
    Value source = op.getSource();
    // collapseContiguousMerges rebuilds the transform chain and rewires this
    // op's source onto it. Isolate the chain first (each rock.transform with a
    // single user) so the rewrite is local to this op and can't perturb the
    // coordinate maps of any other op that shares the original chain.
    b.setInsertionPoint(op);
    Value isolated = isolateTransforms(b, source);
    if (isolated != source)
      op.getSourceMutable().assign(isolated);
    collapseContiguousMerges(isolated);
  });
}
