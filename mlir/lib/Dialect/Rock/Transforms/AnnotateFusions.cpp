//===- AnnotateFusions.cpp - name a kernel's fusions ----------------------===//
//
// Copyright Advanced Micro Devices, Inc.
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
// ============================================================
//
// Records the pointwie/elementwise operations fused around a kernel as
// `rock.input_fusions` and `rock.output_fusions`, which getTuningProblemStr
// will later serialize into the tuning key.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKANNOTATEFUSIONSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

void annotateFusions(func::FuncOp func) {
  // By this point in the pipeline the kernel is one operation. Whatever the
  // attention, gemm-elementwise-gemm and conv-elementwise-gemm patterns
  // absorbed sits inside it (a hand-spelled softmax in its region, a
  // log-sum-exp in its `lse` result, a mask in its `causal` flag). These
  // kind of operations can have inter-gemm elementwise operations, but they
  // are intentionally not supported yet by this pass, and are left for future
  // work if needed.
  SmallVector<Operation *> roots;
  func.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::rock::FusionRoot>())
      roots.push_back(op);
  });
  if (roots.size() != 1)
    return;
  Operation *root = roots.front();

  BackwardSliceOptions sliceOpts;
  sliceOpts.omitBlockArguments = true;

  llvm::SetVector<Operation *> above;
  llvm::SetVector<Operation *> below;
  (void)getBackwardSlice(root, &above, sliceOpts);
  getForwardSlice(root, &below);

  // The epilogue reads operands of its own (the bias of an `addf(result, bias)`
  // and the view feeding it) which neither slice from the root contains. Only
  // an op that computes seeds this: the forward slice also reaches the
  // terminator, and slicing backward from that would pull in every other result
  // of the function, which is someone else's work rather than this kernel's
  // epilogue. The slice runs back through the kernel into its operands, so drop
  // whatever the backward slice already claimed.
  llvm::SetVector<Operation *> epilogueInputs;
  for (Operation *op : below)
    if (isFusionOp(op))
      (void)getBackwardSlice(op, &epilogueInputs, sliceOpts);
  for (Operation *op : epilogueInputs)
    if (op != root && !above.contains(op))
      below.insert(op);

  // `isFusionOp` is the dialect's own answer to "does this compute something":
  // an arith or math operation taking operands and returning one result. Emit
  // in program order, which the walks above do not preserve, because the key
  // has to be stable.
  MLIRContext *ctx = func.getContext();
  SmallVector<Attribute> inputFusions;
  SmallVector<Attribute> outputFusions;
  func.walk([&](Operation *op) {
    // What the kernel absorbed lives in its regions, and the forward slice
    // descends into those before it follows the results.
    if (op == root || root->isProperAncestor(op) || !isFusionOp(op))
      return;
    // The mnemonic verbatim, so a token names exactly one operation and can be
    // traced back to the IR it came from.
    auto nameAttr = StringAttr::get(ctx, op->getName().stripDialect());
    if (above.contains(op))
      inputFusions.push_back(nameAttr);
    else if (below.contains(op))
      outputFusions.push_back(nameAttr);
  });

  // Set only when non-empty, so an unfused kernel keeps the key earlier
  // releases wrote and its tuning database rows stay reachable.
  if (!inputFusions.empty())
    func->setAttr(InputFusionsAttr::getMnemonic(),
                  ArrayAttr::get(ctx, inputFusions));
  if (!outputFusions.empty())
    func->setAttr(OutputFusionsAttr::getMnemonic(),
                  ArrayAttr::get(ctx, outputFusions));
}

struct RockAnnotateFusionsPass
    : public rock::impl::RockAnnotateFusionsPassBase<RockAnnotateFusionsPass> {
  void runOnOperation() override { annotateFusions(getOperation()); }
};

} // namespace
