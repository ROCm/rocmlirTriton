//===- AnnotateFusions.cpp - name a kernel's fusions ------------------===//
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
// Records the operations fused around a kernel as `rock.input_fusions` and
// `rock.output_fusions`, which getTuningProblemStr serializes into the tuning
// key. Runs first in the MIGraphX pipeline because it is the last point at
// which the graph still says what MIGraphX asked for.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace migraphx {
#define GEN_PASS_DEF_MIGRAPHXANNOTATEFUSIONSPASS
#include "mlir/Dialect/MIGraphX/Passes.h.inc"
} // namespace migraphx
} // namespace mlir

using namespace mlir;
using namespace mlir::migraphx;

namespace {

// The operations that become an `OpTrait::rock::FusionRoot`, which cannot be
// asked for directly here because no Rock operation exists this early. The two
// are not one-to-one and that is why this pass has to run at all: a lone one of
// these becomes `rock.gemm` or `rock.conv`, but a *pair* of them collapses into
// a single `rock.attention`, `rock.gemm_elementwise_gemm` or
// `rock.conv_elementwise_gemm`, taking whatever sits between them along.
bool isFusionRootCandidate(Operation *op) {
  return isa<DotOp, QuantDotOp, ConvolutionOp, QuantConvolutionOp,
             ConvolutionBwdDataOp>(op);
}

void walkBackward(ValueRange values, llvm::SmallPtrSetImpl<Operation *> &out) {
  SmallVector<Value> worklist(values.begin(), values.end());
  while (!worklist.empty()) {
    Operation *def = worklist.pop_back_val().getDefiningOp();
    if (!def || !out.insert(def).second)
      continue;
    worklist.append(def->operand_begin(), def->operand_end());
  }
}

void walkForward(ValueRange values, llvm::SmallPtrSetImpl<Operation *> &out) {
  SmallVector<Value> worklist(values.begin(), values.end());
  while (!worklist.empty()) {
    for (Operation *user : worklist.pop_back_val().getUsers()) {
      if (!out.insert(user).second)
        continue;
      worklist.append(user->result_begin(), user->result_end());
    }
  }
}

void annotateFusions(func::FuncOp func) {
  SmallVector<Operation *> roots;
  func.walk([&](Operation *op) {
    if (isFusionRootCandidate(op))
      roots.push_back(op);
  });
  if (roots.empty())
    return;

  SmallVector<Value> rootOperands;
  SmallVector<Value> rootResults;
  for (Operation *op : roots) {
    rootOperands.append(op->operand_begin(), op->operand_end());
    rootResults.append(op->result_begin(), op->result_end());
  }
  llvm::SmallPtrSet<Operation *, 8> above;
  llvm::SmallPtrSet<Operation *, 8> below;
  walkBackward(rootOperands, above);
  walkForward(rootResults, below);

  // An operation both above and below a candidate sits between two of them, and
  // that is precisely what the attention, gemm-elementwise-gemm and
  // conv-elementwise-gemm patterns absorb into the single operation they
  // synthesize. It is therefore kernel, not fusion, which is what keeps an
  // attention's hand-spelled softmax out of the key.
  llvm::SmallPtrSet<Operation *, 8> core(roots.begin(), roots.end());
  for (Operation *op : above)
    if (below.contains(op))
      core.insert(op);

  // The epilogue reads operands of its own (the bias of an `add(result, bias)`
  // and the broadcast behind it) which neither walk from a root reaches.
  llvm::SmallPtrSet<Operation *, 8> epilogueInputs;
  for (Operation *op : below)
    if (!core.contains(op))
      walkBackward(op->getOperands(), epilogueInputs);
  for (Operation *op : epilogueInputs)
    if (!above.contains(op) && !core.contains(op))
      below.insert(op);

  // Emit in program order, which the walks above do not preserve, because the
  // key has to be stable. A literal is a baked-in constant rather than work; the
  // broadcast that spreads it still counts.
  MLIRContext *ctx = func.getContext();
  SmallVector<Attribute> inputFusions;
  SmallVector<Attribute> outputFusions;
  func.walk([&](Operation *op) {
    if (core.contains(op) || isa<LiteralOp>(op) ||
        op->getDialect() != ctx->getLoadedDialect<MIGraphXDialect>())
      return;
    auto name = StringAttr::get(ctx, op->getName().stripDialect());
    if (above.contains(op))
      inputFusions.push_back(name);
    else if (below.contains(op))
      outputFusions.push_back(name);
  });

  // Set only when non-empty, so an unfused kernel keeps the key earlier
  // releases wrote and its tuning database rows stay reachable.
  if (!inputFusions.empty())
    func->setAttr(rock::InputFusionsAttr::getMnemonic(),
                  ArrayAttr::get(ctx, inputFusions));
  if (!outputFusions.empty())
    func->setAttr(rock::OutputFusionsAttr::getMnemonic(),
                  ArrayAttr::get(ctx, outputFusions));
}

struct MIGraphXAnnotateFusionsPass
    : public migraphx::impl::MIGraphXAnnotateFusionsPassBase<
          MIGraphXAnnotateFusionsPass> {
  void runOnOperation() override { annotateFusions(getOperation()); }
};

} // namespace
