//===- LinkSplitKernels.cpp - host driver for split kernels --------------===//
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
// AIROCMLIR-709 Phase 3: host driver for cross-tile fusion splits.
//
// rock-split-cross-tile-fusion (SplitCrossTileFusion.cpp) splits one fused
// kernel `@orig` into `@orig_gemm` + `@orig_elementwise` joined by an
// intermediate buffer, records the linkage in a typed `rock.split_link` op, and
// leaves a private stub `@orig`. But the host still calls `@orig` by name. This
// pass recreates a host function `@orig` that:
//
//   func.func @orig(%a0, ..., %aN) -> Tret {         // NOT rock.kernel (host)
//     %dev   = gpu.alloc() : memref<flat>            // intermediate device buf
//     %inter = bufferization.to_tensor %dev ...
//     %g = func.call @orig_gemm(<gemm inputs>, %inter)
//     %o = func.call @orig_elementwise(%g, <tail inputs>, <outputs>)
//     gpu.dealloc %dev
//     return %o
//   }
//
// It runs late (after rock-elementwise-to-gridwise has appended the elementwise
// output argument, so kernel signatures are final) but before
// rock-serialize-host-funcs. The two func.call ops are rewritten to
// gpu.launch_func by rock-emit-gpu-binary (which handles tensor operands and
// destination-passing results), so launches execute in order and the
// intermediate buffer carries data from the gemm to the elementwise kernel.
//
// The original-kernel signature is reconstructed from the arg-source maps on
// the rock.split_link op (see RockOps.td / utility/splitLink.h), which is then
// consumed and erased.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/splitLink.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLINKSPLITKERNELSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockLinkSplitKernelsPass
    : public rock::impl::RockLinkSplitKernelsPassBase<
          RockLinkSplitKernelsPass> {
  void runOnOperation() override;
};
} // namespace

// Reconstruct the original (pre-split) kernel signature and emit the host
// driver function described by one rock.split_link op.
static LogicalResult emitHostDriver(OpBuilder &b, SplitLinkOp link) {
  ModuleOp moduleOp = link->getParentOfType<ModuleOp>();
  auto gemm = moduleOp.lookupSymbol<func::FuncOp>(link.getGemmAttr());
  auto pw = moduleOp.lookupSymbol<func::FuncOp>(link.getElementwiseAttr());
  if (!gemm || !pw)
    return link.emitError("split-link: could not resolve gemm/elementwise "
                          "kernel referenced by rock.split_link");

  StringRef group = link.getOrigAttr().getValue();
  FunctionType gemmTy = gemm.getFunctionType();
  FunctionType pwTy = pw.getFunctionType();

  ArrayRef<int64_t> gemmArgSrc = link.getGemmArgSrc();
  ArrayRef<int64_t> pwArgSrc = link.getElementwiseArgSrc();
  ArrayRef<int64_t> pwOutSrc = link.getElementwiseOutSrc();

  if (gemmArgSrc.size() != gemmTy.getNumInputs())
    return gemm.emitError("split-link: gemm arg_src (")
           << gemmArgSrc.size() << ") does not match kernel arity ("
           << gemmTy.getNumInputs()
           << "); kernel signature changed unexpectedly after the split";
  // The elementwise kernel gains its output argument(s) after the split,
  // appended after the arguments described by arg_src.
  if (pwArgSrc.size() + pwOutSrc.size() != pwTy.getNumInputs())
    return pw.emitError("split-link: elementwise arg_src (")
           << pwArgSrc.size() << ") + out_src (" << pwOutSrc.size()
           << ") does not match kernel arity (" << pwTy.getNumInputs() << ")";
  if (pwTy.getNumResults() != 1)
    return pw.emitError("split-link: expected a single elementwise result");

  // Map original-argument index -> type, gathered from both halves.
  llvm::DenseMap<int64_t, Type> origTypes;
  Type interType;
  auto record = [&](int64_t src, Type ty) {
    if (src == SplitIntermediateArg)
      interType = ty;
    else
      origTypes[src] = ty;
  };
  for (auto [i, src] : llvm::enumerate(gemmArgSrc))
    record(src, gemmTy.getInput(i));
  for (auto [i, src] : llvm::enumerate(pwArgSrc))
    record(src, pwTy.getInput(i));
  for (auto [k, src] : llvm::enumerate(pwOutSrc))
    record(src, pwTy.getInput(pwArgSrc.size() + k));

  if (!interType)
    return gemm.emitError("split-link: no intermediate buffer in metadata");
  auto interTensorTy = dyn_cast<RankedTensorType>(interType);
  if (!interTensorTy)
    return gemm.emitError("split-link: intermediate buffer is not a ranked "
                          "tensor");

  int64_t numOrig = 0;
  for (auto &kv : origTypes)
    numOrig = std::max(numOrig, kv.first + 1);
  SmallVector<Type> origArgTypes(numOrig);
  for (int64_t i = 0; i < numOrig; ++i) {
    auto it = origTypes.find(i);
    if (it == origTypes.end())
      return gemm.emitError("split-link: missing source for original "
                            "argument ")
             << i;
    origArgTypes[i] = it->second;
  }

  Location loc = pw.getLoc();
  Type resultType = pwTy.getResult(0);
  auto orchType = b.getFunctionType(origArgTypes, {resultType});

  // rock-split-cross-tile-fusion leaves a non-kernel declaration named `group`
  // (the original kernel name) so the host's call stays resolvable. Fill in its
  // body here; if absent (e.g. this pass is run standalone in a test), create
  // the function fresh. Intentionally NOT marked rock.kernel: it is a host
  // function.
  func::FuncOp orchFunc = moduleOp.lookupSymbol<func::FuncOp>(group);
  if (orchFunc) {
    if (!orchFunc.isDeclaration())
      return orchFunc.emitError("split-link: function '")
             << group << "' already has a body";
    orchFunc.setType(orchType);
  } else {
    b.setInsertionPoint(gemm);
    orchFunc = func::FuncOp::create(b, loc, group, orchType);
  }

  Block *body = orchFunc.addEntryBlock();
  OpBuilder bb(body, body->end());

  // Allocate the intermediate device buffer and view it as a tensor.
  auto memrefTy = MemRefType::get(interTensorTy.getShape(),
                                  interTensorTy.getElementType());
  auto devBuf = gpu::AllocOp::create(bb, loc, memrefTy, /*asyncToken=*/Type(),
                                     /*asyncDependencies=*/ValueRange{},
                                     /*dynamicSizes=*/ValueRange{},
                                     /*symbolOperands=*/ValueRange{})
                    .getResult(0);
  Value interTensor =
      bufferization::ToTensorOp::create(bb, loc, interTensorTy, devBuf,
                                        /*restrict=*/true, /*writable=*/true);

  // gemm(<gemm inputs>, intermediate) -> intermediate.
  SmallVector<Value> gemmOperands;
  for (int64_t src : gemmArgSrc)
    gemmOperands.push_back(src == SplitIntermediateArg
                               ? interTensor
                               : orchFunc.getArgument(src));
  auto gemmCall = func::CallOp::create(bb, loc, gemm, gemmOperands);

  // elementwise(intermediate, <tail inputs>, <outputs>) -> output. The gemm
  // result feeds the elementwise intermediate input (a data dependency that
  // becomes the buffer pointer once both calls are lowered to gpu.launch_func).
  SmallVector<Value> pwOperands;
  for (int64_t src : pwArgSrc)
    pwOperands.push_back(src == SplitIntermediateArg
                             ? gemmCall.getResult(0)
                             : orchFunc.getArgument(src));
  for (int64_t src : pwOutSrc)
    pwOperands.push_back(orchFunc.getArgument(src));
  auto pwCall = func::CallOp::create(bb, loc, pw, pwOperands);

  gpu::DeallocOp::create(bb, loc, TypeRange{}, ValueRange{devBuf});
  func::ReturnOp::create(bb, loc, pwCall.getResult(0));
  return success();
}

void RockLinkSplitKernelsPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  SmallVector<SplitLinkOp> links;
  moduleOp.walk([&](SplitLinkOp op) { links.push_back(op); });
  if (links.empty())
    return;

  OpBuilder b(moduleOp);
  for (SplitLinkOp link : links) {
    if (failed(emitHostDriver(b, link)))
      return signalPassFailure();
    // The linkage is consumed; erase the op so it does not reach serialization.
    link.erase();
  }
}
