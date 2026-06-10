//===- SetMatmulOutputTranspose.cpp - Pick accel result transpose --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs right after Triton's tritonamdgpu-accelerate-matmul, which rewrites
// tt.dot into an accelerator-encoded dot (e.g. WMMA) using a fixed default
// result transposition. For each accelerator dot carrying the rock.o_transposed
// metadata (set by rock-add-triton-metadata and preserved across the
// accelerate-matmul rewrite), this pass overrides the result layout's
// transposition so the consuming epilogue store can be vectorized/coalesced.
//
// The orientation is read from rock.o_transposed (computed at the rock level)
// instead of walking forward to the store. Only the WMMA (ttg.amd_wmma) result
// layout is adjusted for now; other accelerator layouts keep their default.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSETMATMULOUTPUTTRANSPOSEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-set-matmul-output-transpose"

using namespace mlir;
using namespace mlir::rock;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {
struct RockSetMatmulOutputTransposePass
    : public rock::impl::RockSetMatmulOutputTransposePassBase<
          RockSetMatmulOutputTransposePass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// Map an output layout (oTransposed: true = column-major / M fast) to the WMMA
// result `isTransposed` flag, accounting for the version-dependent flip.
//   row-major output (oTransposed = false): isTransposed = (version > 1)
//   col-major output (oTransposed = true):  isTransposed = (version == 1)
static bool wmmaIsTransposedFor(bool oTransposed, unsigned wmmaVersion) {
  return oTransposed ? (wmmaVersion == 1) : (wmmaVersion > 1);
}

// Recompute the encoding of an operand of a WMMA dot so that it references
// `newWmma` instead of `oldWmma`. Returns nullptr if the operand encoding does
// not reference the WMMA layout (e.g. block-scaled scale operands).
static Attribute remapOperandEncoding(Attribute enc,
                                      ttg::AMDWmmaEncodingAttr oldWmma,
                                      ttg::AMDWmmaEncodingAttr newWmma) {
  if (enc == oldWmma)
    return newWmma;
  if (auto dotEnc = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(enc)) {
    if (dotEnc.getParent() == oldWmma)
      return ttg::DotOperandEncodingAttr::get(
          enc.getContext(), dotEnc.getOpIdx(), newWmma, dotEnc.getKWidth());
  }
  return nullptr;
}

namespace {
// For each tt.dot / tt.dot_scaled carrying rock.o_transposed, pick the WMMA
// result `isTranspose` flag from the metadata. When the flag has to change, the
// dot is rebuilt onto the new WMMA layout with convert_layout ops bridging the
// old operand/result encodings; the downstream remove-layout-conversions pass
// folds the redundant conversions away.
template <typename DotOpTy>
struct SetWmmaResultTransposePattern : public OpRewritePattern<DotOpTy> {
  using OpRewritePattern<DotOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOpTy op,
                                PatternRewriter &rewriter) const override {
    Operation *dotOp = op;
    // This pass only rewrites the dots tagged by rock-add-triton-metadata;
    // untagged dots are not ours to touch.
    auto attr = dyn_cast_or_null<rock::OTransposedAttr>(
        dotOp->getDiscardableAttr(rock::OTransposedAttr::getNameStr()));
    if (!attr)
      return failure();
    bool oTransposed = attr.getValue();

    auto resTy = dyn_cast<RankedTensorType>(dotOp->getResult(0).getType());
    auto wmma =
        resTy ? dyn_cast_or_null<ttg::AMDWmmaEncodingAttr>(resTy.getEncoding())
              : nullptr;
    // Only WMMA is handled for now.
    if (!wmma)
      return failure();

    bool desired = wmmaIsTransposedFor(oTransposed, wmma.getVersion());
    if (desired == wmma.getIsTransposed())
      return failure();

    auto newWmma = ttg::AMDWmmaEncodingAttr::get(
        wmma.getContext(), wmma.getVersion(), wmma.getCtaLayout(), desired,
        wmma.getCGALayout(), wmma.getInstrShape());

    Location loc = dotOp->getLoc();
    SmallVector<Value> newOperands;
    newOperands.reserve(dotOp->getNumOperands());
    for (Value operand : dotOp->getOperands()) {
      auto ty = dyn_cast<RankedTensorType>(operand.getType());
      if (!ty) {
        newOperands.push_back(operand);
        continue;
      }
      Attribute newEnc = remapOperandEncoding(ty.getEncoding(), wmma, newWmma);
      if (!newEnc) {
        newOperands.push_back(operand);
        continue;
      }
      auto newTy =
          RankedTensorType::get(ty.getShape(), ty.getElementType(), newEnc);
      newOperands.push_back(
          ttg::ConvertLayoutOp::create(rewriter, loc, newTy, operand));
    }

    Operation *newDot = rewriter.clone(*dotOp);
    newDot->setOperands(newOperands);
    auto newResTy = RankedTensorType::get(resTy.getShape(),
                                          resTy.getElementType(), newWmma);
    newDot->getResult(0).setType(newResTy);

    // The dot result now has the new (newWmma) encoding, but its consumers
    // still expect the original layout; bridge back with a convert_layout that
    // remove-layout-conversions folds away.
    Value back = ttg::ConvertLayoutOp::create(rewriter, loc, resTy,
                                              newDot->getResult(0));
    rewriter.replaceOp(op, back);
    return success();
  }
};
} // end anonymous namespace

void RockSetMatmulOutputTransposePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<SetWmmaResultTransposePattern<tt::DotOp>,
               SetWmmaResultTransposePattern<tt::DotScaledOp>>(ctx);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
