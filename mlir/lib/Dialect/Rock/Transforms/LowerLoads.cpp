//===- LowerLoads.cpp - Lower rock.load_marker ops to blockwise loads -----===//
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
// =============================================================================
//
// This pass runs AFTER RegularizeInput. Each rock.load_marker's source is
// guaranteed to be a pure transform chain (no fusions). The pass applies the
// marker's extraViews transforms on top of the source and creates a
// rock.blockwise_load.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERLOADSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

struct LowerLoadMarker : public OpRewritePattern<LoadMarkerOp> {
  using OpRewritePattern<LoadMarkerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadMarkerOp markerOp,
                                PatternRewriter &rewriter) const override {
    Value source = markerOp.getSource();
    ArrayAttr extraViews = markerOp.getExtraViews();

    if (extraViews.size() > 0)
      source = rock::transform(rewriter, source, extraViews);

    auto loadTileOp = BlockwiseLoadOp::create(
        rewriter, markerOp.getLoc(), markerOp.getResult().getType(), source,
        markerOp.getExtraIndices());

    rewriter.replaceOp(markerOp, loadTileOp);
    return success();
  }
};

struct RockLowerLoadsPass
    : public rock::impl::RockLowerLoadsPassBase<RockLowerLoadsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockLowerLoadsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  MLIRContext *ctx = &getContext();

  ConversionTarget target(*ctx);
  target.addIllegalOp<LoadMarkerOp>();
  target.addLegalOp<TransformOp, BlockwiseLoadOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<LowerLoadMarker>(ctx);
  if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
    signalPassFailure();
}
