//===- EmulateNarrowTypes.cpp - Sub-byte -> packed i8 emulation -----------===//
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
// Runs the upstream memref/arith narrow-type emulation patterns to rewrite
// loads, stores, allocs, and arithmetic on sub-byte types into packed i8
// storage with bit manipulation.
//
// Must run after RockConvertNarrowTypeSignaturesPass, which converts function
// signatures first.  This ordering avoids a crash in the upstream
// ConvertMemRefLoad/ConvertMemrefStore patterns that call
// extract_strided_metadata on the original (pre-conversion) memref value.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKEMULATENARROWTYPESPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;

namespace {

/// Workaround for upstream MLIR bug: ConvertMemrefStore missing TypeConverter.
///
/// Commit 20b925a28a29 (PR #178498, Jan 2026) refactored ConvertMemrefStore
/// to support disableAtomicRMW but dropped the TypeConverter from its
/// constructor, registering it via patterns.insert<>(context, flag) instead
/// of patterns.add<>(typeConverter, context).  Without a TypeConverter the
/// dialect-conversion framework cannot remap operands that flow through
/// unrealized_conversion_cast ops (e.g. from a prior signature-conversion
/// pass), causing "failed to legalize operation 'memref.store'".
/// TODO: Remove this workaround once the upstream fix lands.
///
/// This pattern folds unrealized_conversion_cast ops whose input already
/// has the type the converter expects.  The preceding
/// RockConvertNarrowTypeSignaturesPass creates casts like
/// `memref<16xi8> -> memref<32xi4>`.  The type converter maps
/// `memref<32xi4>` back to `memref<16xi8>`, so the cast is an identity
/// through conversion and can be replaced by its input.
struct FoldNarrowTypeCast
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1 || adaptor.getInputs().size() != 1)
      return failure();

    Type targetType =
        getTypeConverter()->convertType(op.getResultTypes().front());
    if (!targetType || adaptor.getInputs().front().getType() != targetType)
      return failure();

    rewriter.replaceOp(op, adaptor.getInputs().front());
    return success();
  }
};

static bool has4BitOps(func::FuncOp funcOp) {
  auto is4Bit = [](Type t) {
    Type elem = getElementTypeOrSelf(t);
    return elem.isIntOrFloat() && elem.getIntOrFloatBitWidth() == 4;
  };
  auto result = funcOp.walk([&](Operation *op) {
    for (Type t : op->getResultTypes())
      if (is4Bit(t))
        return WalkResult::interrupt();
    for (Type t : op->getOperandTypes())
      if (is4Bit(t))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

struct RockEmulateNarrowTypesPass
    : public rock::impl::RockEmulateNarrowTypesPassBase<
          RockEmulateNarrowTypesPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    if (!has4BitOps(funcOp))
      return;

    // Function signatures are already converted to packed i8 by the
    // preceding RockConvertNarrowTypeSignaturesPass.  Here we only need
    // to rewrite loads, stores, allocs, and arithmetic on sub-byte types.
    auto typeConverter = rock::create4BitTypeConverter();

    ConversionTarget target(*ctx);

    auto isLegal = [&typeConverter](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalDialect<arith::ArithDialect,
                                      memref::MemRefDialect>(isLegal);

    // The upstream patterns generate affine.apply ops for index
    // linearization (via makeComposedFoldedAffineApply).  Mark the affine
    // dialect legal so the conversion framework accepts them.  A subsequent
    // lower-affine pass in the pipeline will lower them to standard arith.
    target.addLegalDialect<affine::AffineDialect>();

    // The upstream ConvertMemRefLoad/ConvertMemrefStore patterns create
    // extract_strided_metadata on the original (pre-conversion) memref
    // operand.  After the signatures pass converted function signatures,
    // that operand is an unrealized_conversion_cast result whose metadata
    // folds to constants for static memrefs.  The op itself may remain
    // with a sub-byte result type (the base buffer), but it is dead and
    // will be cleaned up by canonicalization.  Mark it legal so it doesn't
    // block the conversion.
    target.addLegalOp<memref::ExtractStridedMetadataOp>();

    // func ops are already legal after signature conversion.
    target.addLegalOp<func::FuncOp, func::CallOp, func::ReturnOp>();

    // Upstream MLIR bug workaround (see FoldNarrowTypeCast above):
    // fold unrealized casts left by the signatures pass so that
    // ConvertMemrefStore (which lacks a TypeConverter) sees remapped operands.
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&typeConverter](UnrealizedConversionCastOp op) {
          return typeConverter.isLegal(op.getResultTypes());
        });

    RewritePatternSet patterns(ctx);
    patterns.add<FoldNarrowTypeCast>(typeConverter, ctx);
    arith::populateArithNarrowTypeEmulationPatterns(typeConverter, patterns);
    memref::populateMemRefNarrowTypeEmulationPatterns(typeConverter, patterns);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                        patterns, target);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // end anonymous namespace
