//===- LegalizeMathForTriton.cpp - math ops Triton cannot carry -----------===//
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
//===----------------------------------------------------------------------===//
//
// Rewrite math.tanh and math.powf into the op subset that Triton's
// TritonToTritonGPU conversion accepts (populateMathPatternsAndLegality in
// TritonToTritonGPUPass.cpp).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLEGALIZEMATHFORTRITONPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

/// The Triton frontend's HIP libdevice bindings declare an f32 and an f64 entry
/// point for each of these (third_party/amd/language/hip/libdevice.py), so a
/// call is evaluated at whichever of the two fits the operand. Hand-mirrored
/// from that file; re-audited on every Triton bump (see
/// docs/bump_triton_version.md section 5.4.2).
constexpr StringLiteral kOcmlTanhF32 = "__ocml_tanh_f32";
constexpr StringLiteral kOcmlTanhF64 = "__ocml_tanh_f64";
constexpr StringLiteral kOcmlPowF32 = "__ocml_pow_f32";
constexpr StringLiteral kOcmlPowF64 = "__ocml_pow_f64";

/// `type` with its element type replaced, preserving shape for tensors.
static Type withElementType(Type type, Type elemTy) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.clone(elemTy);
  return elemTy;
}

/// tanh(x) = 2 / (1 + exp2(-2*log2(e)*x)) - 1
///
/// Folding the -2*log2(e) into the exp2 argument brings this down to five VALU
/// ops: mul, exp, add, rcp, fma. The identity is exact for either sign, so
/// there is no abs and no select, and the ends saturate on their own: a large
/// negative input overflows the exponential to +inf, 1/+inf is exactly +0, and
/// fma(2, 0, -1) is exactly -1. `ninf` is therefore the one fast-math flag it
/// cannot survive: the ops go out unflagged, rock-allow-fast-math-flags
/// attaches the rest including the `arcp` behind the v_rcp_f32, and
/// tanh-no-ninf.mlir pins that ninf is not among them.
///
/// Always evaluated at f32, and only for a result about to be rounded to a
/// narrower float; TanhLowering enforces both. The error is a roughly fixed
/// 2^-22 in absolute terms, which a round down to f16 or below discards
/// entirely and an f32 consumer does not.
///
/// Accuracy, measured exhaustively over every bit pattern of each type on
/// gfx1100 against __ocml_tanh_f64, driving the same instructions this lowers
/// to so the hardware's own error in exp2 and rcp is included.
/// scripts/tanh-accuracy reproduces the table; re-run it if this changes.
///
///                                    ULP       ULP        abs
///                                   here      ocml       here
///   f16    |x| >= 2^-11              0.5       0.5    2.4e-04
///          anywhere                  1.0       0.5    2.4e-04
///   bf16   |x| >= 2^-16              0.5       0.5    1.9e-03
///          anywhere                334.0       0.5    1.9e-03
///   f8E4M3 anywhere                  0.48      0.48   3.0e-02
///   f8E5M2 anywhere                  0.47      0.47   5.9e-02
///   f4E2M1 anywhere                  0.48      0.48   2.4e-01
///
/// At or below 0.5 ULP is correctly rounded; for fp8 and fp4 the sweep compares
/// code points and all 1028 of them match the library call. No sweep produced a
/// NaN and every overflow saturated to exactly +/-1.
///
/// The large ULP figures are a near-zero artifact: below about 2^-24 the
/// reciprocal rounds to exactly 1/2, the result collapses to zero, and the
/// error is all of x measured against x's own ULP. f16 never gets there, bf16
/// does at 2^-25, and so does f32 -- but so does the expansion, so that is not
/// why f32 keeps the expansion. What f32 is spared is the rest of the range,
/// where the expansion holds 2.2 ULP against this one's 3.7: a narrower result
/// rounds that gap away and an f32 result does not.
///
/// OCML reaches its 0.5 ULP on the same identity with a small-|x| polynomial
/// branch and a compensated argument reduction, some 25 instructions and a
/// branch against five, so this stays behind fast math and --disable-fast-math
/// emits the call.
static Value emitApproxTanh(OpBuilder &b, Location loc, Value x) {
  Type ty = x.getType();
  Type elemTy = getElementTypeOrSelf(ty);
  auto constant = [&](float value) {
    return rock::createConstantFloatOp(b, loc, ty, elemTy, value);
  };

  // -2*log2(e), spelled as the exact f32 the accuracy sweep measured,
  // 0xc038aa3b.
  Value scaled = arith::MulFOp::create(b, loc, x, constant(-0x1.715476p+1f));
  Value exp = math::Exp2Op::create(b, loc, scaled);
  Value denom = arith::AddFOp::create(b, loc, exp, constant(1.0));
  Value recip = arith::DivFOp::create(b, loc, constant(1.0), denom);
  return math::FmaOp::create(b, loc, constant(2.0), recip, constant(-1.0));
}

/// Replace `op` with the libdevice entry point for its type: f64 keeps its
/// precision, everything narrower is computed at f32 with the operands widened
/// around the call and the result rounded back. Anything wider than f64 has
/// neither an entry point nor any AMD hardware, so it is declined and left to
/// the fallback expansion.
template <typename OpTy>
static LogicalResult rewriteViaOcml(OpTy op, PatternRewriter &rewriter,
                                    StringRef f32Symbol, StringRef f64Symbol) {
  Location loc = op.getLoc();
  Type origTy = op.getType();
  auto elemTy = dyn_cast<FloatType>(getElementTypeOrSelf(origTy));
  if (!elemTy || elemTy.getWidth() > 64)
    return rewriter.notifyMatchFailure(op, "no libdevice entry point");

  bool isF64 = isa<Float64Type>(elemTy);
  Type computeTy = withElementType(origTy, isF64 ? Type(rewriter.getF64Type())
                                                 : Type(rewriter.getF32Type()));
  bool widen = computeTy != origTy;

  SmallVector<Value> args;
  for (Value operand : op->getOperands())
    args.push_back(
        widen ? arith::ExtFOp::create(rewriter, loc, computeTy, operand)
                    .getResult()
              : operand);

  // Empty libname and libpath match what `core.extern_elementwise("", "", ...)`
  // emits in the Triton frontend: the symbol is resolved either by an AMD
  // backend rewrite or by the `ocml.bc` that TritonToHsaco auto-links for
  // leftover `__ocml_*` references.
  Value result = triton::ExternElementwiseOp::create(
                     rewriter, loc, computeTy, args, /*libname=*/"",
                     /*libpath=*/"", isF64 ? f64Symbol : f32Symbol,
                     /*pure=*/true)
                     .getResult();
  if (widen)
    result = arith::TruncFOp::create(rewriter, loc, origTy, result).getResult();
  rewriter.replaceOp(op, result);
  return success();
}

struct TanhLowering : public OpRewritePattern<math::TanhOp> {
  TanhLowering(MLIRContext *ctx, bool hasNativeTanh, bool allowApprox,
               PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), hasNativeTanh(hasNativeTanh),
        allowApprox(allowApprox) {}

  /// Replaces `op` with the approximation evaluated at f32, widening the
  /// operand and rounding the result back when the op is narrower than that.
  /// f32 is the only width the error figures in emitApproxTanh were measured
  /// at, so it is fixed here rather than left to callers.
  static LogicalResult approximateAtF32(math::TanhOp op,
                                        PatternRewriter &rewriter) {
    Location loc = op.getLoc();
    Type origTy = op.getType();
    Type computeTy = withElementType(origTy, rewriter.getF32Type());
    if (computeTy == origTy) {
      rewriter.replaceOp(op, emitApproxTanh(rewriter, loc, op.getOperand()));
      return success();
    }
    Value in = arith::ExtFOp::create(rewriter, loc, computeTy, op.getOperand())
                   .getResult();
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(
        op, origTy, emitApproxTanh(rewriter, loc, in));
    return success();
  }

  LogicalResult matchAndRewrite(math::TanhOp op,
                                PatternRewriter &rewriter) const override {
    Type origTy = op.getType();
    auto elemTy = dyn_cast<FloatType>(getElementTypeOrSelf(origTy));

    // On a target with a native tanh the OCML call is both the fastest and the
    // most accurate option, so it wins even under fast math. Without fast math
    // every type takes it as well, that being the point of the flag.
    if (hasNativeTanh || !allowApprox || !elemTy)
      return rewriteViaOcml(op, rewriter, kOcmlTanhF32, kOcmlTanhF64);

    // Narrower than f32: approximate, evaluated at f32 rather than at the op's
    // own type. Rounding the result back down discards the approximation's
    // error, so an f16 lands within 1 ULP where evaluating at f16 is out by
    // thousands. See emitApproxTanh for the measurements.
    //
    // Seeing bf16 and the fp8/fp4 types here at all is why this pass runs ahead
    // of math-extend-to-supported-types, which promotes them to f32.
    if (elemTy.getWidth() < 32)
      return approximateAtF32(op, rewriter);

    // f32 keeps the math dialect's own expansion, which is what the rest of the
    // toolchain already lives with. Nothing rounds the result afterwards, so
    // the approximation has nowhere to hide its error.
    if (elemTy.getWidth() == 32)
      return rewriter.notifyMatchFailure(op, "left to the generic expansion");

    // Wider than f32, so f64 in practice. The approximation has no more
    // accuracy to offer than the library call there and would need the hardware
    // exp2/rcp it does not have, so f64 takes __ocml_tanh_f64. f80 and f128
    // reach this too if a front end ever emits them, and rewriteViaOcml
    // declines those for having no entry point.
    return rewriteViaOcml(op, rewriter, kOcmlTanhF32, kOcmlTanhF64);
  }

  bool hasNativeTanh;
  bool allowApprox;
};

/// powf always becomes the library call. Deep learning barely uses it, so the
/// cheaper paths this used to take, folding small integral exponents into
/// products and handing the rest to the math dialect's exp/log expansion, were
/// special cases earning their complexity on nothing. OCML also gets the
/// negative-base and integral-exponent corners right for free.
struct PowFLowering : public OpRewritePattern<math::PowFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::PowFOp op,
                                PatternRewriter &rewriter) const override {
    return rewriteViaOcml(op, rewriter, kOcmlPowF32, kOcmlPowF64);
  }
};

struct RockLegalizeMathForTritonPass
    : public rock::impl::RockLegalizeMathForTritonPassBase<
          RockLegalizeMathForTritonPass> {
  using rock::impl::RockLegalizeMathForTritonPassBase<
      RockLegalizeMathForTritonPass>::RockLegalizeMathForTritonPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // This runs before rock-serialize-host-funcs, so the host functions reach
    // it too. They carry no arch and are not lowered for the GPU.
    if (!func->hasAttr(rock::KernelAttr::getMnemonic()))
      return;

    StringAttr arch = rock::getArchValueOnFunc(func);

    MLIRContext *ctx = &getContext();
    bool allowApprox = !disableFastMath;

    RewritePatternSet patterns(ctx);
    patterns.add<TanhLowering>(ctx, hasTanhInsts(arch.getValue()), allowApprox,
                               /*benefit=*/2);
    patterns.add<PowFLowering>(ctx, /*benefit=*/2);
    // Fallbacks, at the default benefit so the patterns above win where they
    // apply: f32 tanh, which TanhLowering hands over deliberately, and any
    // element type neither pattern claims (f80 and f128 have no library entry
    // point), which would otherwise reach rock-to-ttir as an op Triton has no
    // conversion for.
    math::populateExpansionPatterns(patterns, {"tanh", "powf"});

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, math::MathDialect,
                           triton::TritonDialect, func::FuncDialect>();
    target.addIllegalOp<math::TanhOp, math::PowFOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
