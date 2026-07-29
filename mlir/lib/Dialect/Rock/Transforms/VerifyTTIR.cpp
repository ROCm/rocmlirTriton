//===- VerifyTTIR.cpp - Check Rock-emitted Triton IR ----------------------===//
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
// Rock emits Triton IR directly from C++ (see RockToTTIR.cpp) instead of going
// through Triton's Python frontend, so none of the legality checks in
// external/triton/python/triton/language/semantic.py run on our kernels. MLIR's
// own verifiers cover a good part of that: triton::verifyDotOpInterface already
// enforces the rank, K-match, batch and output-shape rules from semantic.dot,
// and DotOp::verify enforces equal operand bit widths. This pass adds the
// checks that nothing else performs, so that an illegal kernel fails at compile
// time rather than silently computing the wrong answer.
//
// Currently limited to tt.dot, in two groups:
//
//   * Element-type policy (semantic.py:1433-1499). A violation here means a
//     Rock pass built a dot Triton's frontend would have rejected outright, so
//     it is a hard diagnostic.
//
//   * Lowering reachability. A dot that neither a matrix-core instruction nor a
//     packed v_dot can serve gets its operands and accumulator recast to a
//     common float type by AccelerateBlocked::tryLegalizeFMA. For an integer
//     accumulator that common type is always f32, which silently rounds sums
//     past 2^24. This depends on the tile shape, i.e. on the perf config, so it
//     is reported as an arch-feature mismatch (rock.not_applicable) so that
//     tuning skips the config instead of aborting.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/tritonUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKVERIFYTTIRPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-verify-ttir"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockVerifyTTIRPass
    : public rock::impl::RockVerifyTTIRPassBase<RockVerifyTTIRPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

/// The FP8 types Triton accepts as a `tt.dot` operand. Mirrors tl.dtype.is_fp8()
/// over semantic.py:1433's "all combinations of supported fp8 x fp8".
static bool isDotFp8Type(Type t) {
  return isa<Float8E4M3FNType, Float8E4M3FNUZType, Float8E5M2Type,
             Float8E5M2FNUZType>(t);
}

/// Mirrors the operand dtype whitelist at semantic.py:1437-1440,
/// {int8, uint8, float16, bfloat16, float32, float64}. MLIR integers are
/// signless here, and semantic.py:1484 narrows the integer case to int8
/// regardless, so only a signless i8 is accepted.
static bool isDotOperandType(Type t) {
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.isSignless() && intTy.getWidth() == 8;
  return isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(t);
}

/// Check the element types of `op` against semantic.dot's rules. `cElemTy` is
/// both the accumulator and the result element type: tt.dot couples them with a
/// TypesMatchWith constraint, so MLIR has already verified they agree.
static LogicalResult verifyDotElementTypes(triton::DotOp op) {
  Type aElemTy = op.getA().getType().getElementType();
  Type bElemTy = op.getB().getType().getElementType();
  Type cElemTy = op.getC().getType().getElementType();

  // Any supported fp8 x fp8 pair is permitted, including mixed ones
  // (semantic.py:1433-1435), so the operands need not agree.
  if (!(isDotFp8Type(aElemTy) && isDotFp8Type(bElemTy))) {
    if (!isDotOperandType(aElemTy))
      return op.emitOpError("unsupported element type for operand A: ")
             << aElemTy;
    if (!isDotOperandType(bElemTy))
      return op.emitOpError("unsupported element type for operand B: ")
             << bElemTy;
    if (aElemTy != bElemTy)
      return op.emitOpError("operands A and B must have the same element type, "
                            "got ")
             << aElemTy << " and " << bElemTy;
  }

  // Accumulator rules, mirroring how semantic.dot derives ret_scalar_ty
  // (semantic.py:1483-1499).
  if (aElemTy.isInteger()) {
    if (!cElemTy.isInteger(32))
      return op.emitOpError("integer dot must accumulate into i32, got ")
             << cElemTy;
    return success();
  }
  if (cElemTy.isBF16())
    return op.emitOpError(
        "bf16 accumulator is unsupported; accumulate in f32 and truncate");
  if (aElemTy.isF64()) {
    if (!cElemTy.isF64())
      return op.emitOpError("f64 dot must accumulate into f64, got ") << cElemTy;
    return success();
  }
  if (aElemTy.isF32() || aElemTy.isBF16()) {
    if (!cElemTy.isF32())
      return op.emitOpError("dot with ")
             << aElemTy << " operands must accumulate into f32, got "
             << cElemTy;
    return success();
  }
  // f16 and fp8 operands may accumulate in either f16 or f32.
  if (!cElemTy.isF16() && !cElemTy.isF32())
    return op.emitOpError("dot with ")
           << aElemTy << " operands must accumulate into f16 or f32, got "
           << cElemTy;
  return success();
}

/// Reject a dot that `AccelerateBlocked::tryLegalizeFMA` would recast in a way
/// that loses values the accumulator can represent.
static LogicalResult
verifyDotIsExactlyLowerable(triton::DotOp op,
                            triton::amdgpu::ISAFamily isaFamily) {
  auto aTy = op.getA().getType();
  Type aElemTy = aTy.getElementType();
  Type bElemTy = op.getB().getType().getElementType();
  Type cElemTy = op.getC().getType().getElementType();
  int64_t kDim = aTy.getShape().back();

  if (rock::classifyDotLowering(isaFamily, aElemTy, bElemTy, cElemTy, kDim) !=
      rock::DotLowering::UpcastedFMA)
    return success();

  // tryLegalizeFMA picks a common float type at least as wide as every operand.
  // For float operands that is lossless (it only ever widens), so the fallback
  // costs performance but not accuracy. With an integer accumulator the common
  // type is f32, whose 24-bit mantissa cannot hold every i32.
  if (!cElemTy.isInteger())
    return success();

  return op.emitOpError("K=")
         << kDim << " leaves this " << aElemTy << " x " << bElemTy << " -> "
         << cElemTy
         << " dot without a matrix-core or v_dot lowering, so Triton would "
            "accumulate it in f32 and round sums past 2^24; a K that is a "
            "multiple of "
         << rock::getRequiredDotKMultiple(isaFamily, aElemTy, bElemTy, cElemTy)
         << " stays exact on every architecture";
}

void RockVerifyTTIRPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  FailureOr<StringAttr> maybeArch = rock::getArchOnFunc(funcOp);
  if (failed(maybeArch)) {
    funcOp.emitError("rock.arch not found on kernel function or module");
    return signalPassFailure();
  }
  triton::amdgpu::ISAFamily isaFamily =
      std::get<0>(rock::getArch(maybeArch->getValue()));

  bool sawError = false;
  funcOp.walk([&](triton::DotOp op) {
    // An element-type violation means a Rock pass built IR the Triton frontend
    // would have rejected: a compiler bug, not a bad perf config.
    if (failed(verifyDotElementTypes(op))) {
      sawError = true;
      return;
    }
    if (failed(verifyDotIsExactlyLowerable(op, isaFamily))) {
      // Shape-derived, so the perf config that produced this tile is simply not
      // applicable to this architecture. Let tuning move on to the next one.
      rock::markAsNotApplicable(op);
      sawError = true;
    }
  });

  if (sawError)
    return signalPassFailure();
}
