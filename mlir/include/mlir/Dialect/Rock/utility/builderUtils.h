#ifndef MLIR_DIALECT_ROCK_UTILITY_BUILDERUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_BUILDERUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace rock {
// Utility to create an APFloat of the requested type
std::pair<APFloat, llvm::detail::opStatus> createAPFloat(Type elemType,
                                                         float value);

/// Utility op to emit constant float op
Value createConstantFloatOp(OpBuilder &b, Location loc, Type type,
                            Type elemType, float value,
                            APFloat::opStatus expectedStatus = APFloat::opOK);

/// Utility op to emit constant int op
Value createConstantIntOp(OpBuilder &b, Location loc, Type type, Type elemType,
                          int64_t value);

/// Utility function to emit constant zero op. Can return scalars or vectors.
Value createZeroConstantOp(OpBuilder &b, Location loc, Type type);

/// Utility function to emit type conversion ops.
Value createTypeConversionOp(OpBuilder &b, Location loc, Value source,
                             Type destType);

/// Insert a unit dimension at `axis` with tt.expand_dims, then broadcast the
/// expanded tensor to `resultType`.
Value expandDimAndBroadcast(OpBuilder &b, Location loc, Value source,
                            int64_t axis, RankedTensorType resultType);

/// Saturating + truncating float-to-int conversion implementing MIGraphX's
/// reference `convert` op semantics. Used by both the CPU lowering path
/// (RocmlirCustomTosaToLinalg) and the GPU/kernel path
/// (RockTosaToElementwise) for the rocmlir-domain `tosa.custom` ops
/// `unsigned_cast` (when input is float) and `fp_to_int_cast`. Works on
/// either a scalar float `input` or a tensor of floats; the result has the
/// same shape as `input` (or is a scalar if `input` is scalar) with element
/// type `dstIntType`.
///
/// Why this helper exists (instead of just `arith.fptosi` / `tosa.cast`):
/// MIGraphX requires saturating + truncating semantics, which neither the
/// arith ops (poison on out-of-range/inf/NaN) nor upstream
/// `tosa-to-linalg` (round-to-nearest-even rather than truncation) provide.
/// See the long comment above the implementation for the full rationale.
///
/// Result for every input class (matching MIGraphX semantics):
///   in-range finite -> truncated int (round-toward-zero)
///   out-of-range positive finite, +inf -> INT_MAX
///   out-of-range negative finite, -inf -> INT_MIN (signed) or 0 (unsigned)
///   NaN                                -> 0
///
/// Three cases are handled depending on the relationship between the float
/// type's precision and the integer type's width:
///   Case 1: int range exceeds float exponent range -> cmp+select for inf
///   Case 2: float mantissa can represent int max exactly -> full FP clamp
///   Case 3: float exponent sufficient but mantissa too narrow -> mixed
///           clamp + overflow fix-up against (intMax + 1)
///
/// Precondition: the source float type must have a representable zero and
/// representable +/-infinity. Exotic micro-float types that lack these
/// (e.g. F8E8M0FNU has no zero, F4E2M1FN has no infinity) must be promoted
/// to a wider float type before invoking this helper.
Value createClampedFPToInt(OpBuilder &b, Location loc, Value input,
                           Type dstIntType, bool isUnsigned);

// Get a 1-D version of the shaped type `type`, preserving memory space.
Type getFlattenedType(Type type);

// Utility function to get a MemRef as a tensor
Value getAsTensor(OpBuilder &builder, Location loc, mlir::Value value,
                  bool isWritable = false);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_BUILDERUTILS_H
