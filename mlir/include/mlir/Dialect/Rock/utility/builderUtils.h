#ifndef MLIR_DIALECT_ROCK_UTILITY_BUILDERUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_BUILDERUTILS_H

#include "mlir/IR/Builders.h"
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

// Get a 1-D version of the shaped type `type`, preserving memory space.
Type getFlattenedType(Type type);

// Utility function to get a MemRef as a tensor
Value getAsTensor(OpBuilder &builder, Location loc, mlir::Value value,
                  bool isWritable = false);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_BUILDERUTILS_H
