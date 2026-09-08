//===- tosa Utility Functions -===//
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
#ifndef MLIR_DIALECT_ROCK_TOSA_UTILITY_H
#define MLIR_DIALECT_ROCK_TOSA_UTILITY_H

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/APFloat.h"

namespace mlir {
namespace rock {

/// Threshold (as a double) used to recognize the "large negative" splat
/// constants that frontends emit as additive attention-mask values.
/// The literal -1.0e4 is not arbitrary: it traces back to the 2018 TensorFlow
/// BERT reference implementation (google-research/bert, modeling.py), where
/// the attention mask was implemented as an additive mask rather than a
/// multiplicative / select mask.
///
/// The MIGraphX frontend performs no additional processing of the causal
/// mask, so these constants are guaranteed to appear in the models we lower.
/// We treat any sufficiently-negative splat (<= this threshold) as a
/// stand-in for -infinity when detecting attention masking patterns.
constexpr double kMaskingConstantThreshold = -1.0e4;

bool isSpecificValueAttribute(Attribute value, double target);
bool isConstantValue(Value v, double target);
bool isConstantZero(Value v);
bool isConstantOne(Value v);
bool isConstNegInf(Value v);
bool isConstRange(Value v);

/// Returns true if v should be treated as a "negative infinity" stand-in
/// for attention-mask detection. Accepts:
///   - Actual -infinity.
///   - The largest negative finite value of v's float semantics (the
///     value frontends often substitute when -inf is not representable
///     after a cast, e.g. f16's 0xFBFF).
///   - Any value <= `kMaskingConstantThreshold` (see that constant for the
///     historical -10000.0 BERT-mask convention).
bool isMaskingNegInfValue(const llvm::APFloat &v);

namespace tosa {

/// Number of convolution groups on `op`, read from the optional `group`
/// attribute that the MIGraphX frontend attaches to `tosa.conv2d` and to the
/// `conv_bwd_data` `tosa.custom`. An op without the attribute is an ordinary
/// convolution, hence the default of 1.
///
/// Defined inline so that callers need only the header, not a dependency on
/// MLIRRockUtility.
inline int64_t getConvGroupCount(Operation *op) {
  if (auto groupAttr = op->getAttrOfType<IntegerAttr>("group"))
    return groupAttr.getInt();
  return 1;
}

/// Whether `op` is a grouped convolution, i.e. one whose group dimension has
/// to be materialized rather than elided.
inline bool isGroupedConv(Operation *op) { return getConvGroupCount(op) != 1; }

/// Adjust conv padding for TOSA's exact-divisibility-by-stride requirement.
/// Floor-mode convolutions may produce partial windows that TOSA rejects.
/// Trims high-side padding first; if that's insufficient, inserts a
/// tosa::SliceOp to shrink the input tensor.
///
/// Both input and filter are expected in TOSA layout (spatial dims at
/// indices 1..nSpatial, i.e. NHWC / OCHW).
///
/// \param pads  Interleaved [padLeft0, padRight0, padLeft1, ...]. Modified
///              in-place.
/// \param input  The input tensor value (may be replaced by a SliceOp result).
/// \param filter The filter/kernel tensor value (used to read spatial sizes).
/// \param strides  Stride per spatial dimension.
/// \param dilations  Dilation per spatial dimension.
/// \returns The (possibly sliced) input tensor.
Value adjustConvPadding(OpBuilder &builder, Location loc, Value input,
                        Value filter, MutableArrayRef<int64_t> pads,
                        ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations);

template <typename TosaOp, typename... Args>
TosaOp createOpAndInfer(OpBuilder &rewriter, Location loc, Type elemType,
                        Args &&...args) {
  auto op =
      TosaOp::create(rewriter, loc, UnrankedTensorType::get(elemType), args...);
  InferShapedTypeOpInterface shapeInterface =
      cast<InferShapedTypeOpInterface>(op.getOperation());
  SmallVector<ShapedTypeComponents> returnShape;
  LogicalResult shapeInferenceStatus = shapeInterface.inferReturnTypeComponents(
      op.getContext(), op.getLoc(), op->getOperands(), op->getAttrDictionary(),
      op->getPropertiesStorage(), op->getRegions(), returnShape);
  assert(shapeInferenceStatus.succeeded());
  Type newOutTy = RankedTensorType::get({returnShape[0].getDims()}, elemType);
  auto result = op->getResult(0);
  result.setType(newOutTy);
  return op;
}

Value getOneTensor(OpBuilder &builder, Location loc, RankedTensorType type);

Value getZeroTensor(OpBuilder &builder, Location loc, RankedTensorType type);

mlir::tosa::TransposeOp getTransposeOp(OpBuilder &builder, Location loc,
                                       Value input,
                                       ArrayRef<int32_t> permutation);

mlir::tosa::MulOp getMulOp(OpBuilder &builder, Location loc, Value input1,
                           Value input2, Type elemType);
} // namespace tosa
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TOSA_UTILITY_H
