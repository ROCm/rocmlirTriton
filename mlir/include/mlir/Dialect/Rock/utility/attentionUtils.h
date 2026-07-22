//===- attentionUtils.h - Attention analysis utilities ----------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_UTILITY_ATTENTIONUTILS_H
#define ROCK_UTILITY_ATTENTIONUTILS_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace rock {

/// Classify each external pre-softmax input by the operation where its
/// provenance first merges with the QK score provenance. Ambiguous or malformed
/// dataflow is classified as Other.
SmallVector<PreSoftmaxInputRole>
classifyPreSoftmaxInputRoles(Region &region, unsigned numInputs,
                             unsigned numDequantInputs = 0);

/// Determine whether the two trailing dimensions of `value` are naturally
/// stored or physically transposed. The analysis is deliberately conservative:
/// unsupported or coupled transforms, unit sequence dimensions, and
/// multi-input preprocessing return Unknown. Broadcasts are classified only
/// when the remaining physical strides prove an orientation.
PreSoftmaxInputOrientation classifyPreSoftmaxInputOrientation(Value value);

} // namespace rock
} // namespace mlir

#endif // ROCK_UTILITY_ATTENTIONUTILS_H
