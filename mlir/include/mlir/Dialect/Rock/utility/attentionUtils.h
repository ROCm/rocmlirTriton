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

struct PreSoftmaxInputRoleInfo {
  bool hasDequant = false;
  bool hasScale = false;
  bool hasBias = false;
  bool hasOther = false;
  unsigned numMerges = 0;

  PreSoftmaxInputRole getSingularRole() const;
};

/// Analyze every operation where an external pre-softmax input independently
/// merges with QK score provenance. This retains multiple semantic roles for
/// tuning while allowing lowering to reject ambiguous inputs conservatively.
SmallVector<PreSoftmaxInputRoleInfo>
analyzePreSoftmaxInputRoles(Region &region, unsigned numInputs,
                            unsigned numDequantInputs = 0);

/// Return one role per input for load metadata. Inputs with multiple merge
/// points or semantic roles are classified as Other.
SmallVector<PreSoftmaxInputRole>
classifyPreSoftmaxInputRoles(Region &region, unsigned numInputs,
                             unsigned numDequantInputs = 0);

/// Resolve the explicit dequant input count, retaining the legacy inference of
/// two leading inputs for i8 attention when the attribute is absent.
unsigned getEffectiveNumDequantInputs(AttentionOp op);
unsigned getEffectiveNumDequantInputs(GridwiseAttentionOp op);

/// Determine whether the two trailing dimensions of `value` are naturally
/// stored or physically transposed. The analysis is deliberately conservative:
/// unsupported or coupled transforms, unit sequence dimensions, and
/// multi-input preprocessing return Unknown. Broadcasts are classified only
/// when the remaining physical strides prove an orientation.
PreSoftmaxInputOrientation classifyPreSoftmaxInputOrientation(Value value);

} // namespace rock
} // namespace mlir

#endif // ROCK_UTILITY_ATTENTIONUTILS_H
