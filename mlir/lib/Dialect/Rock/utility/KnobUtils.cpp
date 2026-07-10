//===- KnobUtils.cpp - Triton knob constants and validation ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/KnobUtils.h"

#include "llvm/Support/raw_ostream.h"

#include <array>
#include <utility>

using namespace mlir;

namespace mlir {
namespace rock {

bool isValidKnobBoolean(int64_t value) {
  return value == kKnobDefault || value == 0 || value == 1;
}

} // namespace rock
} // namespace mlir
