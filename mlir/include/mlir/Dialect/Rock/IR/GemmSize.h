//===--------- GemmSize.h - utility struct for GEMM ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a utility struct, GemmSize, that packages the sizes of a
// matrix multiplication to ensure a cleaner API.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_IR_GEMMCONTEXT_H
#define MLIR_DIALECT_ROCK_IR_GEMMCONTEXT_H

#include "mlir/IR/BuiltinTypeInterfaces.h"

#include <cstdint>

namespace mlir {
namespace rock {
struct ConvolutionDims;
enum class ConvOpType : uint32_t;

/// The size that tuning heuristics assume for a dynamic dimension.
constexpr int64_t kDynamicDimHint = 1024;

/// Structure for holding the sizes of a matrix multiplication operation.
/// Dynamic dimensions are ShapedType::kDynamic.
struct GemmSize {
  int64_t g;
  int64_t m;
  int64_t k;
  int64_t n;

  GemmSize(int64_t g, int64_t m, int64_t k, int64_t n)
      : g(g), m(m), k(k), n(n) {}

  /// Compute the gemm size given a convolution type and its dimensions.
  static GemmSize fromConvolution(ConvOpType type,
                                  const ConvolutionDims &sizes);

  bool isDynamic() const {
    return ShapedType::isDynamic(g) || ShapedType::isDynamic(m) ||
           ShapedType::isDynamic(k) || ShapedType::isDynamic(n);
  }

  /// This size with each dynamic dimension replaced by `hint`.
  GemmSize withDynamicHint(int64_t hint = kDynamicDimHint) const {
    auto h = [&](int64_t v) { return ShapedType::isDynamic(v) ? hint : v; };
    return GemmSize(h(g), h(m), h(k), h(n));
  }

  bool operator==(const GemmSize &other) {
    return (g == other.g) && (m == other.m) && (k == other.k) && (n == other.n);
  }
};
} // end namespace rock
} // end namespace mlir
#endif // MLIR_DIALECT_ROCK_IR_GEMMCONTEXT_H
