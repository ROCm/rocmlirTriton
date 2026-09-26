//===- WarpsOnK.h - Blocked layout with every warp on K ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The layout rock-set-itt-reduction-layout and rock-set-reduction-layout give
// a gather load. Both passes can see the same load, and each recognizes one the
// other already rewrote only by comparing it with this layout, so they must
// build it the same way.
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_TRANSFORMS_WARPSONK_H
#define ROCK_TRANSFORMS_WARPSONK_H

#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace rock {

/// `enc` with every warp on `kDim` and the rest of the layout unchanged, or
/// failure when that layout does not tile `shape`. The result is `enc` itself
/// when its warps are all on `kDim` already.
FailureOr<triton::gpu::BlockedEncodingAttr>
computeLayoutWarpsOnK(triton::gpu::BlockedEncodingAttr enc,
                      ArrayRef<int64_t> shape, unsigned kDim);

} // namespace rock
} // namespace mlir
#endif // ROCK_TRANSFORMS_WARPSONK_H
