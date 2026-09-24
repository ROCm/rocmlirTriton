//===- WarpsOnK.cpp - Blocked layout with every warp on K -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WarpsOnK.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;

FailureOr<ttg::BlockedEncodingAttr>
mlir::rock::computeLayoutWarpsOnK(ttg::BlockedEncodingAttr enc,
                                  ArrayRef<int64_t> shape, unsigned kDim) {
  SmallVector<unsigned> warpsPerCTA(enc.getWarpsPerCTA());
  unsigned numWarps = 1;
  for (unsigned warps : warpsPerCTA)
    numWarps *= warps;
  for (unsigned d = 0; d < warpsPerCTA.size(); ++d)
    warpsPerCTA[d] = d == kDim ? numWarps : 1;
  for (auto [d, size] : llvm::enumerate(shape)) {
    int64_t cover =
        enc.getSizePerThread()[d] * enc.getThreadsPerWarp()[d] * warpsPerCTA[d];
    assert(cover != 0 && "blocked encoding tile factors must be >= 1");
    if (size % cover != 0)
      return failure();
  }
  return ttg::BlockedEncodingAttr::get(enc.getContext(), enc.getSizePerThread(),
                                       enc.getThreadsPerWarp(), warpsPerCTA,
                                       enc.getOrder(), enc.getCGALayout());
}
