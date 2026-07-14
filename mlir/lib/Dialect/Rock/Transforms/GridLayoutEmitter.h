//===- GridLayoutEmitter.h - MLIR helper that contains the layout logic -===//
//
// Copyright 2020 The MLIR Authors.
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
// =============================================================================
//
// Helpers that map a flat block id onto the kernel grid by emitting the
// <group, m-block, n-block> triplet used by the generated gemm/attention
// kernels.
//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_ROCK_TRANSFORMS_MLIR_LAYOUT_EMITTER_H
#define MLIR_LIB_DIALECT_ROCK_TRANSFORMS_MLIR_LAYOUT_EMITTER_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

namespace mlir {
namespace rock {
namespace layout {

/// Struct containing information that guide the layout heuristic selection
struct GridLayoutInfo {
  int64_t gBlocks;
  int64_t mBlocks;
  int64_t nBlocks;
  int64_t numCU;
  int64_t numChiplets;
  Type inputType; // A operand load element type
  int64_t gridGroupSize;
  // Tile sizes and whole-matrix byte counts for the L2-residency (1-D
  // GROUP_SIZE_M) groupSizeM heuristic.
  int64_t mPerBlock;   // M-block tile (elements)
  int64_t kPerBlock;   // K-loop tile (elements)
  int64_t aTotalBytes; // whole A matrix in bytes (G*M*K * sizeof(A elem))
  int64_t bTotalBytes; // whole B matrix in bytes (G*N*K * sizeof(B elem))
  int64_t l2Bytes;     // L2 shared by the concurrent workgroups
};

/// This function emits the right triplet of <group,block_m,block_n>
/// identifiers, given a flat blockId. This has been adapted from:
/// https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py
///
GridCoordinates makeGroupedGridLayout(PatternRewriter &b, Location loc,
                                      Value bid, GridLayoutInfo info,
                                      StringRef arch);

AttnGridCoordinates makeGxNGridLayout(PatternRewriter &b, Location loc,
                                      Value bid, int64_t mBlocks, Value nIter,
                                      int64_t gridSize, StringRef arch,
                                      int64_t numChiplets,
                                      Value splitKV = nullptr);

} // namespace layout
} // namespace rock
} // namespace mlir

#endif // MLIR_LIB_DIALECT_ROCK_TRANSFORMS_MLIR_LAYOUT_EMITTER_H
