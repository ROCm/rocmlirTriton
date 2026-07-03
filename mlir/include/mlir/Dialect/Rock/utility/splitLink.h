//===- splitLink.h - cross-tile split link metadata ----------------------===//
//
// Copyright 2026 Advanced Micro Devices.
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
//
// Shared constant describing how a cross-tile fusion was split into a gemm
// kernel + an elementwise kernel (AIROCMLIR-709). The linkage itself is carried
// by the typed `rock.split_link` op (see RockOps.td), produced by
// rock-split-cross-tile-fusion (SplitCrossTileFusion.cpp) and consumed by
// rock-link-split-kernels (LinkSplitKernels.cpp) once kernel signatures are
// finalized. This header only defines the sentinel used inside that op's
// argument-source arrays.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_UTILITY_SPLITLINK_H
#define MLIR_DIALECT_ROCK_UTILITY_SPLITLINK_H

#include <cstdint>

namespace mlir {
namespace rock {

/// Sentinel used in `rock.split_link`'s arg-source arrays to mark the
/// intermediate buffer that joins the two split kernels (as opposed to an
/// original-kernel argument).
inline constexpr int64_t SplitIntermediateArg = -1;

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_SPLITLINK_H
