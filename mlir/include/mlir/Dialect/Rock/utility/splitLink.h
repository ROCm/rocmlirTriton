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
// Shared attribute names + constants describing how a cross-tile fusion was
// split into a gemm kernel + an elementwise kernel (AIROCMLIR-709). The split is
// produced by rock-split-cross-tile-fusion (SplitCrossTileFusion.cpp), which
// tags each half with this metadata; the host driver that allocates the
// intermediate buffer and launches both kernels is generated later by
// rock-link-split-kernels (LinkSplitKernels.cpp), after kernel signatures are
// finalized.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_UTILITY_SPLITLINK_H
#define MLIR_DIALECT_ROCK_UTILITY_SPLITLINK_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace rock {

/// Sentinel used in `rock.split_arg_src` to mark the intermediate buffer that
/// joins the two split kernels (as opposed to an original-kernel argument).
inline constexpr int64_t SplitIntermediateArg = -1;

/// Original (pre-split) kernel symbol name; both halves of a split share it.
inline constexpr llvm::StringLiteral SplitLinkGroupAttr = "rock.split_group";
/// "gemm" or "elementwise".
inline constexpr llvm::StringLiteral SplitLinkRoleAttr = "rock.split_role";
/// Per kernel argument, the index of the corresponding original-kernel
/// argument, or SplitIntermediateArg for the intermediate buffer. Only the
/// arguments present at split time are described here.
inline constexpr llvm::StringLiteral SplitLinkArgSrcAttr = "rock.split_arg_src";
/// (elementwise only) original-kernel argument indices for output argument(s)
/// appended after the split (by rock-elementwise-to-gridwise).
inline constexpr llvm::StringLiteral SplitLinkOutSrcAttr = "rock.split_out_src";

inline constexpr llvm::StringLiteral SplitLinkRoleGemm = "gemm";
inline constexpr llvm::StringLiteral SplitLinkRoleElementwise = "elementwise";

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_SPLITLINK_H
