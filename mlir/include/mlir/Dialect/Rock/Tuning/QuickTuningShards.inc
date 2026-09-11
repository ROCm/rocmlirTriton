//===- QuickTuningShards.inc - index of quick-tuning shards ---------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The checked-in list of quick-tuning shards. A shard only reaches the binary
// if it is listed here, so adding one is an explicit, reviewable edit. This is
// deliberately not a CMake glob: a glob leaves deleted shards behind in an
// incremental build and bakes absolute paths into the build tree.
//
// Rewritten wholesale, from the shards on disk, by
// mlir/utils/performance/analysis/quickTuningGen.py.
//
// Included once per phase by QuickTuningShardDb.cpp, so it must contain
// nothing but `#include` directives, and must not carry an include guard.
// Keep the list sorted by file name.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1100AttentionBf16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1100AttentionI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1100GemmElementwiseGemmF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1100GemmElementwiseGemmF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1101AttentionF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1101AttentionF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1101ConvF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1101ConvF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1101ConvI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1101GemmF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1101GemmF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1101GemmI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1150AttentionF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1150AttentionF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1150ConvF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1150ConvF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1150ConvI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1150GemmF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1150GemmF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1150GemmI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1151AttentionBf16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1151AttentionF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1151AttentionF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1151AttentionI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1151ConvF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1151ConvF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1151ConvI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1151GemmF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1151GemmF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1151GemmI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1170AttentionF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1170AttentionF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1170ConvF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1170ConvF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1170ConvI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1170GemmF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1170GemmF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1170GemmI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1200AttentionF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1200AttentionF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1200ConvF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1200ConvF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1200ConvI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1200GemmF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1200GemmF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1200GemmI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1201AttentionBf16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1201AttentionI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx1201GemmFp8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx908AttentionBf16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx908AttentionF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx908AttentionF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx908AttentionI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx908ConvF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx908ConvF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx908ConvI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx908GemmF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx908GemmF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx908GemmI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx90aAttentionBf16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx90aAttentionF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx90aAttentionF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx90aAttentionI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx90aConvF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx90aConvF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx90aConvI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx90aGemmF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx90aGemmF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx90aGemmI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx942AttentionBf16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx942AttentionF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx942AttentionF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx942AttentionI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx942ConvF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx942ConvF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx942ConvI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx942GemmF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx942GemmF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx942GemmI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950AttentionBf16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950AttentionF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950AttentionF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950AttentionI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950ConvF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950ConvF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950ConvI8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950GemmElementwiseGemmF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950GemmElementwiseGemmF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950GemmF16.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950GemmF32.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950GemmFp8.inc"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards/Gfx950GemmI8.inc"
