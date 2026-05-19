//===- CacheFlush.h - Cache flush helpers -----------------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCMLIR_TUNING_DRIVER_CACHE_FLUSH_H
#define ROCMLIR_TUNING_DRIVER_CACHE_FLUSH_H

#include "mlir/Support/LogicalResult.h"

#include <hip/hip_runtime.h>

namespace rocmlir::tuningdriver {

/// L2-cache flush strategy. Selected by the ``--flush-l2`` driver flag and
/// surfaced to the produced tuning CSV so each measurement records which
/// strategy was in effect (see AIROCMLIR-858).
enum class L2FlushLevel {
  /// Skip the L2 flush entirely. Useful when the experimenter wants to study
  /// the GPU's warm-cache behaviour explicitly.
  None,
  /// Flush the whole L2: ``hipMemsetAsync`` over a buffer larger than the
  /// L2 cache. This is the historical / default behaviour.
  All,
};

/// \brief Flushes the L2 cache according to the given ``level``.
/// \param stream The HIP stream to use for the flush operation.
/// \param level How aggressively to flush the L2 cache.
/// \return success() if the flush succeeds, failure() otherwise.
mlir::LogicalResult flushL2Cache(hipStream_t stream,
                                 L2FlushLevel level = L2FlushLevel::All);

/// \brief Flushes the instruction cache to ensure that any modified code is
/// visible to the device.
/// \param stream The HIP stream to use for the flush operation.
/// \return success() if the flush succeeds, failure() otherwise.
mlir::LogicalResult flushInstructionCache(hipStream_t stream);

/// \brief Cleans up any artifacts created during cache flush operations.
/// \return success() if cleanup succeeds, failure() otherwise.
mlir::LogicalResult cleanupCacheFlushArtifacts();

} // namespace rocmlir::tuningdriver

#endif // ROCMLIR_TUNING_DRIVER_CACHE_FLUSH_H
