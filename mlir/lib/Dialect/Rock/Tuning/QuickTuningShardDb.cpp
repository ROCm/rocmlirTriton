//===- QuickTuningShardDb.cpp - quick-tuning shard data -------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Materializes the compiled-in quick-tuning database by running the generated
// shards through the two-phase include protocol described in
// QuickTuningShardDb.h. This is the only file allowed to define either phase
// macro.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/QuickTuningShardDb.h"

using namespace mlir;
using namespace mlir::rock;

#define QUICK_TUNING_DB_ARRAYS
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards.inc"
#undef QUICK_TUNING_DB_ARRAYS

static const QuickTuningShard kQuickTuningShards[] = {
#define QUICK_TUNING_DB_ENTRIES
#include "mlir/Dialect/Rock/Tuning/QuickTuningShards.inc"
#undef QUICK_TUNING_DB_ENTRIES
};

ArrayRef<QuickTuningShard> mlir::rock::getQuickTuningShards() {
  return kQuickTuningShards;
}
