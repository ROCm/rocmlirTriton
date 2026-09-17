//===- QuickTuningProblemMapTests.cpp - Per-problem map lookup ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/QuickTuningProblemMap.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::rock;

namespace {
// A map shaped like a generated shard: interned perfconfigs, a ribbon of
// indices into them, and problems sorted by hash. The rows deliberately differ
// in length, share perfconfigs and appear out of index order, as real shards
// do.
const StringRef perfConfigs[] = {"A", "B", "C"};
constexpr uint16_t perfConfigIndices[] = {0, 1, 2, 2, 0};
constexpr QuickTuningProblemRef problems[] = {
    {10, 0, 3}, {20, 3, 1}, {30, 4, 1}};

QuickTuningProblemMap makeMap() {
  return QuickTuningProblemMap(problems, perfConfigIndices, perfConfigs);
}
} // namespace

TEST(QuickTuningProblemMapTest, HitReturnsRowInOrder) {
  EXPECT_EQ(makeMap().lookup(10), SmallVector<StringRef>({"A", "B", "C"}));
}

TEST(QuickTuningProblemMapTest, HitReturnsOnlyItsOwnRow) {
  EXPECT_EQ(makeMap().lookup(20), SmallVector<StringRef>({"C"}));
  EXPECT_EQ(makeMap().lookup(30), SmallVector<StringRef>({"A"}));
}

TEST(QuickTuningProblemMapTest, MissReturnsEmpty) {
  QuickTuningProblemMap map = makeMap();
  // Below, between and above the stored hashes: the binary search must not
  // round to a neighbour, because a ranking only holds for the problem it was
  // measured on.
  EXPECT_TRUE(map.lookup(0).empty());
  EXPECT_TRUE(map.lookup(15).empty());
  EXPECT_TRUE(map.lookup(30 + 1).empty());
}

TEST(QuickTuningProblemMapTest, EmptyMapMisses) {
  QuickTuningProblemMap map({}, {}, {});
  EXPECT_TRUE(map.lookup(10).empty());
}
