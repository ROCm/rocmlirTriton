//===- TuningSearchTests.cpp - The vocabulary the searches share ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TuningSearch.h is the protocol every search speaks: how a benchmarked result
// is reported back, and how a strategy is named on a command line. Both of
// those cross a boundary -- one reaches a trace and a prompt, the other a
// user's argv -- so they are interface rather than implementation detail, and
// the tests here hold them still.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/TuningSearch.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {

using Status = BenchmarkResult::Status;

//===----------------------------------------------------------------------===//
// Naming a benchmark result
//===----------------------------------------------------------------------===//

TEST(BenchmarkStatusTest, EveryStatusHasAName) {
  EXPECT_EQ(getNameForBenchmarkStatus(Status::Success), "success");
  EXPECT_EQ(getNameForBenchmarkStatus(Status::NotApplicable), "notApplicable");
  EXPECT_EQ(getNameForBenchmarkStatus(Status::Failed), "failed");
}

// The insertion operator has to agree with the name, since the two are read
// as the same word in different places: one lands in a JSON trace and the
// other in a diagnostic about the same result.
TEST(BenchmarkStatusTest, StreamingAStatusWritesItsName) {
  for (Status status :
       {Status::Success, Status::NotApplicable, Status::Failed}) {
    std::string streamed;
    llvm::raw_string_ostream os(streamed);
    os << status;
    EXPECT_EQ(streamed, getNameForBenchmarkStatus(status));
  }
}

//===----------------------------------------------------------------------===//
// What counts as a measurement
//===----------------------------------------------------------------------===//

TEST(BenchmarkStatusTest, ASuccessWithATimeIsAMeasurement) {
  BenchmarkResult result;
  result.status = Status::Success;
  result.timeNs = 1234.0;
  EXPECT_TRUE(result.measured());
  EXPECT_EQ(result.effectiveStatus(), Status::Success);
}

// The case the two fields can disagree about. A client that reports success
// but no finite time has produced nothing to learn from, and a search told it
// succeeded would fit a surrogate model to an infinity or show a prompt a
// config that "worked" in no time at all.
TEST(BenchmarkStatusTest, ASuccessWithoutATimeIsNotOne) {
  BenchmarkResult result;
  result.status = Status::Success;
  for (double timeNs : {std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()}) {
    result.timeNs = timeNs;
    EXPECT_FALSE(result.measured());
    EXPECT_EQ(result.effectiveStatus(), Status::Failed);
    EXPECT_EQ(getNameForBenchmarkStatus(result.effectiveStatus()), "failed");
  }
}

// "Not applicable" survives, rather than being folded into a failure: the
// config was refused before anything ran, which says something different
// about the space than a compile that broke.
TEST(BenchmarkStatusTest, NotApplicableIsKeptApartFromFailed) {
  BenchmarkResult result;
  result.status = Status::NotApplicable;
  EXPECT_FALSE(result.measured());
  EXPECT_EQ(result.effectiveStatus(), Status::NotApplicable);
}

// The default, which is what a client gets by declaring one and filling in
// only what it knows.
TEST(BenchmarkStatusTest, ADefaultResultIsAFailure) {
  BenchmarkResult result;
  EXPECT_FALSE(result.measured());
  EXPECT_EQ(result.effectiveStatus(), Status::Failed);
}

//===----------------------------------------------------------------------===//
// Naming a strategy
//===----------------------------------------------------------------------===//

// These names are what `--tuning-space` accepts, so a round trip that lost one
// would either drop a search from the command line or spell it two ways.
TEST(SearchStrategyKindTest, EveryKindRoundTripsThroughItsName) {
  for (SearchStrategyKind kind :
       {SearchStrategyKind::Quick, SearchStrategyKind::Full,
        SearchStrategyKind::Exhaustive, SearchStrategyKind::LFBO,
        SearchStrategyKind::LLM, SearchStrategyKind::LLMSeededLFBO}) {
    StringRef name = getSearchStrategyKindName(kind);
    EXPECT_FALSE(name.empty());
    EXPECT_EQ(parseSearchStrategyKind(name), kind) << name;
  }
}

TEST(SearchStrategyKindTest, RefusesAKindItDoesNotHave) {
  EXPECT_FALSE(parseSearchStrategyKind("").has_value());
  EXPECT_FALSE(parseSearchStrategyKind("bayesian").has_value());
  // The names are matched exactly; a search is picked, not guessed at.
  EXPECT_FALSE(parseSearchStrategyKind("LLM").has_value());
  EXPECT_FALSE(parseSearchStrategyKind("llm-").has_value());
}

} // namespace
