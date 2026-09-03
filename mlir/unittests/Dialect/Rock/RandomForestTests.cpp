//===- RandomForestTests.cpp - Tests for the LFBO surrogate model ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The forest is not meant to reproduce scikit-learn's numbers, so these tests
// pin the properties the LFBO search actually relies on: that it ranks a
// separable space correctly, that sample weights move the leaf estimates, that
// two runs of the same problem agree, and that the tree internals the search
// walks (decision paths and leaf ids) stay self-consistent.
//
//===----------------------------------------------------------------------===//

#include "RandomForest.h"

#include "gtest/gtest.h"

#include <vector>

using namespace mlir;
using namespace mlir::rock;

namespace {

// A linearly separable problem: feature 0 decides the label at 0.5, and the
// remaining features are constant, so they can never yield entropy gain and the
// trees are forced onto the informative one.
struct SeparableData {
  std::vector<std::vector<float>> samples;
  std::vector<float> labels;
  std::vector<float> weights;

  explicit SeparableData(unsigned numPoints = 40, unsigned numFeatures = 3) {
    for (unsigned i = 0; i < numPoints; ++i) {
      float x = static_cast<float>(i) / (numPoints - 1);
      std::vector<float> sample(numFeatures, 0.0f);
      sample[0] = x;
      samples.push_back(std::move(sample));
      labels.push_back(x > 0.5f ? 1.0f : 0.0f);
      weights.push_back(1.0f);
    }
  }
};

std::vector<float> point(float x0, unsigned numFeatures = 3) {
  std::vector<float> sample(numFeatures, 0.0f);
  sample[0] = x0;
  return sample;
}

TEST(RandomForestTest, UnfittedForestIsInert) {
  RandomForestClassifier forest;
  EXPECT_FALSE(forest.isFitted());
  EXPECT_EQ(forest.getNumTrees(), 0u);
  EXPECT_FLOAT_EQ(forest.predictProba(point(0.9f)), 0.0f);
}

TEST(RandomForestTest, EmptyTrainingSetLeavesForestUnfitted) {
  RandomForestClassifier forest;
  forest.fit({}, {}, {});
  EXPECT_FALSE(forest.isFitted());

  // A sample with no features carries no information to split on.
  std::vector<std::vector<float>> degenerate = {{}};
  std::vector<float> labels = {1.0f};
  std::vector<float> weights = {1.0f};
  forest.fit(degenerate, labels, weights);
  EXPECT_FALSE(forest.isFitted());
}

TEST(RandomForestTest, SeparableDataIsRankedCorrectly) {
  SeparableData data;
  RandomForestClassifier forest;
  forest.fit(data.samples, data.labels, data.weights);

  ASSERT_TRUE(forest.isFitted());
  EXPECT_EQ(forest.getNumTrees(), RandomForestOptions().numTrees);

  EXPECT_LT(forest.predictProba(point(0.05f)), 0.2f);
  EXPECT_GT(forest.predictProba(point(0.95f)), 0.8f);

  // Ranking is what the search consumes. The scores may not increase strictly -
  // a forest that separates the two classes perfectly is flat on either side of
  // the boundary - but they must never decrease, and must rise across it.
  float previous = -1.0f;
  for (int i = 0; i <= 10; ++i) {
    float proba = forest.predictProba(point(static_cast<float>(i) / 10.0f));
    EXPECT_GE(proba, previous) << "score dropped at x=" << (i / 10.0f);
    previous = proba;
  }
  EXPECT_LT(forest.predictProba(point(0.4f)), forest.predictProba(point(0.6f)));
}

TEST(RandomForestTest, ProbabilitiesStayInUnitRange) {
  SeparableData data;
  RandomForestClassifier forest;
  forest.fit(data.samples, data.labels, data.weights);

  for (int i = -5; i <= 15; ++i) {
    float x = static_cast<float>(i) / 10.0f;
    float proba = forest.predictProba(point(x));
    EXPECT_GE(proba, 0.0f) << "at x=" << x;
    EXPECT_LE(proba, 1.0f) << "at x=" << x;
  }
}

TEST(RandomForestTest, SameSeedGivesIdenticalPredictions) {
  SeparableData data;
  RandomForestOptions options;
  options.seed = 1234;

  RandomForestClassifier first(options);
  RandomForestClassifier second(options);
  first.fit(data.samples, data.labels, data.weights);
  second.fit(data.samples, data.labels, data.weights);

  for (int i = 0; i <= 10; ++i) {
    std::vector<float> sample = point(static_cast<float>(i) / 10.0f);
    EXPECT_FLOAT_EQ(first.predictProba(sample), second.predictProba(sample));
  }
}

TEST(RandomForestTest, RefittingReplacesTheEnsemble) {
  SeparableData data;
  RandomForestOptions options;
  options.numTrees = 7;
  RandomForestClassifier forest(options);

  forest.fit(data.samples, data.labels, data.weights);
  EXPECT_EQ(forest.getNumTrees(), 7u);
  // Fitting again must not append to the previous ensemble; the search refits
  // the same object once per generation.
  forest.fit(data.samples, data.labels, data.weights);
  EXPECT_EQ(forest.getNumTrees(), 7u);
}

TEST(RandomForestTest, SampleWeightsDecideAmbiguousLabels) {
  // Both points sit at the same coordinates with opposite labels, so no split
  // can separate them and the root leaf holds the weighted positive fraction.
  std::vector<std::vector<float>> samples = {point(0.5f), point(0.5f)};
  std::vector<float> labels = {0.0f, 1.0f};
  std::vector<float> weights = {1.0f, 9.0f};

  RandomForestOptions options;
  // Every tree sees the same data, so the expected value is exact.
  options.bootstrap = false;
  RandomForestClassifier forest(options);
  forest.fit(samples, labels, weights);

  EXPECT_NEAR(forest.predictProba(point(0.5f)), 0.9f, 1e-5f);

  weights = {9.0f, 1.0f};
  forest.fit(samples, labels, weights);
  EXPECT_NEAR(forest.predictProba(point(0.5f)), 0.1f, 1e-5f);
}

TEST(RandomForestTest, SingleClassTrainingIsConstant) {
  SeparableData data;
  std::fill(data.labels.begin(), data.labels.end(), 1.0f);

  RandomForestClassifier forest;
  forest.fit(data.samples, data.labels, data.weights);

  ASSERT_TRUE(forest.isFitted());
  EXPECT_FLOAT_EQ(forest.predictProba(point(0.05f)), 1.0f);
  EXPECT_FLOAT_EQ(forest.predictProba(point(0.95f)), 1.0f);
}

TEST(RandomForestTest, DepthLimitOfZeroGivesTheBaseRate) {
  SeparableData data(/*numPoints=*/40);
  RandomForestOptions options;
  options.maxDepth = 0;
  options.bootstrap = false;
  RandomForestClassifier forest(options);
  forest.fit(data.samples, data.labels, data.weights);

  // 40 evenly spaced points in [0,1], positive when x > 0.5: 20 of them.
  EXPECT_NEAR(forest.predictProba(point(0.05f)), 0.5f, 1e-5f);
  EXPECT_NEAR(forest.predictProba(point(0.95f)), 0.5f, 1e-5f);
  for (unsigned i = 0; i < forest.getNumTrees(); ++i)
    EXPECT_EQ(forest.getTree(i).getNodes().size(), 1u);
}

TEST(RandomForestTest, ApplyLeavesIsPerTreeAndStable) {
  SeparableData data;
  RandomForestClassifier forest;
  forest.fit(data.samples, data.labels, data.weights);

  SmallVector<int> low, lowAgain, high;
  forest.applyLeaves(point(0.05f), low);
  forest.applyLeaves(point(0.05f), lowAgain);
  forest.applyLeaves(point(0.95f), high);

  EXPECT_EQ(low.size(), forest.getNumTrees());
  EXPECT_EQ(low, lowAgain);
  // Points on opposite sides of the boundary must part company somewhere.
  EXPECT_NE(low, high);

  for (unsigned i = 0; i < forest.getNumTrees(); ++i) {
    const DecisionTree &tree = forest.getTree(i);
    ASSERT_LT(static_cast<size_t>(low[i]), tree.getNodes().size());
    EXPECT_TRUE(tree.getNodes()[low[i]].isLeaf());
  }
}

TEST(RandomForestTest, DecisionPathFollowsTheComparisons) {
  SeparableData data;
  RandomForestClassifier forest;
  forest.fit(data.samples, data.labels, data.weights);

  std::vector<float> sample = point(0.7f);
  for (unsigned i = 0; i < forest.getNumTrees(); ++i) {
    const DecisionTree &tree = forest.getTree(i);
    ASSERT_FALSE(tree.empty());

    SmallVector<int> path;
    tree.decisionPath(sample, path);
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(path.front(), 0) << "path must start at the root, tree " << i;
    EXPECT_EQ(path.back(), tree.applyLeaf(sample))
        << "path must end where applyLeaf lands, tree " << i;

    ArrayRef<DecisionTree::Node> nodes = tree.getNodes();
    EXPECT_TRUE(nodes[path.back()].isLeaf());
    for (size_t step = 0; step + 1 < path.size(); ++step) {
      const DecisionTree::Node &node = nodes[path[step]];
      ASSERT_FALSE(node.isLeaf());
      int expected =
          sample[node.feature] <= node.threshold ? node.left : node.right;
      EXPECT_EQ(path[step + 1], expected);
    }
  }
}

TEST(RandomForestTest, TreeStructureIsWellFormed) {
  SeparableData data;
  RandomForestClassifier forest;
  forest.fit(data.samples, data.labels, data.weights);

  for (unsigned i = 0; i < forest.getNumTrees(); ++i) {
    ArrayRef<DecisionTree::Node> nodes = forest.getTree(i).getNodes();
    ASSERT_FALSE(nodes.empty());
    for (const DecisionTree::Node &node : nodes) {
      EXPECT_GE(node.proba, 0.0f);
      EXPECT_LE(node.proba, 1.0f);
      if (node.isLeaf())
        continue;
      EXPECT_GE(node.feature, 0);
      EXPECT_LT(static_cast<size_t>(node.feature), data.samples[0].size());
      // Children are appended after their parent, so indices only ever grow.
      EXPECT_GT(node.left, 0);
      EXPECT_GT(node.right, node.left);
      EXPECT_LT(static_cast<size_t>(node.right), nodes.size());
    }
  }
}

} // namespace
