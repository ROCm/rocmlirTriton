//===- RandomForest.cpp - Small random forest classifier ------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RandomForest.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>

using namespace mlir;
using namespace mlir::rock;

//===----------------------------------------------------------------------===//
// DecisionTree
//===----------------------------------------------------------------------===//

int DecisionTree::applyLeaf(ArrayRef<float> sample) const {
  assert(!nodes.empty() && "querying an untrained tree");
  int node = 0;
  while (!nodes[node].isLeaf()) {
    const Node &current = nodes[node];
    node = sample[current.feature] <= current.threshold ? current.left
                                                        : current.right;
  }
  return node;
}

void DecisionTree::decisionPath(ArrayRef<float> sample,
                                SmallVectorImpl<int> &path) const {
  assert(!nodes.empty() && "querying an untrained tree");
  path.clear();
  int node = 0;
  while (true) {
    path.push_back(node);
    const Node &current = nodes[node];
    if (current.isLeaf())
      return;
    node = sample[current.feature] <= current.threshold ? current.left
                                                        : current.right;
  }
}

float DecisionTree::predictProba(ArrayRef<float> sample) const {
  return nodes[applyLeaf(sample)].proba;
}

//===----------------------------------------------------------------------===//
// Training
//===----------------------------------------------------------------------===//

namespace {
/// Binary entropy in bits, i.e. scikit-learn's "log_loss" split criterion.
double entropy(double positiveFraction) {
  if (positiveFraction <= 0.0 || positiveFraction >= 1.0)
    return 0.0;
  double negativeFraction = 1.0 - positiveFraction;
  return -positiveFraction * std::log2(positiveFraction) -
         negativeFraction * std::log2(negativeFraction);
}

/// Grows one weighted CART tree. Each node considers a fresh random subset of
/// the features and splits on the one giving the largest entropy reduction.
class TreeBuilder {
public:
  TreeBuilder(ArrayRef<std::vector<float>> samples, ArrayRef<float> labels,
              ArrayRef<float> weights, const RandomForestOptions &options,
              unsigned numFeatures, std::mt19937_64 &rng)
      : samples(samples), labels(labels), weights(weights), options(options),
        numFeatures(numFeatures), rng(rng),
        // The classic random-forest default: consider sqrt(numFeatures)
        // features per split, which decorrelates the trees.
        maxFeatures(std::max(
            1u, static_cast<unsigned>(std::sqrt(double(numFeatures))))) {}

  void build(std::vector<size_t> indices,
             std::vector<DecisionTree::Node> &out) {
    nodes = &out;
    addNode(std::move(indices), /*depth=*/0);
  }

private:
  /// Weighted totals of a set of training points.
  struct Totals {
    double weight = 0.0;
    double positiveWeight = 0.0;

    double positiveFraction() const {
      return weight > 0.0 ? positiveWeight / weight : 0.0;
    }
  };

  Totals totalsOf(ArrayRef<size_t> indices) const {
    Totals totals;
    for (size_t idx : indices) {
      totals.weight += weights[idx];
      if (labels[idx] > 0.5f)
        totals.positiveWeight += weights[idx];
    }
    return totals;
  }

  /// Appends a node for `indices` and returns its index, recursing into its
  /// children when a worthwhile split exists.
  int addNode(std::vector<size_t> indices, unsigned depth) {
    int nodeIdx = nodes->size();
    nodes->emplace_back();
    Totals totals = totalsOf(indices);
    (*nodes)[nodeIdx].proba = totals.positiveFraction();

    bool isPure =
        totals.positiveWeight <= 0.0 || totals.positiveWeight >= totals.weight;
    if (isPure || depth >= options.maxDepth ||
        indices.size() < options.minSamplesSplit)
      return nodeIdx;

    int bestFeature = DecisionTree::Node::kLeaf;
    float bestThreshold = 0.0f;
    double bestGain = 0.0;
    double parentEntropy = entropy(totals.positiveFraction());

    // Like scikit-learn, keep drawing features until `maxFeatures` of them have
    // actually been examined. A feature that is constant across this node's
    // points offers no split, so letting it consume one of the draws would turn
    // unlucky nodes into leaves and leave whole trees as stumps.
    unsigned examined = 0;
    for (unsigned feature : shuffledFeatures()) {
      if (examined >= maxFeatures)
        break;
      if (isConstantAt(indices, feature))
        continue;
      ++examined;

      // Sweep the points in increasing feature order, moving one distinct value
      // at a time from the right child to the left one and scoring the split
      // that would sit between them.
      sorted.clear();
      sorted.reserve(indices.size());
      for (size_t idx : indices)
        sorted.emplace_back(samples[idx][feature], idx);
      llvm::sort(sorted, [](const std::pair<float, size_t> &lhs,
                            const std::pair<float, size_t> &rhs) {
        return lhs.first < rhs.first;
      });

      Totals left;
      for (size_t i = 0, e = sorted.size() - 1; i < e; ++i) {
        size_t idx = sorted[i].second;
        left.weight += weights[idx];
        if (labels[idx] > 0.5f)
          left.positiveWeight += weights[idx];

        // Only thresholds strictly between two distinct values separate the
        // points; equal values must stay on the same side.
        if (sorted[i].first == sorted[i + 1].first)
          continue;
        size_t leftCount = i + 1;
        size_t rightCount = sorted.size() - leftCount;
        if (leftCount < options.minSamplesLeaf ||
            rightCount < options.minSamplesLeaf)
          continue;

        Totals right;
        right.weight = totals.weight - left.weight;
        right.positiveWeight = totals.positiveWeight - left.positiveWeight;
        if (left.weight <= 0.0 || right.weight <= 0.0)
          continue;

        double gain =
            parentEntropy -
            (left.weight / totals.weight) * entropy(left.positiveFraction()) -
            (right.weight / totals.weight) * entropy(right.positiveFraction());
        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = feature;
          bestThreshold =
              sorted[i].first + (sorted[i + 1].first - sorted[i].first) / 2.0f;
        }
      }
    }

    if (bestFeature == DecisionTree::Node::kLeaf)
      return nodeIdx;

    std::vector<size_t> leftIndices;
    std::vector<size_t> rightIndices;
    for (size_t idx : indices) {
      if (samples[idx][bestFeature] <= bestThreshold)
        leftIndices.push_back(idx);
      else
        rightIndices.push_back(idx);
    }
    // A threshold that rounds to one of the two neighbouring values can put
    // every point on one side; keep the node a leaf rather than recursing
    // forever on the same set.
    if (leftIndices.empty() || rightIndices.empty())
      return nodeIdx;

    indices.clear();
    indices.shrink_to_fit();
    int left = addNode(std::move(leftIndices), depth + 1);
    int right = addNode(std::move(rightIndices), depth + 1);
    // `nodes` may have been reallocated by the recursive calls.
    DecisionTree::Node &node = (*nodes)[nodeIdx];
    node.feature = bestFeature;
    node.threshold = bestThreshold;
    node.left = left;
    node.right = right;
    return nodeIdx;
  }

  /// Whether `feature` takes the same value at every point of this node, in
  /// which case no threshold can separate them.
  bool isConstantAt(ArrayRef<size_t> indices, unsigned feature) const {
    if (indices.empty())
      return true;
    float first = samples[indices.front()][feature];
    return llvm::all_of(
        indices, [&](size_t idx) { return samples[idx][feature] == first; });
  }

  /// The features in a fresh random order (Fisher-Yates over a persistent
  /// permutation, so no allocation per node). The caller walks the order until
  /// it has seen enough usable features, which is why the whole permutation is
  /// produced rather than just a prefix of it.
  ArrayRef<unsigned> shuffledFeatures() {
    if (featurePool.empty()) {
      featurePool.resize(numFeatures);
      std::iota(featurePool.begin(), featurePool.end(), 0u);
    }
    for (unsigned i = 0; i + 1 < numFeatures; ++i) {
      std::uniform_int_distribution<unsigned> pick(i, numFeatures - 1);
      std::swap(featurePool[i], featurePool[pick(rng)]);
    }
    return featurePool;
  }

  ArrayRef<std::vector<float>> samples;
  ArrayRef<float> labels;
  ArrayRef<float> weights;
  const RandomForestOptions &options;
  unsigned numFeatures;
  std::mt19937_64 &rng;
  unsigned maxFeatures;

  std::vector<DecisionTree::Node> *nodes = nullptr;
  std::vector<unsigned> featurePool;
  std::vector<std::pair<float, size_t>> sorted;
};
} // namespace

//===----------------------------------------------------------------------===//
// RandomForestClassifier
//===----------------------------------------------------------------------===//

void RandomForestClassifier::fit(ArrayRef<std::vector<float>> samples,
                                 ArrayRef<float> labels,
                                 ArrayRef<float> weights) {
  trees.clear();
  assert(samples.size() == labels.size() && samples.size() == weights.size() &&
         "fit() expects one label and one weight per sample");
  if (samples.empty() || samples[0].empty())
    return;

  unsigned numFeatures = samples[0].size();
  trees.resize(options.numTrees);
  for (unsigned treeIdx = 0; treeIdx < options.numTrees; ++treeIdx) {
    // Seed per tree rather than threading one generator through, so a tree's
    // shape does not depend on how many trees were built before it.
    std::mt19937_64 rng(options.seed + treeIdx);

    std::vector<size_t> indices(samples.size());
    if (options.bootstrap) {
      std::uniform_int_distribution<size_t> pick(0, samples.size() - 1);
      for (size_t &idx : indices)
        idx = pick(rng);
    } else {
      std::iota(indices.begin(), indices.end(), size_t(0));
    }

    TreeBuilder builder(samples, labels, weights, options, numFeatures, rng);
    builder.build(std::move(indices), trees[treeIdx].nodes);
  }
}

float RandomForestClassifier::predictProba(ArrayRef<float> sample) const {
  if (trees.empty())
    return 0.0f;
  double sum = 0.0;
  for (const DecisionTree &tree : trees)
    sum += tree.predictProba(sample);
  return sum / trees.size();
}

void RandomForestClassifier::applyLeaves(ArrayRef<float> sample,
                                         SmallVectorImpl<int> &leaves) const {
  leaves.clear();
  leaves.reserve(trees.size());
  for (const DecisionTree &tree : trees)
    leaves.push_back(tree.applyLeaf(sample));
}
