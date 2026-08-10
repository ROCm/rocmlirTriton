//===- RandomForest.h - Small random forest classifier ----------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A self-contained binary random forest classifier, used as the surrogate
// model of the LFBO tuning search (see LFBOSearch.h).
//
// It is a small reimplementation of the parts of scikit-learn's
// RandomForestClassifier that the search needs: weighted CART trees grown on
// bootstrap samples with a random feature subset per split, plus access to the
// individual trees so the search can inspect decision paths and leaf
// assignments. It deliberately does not try to reproduce scikit-learn's
// numbers; the search only ever uses the model to rank candidates.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_ROCK_TUNING_RANDOMFOREST_H
#define MLIR_LIB_DIALECT_ROCK_TUNING_RANDOMFOREST_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <vector>

namespace mlir {
namespace rock {

/// A binary classification tree. Nodes are held in one vector; a node is a leaf
/// when its feature is `Node::kLeaf`, and otherwise sends samples with
/// `features[feature] <= threshold` to `left` and the rest to `right`.
class DecisionTree {
public:
  struct Node {
    static constexpr int kLeaf = -1;

    int feature = kLeaf;
    float threshold = 0.0f;
    int left = kLeaf;
    int right = kLeaf;
    /// For a leaf, the weighted fraction of its training samples labelled 1.
    float proba = 0.0f;

    bool isLeaf() const { return feature == kLeaf; }
  };

  /// The index of the leaf `sample` ends up in. Leaf indices are only
  /// comparable within one tree.
  int applyLeaf(ArrayRef<float> sample) const;

  /// The nodes `sample` visits, root first and leaf last.
  void decisionPath(ArrayRef<float> sample, SmallVectorImpl<int> &path) const;

  /// This tree's estimate of P(label == 1) for `sample`.
  float predictProba(ArrayRef<float> sample) const;

  ArrayRef<Node> getNodes() const { return nodes; }
  bool empty() const { return nodes.empty(); }

private:
  friend class RandomForestClassifier;

  std::vector<Node> nodes;
};

struct RandomForestOptions {
  /// Number of trees in the ensemble.
  unsigned numTrees = 100;
  /// Depth limit, counting the root as depth 0.
  unsigned maxDepth = 32;
  /// Nodes with fewer samples than this are not split further.
  unsigned minSamplesSplit = 2;
  /// A split is only taken if both sides keep at least this many samples.
  unsigned minSamplesLeaf = 1;
  /// Draw each tree's training set with replacement, as scikit-learn does.
  bool bootstrap = true;
  /// Fixed so that repeated tuning runs of the same problem agree.
  uint64_t seed = 42;
};

/// An ensemble of `DecisionTree`s trained on bootstrap samples, splitting on a
/// random sqrt(numFeatures)-sized subset of the features at each node and using
/// entropy (scikit-learn's "log_loss" criterion) to score splits.
class RandomForestClassifier {
public:
  explicit RandomForestClassifier(const RandomForestOptions &options = {})
      : options(options) {}

  /// Trains the ensemble. `samples` holds one feature vector per training
  /// point, all of the same length; `labels` are 0 or 1 and `weights` scale
  /// each point's contribution to the split scores. If only one label occurs
  /// the fit succeeds but every tree collapses to a single leaf, so the forest
  /// predicts that label everywhere; callers that need a ranking should wait
  /// for both labels before fitting.
  void fit(ArrayRef<std::vector<float>> samples, ArrayRef<float> labels,
           ArrayRef<float> weights);

  /// P(label == 1) for `sample`, averaged over the trees.
  float predictProba(ArrayRef<float> sample) const;

  /// The leaf `sample` reaches in each tree. Two samples sharing a leaf in many
  /// trees are close in the model's view of the space, which the search uses to
  /// keep a batch of candidates diverse.
  void applyLeaves(ArrayRef<float> sample, SmallVectorImpl<int> &leaves) const;

  bool isFitted() const { return !trees.empty(); }
  unsigned getNumTrees() const { return trees.size(); }
  const DecisionTree &getTree(unsigned idx) const { return trees[idx]; }

private:
  RandomForestOptions options;
  std::vector<DecisionTree> trees;
};

} // namespace rock
} // namespace mlir

#endif // MLIR_LIB_DIALECT_ROCK_TUNING_RANDOMFOREST_H
