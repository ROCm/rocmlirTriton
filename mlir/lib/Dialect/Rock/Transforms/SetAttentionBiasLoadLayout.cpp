//===- SetAttentionBiasLoadLayout.cpp - Guarded direct bias loads ---------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinOps.h"

#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LinearLayout.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSETATTENTIONBIASLOADLAYOUTPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-set-attention-bias-load-layout"

using namespace mlir;
using namespace mlir::rock;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {
struct RockSetAttentionBiasLoadLayoutPass
    : public rock::impl::RockSetAttentionBiasLoadLayoutPassBase<
          RockSetAttentionBiasLoadLayoutPass> {
  using RockSetAttentionBiasLoadLayoutPassBase::
      RockSetAttentionBiasLoadLayoutPassBase;
  void runOnOperation() override;
};

struct ScoreInfo {
  Attribute encoding;
  SmallVector<int64_t> shape;
  Operation *dot = nullptr;
  unsigned count = 0;
  bool invalid = false;
};

struct TaggedLoad {
  tt::LoadOp load;
  PreSoftmaxInputAttr metadata;
};

struct BiasPathStep {
  Operation *op;
  unsigned sourceOperand;
};

struct RematerializationPlan {
  tt::LoadOp load;
  SmallVector<BiasPathStep> path;
  Value endpoint;
  Attribute target;
  bool bypassLDS;
};
} // end anonymous namespace

static constexpr llvm::StringLiteral BypassLDSAttrName = "amdg.bypass_lds_load";

static bool isGfx942(ModuleOp module) {
  auto target = module->getAttrOfType<StringAttr>(ttg::AttrTargetName);
  if (!target)
    return false;

  StringRef targetName = target.getValue();
  targetName.consume_front("hip:");
  return targetName.split(':').first == "gfx942";
}

static std::optional<SmallVector<BiasPathStep>>
getBiasRematerializationPath(tt::LoadOp load, Value endpoint) {
  Value loadResult = load.getResult();
  Value current = endpoint;
  llvm::SetVector<Operation *> forwardSlice;
  mlir::getForwardSlice(load.getOperation(), &forwardSlice);
  if (current != loadResult &&
      (!current.getDefiningOp() ||
       !forwardSlice.contains(current.getDefiningOp()))) {
    LLVM_DEBUG(llvm::dbgs()
               << "reject direct bias load: bias does not feed score C\n");
    return std::nullopt;
  }

  SmallVector<BiasPathStep> reversePath;
  while (current != loadResult) {
    if (!current.hasOneUse()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "reject direct bias load: shared accumulator path value\n");
      return std::nullopt;
    }

    Operation *op = current.getDefiningOp();
    if (!op || op->getNumResults() != 1 || op->getNumRegions() != 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "reject direct bias load: unsupported path operation\n");
      return std::nullopt;
    }

    bool isConversion = isa<ttg::ConvertLayoutOp>(op);
    if (!isConversion && !op->hasTrait<OpTrait::Elementwise>()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "reject direct bias load: cannot rematerialize " << *op
                 << '\n');
      return std::nullopt;
    }

    std::optional<unsigned> sourceOperand;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      bool dependsOnLoad = operand == loadResult ||
                           (operand.getDefiningOp() &&
                            forwardSlice.contains(operand.getDefiningOp()));
      if (!dependsOnLoad)
        continue;
      if (sourceOperand) {
        LLVM_DEBUG(llvm::dbgs()
                   << "reject direct bias load: path merges bias values\n");
        return std::nullopt;
      }
      sourceOperand = index;
    }
    if (!sourceOperand) {
      LLVM_DEBUG(llvm::dbgs()
                 << "reject direct bias load: broken accumulator path\n");
      return std::nullopt;
    }

    if (!isConversion) {
      auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!resultType)
        return std::nullopt;
      for (Value operand : op->getOperands()) {
        auto operandType = dyn_cast<RankedTensorType>(operand.getType());
        if (operandType && operandType.getShape() != resultType.getShape()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "reject direct bias load: broadcasting path\n");
          return std::nullopt;
        }
      }
    }

    reversePath.push_back({op, *sourceOperand});
    current = op->getOperand(*sourceOperand);
  }

  if (!loadResult.hasOneUse()) {
    LLVM_DEBUG(llvm::dbgs()
               << "reject direct bias load: bias load has multiple users\n");
    return std::nullopt;
  }

  return SmallVector<BiasPathStep>(reversePath.rbegin(), reversePath.rend());
}

static ttg::ConvertLayoutOp findPostScoreLoadConversion(tt::LoadOp load,
                                                        Operation *dot) {
  if (!dot)
    return {};

  llvm::SetVector<Operation *> dotForwardSlice;
  mlir::getForwardSlice(dot, &dotForwardSlice);

  Value current = load.getResult();
  while (current.hasOneUse()) {
    Operation *user = *current.getUsers().begin();
    if (user == dot || user->getNumResults() != 1 || user->getNumRegions() != 0)
      return {};

    if (auto conversion = dyn_cast<ttg::ConvertLayoutOp>(user)) {
      llvm::SetVector<Operation *> conversionForwardSlice;
      mlir::getForwardSlice(conversion.getOperation(), &conversionForwardSlice);
      if (llvm::any_of(conversionForwardSlice, [&](Operation *candidate) {
            return dotForwardSlice.contains(candidate);
          }))
        return conversion;
    } else if (!user->hasTrait<OpTrait::Elementwise>()) {
      return {};
    }
    current = user->getResult(0);
  }
  return {};
}

static void rematerializeBiasPath(tt::LoadOp load, ArrayRef<BiasPathStep> path,
                                  Value endpoint, Attribute target,
                                  bool bypassLDS) {
  Operation *newLoad = convertDistributedOpEncoding(target, load);
  if (bypassLDS)
    newLoad->setAttr(BypassLDSAttrName,
                     BoolAttr::get(newLoad->getContext(), true));
  Value rematerialized = newLoad->getResult(0);
  OpBuilder builder(*endpoint.getUsers().begin());

  for (const BiasPathStep &step : path) {
    if (isa<ttg::ConvertLayoutOp>(step.op))
      continue;

    SmallVector<Value> operands;
    operands.reserve(step.op->getNumOperands());
    for (auto [index, operand] : llvm::enumerate(step.op->getOperands())) {
      if (index == step.sourceOperand) {
        operands.push_back(rematerialized);
        continue;
      }

      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (operandType && operandType.getEncoding() != target) {
        auto convertedType = operandType.cloneWithEncoding(target);
        operand = ttg::ConvertLayoutOp::create(builder, step.op->getLoc(),
                                               convertedType, operand);
      }
      operands.push_back(operand);
    }

    SmallVector<Type> resultTypes;
    resultTypes.reserve(step.op->getNumResults());
    for (Type resultType : step.op->getResultTypes()) {
      auto tensorType = cast<RankedTensorType>(resultType);
      resultTypes.push_back(tensorType.cloneWithEncoding(target));
    }

    OperationState state(step.op->getLoc(), step.op->getName());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttributes(step.op->getAttrs());
    rematerialized = builder.create(state)->getResult(0);
  }

  endpoint.replaceAllUsesWith(rematerialized);
}

static bool maintainsLoadCoalescing(RankedTensorType loadType, Attribute target,
                                    unsigned pointerContiguity) {
  auto source =
      dyn_cast_or_null<ttg::BlockedEncodingAttr>(loadType.getEncoding());
  auto targetDistributed = dyn_cast<ttg::DistributedEncodingTrait>(target);
  if (!source || !targetDistributed)
    return false;

  unsigned contiguousDim = source.getOrder().front();
  auto targetLinear =
      ttg::toGenericLinearEncoding(targetDistributed, loadType.getShape());
  if (targetLinear.getContigPerThread()[contiguousDim] < pointerContiguity ||
      targetLinear.getContigPerWarp()[contiguousDim] < pointerContiguity)
    return false;

  const mlir::triton::LinearLayout &targetLayout =
      targetLinear.getLinearLayout();
  auto warp = StringAttr::get(loadType.getContext(), "warp");
  auto bases = targetLayout.getBases().find(warp);
  return bases != targetLayout.getBases().end() &&
         llvm::all_of(bases->second, [](const auto &basis) {
           return llvm::any_of(basis, [](int32_t value) { return value != 0; });
         });
}

static bool
isSafeRematerializedLoad(tt::LoadOp load, Attribute target, int32_t numStages,
                         tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  auto loadType = dyn_cast<RankedTensorType>(load.getType());
  auto pointerType = dyn_cast<RankedTensorType>(load.getPtr().getType());
  if (!loadType || !pointerType || loadType.getRank() != 2 ||
      pointerType.getRank() != 2 ||
      loadType.getShape() != pointerType.getShape() ||
      !isa<tt::PointerType>(pointerType.getElementType()) || load.getMask() ||
      load.getOther())
    return false;

  auto source =
      dyn_cast_or_null<ttg::BlockedEncodingAttr>(loadType.getEncoding());
  if (!source || !isa<ttg::DistributedEncodingTrait>(target) || numStages < 1 ||
      numStages > 2)
    return false;

  ModuleOp module = load->getParentOfType<ModuleOp>();
  int numWarps = ttg::lookupNumWarps(load);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(module);
  if (threadsPerWarp != 64 || numWarps < 2 || numWarps > 8 || numCTAs != 1)
    return false;

  if (ttg::getCGALayout(source) != ttg::getCGALayout(target))
    return false;

  auto targetType = loadType.cloneWithEncoding(target);
  // This rewrite is useful only when exchanging the already-loaded values
  // would require LDS. Rematerializing the pointer DAG in `target` avoids that
  // exchange.
  if (!cvtNeedsSharedMemory(loadType, targetType))
    return false;

  // The validated path deliberately uses scalar strided loads. Do not change
  // the vectorization chosen for naturally contiguous inputs.
  unsigned contiguity = axisInfoAnalysis.getContiguity(load.getPtr());
  if (contiguity != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "reject direct bias load: expected scalar pointer "
                  "contiguity\n");
    return false;
  }
  if (!maintainsLoadCoalescing(loadType, target, contiguity))
    return false;

  mlir::triton::LinearLayout targetLayout = ttg::toLinearLayout(targetType);
  if (!targetLayout.isInjective())
    return false;

  auto freeVariableMasks = targetLayout.getFreeVariableMasks();
  for (StringRef inputName : {"lane", "warp"}) {
    auto found =
        freeVariableMasks.find(StringAttr::get(load.getContext(), inputName));
    if (found == freeVariableMasks.end() || found->second != 0)
      return false;
  }

  unsigned sourceElements = ttg::getTotalElemsPerThread(loadType);
  unsigned targetElements = ttg::getTotalElemsPerThread(targetType);
  unsigned elementBits = std::max(8u, loadType.getElementTypeBitWidth());
  uint64_t targetRegisterBits = uint64_t(targetElements) * elementBits;
  if (targetElements > sourceElements || targetElements > 256 ||
      targetRegisterBits > 256 * 32)
    return false;

  return true;
}

static bool isSafeDirectBiasLoad(tt::LoadOp load,
                                 ttg::AMDMfmaEncodingAttr target,
                                 int32_t numStages,
                                 tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  // The direct accumulator path is currently validated only for CDNA3. In
  // particular, MFMA v4 remains disabled until its accumulator distribution
  // has dedicated runtime coverage.
  if (target.getVersion() != 3 || !target.getIsTransposed() ||
      !target.hasUnitTilesPerWarp())
    return false;

  auto loadType = dyn_cast<RankedTensorType>(load.getType());
  if (!loadType)
    return false;
  ArrayRef<int64_t> shape = loadType.getShape();
  ArrayRef<unsigned> instructionShape = target.getInstrShape();
  if (instructionShape.size() != 3 ||
      (instructionShape[0] != 16 && instructionShape[0] != 32) ||
      instructionShape[1] != instructionShape[0] || shape[0] <= 0 ||
      shape[1] <= 0 || shape[0] % instructionShape[0] != 0 ||
      shape[1] % instructionShape[1] != 0)
    return false;

  unsigned targetWarps = std::accumulate(target.getWarpsPerCTA().begin(),
                                         target.getWarpsPerCTA().end(), 1u,
                                         std::multiplies<unsigned>());
  if (targetWarps != static_cast<unsigned>(ttg::lookupNumWarps(load)))
    return false;

  return isSafeRematerializedLoad(load, target, numStages, axisInfoAnalysis);
}

void RockSetAttentionBiasLoadLayoutPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (!isGfx942(module))
    return;

  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(module);

  for (tt::FuncOp func : module.getOps<tt::FuncOp>()) {
    llvm::DenseMap<uint32_t, ScoreInfo> scores;
    func.walk([&](Operation *candidate) {
      if (!isa<tt::DotOp, tt::DotScaledOp>(candidate))
        return;
      auto group = dyn_cast_or_null<AttentionGroupAttr>(
          candidate->getDiscardableAttr(AttentionGroupAttr::getNameStr()));
      if (!group)
        return;

      ScoreInfo &score = scores[group.getGroupId()];
      ++score.count;
      auto resultType =
          dyn_cast<RankedTensorType>(candidate->getResult(0).getType());
      auto mfma = resultType ? dyn_cast_or_null<ttg::AMDMfmaEncodingAttr>(
                                   resultType.getEncoding())
                             : nullptr;
      if (!mfma || (score.encoding && score.encoding != mfma) ||
          (!score.shape.empty() &&
           ArrayRef<int64_t>(score.shape) != resultType.getShape())) {
        score.invalid = true;
        return;
      }
      score.encoding = mfma;
      score.shape.assign(resultType.getShape().begin(),
                         resultType.getShape().end());
      score.dot = candidate;
    });

    llvm::DenseMap<uint32_t, SmallVector<TaggedLoad>> loadsByGroup;
    func.walk([&](tt::LoadOp load) {
      auto metadata = dyn_cast_or_null<PreSoftmaxInputAttr>(
          load->getDiscardableAttr(PreSoftmaxInputAttr::getNameStr()));
      if (metadata)
        loadsByGroup[metadata.getGroupId()].push_back({load, metadata});
    });

    for (auto &[groupId, loads] : loadsByGroup) {
      auto scoreIt = scores.find(groupId);
      if (scoreIt == scores.end() || scoreIt->second.invalid ||
          scoreIt->second.count != 1 || !scoreIt->second.encoding)
        continue;

      llvm::SmallDenseSet<uint32_t, 4> inputIndices;
      TaggedLoad *directCandidate = nullptr;
      bool ambiguous = false;
      for (TaggedLoad &tagged : loads) {
        if (!inputIndices.insert(tagged.metadata.getInputIndex()).second) {
          ambiguous = true;
          break;
        }
        if (tagged.metadata.getRole() != PreSoftmaxInputRole::Bias ||
            tagged.metadata.getOrientation() !=
                PreSoftmaxInputOrientation::Transposed)
          continue;
        if (directCandidate) {
          ambiguous = true;
          break;
        }
        directCandidate = &tagged;
      }
      if (ambiguous || !directCandidate)
        continue;

      auto scoreEncoding =
          dyn_cast<ttg::AMDMfmaEncodingAttr>(scoreIt->second.encoding);
      auto candidateType =
          dyn_cast<RankedTensorType>(directCandidate->load.getType());
      if (!candidateType ||
          candidateType.getShape() !=
              ArrayRef<int64_t>(scoreIt->second.shape) ||
          !isSafeDirectBiasLoad(directCandidate->load, scoreEncoding, numStages,
                                axisInfoAnalysis))
        continue;

      Operation *scoreDot = scoreIt->second.dot;
      if (!scoreDot || scoreDot->getNumOperands() < 3)
        continue;

      SmallVector<RematerializationPlan> plans;
      Value accumulator = scoreDot->getOperand(2);
      auto accumulatorPath =
          getBiasRematerializationPath(directCandidate->load, accumulator);
      if (!accumulatorPath)
        continue;
      plans.push_back({directCandidate->load, std::move(*accumulatorPath),
                       accumulator, scoreEncoding, /*bypassLDS=*/true});

      // Coalescing may have selected the transposed load's warp ownership for
      // other inputs in the same score expression. Rematerialize only those
      // dataflow-connected loads that have an explicit LDS-requiring
      // conversion to the score's post-dot layout. The target is inferred from
      // that conversion; unrelated inputs remain untouched.
      for (TaggedLoad &tagged : loads) {
        if (tagged.load == directCandidate->load)
          continue;

        ttg::ConvertLayoutOp conversion =
            findPostScoreLoadConversion(tagged.load, scoreDot);
        if (!conversion)
          continue;
        auto targetType =
            dyn_cast<RankedTensorType>(conversion.getResult().getType());
        Attribute target = targetType ? targetType.getEncoding() : nullptr;
        if (!target || !isSafeRematerializedLoad(tagged.load, target, numStages,
                                                 axisInfoAnalysis))
          continue;

        auto path =
            getBiasRematerializationPath(tagged.load, conversion.getResult());
        if (!path)
          continue;
        plans.push_back({tagged.load, std::move(*path), conversion.getResult(),
                         target,
                         /*bypassLDS=*/false});
      }

      for (RematerializationPlan &plan : plans)
        rematerializeBiasPath(plan.load, plan.path, plan.endpoint, plan.target,
                              plan.bypassLDS);
    }
  }
}
