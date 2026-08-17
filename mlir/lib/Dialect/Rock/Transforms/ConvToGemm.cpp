//===- ConvToGemm.cpp - MLIR Rock ops lowering passes ------------===//
//
// Copyright 2020 The MLIR Authors.
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
// This pass converts rock.conv into rock.transform and
// rock.gemm. Additionally, it also converts rock.conv_elementwise_gemm
// into rock.gemm_elementwise_gemm.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockConvInterface.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/UtilityParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <iterator>
#include <numeric>
#include <tuple>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKCONVTOGEMMPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-conv-to-gemm"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;
//===----------------------------------------------------------------------===//
// Conv (forward, backward) lowering.
//===----------------------------------------------------------------------===//
// High level convolution operation always have
// [filter, input, output]
// as the convolution argument. The only difference between different
// height level convolution operations is the argument sequence. For
// simplicity, we always arrange the first two arguments to be input
// and the last argument to be output

namespace {
// The ArgumentFields keep track of differences between conv operations
struct ArgumentFields {
  int gridwiseGemmArgumentPosition[3];
  StringRef gemmTargetCharName[3];
};

struct RockConvToGemmPass
    : public rock::impl::RockConvToGemmPassBase<RockConvToGemmPass> {
  void runOnOperation() override;
};

template <typename T>
LogicalResult checkNames(ArrayRef<StringRef> actual,
                         ArrayRef<StringRef> expected, StringRef argName,
                         T op) {
  if (actual.size() != expected.size()) {
    return op.emitOpError("Layout mismatch in ")
           << argName << " tensor: Expected " << expected.size()
           << " dimensions but have " << actual.size();
  }
  for (StringRef name : expected) {
    if (llvm::find(actual, name) == actual.end()) {
      return op.emitOpError("Layout mismatch in ")
             << argName << " tensor: Expected it to have a `" << name
             << "` dimension";
    }
  }
  return success();
}

// Sort the dimensions in `names` so that they are in the order they appear in
// within `transform`. This allows Merge{} operations to not preform
// transposes that are not needed.
void matchUnderlyingOrder(SmallVectorImpl<StringRef> &names,
                          BottomUpTMBuilder &transform) {
  std::sort(names.begin(), names.end(),
            [&transform](const StringRef &v1, const StringRef &v2) -> bool {
              return transform.startIndex(v1) < transform.startIndex(v2);
            });
}

// If `dest` is defined after `op` (e.g. because RegularizeOutput added
// transforms to the store dest after the conv), move the builder's insertion
// point past `dest`'s definition so that new ops don't violate dominance.
static void ensureInsertionAfterDef(PatternRewriter &b, Operation *op,
                                    Value dest) {
  if (Operation *defOp = dest.getDefiningOp()) {
    if (defOp->getBlock() == op->getBlock() && op->isBeforeInBlock(defOp))
      b.setInsertionPointAfter(defOp);
  }
}

/// Update any StoreOp that uses the conv result to use the gemm result instead.
/// The conv result type differs from the gemm result type (due to shape
/// transformations), so we need to update the StoreOp to use the gemm result
/// and the transformed destination tensor.
void updateStoreOpForGemm(PatternRewriter &b, Location loc, Value convResult,
                          Value gemmResult, Value gemmDest,
                          StoreMethodAttr storeMethod) {
  for (Operation *user : llvm::make_early_inc_range(convResult.getUsers())) {
    if (auto storeOp = dyn_cast<StoreOp>(user)) {
      Type storeResultType = storeOp.getResult().getType();
      auto newStoreOp =
          StoreOp::create(b, loc, storeResultType, gemmResult, gemmDest,
                          storeOp.getResultAlias(), storeMethod);
      b.replaceOp(storeOp, newStoreOp.getResult());
    }
  }
}

/// Get the dimension names for the given `op` into `filterNames`, `inputNames`
/// and `outputNames`, returning failure if `op`'s layout doesn't contain all of
/// the expected dimension names.
template <typename T>
LogicalResult getConvDimNames(T op, SmallVectorImpl<StringRef> &filterNames,
                              SmallVectorImpl<StringRef> &inputNames,
                              SmallVectorImpl<StringRef> &outputNames,
                              bool enableOutput = true) {
  auto filterLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->template getAttrOfType<ArrayAttr>("input_layout");

  unsigned size = filterLayoutAttr.size();
  ArrayAttr outputLayoutAttr;
  if (enableOutput) {
    outputLayoutAttr = op->template getAttrOfType<ArrayAttr>("output_layout");
    if (size != outputLayoutAttr.size())
      return op.emitOpError(
          "All convolution layouts must have the same length");
  }

  if (size != inputLayoutAttr.size())
    return op.emitOpError("All convolution layouts must have the same length");

  filterNames.reserve(size);
  inputNames.reserve(size);
  if (enableOutput)
    outputNames.reserve(size);

  auto updateOldName = [](StringAttr name) {
    auto *ctx = name.getContext();
    if (name == "y")
      return StringAttr::get(ctx, "0");
    if (name == "x")
      return StringAttr::get(ctx, "1");
    if (name.strref().starts_with_insensitive("h")) {
      auto namestr = name.str();
      return StringAttr::get(ctx, std::string("0") +
                                      namestr.substr(1, namestr.length() - 1));
    }
    if (name.strref().starts_with_insensitive("w")) {
      auto namestr = name.str();
      return StringAttr::get(ctx, std::string("1") +
                                      namestr.substr(1, namestr.length() - 1));
    }
    return name;
  };

  for (unsigned i = 0; i < size; ++i) {
    auto filterAttr =
        updateOldName(cast<StringAttr>(filterLayoutAttr.getValue()[i]));
    auto inputAttr =
        updateOldName(cast<StringAttr>(inputLayoutAttr.getValue()[i]));

    filterNames.push_back(filterAttr.getValue());
    inputNames.push_back(inputAttr.getValue());

    if (enableOutput) {
      auto outputAttr =
          updateOldName(cast<StringAttr>(outputLayoutAttr.getValue()[i]));
      outputNames.push_back(outputAttr.getValue());
    }
  }

  SmallVector<StringRef> filterCheck{"k", "g", "c"};
  SmallVector<StringRef> inputCheck{"ni", "gi", "ci"};
  SmallVector<StringRef> outputCheck{"no", "go", "ko"};
  auto ctx = op.getContext();
  for (size_t i = 0; i < filterNames.size() - 3; i++) {
    filterCheck.push_back(StringAttr::get(ctx, std::to_string(i)));
    inputCheck.push_back(StringAttr::get(ctx, std::to_string(i) + "i"));
    outputCheck.push_back(StringAttr::get(ctx, std::to_string(i) + "o"));
  }

  if (failed(checkNames(filterNames, filterCheck, "filter", op)) ||
      failed(checkNames(inputNames, inputCheck, "input", op)) ||
      (enableOutput &&
       failed(checkNames(outputNames, outputCheck, "output", op)))) {
    return failure();
  }

  return success();
}

/// Return the type of v if the underlying convolution has a result, otherwise
/// return null, allowing the lowering here to be, in principle, generic over
/// tensors and memrefs.
/// Uses the shape from outArg (which carries the GEMM-layout shape after
/// transforms) but the element type from the conv's result. This is necessary
/// when the output fusion chain changes the element type (e.g. arith.fptoui
/// from f32 to i32): the store destination carries the final type, but the
/// GEMM must produce the same element type as the conv.
Type getResultType(Operation *convOp, Value outArg) {
  if (convOp->getNumResults() == 1) {
    auto outArgType = cast<RankedTensorType>(outArg.getType());
    auto convElemType =
        cast<RankedTensorType>(convOp->getResult(0).getType()).getElementType();
    return RankedTensorType::get(outArgType.getShape(), convElemType);
  }
  return nullptr;
}

/// Layout normalization.

/// Make the dimensions that are the values in `mapping` and exist within
/// `toLayout` be in the same relative order as the dimensions that the keys of
/// `mapping` have within `fromLayout`, where both layout are given by the
/// names of the attributes containing them.
///
/// To enable usage in rewrite patterns, returns failure() when no change is
/// made.
static LogicalResult makeToLayoutLikeFromLayoutAlong(
    PatternRewriter &b, Operation *op, StringRef fromLayoutAttrName,
    TypedValue<ShapedType> toArg, StringRef toLayoutAttrName,
    const llvm::StringMap<StringAttr> &mapping) {
  llvm::SmallVector<StringAttr> expectedOrder;
  auto fromLayout = op->getAttrOfType<ArrayAttr>(fromLayoutAttrName);
  auto toLayout = op->getAttrOfType<ArrayAttr>(toLayoutAttrName);
  for (StringRef fromName : fromLayout.getAsValueRange<StringAttr>()) {
    auto maybeCorresponding = mapping.find(fromName);
    if (maybeCorresponding != mapping.end())
      expectedOrder.push_back(maybeCorresponding->getValue());
  }

  llvm::SmallDenseMap<StringAttr, size_t> toLayoutIdxs;
  for (auto pair : llvm::enumerate(toLayout.getAsRange<StringAttr>()))
    toLayoutIdxs.insert({pair.value(), pair.index()});

  bool inOrder = true;
  size_t prevIndex = 0;
  for (StringAttr expected : expectedOrder) {
    auto foundp = toLayoutIdxs.find(expected);
    if (foundp == toLayoutIdxs.end())
      return failure();
    size_t thisIndex = foundp->getSecond();
    if (thisIndex <
        prevIndex) { // the values are not in the relative expected order
      inOrder = false;
      break;
    }
    prevIndex = thisIndex;
  }
  if (inOrder)
    return failure();

  /// And now we have to actually do the thing
  // Is just an attribute to allow array builder
  SmallVector<Attribute> newToLayout;
  llvm::SmallDenseSet<StringAttr> permutedDimsSet{expectedOrder.begin(),
                                                  expectedOrder.end()};

  SmallVector<StringAttr>::const_iterator expectedOrderIter =
      expectedOrder.begin();
  for (StringAttr dim : toLayout.getAsRange<StringAttr>()) {
    if (permutedDimsSet.contains(dim)) {
      newToLayout.push_back(*expectedOrderIter);
      ++expectedOrderIter;
    } else {
      newToLayout.push_back(dim);
    }
  }

  SmallVector<StringRef> oldToLayoutRefs;
  llvm::copy(toLayout.getAsValueRange<StringAttr>(),
             std::back_inserter(oldToLayoutRefs));
  ArrayRef<int64_t> toShape = toArg.getType().getShape();

  BottomUpTMBuilder relayout(b, oldToLayoutRefs, toShape, op->getLoc());
  llvm::StringMap<uint32_t> newToLayoutIdxs;
  for (auto pair : llvm::enumerate(newToLayout)) {
    StringRef value = cast<StringAttr>(pair.value()).getValue();
    newToLayoutIdxs.insert({value, pair.index()});
  }
  BottomUpTMTopDimsWrapper relayoutWrapped(relayout,
                                           std::move(newToLayoutIdxs));

  relayoutWrapped.passThrough(oldToLayoutRefs);
  TransformMapAttr relayoutAttr = relayout.get();

  Value transformed = TransformOp::create(b, op->getLoc(), toArg, relayoutAttr);
  for (OpOperand &operand : op->getOpOperands())
    if (operand.get() == toArg)
      operand.set(transformed);

  op->setAttr(toLayoutAttrName, b.getArrayAttr(newToLayout));
  return success();
}

// Build a mapping from input layout dimension names to filter layout
// dimension names, used for layout regularization.
static llvm::StringMap<StringAttr> buildInputToFilterMapping(PatternRewriter &b,
                                                             int rank) {
  llvm::StringMap<StringAttr> mapping = {{"ci", b.getStringAttr("c")},
                                         {"hi", b.getStringAttr("y")},
                                         {"wi", b.getStringAttr("x")}};
  for (int i = 0; i < rank - 3; i++)
    mapping.insert_or_assign(b.getStringAttr(Twine(i) + "i"),
                             b.getStringAttr(Twine(i)));
  return mapping;
}

// Build a mapping from input layout dimension names to output layout
// dimension names, used for layout regularization.
static llvm::StringMap<StringAttr> buildInputToOutputMapping(PatternRewriter &b,
                                                             int rank) {
  llvm::StringMap<StringAttr> mapping = {{"ni", b.getStringAttr("no")},
                                         {"hi", b.getStringAttr("ho")},
                                         {"wi", b.getStringAttr("wo")}};
  for (int i = 0; i < rank - 3; i++)
    mapping.insert_or_assign(b.getStringAttr(Twine(i) + "i"),
                             b.getStringAttr(Twine(i) + "o"));
  return mapping;
}

// Apply layout regularization to a dest buffer value. Reorders dimensions
// of `destValue` so that the dimensions in `toLayout` that correspond
// (via `mapping`) to dimensions in `fromLayout` appear in the same relative
// order. If reordering is needed, creates a PassThrough transform and
// updates `names` in-place to reflect the new ordering.
// Returns the (possibly transformed) value.
static Value regularizeDestLayout(PatternRewriter &b, Location loc,
                                  ArrayAttr fromLayout, Value destValue,
                                  ArrayAttr toLayout,
                                  const llvm::StringMap<StringAttr> &mapping,
                                  SmallVectorImpl<StringRef> &names) {
  // Determine expected order of mapped dimensions.
  SmallVector<StringAttr> expectedOrder;
  for (StringRef fromName : fromLayout.getAsValueRange<StringAttr>()) {
    auto it = mapping.find(fromName);
    if (it != mapping.end())
      expectedOrder.push_back(it->getValue());
  }

  // Compute new layout by inserting mapped dims in expected order.
  // If already in the correct order this produces an identity PassThrough,
  // which has no effect on indexing arithmetic.
  SmallVector<StringRef> newNames;
  llvm::SmallDenseSet<StringAttr> permutedSet{expectedOrder.begin(),
                                              expectedOrder.end()};
  auto expectedIter = expectedOrder.begin();
  for (StringAttr dim : toLayout.getAsRange<StringAttr>()) {
    if (permutedSet.contains(dim)) {
      newNames.push_back((*expectedIter).getValue());
      ++expectedIter;
    } else {
      newNames.push_back(dim.getValue());
    }
  }

  // Build the PassThrough transform to reorder dimensions.
  SmallVector<StringRef> oldNames;
  llvm::copy(toLayout.getAsValueRange<StringAttr>(),
             std::back_inserter(oldNames));
  ArrayRef<int64_t> shape = cast<ShapedType>(destValue.getType()).getShape();

  BottomUpTMBuilder relayout(b, oldNames, shape, loc);
  llvm::StringMap<uint32_t> newIdxs;
  for (auto [idx, name] : llvm::enumerate(newNames))
    newIdxs.insert({name, idx});
  BottomUpTMTopDimsWrapper relayoutWrapped(relayout, std::move(newIdxs));
  relayoutWrapped.passThrough(oldNames);
  TransformMapAttr relayoutAttr = relayout.get();

  Value transformed = TransformOp::create(b, loc, destValue, relayoutAttr);

  // Update the names in-place to reflect the new ordering.
  names.clear();
  names.append(newNames.begin(), newNames.end());

  return transformed;
}

struct MatchLayoutsToInput final
    : public OpInterfaceRewritePattern<RockConvInterface> {
  using OpInterfaceRewritePattern<RockConvInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(RockConvInterface op,
                                PatternRewriter &b) const override {
    TypedValue<ShapedType> filter = op.getConvFilter();
    TypedValue<ShapedType> output = op.getConvOutput();
    llvm::StringMap<StringAttr> inputToFilter = {{"ci", b.getStringAttr("c")},
                                                 {"hi", b.getStringAttr("y")},
                                                 {"wi", b.getStringAttr("x")}};
    llvm::StringMap<StringAttr> inputToOutput = {{"ni", b.getStringAttr("no")},
                                                 {"hi", b.getStringAttr("ho")},
                                                 {"wi", b.getStringAttr("wo")}};

    for (auto i = 0; i < filter.getType().getRank() - 3; i++) {
      auto key = b.getStringAttr(Twine(i) + Twine("i"));
      inputToFilter.insert_or_assign(key, b.getStringAttr(Twine(i)));
      inputToOutput.insert_or_assign(key,
                                     b.getStringAttr(Twine(i) + Twine("o")));
    }

    // Only re-layout the filter/output if it's an input operand of this op,
    // not if it's the op's own result.
    LogicalResult didReLayoutFilter = failure();
    if (filter.getDefiningOp() != op.getOperation())
      didReLayoutFilter = makeToLayoutLikeFromLayoutAlong(
          b, op, "input_layout", filter, "filter_layout", inputToFilter);
    LogicalResult didReLayoutOutput = failure();
    if (output.getDefiningOp() != op.getOperation())
      didReLayoutOutput = makeToLayoutLikeFromLayoutAlong(
          b, op, "input_layout", output, "output_layout", inputToOutput);
    return success(didReLayoutFilter.succeeded() ||
                   didReLayoutOutput.succeeded());
  }
};

struct MatchFilterToInput final
    : public OpRewritePattern<ConvElementwiseGemmOp> {
  using OpRewritePattern<ConvElementwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvElementwiseGemmOp op,
                                PatternRewriter &b) const override {
    TypedValue<ShapedType> filter = op.getFilter();
    llvm::StringMap<StringAttr> inputToFilter = {{"ci", b.getStringAttr("c")},
                                                 {"hi", b.getStringAttr("y")},
                                                 {"wi", b.getStringAttr("x")}};

    for (auto i = 0; i < filter.getType().getRank() - 3; i++) {
      auto key = b.getStringAttr(Twine(i) + Twine("i"));
      inputToFilter.insert_or_assign(key, b.getStringAttr(Twine(i)));
    }

    LogicalResult didReLayoutFilter = makeToLayoutLikeFromLayoutAlong(
        b, op, "input_layout", filter, "filter_layout", inputToFilter);
    return didReLayoutFilter;
  }
};

/// Lowerings for particular convolution algorithms (TODO, new file?)
FailureOr<std::tuple<Value, Value, Value>>
backwardWeightAtomicAdd(ConvBwdWeightOp op, PatternRewriter &b) {
  Location loc = op.getLoc();

  Attribute tuningParams = op.getParamsAttr();
  if (!tuningParams) {
    return op.emitOpError("can't lower without tuning parameters\n");
  }

  if (!op.getKBlocks().has_value())
    return op.emitOpError("must have kBlocks set at lowering");
  int64_t gemmKBlocks = op.getKBlocks()->getZExtValue();

  ConvolutionContext ctx = populateConvContext(op);

  // Get shape of filter tensor (filter is the result for BwdWeight).
  ShapedType filterType = cast<ShapedType>(op.getResult().getType());
  auto filterShape = filterType.getShape();

  // Get shape of input tensor.
  ShapedType inputType = op.getInput().getType();
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Get shape of gradient tensor (forward output gradient).
  ShapedType gradientType = op.getGradient().getType();
  ArrayRef<int64_t> gradientShape = gradientType.getShape();

  // Obtain convolution parameters: padding / dilation / stride.
  auto pads = ctx.getPaddingVal();
  auto dilations = ctx.getDilationVal();
  auto strides = ctx.getStrideVal();
  ConvolutionDims convDims = ctx.getConvDims();

  llvm::SmallVector<StringRef, 5> filterNames, inputNames, outputNames;
  if (failed(getConvDimNames(op, filterNames, inputNames, outputNames)))
    return failure();

  // Find the StoreOp dest buffer for the filter result.
  auto maybeStores = traceRootOutputToStoreOps(op.getResult());
  if (failed(maybeStores))
    return op.emitOpError("cannot trace bwd_weight result to rock::StoreOp");
  assert(maybeStores->size() == 1 &&
         "bwd_weight has no fusions, expected exactly one store");
  StoreOp firstStore = maybeStores->front();
  Value filterDest = firstStore.getDest();
  ensureInsertionAfterDef(b, op, filterDest);

  // Regularize filter dest layout to match input layout ordering.
  // This must happen before building transforms so that filterNames,
  // filterShape, and filterDest are all consistent.
  {
    int rank = static_cast<int>(filterNames.size());
    auto mapping = buildInputToFilterMapping(b, rank);
    filterDest = regularizeDestLayout(
        b, loc, op->getAttrOfType<ArrayAttr>("input_layout"), filterDest,
        op->getAttrOfType<ArrayAttr>("filter_layout"), mapping, filterNames);
    filterShape = cast<ShapedType>(filterDest.getType()).getShape();
  }

  Value gemmFilter, gemmInput, gemmOutput;
  // Transform filter tensor.
  {
    SmallVector<StringRef, 5> nonKDims;
    for (StringRef name : filterNames)
      if (name != "g" && name != "k")
        nonKDims.push_back(name);
    // Add a dimension, that'll be ignored when writing the output, for KBlock
    // The existence of this dimension makes the mapping between the C matrix
    // and the filter tensor uninvertable, hence the need for atomic add

    llvm::StringMap<uint32_t> kBlockDims =
        expandNamesInPlace(filterNames, {{{"k", {"kBlock", "k"}}}});
    BottomUpTMBuilder addKBlockTransform(b, filterNames, filterShape, loc);
    BottomUpTMTopDimsWrapper addKBlockWrap(addKBlockTransform,
                                           std::move(kBlockDims));
    addKBlockWrap.passThrough("g");
    addKBlockWrap.addDim("kBlock", gemmKBlocks);
    SmallVector<StringRef, 5> throughDims{"k", "c"};
    for (size_t i = 0; i < convDims.fil.size(); i++)
      throughDims.push_back(b.getStringAttr(Twine(i)));
    addKBlockWrap.passThrough(throughDims);

    TransformMapAttr addKBlockTransformAttr = addKBlockTransform.get();
    Value withKBlock =
        rock::TransformOp::create(b, loc, filterDest, addKBlockTransformAttr);

    // Create GEMM filter tensor
    // Here, we merge the KBlock dimension into the G dimension
    // keeping the kBlock dimension as the minor index
    // and send K to the M dimension and CYX to the N dimension as usual
    auto gemmTransform =
        BottomUpTMBuilder::above(addKBlockTransform, addKBlockTransformAttr);
    gemmTransform.merge("gemmG", 0, {"g", "kBlock"});
    gemmTransform.passThrough({"gemmM"}, {1}, {"k"});
    gemmTransform.merge("gemmN", 2, nonKDims);

    TransformMapAttr gemmTransformAttr = gemmTransform.get();
    gemmFilter = TransformOp::create(b, loc, withKBlock, gemmTransformAttr);
    // This kernel is only invoked when there's no need for gemm padding
  }

  // Transform input tensor
  {
    // Pad H and W and split N into  n0 and n1 where n0 has size kBlocks and n1
    // is what's left
    llvm::StringMap<SmallVector<StringRef, 2>> expansions;
    expansions.insert({"ni", {"n0", "n1"}});
    for (size_t i = 0; i < convDims.in.size(); i++) {
      StringAttr key = b.getStringAttr(Twine(i) + "i");
      StringAttr val = b.getStringAttr(Twine(i) + "ipad");
      expansions.insert({key, {val}});
    }
    llvm::StringMap<uint32_t> firstTransformOutDims =
        expandNamesInPlace(inputNames, expansions);

    BottomUpTMBuilder firstTransform(b, inputNames, inputShape, loc);
    BottomUpTMTopDimsWrapper firstWrap(firstTransform,
                                       std::move(firstTransformOutDims));
    firstWrap.passThrough("gi");
    firstWrap.unmerge({"n0", "n1"}, "ni",
                      {gemmKBlocks, convDims.n / gemmKBlocks});
    firstWrap.passThrough("ci");
    SmallVector<StringRef, 3> outs;
    SmallVector<StringRef, 3> ins;
    for (size_t i = 0; i < convDims.in.size(); i++) {
      outs.push_back(b.getStringAttr(Twine(i) + "ipad"));
      ins.push_back(b.getStringAttr(Twine(i) + "i"));
    }
    firstWrap.pad(outs, ins, pads);

    TransformMapAttr firstTransformAttr = firstTransform.get();
    Value firstTransformed =
        TransformOp::create(b, loc, op.getInput(), firstTransformAttr);

    // The usual mapping of input space to dimensions such that filter elements
    // get multiplied by the right thing
    expansions.clear();
    for (size_t i = 0; i < convDims.out.size(); i++) {
      StringAttr key = b.getStringAttr(Twine(i) + "ipad");
      StringAttr val1 = b.getStringAttr(Twine(i));
      StringAttr val2 = b.getStringAttr(Twine(i) + "o");
      expansions.insert({key, {val1, val2}});
    }
    llvm::StringMap<uint32_t> embedOutDims =
        expandNamesInPlace(firstTransform, expansions);
    auto embedTransform =
        BottomUpTMBuilder::above(firstTransform, firstTransformAttr);
    BottomUpTMTopDimsWrapper embedWrap(embedTransform, std::move(embedOutDims));
    embedWrap.passThrough({"gi", "n0", "n1", "ci"});
    assert(convDims.fil.size() == convDims.out.size());
    for (auto [i, filLen] : llvm::enumerate(convDims.fil)) {
      StringAttr val1 = b.getStringAttr(Twine(i));
      StringAttr val2 = b.getStringAttr(Twine(i) + "o");
      StringAttr val3 = b.getStringAttr(Twine(i) + "ipad");
      if (filLen != 1) {
        embedWrap.embed({val1, val2}, {filLen, convDims.out[i]}, val3,
                        {dilations[i], strides[i]});
      } else if (strides[i] != 1) {
        embedWrap.addDim(val1, filLen);
        embedWrap.embed({val2}, {convDims.out[i]}, val3, {strides[i]});
      } else {
        embedWrap.addDim(val1, filLen);
        embedWrap.passThrough(val2, val3);
      }
    }

    TransformMapAttr embedTransformAttr = embedTransform.get();
    Value embedded =
        TransformOp::create(b, loc, firstTransformed, embedTransformAttr);

    // Merge N1HoWO to gemmK and CYX to gemmN
    auto gemmInputTransform =
        BottomUpTMBuilder::above(embedTransform, embedTransformAttr);

    llvm::SmallVector<StringRef, 3> nonNHWDims = {"ci"};
    for (size_t i = 0; i < convDims.in.size(); i++)
      nonNHWDims.push_back(b.getStringAttr(Twine(i)));
    matchUnderlyingOrder(nonNHWDims, gemmInputTransform);
    llvm::SmallVector<StringRef, 3> nhwDims = {"n1"};
    for (size_t i = 0; i < convDims.out.size(); i++)
      nhwDims.push_back(b.getStringAttr(Twine(i) + "o"));
    matchUnderlyingOrder(nhwDims, gemmInputTransform);

    // In the gemmG dimension, unlike with gemmN, we don't have the same
    // traversal order concerns - a step in the G dimension always first visits
    // kBlock/N0 and then moves on to the next G
    gemmInputTransform.merge("gemmG", 0, {"gi", "n0"});
    gemmInputTransform.merge("gemmK", 1, nhwDims);
    gemmInputTransform.merge("gemmN", 2, nonNHWDims);

    TransformMapAttr gemmInputTransformAttr = gemmInputTransform.get();
    gemmInput = TransformOp::create(b, loc, embedded, gemmInputTransformAttr);
  }

  // Transform gradient tensor (forward output gradient)
  {
    // First, split the N dimension as in the input
    llvm::StringMap<uint32_t> outDims =
        expandNamesInPlace(outputNames, {{"no", {"n0", "n1"}}});
    BottomUpTMBuilder firstTransform(b, outputNames, gradientShape, loc);
    BottomUpTMTopDimsWrapper firstWrap(firstTransform, std::move(outDims));
    firstWrap.passThrough("go");
    firstWrap.unmerge({"n0", "n1"}, "no",
                      {gemmKBlocks, convDims.n / gemmKBlocks});
    SmallVector<StringRef, 3> names{"ko"};
    for (size_t i = 0; i < convDims.out.size(); i++)
      names.push_back(b.getStringAttr(Twine(i) + "o"));
    firstWrap.passThrough(names);

    TransformMapAttr firstTransformAttr = firstTransform.get();
    Value transformed =
        TransformOp::create(b, loc, op.getGradient(), firstTransformAttr);

    // Map G and N0 to gemmG, N1HW to gemmK and K to gemmM
    auto gemmOutputTransform =
        BottomUpTMBuilder::above(firstTransform, firstTransformAttr);
    llvm::SmallVector<StringRef, 3> nhwDims = {"n1"};
    for (size_t i = 0; i < convDims.out.size(); i++)
      nhwDims.push_back(b.getStringAttr(Twine(i) + "o"));
    matchUnderlyingOrder(nhwDims, gemmOutputTransform);
    gemmOutputTransform.merge("gemmG", 0, {"go", "n0"});
    gemmOutputTransform.merge("gemmK", 1, nhwDims);
    gemmOutputTransform.passThrough({"gemmM"}, {2}, {"ko"});

    TransformMapAttr gemmOutputTransformAttr = gemmOutputTransform.get();
    gemmOutput =
        TransformOp::create(b, loc, transformed, gemmOutputTransformAttr);
  }

  // This kernel is not run when there is padding on the GEMM
  auto storeMethod = b.getAttr<StoreMethodAttr>(StoreMethod::AtomicAdd);
  auto gemm = GemmOp::create(
      b, loc, getResultType(op, gemmFilter), gemmOutput, gemmInput,
      /*scaleA=*/nullptr, /*scaleB=*/nullptr,
      /*aTransposed=*/b.getUnitAttr(), /*bTransposed=*/nullptr,
      /*cTransposed=*/nullptr,
      /*aScaleTransposed=*/nullptr, /*bScaleTransposed=*/nullptr,
      /*quantBlockSize=*/nullptr, op.getParamsAttr());

  // Update any StoreOp that uses the conv result to use the gemm result.
  updateStoreOpForGemm(b, loc, op.getResult(), gemm.getResult(), gemmFilter,
                       storeMethod);

  // Finally, erase the original Conv op.
  b.eraseOp(op);

  return std::make_tuple(Value(), Value(), Value());
}

FailureOr<std::pair<Value, Value>>
backwardDataGemmForKernelId(ConvBwdDataOp op, PatternRewriter &b,
                            int64_t kernelId, Value destBuffer) {
  Location loc = op.getLoc();

  ConvolutionContext ctx = populateConvContext(op);

  // Get shape of filter tensor.
  ShapedType filterType = op.getFilter().getType();
  ArrayRef<int64_t> filterShape = filterType.getShape();

  // Get shape of result tensor (gradient w.r.t. input, same shape as fwd
  // input).
  ShapedType resultType = cast<ShapedType>(op.getResult().getType());
  ArrayRef<int64_t> resultShape = resultType.getShape();

  // Get shape of gradient tensor (from forward output).
  ShapedType gradientType = op.getGradient().getType();
  ArrayRef<int64_t> gradientShape = gradientType.getShape();

  // Obtain convolution parameters: padding / dilation / stride.
  auto pads = ctx.getPaddingVal();
  auto dilations = ctx.getDilationVal();
  auto strides = ctx.getStrideVal();
  ConvolutionDims convDims = ctx.getConvDims();
  SmallVector<StringRef, 5> filterNames, inputNames, outputNames;
  if (failed(getConvDimNames(op, filterNames, inputNames, outputNames))) {
    return failure();
  }

  SmallVector<int64_t, 5> gcdStrideDilations;
  assert(strides.size() == dilations.size());
  for (const auto &[stride, dilation] : zip(strides, dilations)) {
    gcdStrideDilations.push_back(std::gcd(stride, dilation));
  }

  SmallVector<int64_t, 5> filTilda;
  for (const auto &[stride, gcdSD] : zip(strides, gcdStrideDilations)) {
    filTilda.push_back(stride / gcdSD);
  }

  SmallVector<int64_t, 5> filDots;
  for (const auto &[fil, tilda] : zip(convDims.fil, filTilda)) {
    filDots.push_back(llvm::divideCeil(fil, tilda));
  }

  SmallVector<int64_t, 5> outTilda;
  for (const auto &[out, dilation, fil, stride] :
       zip(convDims.out, dilations, convDims.fil, strides)) {
    outTilda.push_back(out + llvm::divideCeil(dilation * (fil - 1), stride));
  }

  SmallVector<int64_t, 5> iTildaLeft;
  SmallVector<int64_t, 5> iTildaRight;
  for (const auto &[padindex, dilation, tilda, stride] :
       enumerate(dilations, filTilda, strides)) {
    iTildaLeft.push_back(
        std::max((int64_t)0, pads[2 * padindex] - dilation * (tilda - 1)) /
        stride);
  }
  for (const auto &[padindex, out, in, stride] :
       enumerate(outTilda, convDims.in, strides)) {
    iTildaRight.push_back(
        std::min(out, static_cast<int64_t>(llvm::divideCeil(
                          pads[2 * padindex] + in - 1, stride)) +
                          1));
  }

  // i2tilda = kernelid % filtilda[2]
  // i1tilda = (kernelid % (filtilda[2] * filtilda[1])) / filtilda[2]
  // i0tilda = kernelid / (filtilda[2] * filtilda[1])
  //  get-backward-kernel-count or similar

  SmallVector<int64_t, 3> iTilda;
  SmallVector<int64_t, 3> iDotSlice;
  int64_t product = 1;
  for (size_t i = 1; i < convDims.fil.size(); i++)
    product *= filTilda[i];
  int64_t divisor = 1;
  iTilda.resize(convDims.fil.size());
  switch (convDims.fil.size()) {
  default:
    llvm_unreachable("Only 2-D and 3-D have been implemented.");
    break;
  case 3:
    divisor = filTilda[2];
    iTilda[2] = kernelId % divisor;
    [[fallthrough]];
  case 2:
    iTilda[1] = (kernelId % product) / divisor;
    iTilda[0] = kernelId / product;
  }

  // `kernelId` must come from `backwardDataKernelIds`, which filters out
  // phases where `iTilda[i] >= convDims.fil[i]`. Without that filter,
  // `divideCeil`'s unsigned-converting overload would wrap a negative
  // numerator into a huge value here.
  for (size_t i = 0; i < convDims.fil.size(); i++) {
    assert(iTilda[i] < convDims.fil[i] &&
           "kernelId not pre-filtered by backwardDataKernelIds");
    iDotSlice.push_back(
        llvm::divideCeil(convDims.fil[i] - iTilda[i], filTilda[i]));
  }

  // backward data only, compute iTilda indices for multi-gemm decomposition
  // c is input channels , k is output channels
  // n is batch , yDotSlice,xDotSlice computed in above

  Value gemmFilter, gemmInput, gemmOutput;
  // Transform filter tensor.
  {
    // Embed y/x into {y/x}dot and {y/x}tilda (Why the
    // particular embed coefficients is in a presentation somewhere)
    llvm::StringMap<SmallVector<StringRef, 2>> expansions;
    for (size_t i = 0; i < convDims.fil.size(); i++) {
      StringAttr key = b.getStringAttr(Twine(i));
      StringAttr val1 = b.getStringAttr(Twine(i) + "dot");
      StringAttr val2 = b.getStringAttr(Twine(i) + "tilda");
      expansions.insert({key, {val1, val2}});
    }
    llvm::StringMap<uint32_t> embedDims =
        expandNamesInPlace(filterNames, expansions);
    BottomUpTMBuilder embedTransform(b, filterNames, filterShape, loc);
    BottomUpTMTopDimsWrapper embedWrap(embedTransform, std::move(embedDims));
    // array of smallstring?

    embedWrap.passThrough({"g", "k", "c"});
    for (size_t i = 0; i < convDims.fil.size(); i++) {
      StringAttr upper1 = b.getStringAttr(Twine(i) + "dot");
      StringAttr upper2 = b.getStringAttr(Twine(i) + "tilda");
      StringAttr lower = b.getStringAttr(Twine(i));
      embedWrap.embed({upper1, upper2}, {filDots[i], filTilda[i]}, lower,
                      {strides[i] / gcdStrideDilations[i], 1});
    }

    TransformMapAttr embedTransformAttr = embedTransform.get();
    Value embeddedFilter =
        TransformOp::create(b, loc, op.getFilter(), embedTransformAttr);

    // Take slices in the ydot, ytilda, xdot, and xtilda dimensions
    // to reflect which kernel we're performing
    auto sliceTransform =
        BottomUpTMBuilder::above(embedTransform, embedTransformAttr);
    sliceTransform.passThrough({"g", "k", "c"});
    llvm::SmallVector<StringRef, 2> uppers;
    llvm::SmallVector<StringRef, 2> lowers;
    llvm::SmallVector<int64_t, 2> begins;
    for (size_t i = 0; i < convDims.in.size(); i++) {
      uppers.push_back(b.getStringAttr(Twine(i) + "dotslice"));
      lowers.push_back(b.getStringAttr(Twine(i) + "dot"));
      begins.push_back(0);
    }
    sliceTransform.slice(uppers, lowers, begins, iDotSlice);
    uppers.clear();
    lowers.clear();
    for (size_t i = 0; i < convDims.fil.size(); i++) {
      uppers.push_back(b.getStringAttr(Twine(i) + "tildaslice"));
      lowers.push_back(b.getStringAttr(Twine(i) + "tilda"));
    }
    llvm::SmallVector<int64_t, 3> iTildasPlusOne;
    for (size_t i = 0; i < convDims.fil.size(); i++)
      iTildasPlusOne.push_back(iTilda[i] + 1);
    sliceTransform.slice(uppers, lowers, iTilda, iTildasPlusOne);

    TransformMapAttr sliceTransformAttr = sliceTransform.get();
    Value slicedFilter =
        TransformOp::create(b, loc, embeddedFilter, sliceTransformAttr);

    // Set up gemm by passing g -> gemmG, merging
    // [k, ydotslice, xdotslice] to gemmK, and [c, ytildaslice, xtildaslice]
    // to gemmM
    auto gemmFilterTransform =
        BottomUpTMBuilder::above(sliceTransform, sliceTransformAttr);
    gemmFilterTransform.passThrough({"gemmG"}, {0}, {"g"});
    lowers.clear();
    lowers.push_back("k");
    for (size_t i = 0; i < convDims.fil.size(); i++)
      lowers.push_back(b.getStringAttr(Twine(i) + "dotslice"));
    gemmFilterTransform.merge("gemmK", 1, lowers);
    lowers.clear();
    lowers.push_back("c");
    for (size_t i = 0; i < convDims.fil.size(); i++)
      lowers.push_back(b.getStringAttr(Twine(i) + "tildaslice"));
    gemmFilterTransform.merge("gemmM", 2, lowers);

    TransformMapAttr gemmFilterTransformAttr = gemmFilterTransform.get();
    gemmFilter =
        TransformOp::create(b, loc, slicedFilter, gemmFilterTransformAttr);
  }

  // Transform destination buffer (where gradient w.r.t. input is stored).
  // The dest buffer comes from the StoreOp and has the forward input shape.
  {
    BottomUpTMBuilder padInputTransform(b, inputNames, resultShape, loc);
    padInputTransform.passThrough({"gi", "ni", "ci"});

    llvm::SmallVector<uint32_t, 2> padDims;
    llvm::SmallVector<StringRef, 2> outs;
    llvm::SmallVector<StringRef, 2> ins;
    for (size_t i = 0; i < convDims.in.size(); i++) {
      padDims.push_back(padInputTransform.startIndex(std::to_string(i) + "i"));
      outs.push_back(b.getStringAttr(Twine(i) + "ipad"));
      ins.push_back(b.getStringAttr(Twine(i) + "i"));
    }
    padInputTransform.pad(outs, padDims, ins, pads);

    TransformMapAttr padTransformAttr = padInputTransform.get();
    Value paddedInput =
        TransformOp::create(b, loc, destBuffer, padTransformAttr);

    // Split 0ipad, 1ipad into 0ftilda, 0itilda, 1ftilda, 1itilda
    llvm::StringMap<SmallVector<StringRef, 2>> expansions;
    for (size_t i = 0; i < convDims.in.size(); i++) {
      StringAttr key = b.getStringAttr(Twine(i) + "ipad");
      StringAttr val1 = b.getStringAttr(Twine(i) + "ftilda");
      StringAttr val2 = b.getStringAttr(Twine(i) + "itilda");
      expansions.insert({key, {val1, val2}});
    }
    llvm::StringMap<uint32_t> embedDims =
        expandNamesInPlace(padInputTransform, expansions);
    auto tildaEmbedTransform =
        BottomUpTMBuilder::above(padInputTransform, padTransformAttr);
    BottomUpTMTopDimsWrapper tildaEmbedWrap(tildaEmbedTransform,
                                            std::move(embedDims));
    tildaEmbedWrap.passThrough({"gi", "ni", "ci"});
    for (size_t i = 0; i < convDims.fil.size(); i++) {
      StringAttr upper1 = b.getStringAttr(Twine(i) + "ftilda");
      StringAttr upper2 = b.getStringAttr(Twine(i) + "itilda");
      StringAttr lower = b.getStringAttr(Twine(i) + "ipad");
      tildaEmbedWrap.embed({upper1, upper2}, {filTilda[i], outTilda[i]}, lower,
                           {dilations[i], strides[i]});
    }

    TransformMapAttr tildaEmbedTransformAttr = tildaEmbedTransform.get();
    Value tildaEmbedded =
        TransformOp::create(b, loc, paddedInput, tildaEmbedTransformAttr);

    // Slice all the tilda dimensions: ytilda and xtilda get slices of length
    // 1 while htilda and wtilda have slice indices computed above
    auto sliceTransform =
        BottomUpTMBuilder::above(tildaEmbedTransform, tildaEmbedTransformAttr);
    sliceTransform.passThrough({"gi", "ni", "ci"});
    llvm::SmallVector<StringRef, 2> uppers;
    llvm::SmallVector<StringRef, 2> lowers;
    for (size_t i = 0; i < convDims.in.size(); i++) {
      uppers.push_back(b.getStringAttr(Twine(i) + "slice"));
      lowers.push_back(b.getStringAttr(Twine(i) + "ftilda"));
    }
    llvm::SmallVector<int64_t, 3> iTildasPlusOne;
    for (size_t i = 0; i < convDims.fil.size(); i++)
      iTildasPlusOne.push_back(iTilda[i] + 1);
    sliceTransform.slice(uppers, lowers, iTilda, iTildasPlusOne);
    uppers.clear();
    lowers.clear();
    for (size_t i = 0; i < convDims.fil.size(); i++) {
      uppers.push_back(b.getStringAttr(Twine(i) + "islice"));
      lowers.push_back(b.getStringAttr(Twine(i) + "itilda"));
    }
    sliceTransform.slice(uppers, lowers, iTildaLeft, iTildaRight);

    TransformMapAttr sliceTransformAttr = sliceTransform.get();
    Value sliced =
        TransformOp::create(b, loc, tildaEmbedded, sliceTransformAttr);

    // C plus the length 1 slices (yslice and xslice) become the gemmM
    // dimension G, N, and the h and w slices become gemmN
    auto gemmTransform =
        BottomUpTMBuilder::above(sliceTransform, sliceTransformAttr);
    gemmTransform.passThrough({"gemmG"}, {0}, {"gi"});
    lowers.clear();
    lowers.push_back("ci");
    for (size_t i = 0; i < convDims.fil.size(); i++)
      lowers.push_back(b.getStringAttr(Twine(i) + "slice"));
    gemmTransform.merge("gemmM", 1, lowers);
    lowers.clear();
    lowers.push_back("ni");
    for (size_t i = 0; i < convDims.fil.size(); i++)
      lowers.push_back(b.getStringAttr(Twine(i) + "islice"));
    gemmTransform.merge("gemmN", 2, lowers);

    TransformMapAttr gemmTransformAttr = gemmTransform.get();
    gemmInput = TransformOp::create(b, loc, sliced, gemmTransformAttr);
  }

  // Transform gradient tensor (forward output gradient, B-matrix for GEMM)
  {
    // Embed 0o to 0dot and 0tilda and 1o to 1dot and 1tilda
    llvm::StringMap<SmallVector<StringRef, 2>> expansions;
    for (size_t i = 0; i < convDims.out.size(); i++) {
      StringAttr key = b.getStringAttr(Twine(i) + "o");
      StringAttr val1 = b.getStringAttr(Twine(i) + "dot");
      StringAttr val2 = b.getStringAttr(Twine(i) + "tilda");
      expansions.insert({key, {val1, val2}});
    }
    llvm::StringMap<uint32_t> embedDims =
        expandNamesInPlace(outputNames, expansions);
    BottomUpTMBuilder embedTransform(b, outputNames, gradientShape, loc);
    BottomUpTMTopDimsWrapper embedWrap(embedTransform, std::move(embedDims));
    embedWrap.passThrough({"go", "no", "ko"});
    for (size_t i = 0; i < convDims.fil.size(); i++) {
      StringAttr upper1 = b.getStringAttr(Twine(i) + "dot");
      StringAttr upper2 = b.getStringAttr(Twine(i) + "tilda");
      StringAttr lower = b.getStringAttr(Twine(i) + "o");
      embedWrap.embed({upper1, upper2}, {filDots[i], outTilda[i]}, lower,
                      {(-dilations[i]) / gcdStrideDilations[i], 1});
    }

    TransformMapAttr embedTransformAttr = embedTransform.get();
    Value embedded =
        TransformOp::create(b, loc, op.getGradient(), embedTransformAttr);

    // Take the same slices in ydot, xdot, 0tilda, and 1tilda as were taken in
    // the filter and input
    auto sliceTransform =
        BottomUpTMBuilder::above(embedTransform, embedTransformAttr);
    sliceTransform.passThrough({"go", "no", "ko"});
    llvm::SmallVector<StringRef, 2> uppers;
    llvm::SmallVector<StringRef, 2> lowers;
    llvm::SmallVector<int64_t, 2> begins;
    for (size_t i = 0; i < convDims.out.size(); i++) {
      uppers.push_back(b.getStringAttr(Twine(i) + "slice"));
      lowers.push_back(b.getStringAttr(Twine(i) + "dot"));
      begins.push_back(0);
    }
    sliceTransform.slice(uppers, lowers, begins, iDotSlice);
    lowers.clear();
    uppers.clear();
    for (size_t i = 0; i < convDims.out.size(); i++) {
      uppers.push_back(b.getStringAttr(Twine(i) + "islice"));
      lowers.push_back(b.getStringAttr(Twine(i) + "tilda"));
    }
    sliceTransform.slice(uppers, lowers, iTildaLeft, iTildaRight);

    TransformMapAttr sliceTransformAttr = sliceTransform.get();
    Value sliced = TransformOp::create(b, loc, embedded, sliceTransformAttr);

    // Merge k, yslice, and xslice to gemmK and n, hslice, and wslice to gemmN
    auto gemmOutputTransform =
        BottomUpTMBuilder::above(sliceTransform, sliceTransformAttr);
    gemmOutputTransform.passThrough({"gemmG"}, {0}, {"go"});
    lowers.clear();
    lowers.push_back("ko");
    for (size_t i = 0; i < convDims.out.size(); i++)
      lowers.push_back(b.getStringAttr(Twine(i) + "slice"));
    gemmOutputTransform.merge("gemmK", 1, lowers);
    lowers.clear();
    lowers.push_back("no");
    for (size_t i = 0; i < convDims.out.size(); i++)
      lowers.push_back(b.getStringAttr(Twine(i) + "islice"));
    gemmOutputTransform.merge("gemmN", 2, lowers);

    TransformMapAttr gemmOutputTransformAttr = gemmOutputTransform.get();
    gemmOutput = TransformOp::create(b, loc, sliced, gemmOutputTransformAttr);
  }

  // Emit rock.gemm op.
  auto gemm = GemmOp::create(
      b, loc, getResultType(op, gemmInput), gemmFilter, gemmOutput,
      /*scaleA=*/nullptr, /*scaleB=*/nullptr,
      /*aTransposed=*/b.getUnitAttr(), /*bTransposed=*/nullptr,
      /*cTransposed=*/nullptr,
      /*aScaleTransposed=*/nullptr, /*bScaleTransposed=*/nullptr,
      /*quantBlockSize=*/nullptr, op.getParamsAttr());

  return std::pair<Value, Value>(gemm.getResult(), gemmInput);
}

// Lower a conv op (forward, backward-weight, or backward-data) into one or
// more rock.gemm ops by computing the necessary layout transforms for the
// filter, input, and output tensors. For BwdData, the conv is expanded into
// multiple gemms (one per kernel ID) inline and the function returns early.
// For Fwd and BwdWeight, it builds gemm-compatible views (gemmG/gemmM/gemmK/
// gemmN) of the three conv operands and returns them as (filter, input,
// output). When Fwd, `outputViews` and `fusionInputMap` are also transformed
// to carry the output layout so the caller can wire up fusion stores.
template <typename T>
static FailureOr<std::tuple<Value, Value, Value>>
commonConvRewrite(T op, PatternRewriter &b, ConvolutionContext &ctx,
                  ConvOpType convOpType, DenseMap<Value, Value> &fusionInputMap,
                  SmallVector<Value> &outputViews) {
  // Type dataType = op.getInput().getType().getElementType();
  if (ConvOpType::BwdData == convOpType) {
    auto bwdDataOp = cast<ConvBwdDataOp>(op);
    Location loc = bwdDataOp.getLoc();
    // Single function, expand all kernel IDs into
    // multiple gemms within this function.
    auto strideDims = ctx.getStrideVal();
    auto dilationDims = ctx.getDilationVal();
    auto filterDims = ctx.getConvDims().fil;
    auto kernelIds =
        rock::backwardDataKernelIds(strideDims, dilationDims, filterDims);

    // ConvBwdData rewrite currently expects a single consuming rock.store.
    auto maybeStores = traceRootOutputToStoreOps(bwdDataOp.getResult());
    if (failed(maybeStores))
      return bwdDataOp.emitOpError(
          "cannot trace bwd_data result to rock::StoreOp");
    assert(maybeStores->size() == 1 &&
           "bwd_data has no fusions, expected exactly one store");
    StoreOp originalStoreOp = maybeStores->front();

    Type storeResultType = originalStoreOp.getResult().getType();
    auto storeMethod = originalStoreOp.getStoreMethodAttr();

    // Create a gemm + store pair for each kernel ID.
    // Note: no layout regularization is needed for the BwdData dest buffer
    // because it represents the input gradient, and the input layout is the
    // reference layout that everything else is regularized against.
    Value destBuffer = originalStoreOp.getDest();
    ensureInsertionAfterDef(b, bwdDataOp, destBuffer);

    // Thread only the store result alias through each per-kernel store so the
    // single returned tensor represents all disjoint bwd_data phase writes. The
    // actual destination view stays rooted at the original destination buffer.
    // If the original store had no explicit alias, the first generated store
    // starts the SSA chain and later stores alias the previous store result.
    Value currentResultAlias = originalStoreOp.getResultAlias();
    Value finalStoreResult;
    for (auto [idx, kid] : llvm::enumerate(kernelIds)) {
      auto maybe = backwardDataGemmForKernelId(bwdDataOp, b, kid, destBuffer);
      if (failed(maybe))
        return failure();

      auto [gemmResult, gemmDest] = maybe.value();
      auto newStoreOp =
          StoreOp::create(b, loc, storeResultType, gemmResult, gemmDest,
                          currentResultAlias, storeMethod);
      finalStoreResult = newStoreOp.getResult();
      if (idx + 1 < kernelIds.size()) {
        currentResultAlias = finalStoreResult;
      }
    }

    // BwdData with multiple kernel IDs emits N independent gemm + store pairs,
    // each writing a disjoint slice of the same output buffer. Since
    // `rock.store` is Pure, each store result must be live; threading later
    // stores through the previous store result makes the final result represent
    // the full logical output without exposing per-phase stores in the function
    // ABI.
    b.replaceOp(originalStoreOp, finalStoreResult);
    b.eraseOp(bwdDataOp);

    return std::make_tuple(Value(), Value(), Value());
  }
  Location loc = op.getLoc();

  // Determine the filter and input tensor values based on op type.
  // Each op variant has different operand names:
  //   ConvOp:                $filter, $input
  //   ConvBwdWeightOp:       $input, $gradient
  //   ConvElementwiseGemmOp: $filter, $input
  // For BwdWeight, the filter is the result (dest comes from StoreOp) and
  // the gradient is the "output" in convolution terms.
  Value filterValue, inputValue;
  if constexpr (std::is_same_v<T, ConvBwdDataOp>) {
    llvm_unreachable("BwdData should have been handled above");
  } else if constexpr (std::is_same_v<T, ConvElementwiseGemmOp>) {
    filterValue = op.getFilter();
    inputValue = op.getInput();
  } else {
    filterValue = op.getConvFilter();
    inputValue = op.getConvInput();
  }

  // Get shapes — for BwdWeight, filter shape comes from the result type.
  ArrayRef<int64_t> filterShape;
  if constexpr (std::is_same_v<T, ConvBwdWeightOp>)
    filterShape = cast<ShapedType>(op.getResult().getType()).getShape();
  else
    filterShape = cast<ShapedType>(filterValue.getType()).getShape();
  ArrayRef<int64_t> inputShape =
      cast<ShapedType>(inputValue.getType()).getShape();

  Type dataType = cast<ShapedType>(inputValue.getType()).getElementType();

  // Obtain convolution parameters: padding / dilation / stride.
  auto dilations = ctx.getDilationVal();
  auto strides = ctx.getStrideVal();
  ConvolutionDims convDims = ctx.getConvDims();
  const bool notConvGemm = !std::is_same_v<T, ConvElementwiseGemmOp>;

  llvm::SmallVector<StringRef, 5> filterNames, inputNames, outputNames;
  if (failed(getConvDimNames(op, filterNames, inputNames, outputNames,
                             notConvGemm))) {
    return failure();
  }

  // For non-ConvElementwiseGemmOp, find the StoreOp dest to get the output
  // buffer
  Value destBuffer;
  if constexpr (notConvGemm) {
    if constexpr (std::is_same_v<T, ConvBwdWeightOp>)
      destBuffer = op.getConvOutput();
    else
      destBuffer = outputViews[0];

    auto tuningParams = op.getParamsAttr();
    GemmSize gemmSize = op.getGemmSize();
    std::optional<GemmSize> maybeGemmExtraPad;

    if (tuningParams) {
      maybeGemmExtraPad = requiredPadding(tuningParams, gemmSize);
    } else {
      // We don't know if this'll be a padding kernel, so we can't promise a
      // merge or rely on atomic add, and so set the extraPad to a nonsense but
      // existing value.
      maybeGemmExtraPad = GemmSize{-1, -1, -1, -1};
    }

    if (ConvOpType::BwdWeight == convOpType &&
        isWrWAtomicKernel(dataType, maybeGemmExtraPad.has_value())) {
      return backwardWeightAtomicAdd(cast<ConvBwdWeightOp>(op), b);
    }
  }

  // Apply layout regularization to the dest buffer for result tensors.
  // MatchLayoutsToInput handles operand regularization; for result values
  // (filter for BwdWeight, output for ConvOp) we regularize the StoreOp
  // dest buffer here.
  if constexpr (notConvGemm) {
    int rank = static_cast<int>(filterNames.size());
    if constexpr (std::is_same_v<T, ConvBwdWeightOp>) {
      // BwdWeight: the regularized filterValue feeds into filter transforms
      // that become gemm operands, so keep the insertion point moved.
      // Using `outputViews[0]` instead of `destBuffer`: for BwdWeight 
      // `destBuffer` is the output-gradient operand which may be
      // defined before `outputViews[0]`. Using `destBuffer` could
      // therefore place the new TransformOp before its operand's defining
      // op and break SSA dominance.
      ensureInsertionAfterDef(b, op, outputViews[0]);
      auto mapping = buildInputToFilterMapping(b, rank);
      filterValue = regularizeDestLayout(
          b, loc, op->template getAttrOfType<ArrayAttr>("input_layout"),
          outputViews[0],
          op->template getAttrOfType<ArrayAttr>("filter_layout"), mapping,
          filterNames);
      filterShape = cast<ShapedType>(filterValue.getType()).getShape();
    } else {
      // ConvOp: regularize at the dest buffer's location, then restore the
      // insertion point to the conv so filter/input transforms and the gemm
      // are placed there. The gemm doesn't take gemmOutput as an SSA operand.
      TransformMapAttr relayoutAttr;
      {
        OpBuilder::InsertionGuard guard(b);
        ensureInsertionAfterDef(b, op, destBuffer);
        auto mapping = buildInputToOutputMapping(b, rank);
        destBuffer = regularizeDestLayout(
            b, loc, op->template getAttrOfType<ArrayAttr>("input_layout"),
            destBuffer, op->template getAttrOfType<ArrayAttr>("output_layout"),
            mapping, outputNames);
        relayoutAttr =
            cast<TransformOp>(destBuffer.getDefiningOp()).getTransform();
      }
      // Apply the relayout to each output view and fusion extra input,
      // placing each transform right after its input's defining op so it
      // dominates all consumers (fusion ops, stores).
      auto applyRelayout = [&](Value &v) {
        if (Operation *defOp = v.getDefiningOp()) {
          OpBuilder::InsertionGuard vg(b);
          b.setInsertionPointAfter(defOp);
          v = TransformOp::create(b, loc, v, relayoutAttr);
        } else {
          v = TransformOp::create(b, loc, v, relayoutAttr);
        }
      };
      for (auto &view : outputViews)
        applyRelayout(view);
      for (auto &[orig, view] : fusionInputMap)
        applyRelayout(view);
    }
  }

  // Transform filter tensor.

  // set layout attribute.
  // Weight tensor transformation for ConvOp
  // - PassThrough G dimension to dimension 0, name it gemmG.
  // - Merge non-K dimensions to dimension 1, name it as gemmK.
  //   Optimization: If non-K dimensions are consecutive, apply merge.
  // - PassThrough K dimension to dimension 2, name it as gemmM.
  //
  // Weight tensor transformation for ConvBwdWeightOp
  // - PassThrough G dimension to dimension 0, name it gemmG
  // - PassThrough K dimension to dimension 1, name it as gemmM.
  // - Merge non-K dimensions to dimension 2, name it as gemmN.
  SmallVector<StringRef, 5> filterNonKDims;
  for (StringRef name : filterNames)
    if (name != "g" && name != "k")
      filterNonKDims.push_back(name);

  BottomUpTMBuilder filterTransform(b, filterNames, filterShape, loc);
  filterTransform.passThrough({"gemmG"}, {0}, {"g"});
  switch (convOpType) {
  case ConvOpType::Fwd:
    filterTransform.merge("gemmK", 1, filterNonKDims);
    filterTransform.passThrough({"gemmM"}, {2}, {"k"});
    break;
  case ConvOpType::BwdWeight:
    filterTransform.passThrough({"gemmM"}, {1}, {"k"});
    filterTransform.merge("gemmN", 2, filterNonKDims);
    break;
  case ConvOpType::BwdData:
    llvm_unreachable("Backward data has been sent elsewhere");
    break;
  }

  TransformMapAttr filterTransformAttr = filterTransform.get();
  Value gemmFilter =
      TransformOp::create(b, loc, filterValue, filterTransformAttr);

  // Transform input tensor.
  // Input tensor step 1: padded input.

  // set layout attribute.
  // Padded input tensor transformation:
  // - Pass through ni, gi, and ci, not renaming them
  // - Pad hi and wi as specified in padding attributes, renaming them to
  // 0ipad and 1ipad
  BottomUpTMBuilder padInputTransform(b, inputNames, inputShape, loc);
  padInputTransform.passThrough("ni");
  padInputTransform.passThrough("gi");
  padInputTransform.passThrough("ci");

  llvm::SmallVector<uint32_t, 2> padOutDims;
  llvm::SmallVector<StringRef, 2> outs;
  llvm::SmallVector<StringRef, 2> ins;
  for (size_t i = 0; i < convDims.in.size(); i++) {
    padOutDims.push_back(padInputTransform.startIndex(std::to_string(i) + "i"));
    outs.push_back(b.getStringAttr(Twine(i) + "ipad"));
    ins.push_back(b.getStringAttr(Twine(i) + "i"));
  }
  padInputTransform.pad(outs, padOutDims, ins, ctx.getPaddingVal());

  TransformMapAttr padInputTransformAttr = padInputTransform.get();

  Value paddedInput =
      TransformOp::create(b, loc, inputValue, padInputTransformAttr);

  // Input tensor step 2 : embedded input.
  // Embedded input tensor transformation:
  // - PassThrough gi, ni, and ci
  // - Embed 0ipad to y and ho with size filter y by output h and
  //   coefficients dilations[0] and strides[0]
  // - Embed 1ipad to x and wo with size filter x by output h and
  //   coefficients dilations[1] and strides[1]

  llvm::StringMap<SmallVector<StringRef, 2>> expansions;
  for (size_t i = 0; i < convDims.in.size(); i++) {
    StringAttr key = b.getStringAttr(Twine(i) + "ipad");
    StringAttr val1 = b.getStringAttr(Twine(i));
    StringAttr val2 = b.getStringAttr(Twine(i) + "o");
    expansions.insert({key, {val1, val2}});
  }
  llvm::StringMap<uint32_t> embeddedInputDims =
      expandNamesInPlace(padInputTransform, expansions);

  BottomUpTMBuilder embedInputTransform =
      BottomUpTMBuilder::above(padInputTransform, padInputTransformAttr);
  BottomUpTMTopDimsWrapper embedInputWrap(embedInputTransform,
                                          std::move(embeddedInputDims));
  embedInputWrap.passThrough({"ni", "gi", "ci"});
  assert(convDims.fil.size() == convDims.out.size());
  for (auto [i, filLen] : llvm::enumerate(convDims.fil)) {
    StringAttr val1 = b.getStringAttr(Twine(i));
    StringAttr val2 = b.getStringAttr(Twine(i) + "o");
    StringAttr val3 = b.getStringAttr(Twine(i) + "ipad");
    if (filLen != 1) {
      embedInputWrap.embed({val1, val2}, {filLen, convDims.out[i]}, val3,
                           {dilations[i], strides[i]});
    } else if (strides[i] != 1) {
      embedInputWrap.addDim(val1, filLen);
      embedInputWrap.embed({val2}, {convDims.out[i]}, val3, {strides[i]});
    } else {
      embedInputWrap.addDim(val1, filLen);
      embedInputWrap.passThrough(val2, val3);
    }
  }

  TransformMapAttr embedInputTransformAttr = embedInputTransform.get();
  Value embeddedInput =
      TransformOp::create(b, loc, paddedInput, embedInputTransformAttr);

  // Input tensor step 3: GEMM'd input
  //
  // - PassThrough gi to dimension 0 and name it gemmG, then
  // For ConvOp:
  // - Merge ci, y, x dimensions to dimension 1, name it as gemmK.
  // - Merge ni, ho, wo dimensions to dimension 2, name it as gemmN.
  //
  // For ConvBwdWeightOp:
  // - Part 1: Merge ni, ho, wo dimensions to dimension 1, name it as gemmK.
  // - Part 2: Merge ci, y, x dimensions to dimension 2, name it as gemmN.

  auto gemmInputTransform =
      BottomUpTMBuilder::above(embedInputTransform, embedInputTransformAttr);
  gemmInputTransform.passThrough({"gemmG"}, {0}, {"gi"});

  llvm::SmallVector<StringRef, 3> nonNHWDims = {"ci"};
  for (size_t i = 0; i < convDims.in.size(); i++)
    nonNHWDims.push_back(b.getStringAttr(Twine(i)));
  matchUnderlyingOrder(nonNHWDims, gemmInputTransform);
  llvm::SmallVector<StringRef, 3> nhwDims = {"ni"};
  for (size_t i = 0; i < convDims.out.size(); i++)
    nhwDims.push_back(b.getStringAttr(Twine(i) + "o"));
  matchUnderlyingOrder(nhwDims, gemmInputTransform);

  llvm::SmallVector<StringRef, 3> mergeToK, mergeToN;
  switch (convOpType) {
  case ConvOpType::Fwd:
    gemmInputTransform.merge("gemmK", 1, nonNHWDims);
    gemmInputTransform.merge("gemmN", 2, nhwDims);
    break;
  case ConvOpType::BwdWeight:
    gemmInputTransform.merge("gemmK", 1, nhwDims);
    gemmInputTransform.merge("gemmN", 2, nonNHWDims);
    break;
  case ConvOpType::BwdData:
    llvm_unreachable("Backward data has been sent elsewhere");
    break;
  }

  TransformMapAttr gemmInputTransformAttr = gemmInputTransform.get();
  Value gemmInput =
      TransformOp::create(b, loc, embeddedInput, gemmInputTransformAttr);

  Value gemmOutput;
  if constexpr (notConvGemm) {
    Value outputValue = destBuffer;

    ArrayRef<int64_t> outputShape =
        cast<ShapedType>(outputValue.getType()).getShape();

    // Transform output tensor.
    // - PassThrough G to dimension 0, name it gemmG, then
    // Output tensor transformation for ConvOp:
    // - PassThrough K dimension to dimension 1, named gemmM
    // - Merge non-K dimensions to dimension2, named gemmN

    // Output tensor transformation for backward weight:
    // - Merge non-K dimensions to dimension 1, named gemmK
    // - PassThrough K dimension to dimension 2, name it gemmM
    SmallVector<StringRef, 5> outputNonKDims;
    for (StringRef name : outputNames)
      if (name != "go" && name != "ko")
        outputNonKDims.push_back(name);

    BottomUpTMBuilder outputTransform(b, outputNames, outputShape, loc);
    outputTransform.passThrough({"gemmG"}, {0}, {"go"});
    switch (convOpType) {
    case ConvOpType::Fwd:
      outputTransform.passThrough({"gemmM"}, {1}, {"ko"});
      outputTransform.merge("gemmN", 2, outputNonKDims);
      break;
    case ConvOpType::BwdWeight:
      outputTransform.merge("gemmK", 1, outputNonKDims);
      outputTransform.passThrough({"gemmM"}, {2}, {"ko"});
      break;
    case ConvOpType::BwdData:
      llvm_unreachable("Backward data has been sent elsewhere");
      break;
    }

    TransformMapAttr outputTransformAttr = outputTransform.get();

    // For forward conv, also transform the store destinations and fusion
    // inputs with the output transform. For BwdWeight, the store destination
    // is the filter (not the output), so the caller uses gemmC instead.
    if (convOpType == ConvOpType::Fwd) {
      for (auto &outputView : outputViews) {
        if (Operation *defOp = outputView.getDefiningOp()) {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointAfter(defOp);
          outputView =
              TransformOp::create(b, loc, outputView, outputTransformAttr);
        } else {
          outputView =
              TransformOp::create(b, loc, outputView, outputTransformAttr);
        }
      }
      for (auto &[orig, view] : fusionInputMap) {
        if (Operation *defOp = view.getDefiningOp()) {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointAfter(defOp);
          view = TransformOp::create(b, loc, view, outputTransformAttr);
        } else {
          view = TransformOp::create(b, loc, view, outputTransformAttr);
        }
      }
      // All of them have the same shape, so get the first one.
      // Note that we always have at least one outputView, otherwise
      // traceRootOutputToStoreOps() would have failed
      assert(!outputViews.empty() && "outputViews is empty!");
      gemmOutput = outputViews[0];
    } else {
      // Note that we don't support backward conv fusions!
      gemmOutput =
          TransformOp::create(b, loc, outputValue, outputTransformAttr);
    }
  }

  return std::make_tuple(gemmFilter, gemmInput, gemmOutput);
}

struct ConvGemmRewritePattern : public OpRewritePattern<ConvElementwiseGemmOp> {
  using OpRewritePattern<ConvElementwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvElementwiseGemmOp op,
                                PatternRewriter &b) const override {

    ConvolutionContext ctx = populateConvContextFromConvGemm(op);

    // pass empty tensors because it's not used for conv+gemm
    // because conv+gemm doesn't support fusions
    DenseMap<Value, Value> fusionInputMap;
    SmallVector<Value> outputViews;
    auto maybeArgs = commonConvRewrite(op, b, ctx, ConvOpType::Fwd,
                                       fusionInputMap, outputViews);
    if (failed(maybeArgs))
      return failure();
    Value gemmFilter, gemmInput;
    std::tie(gemmFilter, gemmInput, std::ignore) = maybeArgs.value();

    // emit rock.gemm_elementwise_gemm op
    Location loc = op.getLoc();

    // note that here A = input, B = filter, ConvToGemm is the opposite
    auto newOp = rock::GemmElementwiseGemmOp::create(
        b, loc, op->getResultTypes(), gemmInput, gemmFilter, op.getC(),
        op.getElemwiseInputs(),
        /*aTransposed=*/b.getUnitAttr(), /*bTransposed=*/nullptr,
        op.getCTransposedAttr(), op.getOTransposedAttr(), op.getParams0Attr(),
        op.getParams1Attr());

    // copy fusions if there are any
    bool hasFusion = rock::gemmGemmHasPreSecondGemmFusion(op);
    if (hasFusion) {
      b.inlineRegionBefore(op.getPreSecondGemmBody(),
                           newOp.getPreSecondGemmBody(),
                           newOp.getPreSecondGemmBody().begin());
    }
    b.replaceOp(op, newOp);

    return success();
  }
};

template <typename T>
struct ConvRewritePattern : public OpRewritePattern<T> {
  const static ArgumentFields fields;
  const static ConvOpType convOpType;
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &b) const override {
    ConvolutionContext ctx = populateConvContext(op);

    auto maybeViews = rock::traceOutputsAndFusionInputs(op.getResult());
    if (failed(maybeViews))
      return op.emitOpError("cannot trace to rock::StoreOp");
    auto &[stores, outputViews, fusionInputMap] = maybeViews.value();

    auto maybeArgs =
        commonConvRewrite(op, b, ctx, convOpType, fusionInputMap, outputViews);
    if (failed(maybeArgs))
      return failure();
    Value gemmFilter, gemmInput, gemmOutput;
    std::tie(gemmFilter, gemmInput, gemmOutput) = maybeArgs.value();

    // backward conv was run, no need to keep running the pass
    if (gemmFilter == nullptr && gemmInput == nullptr &&
        gemmOutput == nullptr) {
      assert(convOpType != ConvOpType::Fwd);
      return success();
    }

    SmallVector<Value, 3> arguments = {gemmFilter, gemmInput, gemmOutput};

    Value gemmA, gemmB, gemmC;
    gemmA = arguments[fields.gridwiseGemmArgumentPosition[0]];
    gemmB = arguments[fields.gridwiseGemmArgumentPosition[1]];
    gemmC = arguments[fields.gridwiseGemmArgumentPosition[2]];

    // Emit rock.gemm op.
    Location loc = op.getLoc();
    auto tuningParams = op.getParamsAttr();
    auto newGemmOp = GemmOp::create(
        b, loc, getResultType(op, gemmC), gemmA, gemmB,
        /*scaleA=*/nullptr, /*scaleB=*/nullptr,
        /*aTransposed=*/b.getUnitAttr(), /*bTransposed=*/nullptr,
        /*cTransposed=*/nullptr,
        /*aScaleTransposed=*/nullptr,
        /*bScaleTransposed=*/nullptr, /*quantBlockSize=*/nullptr, tuningParams);

    Value result = newGemmOp.getResult();

    // Propagate the new output type through any fusion ops
    // between the gemm result and the store ops. This replaces uses of the old
    // gemm result inside fusion ops with the gridwise result and updates their
    // result types to match the new shape.
    rock::propagateOutputType(op.getResult(), result);

    // Replace extra fusion input operands with their gemm versions.
    rock::replaceFusionExtraInputs(result, fusionInputMap);

    for (size_t i = 0; i < stores.size(); ++i) {
      StoreOp storeOp = stores[i];
      // For Fwd, the store destination has been output-transformed.
      // For BwdWeight, the store destination is the filter; use gemmC which
      // already carries the filter transform.
      Value view = (convOpType == ConvOpType::Fwd) ? outputViews[i] : gemmC;
      // adjust the store method
      StoreMethodAttr storeMethod = storeOp.getStoreMethodAttr();
      // If the store's source is the gemm result directly (no fusions),
      // use the gridwise result. Otherwise, propagateOutputType has already
      // updated the fusion chain, and storeOp.getSource() has the correct type.
      Value source = storeOp.getSource();
      if (source == op.getResult())
        source = result;
      b.setInsertionPoint(storeOp);
      auto newStoreOp = rock::StoreOp::create(
          b, storeOp.getLoc(), storeOp.getResult().getType(), source, view,
          storeOp.getResultAlias(), storeMethod);
      b.replaceOp(storeOp, newStoreOp.getResult());
    }

    // Finally, erase the original Conv op.
    b.eraseOp(op);

    return success();
  }
};

template <>
const ArgumentFields ConvRewritePattern<ConvOp>::fields = {
    {0, 1, 2},
    {"KM", "KN", "MN"},
};
template <>
const ConvOpType ConvRewritePattern<ConvOp>::convOpType = ConvOpType::Fwd;

template <>
const ArgumentFields ConvRewritePattern<ConvBwdDataOp>::fields = {
    {0, 2, 1},
    {"KM", "MN", "KN"},
};

template <>
const ConvOpType ConvRewritePattern<ConvBwdDataOp>::convOpType =
    ConvOpType::BwdData;

template <>
const ArgumentFields ConvRewritePattern<ConvBwdWeightOp>::fields = {
    {2, 1, 0},
    {"MN", "KN", "KM"},
};

template <>
const ConvOpType ConvRewritePattern<ConvBwdWeightOp>::convOpType =
    ConvOpType::BwdWeight;

// Explicitly instantiate the template to operation type
template struct ConvRewritePattern<ConvOp>;
template struct ConvRewritePattern<ConvBwdDataOp>;
template struct ConvRewritePattern<ConvBwdWeightOp>;

void RockConvToGemmPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  func::FuncOp func = getOperation();

  // Annotate the function as a convolution kernel.
  WalkResult convWalk = func.walk([](Operation *op) {
    if (isa<rock::ConvOp, rock::ConvBwdDataOp, rock::ConvBwdWeightOp,
            rock::ConvElementwiseGemmOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (convWalk.wasInterrupted())
    func->setAttr(rock::ConvKernelAttr::getMnemonic(), UnitAttr::get(ctx));

  RewritePatternSet preConvToGemmPatterns(ctx);
  preConvToGemmPatterns.add<MatchLayoutsToInput, MatchFilterToInput>(ctx);

  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(preConvToGemmPatterns)))) {
    signalPassFailure();
    return;
  }

  ConversionTarget target(*ctx);

  target.addIllegalOp<rock::ConvOp, rock::ConvBwdDataOp, rock::ConvBwdWeightOp,
                      rock::ConvElementwiseGemmOp>();
  target.addLegalOp<rock::TransformOp, rock::GemmOp,
                    rock::GemmElementwiseGemmOp, rock::StoreOp>();
  // Below are required legalize for the lowering of ConvBwdWeightOp
  target.addLegalDialect<arith::ArithDialect, scf::SCFDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<ConvRewritePattern<ConvOp>, ConvRewritePattern<ConvBwdDataOp>,
               ConvRewritePattern<ConvBwdWeightOp>, ConvGemmRewritePattern>(
      ctx);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // end anonymous namespace
