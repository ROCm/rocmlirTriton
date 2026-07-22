//===- attentionUtils.cpp - Attention analysis utilities -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/attentionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::rock;

PreSoftmaxInputRole PreSoftmaxInputRoleInfo::getSingularRole() const {
  if (numMerges != 1 || hasOther)
    return PreSoftmaxInputRole::Other;
  unsigned numRoles = hasDequant + hasScale + hasBias;
  if (numRoles != 1)
    return PreSoftmaxInputRole::Other;
  if (hasDequant)
    return PreSoftmaxInputRole::Dequant;
  if (hasScale)
    return PreSoftmaxInputRole::Scale;
  return PreSoftmaxInputRole::Bias;
}

SmallVector<PreSoftmaxInputRoleInfo>
mlir::rock::analyzePreSoftmaxInputRoles(Region &region, unsigned numInputs,
                                        unsigned numDequantInputs) {
  SmallVector<PreSoftmaxInputRoleInfo> roles(numInputs);
  if (region.empty() || !llvm::hasSingleElement(region))
    return roles;

  Block &block = region.front();
  if (block.getNumArguments() != numInputs + 1)
    return roles;

  numDequantInputs = std::min(numDequantInputs, numInputs);
  SmallVector<bool> pendingDequantMerge(numInputs, false);
  for (unsigned inputIndex = 0; inputIndex < numDequantInputs; ++inputIndex) {
    roles[inputIndex].hasDequant = true;
    roles[inputIndex].numMerges = 1;
    pendingDequantMerge[inputIndex] = true;
  }

  const unsigned numProvenanceBits = numInputs + 1;
  llvm::DenseMap<Value, llvm::SmallBitVector> provenance;
  for (auto [index, argument] : llvm::enumerate(block.getArguments())) {
    llvm::SmallBitVector bits(numProvenanceBits);
    bits.set(index);
    provenance.try_emplace(argument, std::move(bits));
  }

  for (Operation &op : block) {
    llvm::SmallBitVector resultProvenance(numProvenanceBits);
    for (Value operand : op.getOperands()) {
      auto found = provenance.find(operand);
      if (found != provenance.end())
        resultProvenance |= found->second;
    }

    for (unsigned inputIndex = 0; inputIndex < numInputs; ++inputIndex) {
      unsigned inputBit = inputIndex + 1;
      bool hasScoreOperand = false;
      bool hasFreshInputOperand = false;
      for (Value operand : op.getOperands()) {
        auto found = provenance.find(operand);
        if (found == provenance.end())
          continue;
        hasScoreOperand |= found->second.test(0);
        hasFreshInputOperand |=
            found->second.test(inputBit) && !found->second.test(0);
      }
      if (!hasScoreOperand || !hasFreshInputOperand)
        continue;

      if (pendingDequantMerge[inputIndex]) {
        pendingDequantMerge[inputIndex] = false;
        continue;
      }

      PreSoftmaxInputRoleInfo &role = roles[inputIndex];
      ++role.numMerges;
      if (isa<arith::AddFOp>(op))
        role.hasBias = true;
      else if (isa<arith::MulFOp>(op))
        role.hasScale = true;
      else
        role.hasOther = true;
    }

    for (Value result : op.getResults())
      provenance[result] = resultProvenance;
  }

  return roles;
}

SmallVector<PreSoftmaxInputRole>
mlir::rock::classifyPreSoftmaxInputRoles(Region &region, unsigned numInputs,
                                         unsigned numDequantInputs) {
  SmallVector<PreSoftmaxInputRoleInfo> analysis =
      analyzePreSoftmaxInputRoles(region, numInputs, numDequantInputs);
  return llvm::map_to_vector(analysis, [](const PreSoftmaxInputRoleInfo &info) {
    return info.getSingularRole();
  });
}

static unsigned getExplicitOrLegacyDequantCount(IntegerAttr countAttr,
                                                Type queryType,
                                                unsigned numInputs) {
  if (countAttr) {
    int64_t count = countAttr.getInt();
    return count <= 0
               ? 0
               : std::min<unsigned>(static_cast<unsigned>(count), numInputs);
  }
  return queryType.isInteger(8) ? std::min(2u, numInputs) : 0;
}

unsigned mlir::rock::getEffectiveNumDequantInputs(AttentionOp op) {
  return getExplicitOrLegacyDequantCount(
      op.getNumDequantInputsAttr(), op.getQueries().getType().getElementType(),
      op.getPreSoftmaxElemWiseInputs().size());
}

unsigned mlir::rock::getEffectiveNumDequantInputs(GridwiseAttentionOp op) {
  if (!op.getEnableSoftmax() && !op.getNumDequantInputsAttr())
    return 0;
  return getExplicitOrLegacyDequantCount(
      op.getNumDequantInputsAttr(), op.getQueries().getType().getElementType(),
      op.getPreSoftmaxElemWiseInputs().size());
}

static bool isLayoutNeutralElementwise(Operation *op) {
  return isa<arith::MulFOp, arith::AddFOp, arith::SubFOp, arith::DivFOp,
             arith::MaximumFOp, arith::MinimumFOp, arith::TruncFOp,
             arith::ExtFOp, arith::NegFOp, arith::TruncIOp, arith::ExtSIOp,
             arith::ExtUIOp>(op);
}

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  return !llvm::MulOverflow(lhs, rhs, result);
}

/// Return false when a floor/mod expression couples multiple upper
/// dimensions. In that case, measuring a dimension with all other coordinates
/// set to zero would not prove its physical stride over the whole domain.
static bool hasCoupledDivision(AffineExpr expr, unsigned numDims) {
  bool coupled = false;
  expr.walk([&](AffineExpr nested) {
    if (coupled || (nested.getKind() != AffineExprKind::FloorDiv &&
                    nested.getKind() != AffineExprKind::CeilDiv &&
                    nested.getKind() != AffineExprKind::Mod))
      return;

    auto binary = cast<AffineBinaryOpExpr>(nested);
    if (!isa<AffineConstantExpr>(binary.getRHS())) {
      coupled = true;
      return;
    }

    unsigned dependentDims = 0;
    for (unsigned dim = 0; dim < numDims; ++dim)
      dependentDims += binary.getLHS().isFunctionOfDim(dim);
    coupled = dependentDims > 1;
  });
  return coupled;
}

/// Measure the smallest physical address granularity generated by varying one
/// upper dimension over its complete static domain. Exact enumeration handles
/// merged dimensions represented with floordiv/mod while remaining
/// conservative for large or coupled domains.
static std::optional<int64_t> getPhysicalGranularity(AffineMap physicalOffset,
                                                     unsigned dim,
                                                     int64_t dimSize) {
  constexpr int64_t maxEnumeratedDimension = 65536;
  if (dimSize <= 1)
    return 0;
  if (dimSize > maxEnumeratedDimension)
    return std::nullopt;
  if (!physicalOffset.getResult(0).isFunctionOfDim(dim))
    return 0;

  MLIRContext *context = physicalOffset.getContext();
  auto zero = IntegerAttr::get(IndexType::get(context), 0);
  SmallVector<Attribute> coordinates(physicalOffset.getNumDims(), zero);

  auto evaluate = [&]() -> std::optional<int64_t> {
    SmallVector<Attribute> folded;
    if (failed(physicalOffset.constantFold(coordinates, folded)) ||
        folded.size() != 1)
      return std::nullopt;
    auto value = dyn_cast<IntegerAttr>(folded.front());
    if (!value)
      return std::nullopt;
    return value.getInt();
  };

  std::optional<int64_t> previous = evaluate();
  if (!previous)
    return std::nullopt;

  int64_t granularity = 0;
  for (int64_t coordinate = 1; coordinate < dimSize; ++coordinate) {
    coordinates[dim] = IntegerAttr::get(IndexType::get(context), coordinate);
    std::optional<int64_t> current = evaluate();
    int64_t delta;
    if (!current || llvm::SubOverflow(*current, *previous, delta) ||
        delta == std::numeric_limits<int64_t>::min())
      return std::nullopt;
    granularity = std::gcd(granularity, std::abs(delta));
    previous = current;
  }
  return granularity;
}

struct SequenceCoordinates {
  SmallVector<int64_t> shape;
  unsigned m;
  unsigned n;
  SmallVector<TransformMapAttr> transformsToStorage;
};

/// Validate one sequence axis of the block/iteration tiling map emitted by
/// computeOutputTransforms. Names identify the intended logical axis, while
/// the Unmerge structure and exact bounds prove that the two upper dimensions
/// densely cover it.
static std::optional<SmallVector<unsigned>>
getDenseTileAxis(TransformMapAttr map, StringRef axisName,
                 unsigned lowerDimension) {
  std::optional<TransformAttr> matchingTransform;
  for (TransformAttr transform : map.getOps()) {
    for (auto [name, dimension] :
         llvm::zip(transform.getLowerNames(), transform.getLowerDims())) {
      if (name != axisName)
        continue;
      if (matchingTransform || dimension != lowerDimension)
        return std::nullopt;
      matchingTransform = transform;
    }
  }
  if (!matchingTransform ||
      matchingTransform->getType() != TransformType::Unmerge ||
      matchingTransform->getLowerDims().size() != 1 ||
      matchingTransform->getUpperDims().size() != 2 ||
      matchingTransform->getParams().size() != 2)
    return std::nullopt;

  ArrayRef<uint32_t> upperDimensions = matchingTransform->getUpperDims();
  ArrayRef<int64_t> lengths = matchingTransform->getParams();
  ArrayRef<int64_t> upperBounds = map.getUpperBounds();
  ArrayRef<int64_t> lowerBounds = map.getLowerBounds();
  if (upperDimensions[0] == upperDimensions[1] ||
      upperDimensions[0] >= upperBounds.size() ||
      upperDimensions[1] >= upperBounds.size() ||
      lowerDimension >= lowerBounds.size() || lengths[0] <= 0 ||
      lengths[1] <= 0 || upperBounds[upperDimensions[0]] != lengths[0] ||
      upperBounds[upperDimensions[1]] != lengths[1])
    return std::nullopt;

  int64_t logicalExtent;
  if (!checkedMul(lengths[0], lengths[1], logicalExtent) ||
      lowerBounds[lowerDimension] != logicalExtent)
    return std::nullopt;

  AffineExpr expected =
      getAffineDimExpr(upperDimensions[0], map.getContext()) * lengths[1] +
      getAffineDimExpr(upperDimensions[1], map.getContext());
  if (map.getMap().getAffineMap().getResult(lowerDimension) != expected)
    return std::nullopt;

  return SmallVector<unsigned>{upperDimensions[0], upperDimensions[1]};
}

static std::optional<SequenceCoordinates>
getSequenceCoordinates(ArrayRef<TransformMapAttr> transforms,
                       RankedTensorType upperType) {
  unsigned upperRank = upperType.getRank();
  if (upperRank < 2 || !upperType.hasStaticShape())
    return std::nullopt;

  auto trailingDimensions = [&]() {
    return SequenceCoordinates{llvm::to_vector(upperType.getShape()),
                               upperRank - 2, upperRank - 1,
                               llvm::to_vector(transforms)};
  };
  if (transforms.empty())
    return trailingDimensions();

  TransformMapAttr outer = transforms.front();
  AffineMap outerMap = outer.getMap().getAffineMap();
  SmallVector<StringRef> lowerNames(outerMap.getNumResults());
  for (TransformAttr transform : outer.getOps())
    for (auto [name, dimension] :
         llvm::zip(transform.getLowerNames(), transform.getLowerDims())) {
      if (dimension >= lowerNames.size())
        return std::nullopt;
      lowerNames[dimension] = name;
    }

  auto mName = llvm::find(lowerNames, "gemmM");
  auto nName = llvm::find(lowerNames, "gemmN");
  bool hasM = mName != lowerNames.end();
  bool hasN = nName != lowerNames.end();
  if (!hasM && !hasN)
    return trailingDimensions();
  if (!hasM || !hasN || outerMap.getNumDims() != upperRank ||
      outer.getUpperBounds().asArrayRef() != upperType.getShape())
    return std::nullopt;

  unsigned mResult = std::distance(lowerNames.begin(), mName);
  unsigned nResult = std::distance(lowerNames.begin(), nName);
  std::optional<SmallVector<unsigned>> mUpper =
      getDenseTileAxis(outer, "gemmM", mResult);
  std::optional<SmallVector<unsigned>> nUpper =
      getDenseTileAxis(outer, "gemmN", nResult);
  if (!mUpper || !nUpper || llvm::any_of(*mUpper, [&](unsigned dimension) {
        return llvm::is_contained(*nUpper, dimension);
      }))
    return std::nullopt;

  return SequenceCoordinates{
      llvm::to_vector(outer.getLowerBounds().asArrayRef()), mResult, nResult,
      llvm::to_vector(transforms.drop_front())};
}

static PreSoftmaxInputOrientation classifyPureTransformChain(Value value) {
  auto upperType = dyn_cast<RankedTensorType>(value.getType());
  if (!upperType || upperType.getRank() < 2)
    return PreSoftmaxInputOrientation::Unknown;

  SmallVector<TransformMapAttr> transforms;
  Value root;
  std::tie(root, std::ignore) = untransform(value, transforms);
  auto rootType = dyn_cast<RankedTensorType>(root.getType());
  if (!rootType || rootType.getRank() == 0 || !rootType.hasStaticShape())
    return PreSoftmaxInputOrientation::Unknown;

  // Affine composition drops the validity domain. A nontrivial pad or embed
  // can therefore look like a dense transpose even though most coordinates
  // are invalid.
  if (llvm::any_of(transforms, mapImpactsValidity))
    return PreSoftmaxInputOrientation::Unknown;

  std::optional<SequenceCoordinates> coordinates =
      getSequenceCoordinates(transforms, upperType);
  if (!coordinates || coordinates->shape[coordinates->m] <= 1 ||
      coordinates->shape[coordinates->n] <= 1)
    return PreSoftmaxInputOrientation::Unknown;

  unsigned coordinateRank = coordinates->shape.size();
  AffineMap composed =
      coordinates->transformsToStorage.empty()
          ? AffineMap::getMultiDimIdentityMap(coordinateRank,
                                              value.getContext())
          : composeTransforms(coordinates->transformsToStorage);
  composed = simplifyAffineMap(composed);
  if (!composed || composed.getNumSymbols() != 0 ||
      composed.getNumDims() != coordinateRank ||
      composed.getNumResults() != static_cast<unsigned>(rootType.getRank()))
    return PreSoftmaxInputOrientation::Unknown;

  // Convert the composed upper-to-root map into one row-major physical offset.
  // Comparing address granularities handles flattened storage, broadcasts,
  // and merged group dimensions without relying on transform spelling.
  AffineExpr physicalOffset = getAffineConstantExpr(0, value.getContext());
  int64_t stride = 1;
  for (int64_t resultIndex = rootType.getRank() - 1; resultIndex >= 0;
       --resultIndex) {
    physicalOffset = physicalOffset + composed.getResult(resultIndex) * stride;
    int64_t nextStride;
    if (!checkedMul(stride, rootType.getDimSize(resultIndex), nextStride))
      return PreSoftmaxInputOrientation::Unknown;
    stride = nextStride;
  }

  if (hasCoupledDivision(physicalOffset, coordinateRank))
    return PreSoftmaxInputOrientation::Unknown;

  AffineMap offsetMap = AffineMap::get(coordinateRank, 0, physicalOffset);
  SmallVector<std::optional<int64_t>> granularities;
  granularities.reserve(coordinateRank);
  for (auto [dim, dimSize] : llvm::enumerate(coordinates->shape))
    granularities.push_back(getPhysicalGranularity(offsetMap, dim, dimSize));

  if (llvm::any_of(granularities,
                   [](std::optional<int64_t> value) { return !value; }))
    return PreSoftmaxInputOrientation::Unknown;

  auto positiveGranularity = [&](unsigned dimension) -> std::optional<int64_t> {
    int64_t granularity = *granularities[dimension];
    if (granularity <= 0)
      return std::nullopt;
    return granularity;
  };
  std::optional<int64_t> mGranularity = positiveGranularity(coordinates->m);
  std::optional<int64_t> nGranularity = positiveGranularity(coordinates->n);
  auto isSequenceDimension = [&](unsigned dim) {
    return dim == coordinates->m || dim == coordinates->n;
  };

  if (mGranularity && nGranularity) {
    if (*mGranularity == *nGranularity)
      return PreSoftmaxInputOrientation::Unknown;

    // Sequence-plane dimensions must remain the two fastest non-unit
    // dimensions. A permutation involving a non-sequence dimension is not a
    // bias transpose.
    int64_t maxSequenceGranularity = std::max(*mGranularity, *nGranularity);
    for (unsigned dim = 0; dim < coordinateRank; ++dim) {
      if (isSequenceDimension(dim))
        continue;
      if (coordinates->shape[dim] > 1 && *granularities[dim] > 0 &&
          *granularities[dim] <= maxSequenceGranularity)
        return PreSoftmaxInputOrientation::Unknown;
    }

    return *mGranularity < *nGranularity
               ? PreSoftmaxInputOrientation::Transposed
               : PreSoftmaxInputOrientation::Natural;
  }

  // A column bias can be broadcast across M. Its paired physical dimension
  // may have been folded into the attention group coordinate. Compare N with
  // the fastest remaining varying dimension to retain the original physical
  // orientation through that grouping.
  if (mGranularity || !nGranularity)
    return PreSoftmaxInputOrientation::Unknown;

  std::optional<int64_t> fastestOther;
  for (unsigned dim = 0; dim < coordinateRank; ++dim) {
    if (isSequenceDimension(dim))
      continue;
    int64_t granularity = *granularities[dim];
    if (coordinates->shape[dim] <= 1 || granularity <= 0)
      continue;
    fastestOther =
        fastestOther ? std::min(*fastestOther, granularity) : granularity;
  }
  if (!fastestOther || *fastestOther == *nGranularity)
    return PreSoftmaxInputOrientation::Unknown;
  return *nGranularity < *fastestOther ? PreSoftmaxInputOrientation::Natural
                                       : PreSoftmaxInputOrientation::Transposed;
}

PreSoftmaxInputOrientation
mlir::rock::classifyPreSoftmaxInputOrientation(Value value) {
  while (value) {
    SmallVector<TransformMapAttr> transforms;
    Value source;
    std::tie(source, std::ignore) = untransform(value, transforms);

    Operation *def = source.getDefiningOp();
    if (!def)
      return classifyPureTransformChain(value);

    // A transform outside an elementwise operation changes the coordinate
    // space on only one side of that operation. Do not guess how the two
    // transform runs compose.
    if (!transforms.empty() || !isLayoutNeutralElementwise(def))
      return PreSoftmaxInputOrientation::Unknown;

    Value tensorOperand;
    for (Value operand : def->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType()) ||
          matchPattern(operand, m_Constant()))
        continue;
      if (tensorOperand)
        return PreSoftmaxInputOrientation::Unknown;
      tensorOperand = operand;
    }
    if (!tensorOperand)
      return PreSoftmaxInputOrientation::Unknown;
    value = tensorOperand;
  }
  return PreSoftmaxInputOrientation::Unknown;
}
