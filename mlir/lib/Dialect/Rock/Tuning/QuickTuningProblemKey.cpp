//===- QuickTuningProblemKey.cpp - tuning problem identity ----------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the tuning-problem serialization and the quick-tuning
// problem hash.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/QuickTuningProblemKey.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

using namespace mlir;
using namespace mlir::rock;

static LogicalResult
extractLayouts(Operation *op, llvm::StringMap<unsigned> &fLayoutMap,
               llvm::StringMap<unsigned> &iLayoutMap,
               llvm::StringMap<unsigned> &oLayoutMap, SmallString<6> &fLayout,
               SmallString<6> &iLayout, SmallString<6> &oLayout,
               bool computeOutput = true) {
  // Extract layout information
  auto filterLayoutAttr = op->getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->getAttrOfType<ArrayAttr>("input_layout");
  ArrayAttr outputLayoutAttr;
  if (computeOutput)
    outputLayoutAttr = op->getAttrOfType<ArrayAttr>("output_layout");

  unsigned size = filterLayoutAttr.size();

  for (unsigned i = 0; i < size; ++i) {
    auto filterAttr = cast<StringAttr>(filterLayoutAttr.getValue()[i]);
    StringRef fKey = filterAttr.getValue();
    if (fKey == "y")
      fKey = "0";
    if (fKey == "x")
      fKey = "1";
    fLayoutMap[fKey] = i;
    auto inputAttr = cast<StringAttr>(inputLayoutAttr.getValue()[i]);
    StringRef iKey = inputAttr.getValue();
    if (iKey == "hi")
      iKey = "0i";
    if (iKey == "wi")
      iKey = "1i";
    iLayoutMap[iKey] = i;
    if (computeOutput) {
      auto outputAttr = cast<StringAttr>(outputLayoutAttr.getValue()[i]);
      StringRef oKey = outputAttr.getValue();
      if (oKey == "ho")
        oKey = "0o";
      if (oKey == "wo")
        oKey = "1o";
      oLayoutMap[oKey] = i;
    }
  }

  fLayout.assign(size, '#');
  iLayout.assign(size, '#');
  oLayout.assign(size, '#');

  // dimensions need to be mapped 1 to 1.
  fLayout[fLayoutMap["k"]] = 'N';
  fLayout[fLayoutMap["c"]] = 'C';
  fLayout[fLayoutMap["g"]] = 'G';
  iLayout[iLayoutMap["ni"]] = 'N';
  iLayout[iLayoutMap["ci"]] = 'C';
  iLayout[iLayoutMap["gi"]] = 'G';
  if (computeOutput) {
    oLayout[oLayoutMap["no"]] = 'N';
    oLayout[oLayoutMap["ko"]] = 'C';
    oLayout[oLayoutMap["go"]] = 'G';
  }

  for (unsigned i = 0; i < size - 3; i++) {
    std::string key = std::to_string(i);
    char val = '0' + i;
    fLayout[fLayoutMap[key]] = val;
    iLayout[iLayoutMap[key + "i"]] = val;
    if (computeOutput)
      oLayout[oLayoutMap[key + "o"]] = val;
  }

  if (computeOutput) {
    if (llvm::any_of(llvm::concat<const char>(fLayout, iLayout, oLayout),
                     [](const char c) { return c == '#'; }))
      return failure();
  } else {
    if (llvm::any_of(llvm::concat<const char>(fLayout, iLayout),
                     [](const char c) { return c == '#'; }))
      return failure();
  }
  return success();
}

// Walk backward through a transform chain and report whether any individual
// transform preserves rank and element extents while applying a non-identity
// permutation. This catches layout-only changes, such as transposed attention
// bias, even when they are surrounded by flatten/unflatten transforms.
static bool hasRankPreservingNonIdentityPermutation(Value value) {
  while (auto transformOp = value.getDefiningOp<TransformOp>()) {
    auto valueType = dyn_cast<ShapedType>(transformOp.getResult().getType());
    auto inputType = dyn_cast<ShapedType>(transformOp.getInput().getType());
    if (valueType && inputType && valueType.getRank() == inputType.getRank()) {
      SmallVector<int64_t> valueShape(valueType.getShape());
      SmallVector<int64_t> inputShape(inputType.getShape());
      llvm::sort(valueShape);
      llvm::sort(inputShape);

      AffineMap map = transformOp.getTransform().getMap().getAffineMap();
      if (valueShape == inputShape && map && map.isPermutation() &&
          !isIdentityOnShape(map, valueType.getShape()))
        return true;
    }
    value = transformOp.getInput();
  }
  return false;
}

// Determine whether an attention op fuses a pre-softmax scale and/or bias, as
// created by rocmlir-gen's `--with-attn-scale` / `--with-attn-bias`. Scale
// folds an extra elementwise multiply of the QK^T scores by an external input;
// bias folds an elementwise add. Both change the generated kernel (and thus its
// optimal perf config), so they are part of the tuning-problem identity and
// must appear in the tuning key. A rank-preserving non-identity permutation on
// the bias input records that the bias is loaded transposed.
//
// The fusion is encoded as ops inside the `preSoftmaxBody` region that consume
// the region's block arguments. Block argument 0 is the QK^T product; the
// remaining block arguments map 1:1 to `preSoftmaxElemWiseInputs`. Constant
// scales/biases
// and causal masks are not external inputs (they are folded into the body or
// captured by the `causal` attribute), so they are intentionally not treated as
// attn scale/bias here. For quantized (i8) attention the first two elementwise
// inputs are dequantization operands, not the attention scale/bias, so they are
// skipped.
static void getAttentionScaleBias(AttentionOp attnOp, bool isQuantized,
                                  bool &hasAttnScale, bool &hasAttnBias,
                                  bool &hasTransposedAttnBias) {
  hasAttnScale = false;
  hasAttnBias = false;
  hasTransposedAttnBias = false;
  Region &body = attnOp.getPreSoftmaxBody();
  if (body.empty())
    return;
  Block &entry = body.front();
  unsigned numInputs = attnOp.getPreSoftmaxElemWiseInputs().size();
  unsigned numQuantInputs = isQuantized ? 2u : 0u;
  if (numInputs <= numQuantInputs)
    return;

  // Entry block argument 0 is the QK^T product; arguments 1.. correspond 1:1 to
  // the pre-softmax elementwise inputs. AttentionOp::verify pins this arity
  // (see verifyGemmPlusGemmLikeOp).
  assert(entry.getNumArguments() == 1 + numInputs &&
         "pre-softmax body arguments must match elementwise inputs");

  for (unsigned i = numQuantInputs; i < numInputs; ++i) {
    // Walk forward from the input through its use chain, stepping over any
    // intermediate op (e.g. a `rock.transform` reshaping the input to match the
    // scores) until we reach the multiply (scale) or add (bias) that consumes
    // it.
    SmallVector<Value> worklist{entry.getArgument(i + 1)};
    llvm::SmallPtrSet<Value, 8> seen;
    bool isTransposedInput = hasRankPreservingNonIdentityPermutation(
        attnOp.getPreSoftmaxElemWiseInputs()[i]);
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      for (Operation *user : v.getUsers()) {
        if (isa<arith::MulFOp>(user))
          hasAttnScale = true;
        else if (isa<arith::AddFOp>(user)) {
          hasAttnBias = true;
          hasTransposedAttnBias |= isTransposedInput;
        } else {
          llvm::append_range(worklist, user->getResults());
        }
      }
    }
  }
}

// Keep this problem-key serialization in sync with the corresponding
// configuration parsing and serialization in perfRunner.py.
LogicalResult
mlir::rock::serializeTuningProblem(RockGemmGemmWrapperInterface gemmGemmOp,
                                   TuningProblemFormat format,
                                   SmallVectorImpl<char> &out) {
  // The quick-tuning key drops the architecture prefix and the element types,
  // because the shard it is probed in already selects on both, and names the
  // kernel type instead. See TuningProblemFormat.
  const bool isPerfDb = format == TuningProblemFormat::PerfDb;
  int64_t numCU = rock::getNumCUValue(gemmGemmOp);
  int64_t numChiplets = rock::getNumChipletsValue(gemmGemmOp);
  constexpr char sep = ' ';
  constexpr char tab = '\t';
  int64_t headDimQK;
  int64_t headDimV;
  int64_t seqLenQ;
  int64_t seqLenK;
  llvm::raw_svector_ostream problemOS(out);
  if (isPerfDb) {
    // ARCH string
    problemOS << StringRef(rock::getArchValue(gemmGemmOp)) << tab;
    // Number of Compute Units
    problemOS << numCU << tab;
    // Number of chiplets
    problemOS << numChiplets << tab;
  } else {
    problemOS << stringifyEnum(gemmGemmOp.getKernelType()) << sep;
  }

  ArrayRef<int64_t> qShape = cast<ShapedType>(gemmGemmOp.getAType()).getShape();
  ArrayRef<int64_t> kShape = cast<ShapedType>(gemmGemmOp.getBType()).getShape();
  ArrayRef<int64_t> vShape = cast<ShapedType>(gemmGemmOp.getCType()).getShape();

  bool isAttention = isa<AttentionOp>(gemmGemmOp);
  bool isConvGemm = isa<ConvElementwiseGemmOp>(gemmGemmOp);

  // The element type is validated in both formats so that an unsupported one
  // is rejected the same way, and only printed for the perf-database key.
  Type elemTypeQ = cast<ShapedType>(gemmGemmOp.getAType()).getElementType();
  StringRef elemTypeQStr;
  if (elemTypeQ.isF32()) {
    elemTypeQStr = "f32";
  } else if (elemTypeQ.isF16()) {
    elemTypeQStr = "f16";
  } else if (elemTypeQ.isBF16()) {
    elemTypeQStr = "bf16";
  } else if (elemTypeQ.isInteger(8) && isAttention) {
    elemTypeQStr = "i8";
  } else {
    if (!isPerfDb)
      return failure();
    return gemmGemmOp.emitError("invalid type:") << elemTypeQ << "\n";
  }
  if (isPerfDb)
    problemOS << "-t " << elemTypeQStr << sep;

  // Extract layout information
  llvm::StringMap<unsigned> fLayoutMap, iLayoutMap, oLayoutMap;
  SmallString<6> fLayout, iLayout, oLayout;

  if (isConvGemm) {
    if (failed(extractLayouts(gemmGemmOp, fLayoutMap, iLayoutMap, oLayoutMap,
                              fLayout, iLayout, oLayout, false))) {
      if (!isPerfDb)
        return failure();
      return gemmGemmOp.emitError("layout can't be extracted");
    }

    // filter layout
    problemOS << "-f " << fLayout << sep;
    // input layout
    problemOS << "-I " << iLayout << sep;
  } else {
    // TransQ
    if (isAttention)
      problemOS << "-transQ ";
    else
      problemOS << "-transA ";
    if (gemmGemmOp.getTransposedA()) {
      seqLenQ = qShape[2];
      headDimQK = qShape[1];
      problemOS << "true" << sep;
    } else {
      seqLenQ = qShape[1];
      headDimQK = qShape[2];
      problemOS << "false" << sep;
    }

    // TransK
    if (isAttention)
      problemOS << "-transK ";
    else
      problemOS << "-transB ";
    if (gemmGemmOp.getTransposedB()) {
      seqLenK = kShape[1];
      problemOS << "true" << sep;
    } else {
      seqLenK = kShape[2];
      problemOS << "false" << sep;
    }
  }

  // TransV
  if (isAttention)
    problemOS << "-transV ";
  else
    problemOS << "-transC ";
  if (gemmGemmOp.getTransposedC()) {
    headDimV = vShape[1];
    problemOS << "true" << sep;
  } else {
    headDimV = vShape[2];
    problemOS << "false" << sep;
  }

  // TransO
  problemOS << "-transO ";
  if (gemmGemmOp.getTransposedOut())
    problemOS << "true" << sep;
  else
    problemOS << "false" << sep;

  bool hasAttnScale = false, hasAttnBias = false;
  bool hasTransposedAttnBias = false;
  if (isAttention) {
    auto attentionOp = cast<AttentionOp>(gemmGemmOp);
    getAttentionScaleBias(attentionOp, elemTypeQ.isInteger(8), hasAttnScale,
                          hasAttnBias, hasTransposedAttnBias);
    problemOS << "-causal ";
    if (attentionOp.getCausal())
      problemOS << "true" << sep;
    else
      problemOS << "false" << sep;

    problemOS << "-return_lse ";
    if (attentionOp.getLse())
      problemOS << "true" << sep;
    else
      problemOS << "false" << sep;

    problemOS << "-split_kv " << attentionOp.getSplitKV() << sep;
    // The look-back is optional; only emit it when set so non-sliding problems
    // omit the field from their tuning identity.
    if (auto slidingWindowLookBack = attentionOp.getSlidingWindowLookBack();
        slidingWindowLookBack && *slidingWindowLookBack > 0)
      problemOS << "-sliding_window_look_back " << *slidingWindowLookBack
                << sep;
    problemOS << "-num_heads_q " << attentionOp.getNumHeadsQ() << sep;
    problemOS << "-num_heads_kv " << attentionOp.getNumHeadsKV() << sep;
    problemOS << "-g " << qShape[0] / attentionOp.getNumHeadsQ() << sep;
  }

  if (!isConvGemm && !isAttention)
    problemOS << "-g " << qShape[0] << sep;

  if (isAttention) {
    problemOS << "-seq_len_q " << seqLenQ << sep;
    problemOS << "-seq_len_k " << seqLenK << sep;
    problemOS << "-head_dim_qk " << headDimQK << sep;
    problemOS << "-head_dim_v " << headDimV;
    // Keep these last and in this order to match the layout parsed by
    // AttentionConfiguration.from_command_line() in perfRunner.py.
    problemOS << sep << "-with-attn-scale "
              << (hasAttnScale ? "true" : "false");
    problemOS << sep << "-with-attn-bias " << (hasAttnBias ? "true" : "false");
    problemOS << sep << "-transBias "
              << (hasTransposedAttnBias ? "true" : "false");
  } else if (isConvGemm) {
    auto convGemmOp = cast<ConvElementwiseGemmOp>(gemmGemmOp);
    ArrayRef<int64_t> inShape = convGemmOp.getInput().getType().getShape();
    ArrayRef<int64_t> filShape = convGemmOp.getFilter().getType().getShape();

    // N
    problemOS << "-n " << inShape[iLayoutMap["ni"]] << sep;
    // C
    problemOS << "-c " << inShape[iLayoutMap["ci"]] * inShape[iLayoutMap["gi"]]
              << sep;
    // H
    problemOS << "-H " << inShape[iLayoutMap["0i"]] << sep;
    // W
    problemOS << "-W " << inShape[iLayoutMap["1i"]] << sep;
    // K
    problemOS << "-k " << filShape[fLayoutMap["k"]] * filShape[fLayoutMap["g"]]
              << sep;
    // Y
    problemOS << "-y " << filShape[fLayoutMap["0"]] << sep;
    // X
    problemOS << "-x " << filShape[fLayoutMap["1"]] << sep;

    auto paddingVal =
        extractFromIntegerArrayAttr<int64_t>(convGemmOp.getPadding());
    auto strideVal =
        extractFromIntegerArrayAttr<int64_t>(convGemmOp.getStrides());
    auto dilationVal =
        extractFromIntegerArrayAttr<int64_t>(convGemmOp.getDilations());

    // padding
    problemOS << "-p " << paddingVal[0] << " -q " << paddingVal[2] << sep;
    // stride
    problemOS << "-u " << strideVal[0] << " -v " << strideVal[1] << sep;
    // dilation
    problemOS << "-l " << dilationVal[0] << " -j " << dilationVal[1] << sep;
    // group
    problemOS << "-g " << inShape[iLayoutMap["gi"]] << sep;
    problemOS << "-gemmO " << headDimV;
  } else {
    problemOS << "-m " << seqLenQ << sep;
    problemOS << "-n " << seqLenK << sep;
    problemOS << "-k " << headDimQK << sep;
    problemOS << "-gemmO " << headDimV;
  }
  return success();
}

LogicalResult
mlir::rock::serializeTuningProblem(RockGemmWrapperInterface gemmIF,
                                   TuningProblemFormat format,
                                   SmallVectorImpl<char> &out) {
  // See the sibling overload for what the quick-tuning key leaves out.
  const bool isPerfDb = format == TuningProblemFormat::PerfDb;
  int64_t numCU = rock::getNumCUValue(gemmIF);
  int64_t numChiplets = rock::getNumChipletsValue(gemmIF);
  constexpr char sep = ' ';
  constexpr char tab = '\t';
  llvm::raw_svector_ostream problemOS(out);

  KernelType opType = gemmIF.getKernelType();
  Operation *gemmOp = gemmIF.getOperation();

  auto f8TypeStr = [](const Type &type) -> std::optional<StringLiteral> {
    if (isa<Float8E4M3FNUZType, Float8E4M3FNType>(type))
      return StringLiteral("fp8");
    if (isa<Float8E5M2FNUZType, Float8E5M2Type>(type))
      return StringLiteral("bf8");
    return std::nullopt;
  };

  if (isPerfDb) {
    // ARCH string
    problemOS << StringRef(rock::getArchValue(gemmIF)).trim("\"") << tab;
    // Number of Compute Units
    problemOS << numCU << tab;
    // Number of chiplets
    problemOS << numChiplets << tab;
  } else {
    problemOS << stringifyEnum(opType) << sep;
  }

  if (opType == KernelType::Conv ||
      opType == KernelType::ConvBwdData) { // conv cases
    RockConvInterface convIF = dyn_cast<RockConvInterface>(gemmOp);

    ShapedType inType = convIF.getConvInput().getType();
    ArrayRef<int64_t> inShape = inType.getShape();
    ShapedType filType = convIF.getConvFilter().getType();
    ArrayRef<int64_t> filShape = filType.getShape();

    // Extract layout information
    llvm::StringMap<unsigned> fLayoutMap, iLayoutMap, oLayoutMap;
    SmallString<6> fLayout, iLayout, oLayout;
    if (failed(extractLayouts(gemmOp, fLayoutMap, iLayoutMap, oLayoutMap,
                              fLayout, iLayout, oLayout))) {
      if (!isPerfDb)
        return failure();
      return convIF.emitError("layout can't be extracted");
    }

    // Please keep these in sync with mlir/utils/performance/perfRunner.py

    // OP datatype. This token names the operation as well as its precision, so
    // the quick-tuning key, which omits it, leans on its explicit kernel type
    // to tell a convolution apart from anything else.
    Type inElemType = inType.getElementType();
    Type filElemType = filType.getElementType();
    SmallString<16> opTypeStr;
    if (inElemType.isF32()) {
      opTypeStr = "conv";
    } else if (inElemType.isF16()) {
      opTypeStr = "convfp16";
    } else if (inElemType.isBF16()) {
      opTypeStr = "convbfp16";
    } else if (inElemType.isInteger(8)) {
      opTypeStr = "convint8";
    } else {
      auto inString = f8TypeStr(inElemType);
      auto filString = f8TypeStr(filElemType);
      if (!inString || !filString)
        return failure();
      opTypeStr = llvm::formatv("conv{0}_{1}", *inString, *filString).str();
    }
    if (isPerfDb)
      problemOS << opTypeStr << sep;

    // OP direction
    switch (opType) {
    case KernelType::Conv:
      problemOS << "-F 1" << sep;
      break;
    case KernelType::ConvBwdData:
      problemOS << "-F 2" << sep;
      break;
    default:
      return failure();
    }

    // filter layout
    problemOS << "-f " << fLayout << sep;
    // input layout
    problemOS << "-I " << iLayout << sep;
    // output layout
    problemOS << "-O " << oLayout << sep;
    // N
    problemOS << "-n " << inShape[iLayoutMap["ni"]] << sep;
    // C
    problemOS << "-c " << inShape[iLayoutMap["ci"]] * inShape[iLayoutMap["gi"]]
              << sep;
    // H
    problemOS << "-H " << inShape[iLayoutMap["0i"]] << sep;
    // W
    problemOS << "-W " << inShape[iLayoutMap["1i"]] << sep;
    // K
    problemOS << "-k " << filShape[fLayoutMap["k"]] * filShape[fLayoutMap["g"]]
              << sep;
    // Y
    problemOS << "-y " << filShape[fLayoutMap["0"]] << sep;
    // X
    problemOS << "-x " << filShape[fLayoutMap["1"]] << sep;

    auto paddingVal = extractFromIntegerArrayAttr<int64_t>(convIF.getPadding());
    auto strideVal = extractFromIntegerArrayAttr<int64_t>(convIF.getStrides());
    auto dilationVal =
        extractFromIntegerArrayAttr<int64_t>(convIF.getDilations());
    // padding
    problemOS << "-p " << paddingVal[0] << " -q " << paddingVal[2] << sep;
    // stride
    problemOS << "-u " << strideVal[0] << " -v " << strideVal[1] << sep;
    // dilation
    problemOS << "-l " << dilationVal[0] << " -j " << dilationVal[1] << sep;
    // group
    problemOS << "-g " << inShape[iLayoutMap["gi"]] << sep;

  } else if (opType == KernelType::Gemm) { // gemm case
    rock::GemmOp rGemmOp = dyn_cast<rock::GemmOp>(gemmOp);
    bool isScaledGemm =
        rGemmOp.getScaleA() != nullptr && rGemmOp.getScaleB() != nullptr;
    // Please keep these in sync with mlir/utils/performance/perfRunner.py
    // Data type
    SmallString<16> dataTypeStr;
    Type elemTypeA = gemmIF.getAType(), elemTypeB = gemmIF.getBType();
    if (elemTypeA.isF32() && elemTypeB.isF32()) {
      dataTypeStr = "f32";
    } else if (elemTypeA.isF16() && elemTypeB.isF16()) {
      dataTypeStr = "f16";
    } else if (elemTypeA.isBF16() && elemTypeB.isBF16()) {
      dataTypeStr = "bf16";
    } else if (elemTypeA.isInteger(8) && elemTypeB.isInteger(8)) {
      dataTypeStr = "i8";
    } else if (isa<Float4E2M1FNType>(elemTypeA) &&
               isa<Float4E2M1FNType>(elemTypeB)) {
      dataTypeStr = "f4E2M1FN";
    } else {
      auto aString = f8TypeStr(elemTypeA);
      auto bString = f8TypeStr(elemTypeB);
      if (!aString || !bString)
        return failure();
      dataTypeStr = llvm::formatv("{0}_{1}", *aString, *bString).str();
    }
    if (isPerfDb)
      problemOS << "-t " << dataTypeStr;

    // Output datatype
    Type outType = gemmIF->getResult(0).getType();
    Type elemTypeC;
    if (auto shapedType = dyn_cast<ShapedType>(outType))
      elemTypeC = shapedType.getElementType();
    else
      elemTypeC = outType;
    if (isPerfDb) {
      problemOS << " -out_datatype ";
      auto outStr = f8TypeStr(elemTypeC);
      if (outStr)
        problemOS << *outStr << sep;
      else
        problemOS << elemTypeC << sep;
    }

    // TransA
    problemOS << "-transA ";
    if (rGemmOp.getATransposed())
      problemOS << "true ";
    else
      problemOS << "false ";

    // TransB
    problemOS << "-transB ";
    if (rGemmOp.getBTransposed())
      problemOS << "true ";
    else
      problemOS << "false ";

    // TransO
    problemOS << "-transO ";
    if (rGemmOp.getOTransposed())
      problemOS << "true ";
    else
      problemOS << "false ";

    if (isScaledGemm) {
      problemOS << "-scaledGemm" << sep;
      auto scaleA = rGemmOp.getScaleA();
      auto scaleB = rGemmOp.getScaleB();
      // The scale element types stay in the quick-tuning key: unlike the
      // operand and result types, no shard selects on them.
      problemOS << "-scale_a_dtype ";
      auto scaleAElemType = scaleA.getType().getElementType();
      auto scaleBElemType = scaleB.getType().getElementType();
      if (scaleAElemType.isF32()) {
        problemOS << "f32";
      } else if (isa<Float8E8M0FNUType>(scaleAElemType)) {
        problemOS << "f8E8M0FNU";
      } else {
        llvm_unreachable("Unsupported scale A element type");
      }
      problemOS << sep;
      problemOS << "-scale_b_dtype ";
      if (scaleBElemType.isF32()) {
        problemOS << "f32";
      } else if (isa<Float8E8M0FNUType>(scaleBElemType)) {
        problemOS << "f8E8M0FNU";
      } else {
        llvm_unreachable("Unsupported scale B element type");
      }
      problemOS << sep;
      problemOS << "-transScaleA" << sep;
      if (rGemmOp.getAScaleTransposed()) {
        problemOS << "true" << sep;
      } else {
        problemOS << "false" << sep;
      }
      problemOS << "-transScaleB" << sep;
      if (rGemmOp.getBScaleTransposed()) {
        problemOS << "true" << sep;
      } else {
        problemOS << "false" << sep;
      }
    }

    // Gemmsize G/M/N/K
    problemOS << "-g " << gemmIF.getGemmSize().g << sep;
    problemOS << "-m " << gemmIF.getGemmSize().m << sep;
    problemOS << "-n " << gemmIF.getGemmSize().n << sep;
    problemOS << "-k " << gemmIF.getGemmSize().k << sep;
  } else {
    // Unknown op type, unreachable.
    return failure();
  }

  while (out.back() == sep) {
    // remove trailing whitespace
    out.pop_back();
  }

  return success();
}

uint64_t mlir::rock::hashQuickTuningProblemKey(StringRef key) {
  return llvm::xxh3_64bits(key);
}

// The buffer is sized for the longest key we emit, an attention problem with a
// sliding window; anything longer just spills to the heap.
using ProblemKeyStorage = SmallString<256>;

FailureOr<uint64_t>
mlir::rock::getQuickTuningProblemHash(RockGemmWrapperInterface gemmIF) {
  ProblemKeyStorage key;
  if (failed(serializeTuningProblem(gemmIF, TuningProblemFormat::QuickTuningKey,
                                    key)))
    return failure();
  return hashQuickTuningProblemKey(key);
}

FailureOr<uint64_t>
mlir::rock::getQuickTuningProblemHash(RockGemmGemmWrapperInterface gemmGemmOp) {
  ProblemKeyStorage key;
  if (failed(serializeTuningProblem(gemmGemmOp,
                                    TuningProblemFormat::QuickTuningKey, key)))
    return failure();
  return hashQuickTuningProblemKey(key);
}

FailureOr<uint64_t> mlir::rock::getQuickTuningProblemHash(ModuleOp mod) {
  return visitPrimaryTuningOp<FailureOr<uint64_t>>(
      mod, [](auto op) { return getQuickTuningProblemHash(op); });
}
