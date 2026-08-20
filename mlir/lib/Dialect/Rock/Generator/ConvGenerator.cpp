// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "mlir/Dialect/Rock/Generator/ConvGenerator.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/RocmDeviceName.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <functional>
#include <numeric>

using namespace mlir;
using namespace mlir::rock;

#define DEBUG_TYPE "conv2d-gen"

ConvGenerator::ConvGenerator(
    const std::string &arch, const std::string &chip,
    bool disableSplitKForTuning, const std::string &triple,
    const std::string &chipFeatures, const std::string &perfConfig,
    std::optional<int> num_cu, std::optional<int> num_chiplets,
    const std::optional<ConvOpType> operation,
    const std::string &filterDataTypeStr, const std::string &inputDataTypeStr,
    const std::string &outputDataTypeStr, ArrayRef<int> dilations,
    ArrayRef<int> strides, ArrayRef<int> paddingLeft,
    ArrayRef<int> paddingRight, const std::string &filterLayout,
    const std::string &inputLayout, const std::string &outputLayout,
    const std::string &kernelBaseName)
    : config{arch,
             chip,
             disableSplitKForTuning,
             triple,
             chipFeatures,
             perfConfig,
             num_cu,
             num_chiplets,
             operation,
             filterDataTypeStr,
             inputDataTypeStr,
             outputDataTypeStr,
             {dilations.begin(), dilations.end()},
             {strides.begin(), strides.end()},
             {paddingLeft.begin(), paddingLeft.end()},
             {paddingRight.begin(), paddingRight.end()},
             filterLayout,
             inputLayout,
             outputLayout,
             kernelBaseName,
             {},
             {},
             {},
             {}} {}

ConvGenerator::ConvGenerator(const ConvGenerator::Config &_config)
    : config(_config) {}

static void strToTokens(const std::string &arguments,
                        std::map<std::string, std::string> &argMap) {
  std::istringstream iss(arguments);
  std::string token;
  std::string argKey;
  while (iss >> token) {
    auto pos = token.find("--");
    if (pos != std::string::npos) {
      argKey = token.substr(pos + 2);
    } else {
      if (!argKey.empty()) {
        argMap[argKey] = token;
        argKey.clear();
      }
    }
  }
}

static llvm::StringMap<int64_t> canonicalizeDims(const ArrayRef<int64_t> dims,
                                                 const StringRef layout) {
  llvm::StringMap<int64_t> ret;
  for (const auto &[keych, dim] : llvm::zip(layout, dims)) {
    StringRef key(&keych, 1);
    ret.insert_or_assign(key, dim);
  }
  return ret;
}

static LogicalResult hasDimensions(const llvm::StringMap<int64_t> &map,
                                   const StringRef wantedLayout,
                                   const StringRef operation) {
  for (size_t i = 0; i < wantedLayout.size(); ++i) {
    auto key = wantedLayout.slice(i, i + 1);
    if (map.count(key) == 0) {
      LLVM_DEBUG(llvm::dbgs() << "Layout for " << operation
                              << " tensor missing dimension: " << key << "\n");
      return failure();
    }
  }
  return success();
}

LogicalResult ConvGenerator::isApplicable() const {
  if (failed(hasValidDimension())) {
    return failure();
  }

  return success();
}

LogicalResult ConvGenerator::hasValidDimension() const {
  if (any_of(
          llvm::concat<const int64_t>(config.dilationDims, config.strideDims),
          [](const int64_t &a) { return a <= 0; })) {
    LLVM_DEBUG(llvm::dbgs()
               << "Dilation and stride must be a positive integer\n");
    return failure();
  }

  if (any_of(llvm::concat<const int64_t>(config.paddingLeftDims,
                                         config.paddingRightDims),
             [](const int64_t &a) { return a < 0; })) {
    LLVM_DEBUG(llvm::dbgs() << "Padding values cannot be negative\n");
    return failure();
  }

  static const llvm::StringMap<size_t> typeWidths{
      {"f32", sizeof(float)},     {"fp32", sizeof(float)},
      {"fp16", sizeof(uint16_t)}, {"f16", sizeof(uint16_t)},
      {"bf16", sizeof(uint16_t)}, {"i8", sizeof(int8_t)},
      {"fp8", sizeof(uint8_t)},   {"bf8", sizeof(int8_t)}};

  for (const std::string &type :
       {config.filterDataTypeStr, config.inputDataTypeStr,
        config.outputDataTypeStr}) {
    if (typeWidths.count(type) == 0) {
      LLVM_DEBUG(llvm::dbgs() << type << " is not a known datatype\n");
    }
  }

  auto checkDimSizes = [](const ArrayRef<int64_t> dims) -> bool {
    return all_of(dims, [](const int64_t &a) { return a > 0; });
  };

  if (!checkDimSizes(config.inputDimension)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Input tensor dimensions must be strictly positive\n");
    return failure();
  }
  if (!checkDimSizes(config.filterDimension)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Filter tensor dimensions must be strictly positive\n");
  }
  if (!checkDimSizes(config.outputDimension)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Output tensor dimensions must be strictly positive\n");
    return failure();
  }

  auto inDim = canonicalizeDims(config.inputDimension, config.inputLayout);
  auto filDim = canonicalizeDims(config.filterDimension, config.filterLayout);
  auto outDim = canonicalizeDims(config.outputDimension, config.outputLayout);

  // Note: hasDimensions() prints error messages
  if (failed(hasDimensions(inDim, "ngc01", "input")) ||
      failed(hasDimensions(filDim, "gkc01", "filter")) ||
      failed(hasDimensions(outDim, "ngk01", "output"))) {
    return failure();
  }

  if (inDim["n"] != outDim["n"]) {
    LLVM_DEBUG(llvm::dbgs() << "Input and output batch sizes don't match\n");
    return failure();
  }
  if (inDim["g"] != outDim["g"] || inDim["g"] != filDim["g"]) {
    LLVM_DEBUG(llvm::dbgs() << "Group sizes are not consistent between input, "
                               "output, and filter\n");
    return failure();
  }
  if (inDim["c"] != filDim["c"]) {
    LLVM_DEBUG(llvm::dbgs()
               << "Number of channels in input doesn't match number of "
                  "channels in filter\n");
    return failure();
  }
  if (filDim["k"] != outDim["k"]) {
    LLVM_DEBUG(llvm::dbgs()
               << "Number of channels in output doesn't match number of "
                  "channels in filter\n");
    return failure();
  }

  assert(config.strideDims.size() == config.dilationDims.size() &&
         config.strideDims.size() == config.paddingLeftDims.size() &&
         config.strideDims.size() == config.paddingRightDims.size());

  for (size_t i = 0; i < config.strideDims.size(); i++) {
    auto ii = std::to_string(i);
    int64_t expected =
        outputDim(inDim[ii], filDim[ii], config.paddingLeftDims[i],
                  config.paddingRightDims[i], config.strideDims[i],
                  config.dilationDims[i]);
    if (outDim[ii] != expected) {
      LLVM_DEBUG(llvm::dbgs() << "Output dimension " << i << " " << outDim[ii]
                              << " doesn't match " << expected
                              << " computed from other parameters\n");
      return failure();
    }
  }

  for (size_t i = 0; i < config.paddingLeftDims.size(); i++) {
    auto ii = std::to_string(i);
    if (inDim[ii] + config.paddingLeftDims[i] + config.paddingRightDims[i] <
        filDim[ii]) {
      LLVM_DEBUG(llvm::dbgs() << "Input, including padding, is smaller than "
                                 "the filter in dimension "
                              << i << "\n");
      return failure();
    }
  }

  return success();
}

static Type strToType(StringRef dataTypeStr, OpBuilder &builder) {
  std::optional<Type> type =
      llvm::StringSwitch<std::optional<Type>>(dataTypeStr)
          .Case("f32", builder.getF32Type())
          .Case("fp32", builder.getF32Type())
          .Case("f16", builder.getF16Type())
          .Case("fp16", builder.getF16Type())
          .Case("bf16", builder.getBF16Type())
          .Case("i32", builder.getI32Type())
          .Case("i8", builder.getI8Type())
          .Case("f8E5M2", builder.getType<Float8E5M2Type>())
          .Case("f8E4M3FN", builder.getType<Float8E4M3FNType>())
          .Case("f8E5M2FNUZ", builder.getType<Float8E5M2FNUZType>())
          .Case("f8E4M3FNUZ", builder.getType<Float8E4M3FNUZType>())
          .Default(std::nullopt);
  if (!type) {
    llvm::errs() << "Unknown data type: " << dataTypeStr << "\n";
    exit(1);
  }
  return *type;
}

Type ConvGenerator::getFilterDataType(OpBuilder &builder) const {
  if (config.filterDataTypeStr.empty())
    return getInputDataType(builder);
  return strToType(config.filterDataTypeStr, builder);
}

Type ConvGenerator::getInputDataType(OpBuilder &builder) const {
  return strToType(config.inputDataTypeStr, builder);
}

Type ConvGenerator::getOutputDataType(OpBuilder &builder) const {
  if (config.outputDataTypeStr.empty()) {
    // Special-case i8 -> i32 translation as a default
    Type inType = getInputDataType(builder);
    if (inType.isInteger(8))
      return builder.getIntegerType(32);
  }
  return strToType(config.outputDataTypeStr, builder);
}

LogicalResult ConvGenerator::needExtraPadBwdWeight(OpBuilder &builder,
                                                   bool &needExtraPad) const {
  Type dataType = getInputDataType(builder);
  ConvOpType dir = config.operation.value();
  assert(dir == ConvOpType::BwdWeight &&
         "This method should only be called for wrw ops");

  ConvolutionDims convDims = getConvolutionDims(&config);
  GemmSize gemmSize = GemmSize::fromConvolution(dir, convDims);

  needExtraPad = false;
  // TODO: support mixed-type fp8 here too.
  PopulateParamsInfo info{/*gemmSize=*/gemmSize,
                          /*arch*=*/config.arch,
                          /*gemmAType=*/dataType,
                          /*gemmBType=*/dataType,
                          /*kernelType=*/KernelType::ConvBwdWeight,
                          /*batchSize=*/convDims.n,
                          /*numCU=*/getNumCU()};

  auto populateParamsPtr = std::make_unique<PopulateParams>();
  auto maybeValidParams = populateParamsPtr->obtainTuningParameters(
      builder, info, config.perfConfig);
  if (succeeded(maybeValidParams)) {
    needExtraPad = (populateParamsPtr->calculatePaddingAmount(
                        maybeValidParams.value(), gemmSize) != 0);
    return success();
  }
  return failure();
}

uint32_t ConvGenerator::getNumCU() const {
  return config.num_cu.has_value() ? config.num_cu.value()
                                   : rock::getMinNumCU(config.arch);
}

int64_t ConvGenerator::getNumChiplets() const {
  return config.num_chiplets.has_value()
             ? config.num_chiplets.value()
             : rock::inferNumChiplets(config.arch, getNumCU());
}

LogicalResult ConvGenerator::parseConvConfig(OpBuilder &builder,
                                             const char *arguments) {
  std::map<std::string, std::string> argMap;
  strToTokens(arguments, argMap);

  auto isValid = [&argMap]() {
    // only require tensor configs
    static std::vector<std::string> validKeys = {
        "batchsize",   "groupsize",    "in_layout", "in_type",
        "in_channels", "in_h",         "in_w",      "out_layout",
        "out_type",    "out_channels", "out_h",     "out_w",
        "fil_layout",  "fil_type",     "fil_w",     "fil_h"};
    if (argMap["in_layout"].length() > 5) { // Ie, 3-D.
      validKeys.push_back("in_d");
      validKeys.push_back("out_d");
      validKeys.push_back("fil_d");
    }
    auto isPresent = [&argMap](const std::string &key) {
      return argMap.count(key) > 0;
    };
    if (!llvm::all_of(validKeys, isPresent)) {
      return false;
    }
    return (argMap["fil_layout"].length() == argMap["in_layout"].length()) &&
           (argMap["in_layout"].length() == argMap["out_layout"].length());
  };

  // Proceed only if we have a valid argMap. Otherwise leave the handle to be
  // empty
  if (!isValid()) {
    return failure();
  }

  auto strToLong = [&argMap](const std::string &argKey) {
    return std::stol(argMap[argKey]);
  };

  auto strToInt = [&argMap](const std::string &key, auto &setting) {
    if (argMap.find(key) != argMap.end()) {
      setting = std::stoi(argMap[key]);
    }
  };

  auto strToStr = [&argMap](const std::string &key, std::string &setting) {
    if (argMap.find(key) != argMap.end()) {
      setting = argMap[key];
    }
  };

  std::string arch;
  strToStr(rock::ArchAttr::getMnemonic().str(), arch);
  RocmDeviceName splitter;
  if (failed(splitter.parse(arch))) {
    return failure();
  }
  // Canonicalize architecture name
  SmallString<64> canonicalArch;
  splitter.getFullName(canonicalArch);
  arch = canonicalArch.str();

  config.arch = arch;
  config.chip = splitter.getChip().str();
  config.chipFeatures = splitter.getFeaturesForBackend();
  config.triple = splitter.getTriple().str();

  FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(config.chip);
  if (failed(maybeChipset)) {
    emitError(UnknownLoc::get(builder.getContext()),
              "Invalid chipset name: " + config.chip);
    exit(1);
  }

  strToStr("perf_config", config.perfConfig);
  strToInt(rock::NumCUAttr::getMnemonic().str(), config.num_cu);
  strToInt(rock::NumChipletsAttr::getMnemonic().str(), config.num_chiplets);

  // conv settings
  auto const op = getConvOpTypeForName(argMap["operation"]);
  if (!op.has_value()) {
    return failure();
  }

  auto canonicalizeDataType = [&](const std::string &type) {
    if (type == "fp32")
      return std::string("f32");
    if (type == "fp16")
      return std::string("f16");
    if (type == "fp8") {
      if (amdgpu::hasOcpFp8(maybeChipset.value()))
        return std::string("f8E4M3FN");
      return std::string("f8E4M3FNUZ");
    }
    if (type == "bf8") {
      if (amdgpu::hasOcpFp8(maybeChipset.value()))
        return std::string("f8E5M2");
      return std::string("f8E5M2FNUZ");
    }
    return type;
  };
  config.operation = op.value();
  config.filterDataTypeStr = canonicalizeDataType(argMap["fil_type"]);
  config.inputDataTypeStr = canonicalizeDataType(argMap["in_type"]);
  config.outputDataTypeStr = canonicalizeDataType(argMap["out_type"]);
  strToInt("dilation_h", config.dilationDims[DIM::HEIGHT]);
  strToInt("dilation_w", config.dilationDims[DIM::WIDTH]);
  if (config.dilationDims.size() > DIM::DEPTH)
    strToInt("dilation_d", config.dilationDims[DIM::DEPTH]);
  strToInt("conv_stride_h", config.strideDims[DIM::HEIGHT]);
  strToInt("conv_stride_w", config.strideDims[DIM::WIDTH]);
  if (config.strideDims.size() > DIM::DEPTH)
    strToInt("conv_stride_d", config.strideDims[DIM::DEPTH]);
  strToInt("padding_h", config.paddingLeftDims[DIM::HEIGHT]);
  strToInt("padding_h", config.paddingRightDims[DIM::HEIGHT]);
  strToInt("padding_w", config.paddingLeftDims[DIM::WIDTH]);
  strToInt("padding_w", config.paddingRightDims[DIM::WIDTH]);
  if (config.paddingLeftDims.size() > DIM::DEPTH)
    strToInt("padding_d", config.paddingLeftDims[DIM::DEPTH]);
  if (config.paddingRightDims.size() > DIM::DEPTH)
    strToInt("padding_d", config.paddingRightDims[DIM::DEPTH]);

  strToStr("kernel_name", config.kernelBaseName);

  // Allow only fwd direction for 8-bit types. Reject other directions.
  if (config.operation.value() != ConvOpType::Fwd &&
      (config.inputDataTypeStr == "i8" || config.inputDataTypeStr == "fp8" ||
       config.inputDataTypeStr == "bf8")) {
    return failure();
  }

  // Rock has NCHW as layout string for all three tensors
  config.inputLayout = translateLayout(
      argMap["in_layout"], std::string("NGCHWD012"), std::string("ngchwd012"));
  config.filterLayout = translateLayout(
      argMap["fil_layout"], std::string("GNCHWD012"), std::string("gkcyxz012"));
  config.outputLayout = translateLayout(
      argMap["out_layout"], std::string("NGCHWD012"), std::string("ngkhwd012"));

  // Determine tensor dimensions.
  SmallVector<int64_t> inDims{strToLong("in_h"), strToLong("in_w")};
  if (argMap.count("in_d") > 0)
    inDims.push_back(strToLong("in_d"));
  SmallVector<int64_t> outDims{strToLong("out_h"), strToLong("out_w")};
  if (argMap.count("out_d") > 0)
    outDims.push_back(strToLong("out_d"));
  SmallVector<int64_t> filDims{strToLong("fil_h"), strToLong("fil_w")};
  if (argMap.count("fil_d") > 0)
    filDims.push_back(strToLong("fil_d"));
  auto status = parseConvDims(strToLong("batchsize"), strToLong("groupsize"),
                              strToLong("in_channels"), inDims,
                              strToLong("out_channels"), outDims, filDims);

  if (status.failed()) {
    return failure();
  }

  return success();
}

LogicalResult ConvGenerator::parseConvDims(int64_t batchSize, int64_t groupSize,
                                           int64_t inputChannel,
                                           ArrayRef<int64_t> inputDims,
                                           int64_t outputChannel,
                                           ArrayRef<int64_t> outputDims,
                                           ArrayRef<int64_t> filterDims) {

  if (outputChannel % groupSize != 0 || inputChannel % groupSize != 0)
    return failure();

  config.filterDims.clear();
  for (auto dim : filterDims)
    config.filterDims.push_back(dim);

  llvm::StringMap<int64_t> filterMap = {{"k", outputChannel / groupSize},
                                        {"g", groupSize},
                                        {"c", inputChannel / groupSize},
                                        {"y", filterDims[0]},
                                        {"x", filterDims[1]}};
  for (size_t i = 0; i < filterDims.size(); i++)
    filterMap[std::to_string(i)] = filterDims[i];

  llvm::StringMap<int64_t> inputMap = {{"n", batchSize},
                                       {"g", groupSize},
                                       {"c", inputChannel / groupSize},
                                       {"h", inputDims[0]},
                                       {"w", inputDims[1]}};
  for (size_t i = 0; i < inputDims.size(); i++)
    inputMap[std::to_string(i)] = inputDims[i];

  llvm::StringMap<int64_t> outputMap = {{"n", batchSize},
                                        {"g", groupSize},
                                        {"k", outputChannel / groupSize},
                                        {"h", outputDims[0]},
                                        {"w", outputDims[1]}};
  for (size_t i = 0; i < outputDims.size(); i++)
    outputMap[std::to_string(i)] = outputDims[i];

  auto convertLayout = [](char &key, llvm::StringMap<int64_t> &kmap,
                          auto &dims) {
    auto keyl = std::string{static_cast<char>(std::tolower(key))};
    if (!kmap.contains(keyl) && !isdigit(key)) {
      keyl = "k";
      if (!kmap.contains(keyl))
        return false;
    }
    dims.push_back(kmap[keyl]);
    key = keyl[0];
    return true;
  };

  size_t layoutLen = config.filterLayout.length();
  if (layoutLen != config.inputLayout.length() ||
      layoutLen != config.outputLayout.length()) {
    return failure();
  }
  // Determine dimensions.
  for (size_t i = 0; i < layoutLen; ++i) {
    if (!convertLayout(config.filterLayout[i], filterMap,
                       config.filterDimension)) {
      return failure();
    }
    if (!convertLayout(config.inputLayout[i], inputMap,
                       config.inputDimension)) {
      return failure();
    }
    if (!convertLayout(config.outputLayout[i], outputMap,
                       config.outputDimension)) {
      return failure();
    }
  }

  // Determine kernel name, if there isn't one.
  if (config.kernelBaseName.empty()) {
    assert(config.operation.has_value());
    auto opType = config.operation.value();
    config.kernelBaseName = std::string("rock_") +
                            getNameForConvOpType(opType).str() + "_" +
                            config.filterLayout + "_" + config.inputLayout +
                            "_" + config.outputLayout;
  }

  return success();
}

void ConvGenerator::setDataTypes(const std::string &dataTypeStr) {
  config.filterDataTypeStr = config.inputDataTypeStr =
      config.outputDataTypeStr = dataTypeStr;
}

void ConvGenerator::setPerfConfig(StringRef perfConfig) {
  config.perfConfig = perfConfig.str();
}

ConvolutionDims ConvGenerator::getConvolutionDims(const Config *config) {
  auto inDim = canonicalizeDims(config->inputDimension, config->inputLayout);
  auto filDim = canonicalizeDims(config->filterDimension, config->filterLayout);
  auto outDim = canonicalizeDims(config->outputDimension, config->outputLayout);

  SmallVector<int64_t> inDims;
  for (size_t i = 0; i < config->inputLayout.size() - 3; i++)
    inDims.push_back(inDim[std::to_string(i)]);
  SmallVector<int64_t> filDims;
  for (size_t i = 0; i < config->filterLayout.size() - 3; i++)
    filDims.push_back(filDim[std::to_string(i)]);
  SmallVector<int64_t> outDims;
  for (size_t i = 0; i < config->outputLayout.size() - 3; i++)
    outDims.push_back(outDim[std::to_string(i)]);

  return ConvolutionDims(filDims, outDims, inDims, filDim["k"], filDim["c"],
                         inDim["n"], inDim["g"]);
}

// Helper function to zero-initialize an argument of a FuncOp using the
// prefill attribute
static void zeroInitArg(OpBuilder &builder, func::FuncOp func,
                        unsigned argIndex) {
  auto argToPrefill = func.getArgument(argIndex);
  auto attrName = rock::PrefillAttr::getMnemonic();
  auto elementType = getElementTypeOrSelf(argToPrefill.getType());
  Attribute zero;
  if (isa<FloatType>(elementType)) {
    zero = builder.getFloatAttr(elementType, 0.0);
  } else {
    assert(isa<IntegerType>(elementType) &&
           "Unsupported element type for prefill attribute");
    zero = builder.getIntegerAttr(elementType, 0);
  }
  func.setArgAttrs(argToPrefill.getArgNumber(),
                   builder.getNamedAttr(attrName, zero));
}

LogicalResult ConvGenerator::genConvModule(ModuleOp &module, bool isVerifier,
                                           bool ignoreTuning) {
  OpBuilder builder(module.getContext());

  Type filterDataType = getFilterDataType(builder);
  Type inputDataType = getInputDataType(builder);
  Type outputDataType = getOutputDataType(builder);
  if (!filterDataType || !inputDataType || !outputDataType)
    return failure();

  // Construct a new FuncOp with tensor types.
  auto filterArgType =
      RankedTensorType::get(ArrayRef<int64_t>(config.filterDimension.begin(),
                                              config.filterDimension.end()),
                            filterDataType);
  auto inputArgType =
      RankedTensorType::get(ArrayRef<int64_t>(config.inputDimension.begin(),
                                              config.inputDimension.end()),
                            inputDataType);
  auto outputArgType =
      RankedTensorType::get(ArrayRef<int64_t>(config.outputDimension.begin(),
                                              config.outputDimension.end()),
                            outputDataType);

  // Build argument types in standard [filter, input, output] order, then
  // reorder so the store destination (the result of the conv) is always the
  // last argument. This simplifies downstream passes and host code that
  // assume outputs are at the end of the kernel argument list.
  SmallVector<Type, 3> logicalFuncArgTypes = {filterArgType, inputArgType,
                                              outputArgType};
  reorderConvArgsForKernel(config.operation.value(), logicalFuncArgTypes);
  unsigned storeDestIdx = logicalFuncArgTypes.size() - 1;

  SmallVector<Type, 3> physicalFuncArgTypes =
      llvm::map_to_vector(logicalFuncArgTypes, getFlattenedType);
  Type resultFlatType = physicalFuncArgTypes[storeDestIdx];
  auto funcType =
      builder.getFunctionType(physicalFuncArgTypes, {resultFlatType});

  std::string kernelName = config.kernelBaseName;
  if (isVerifier) {
    kernelName += "_ver";
  }

  func::FuncOp func = module.lookupSymbol<func::FuncOp>(kernelName);
  if (func) {
    assert(func.isDeclaration());
    func.erase();
  }

  StringAttr archStrAttr = builder.getStringAttr(config.arch);
  NamedAttribute archAttr =
      builder.getNamedAttr(rock::ArchAttr::getMnemonic(), archStrAttr);
  IntegerAttr numCUIntAttr =
      builder.getIntegerAttr(builder.getI32Type(), getNumCU());
  NamedAttribute numCUAttr =
      builder.getNamedAttr(rock::NumCUAttr::getMnemonic(), numCUIntAttr);
  IntegerAttr numChipletsIntAttr =
      builder.getIntegerAttr(builder.getI64Type(), getNumChiplets());
  NamedAttribute numChipletsAttr = builder.getNamedAttr(
      rock::NumChipletsAttr::getMnemonic(), numChipletsIntAttr);

  SmallVector<NamedAttribute, 2> kernelAttrs = {
      builder.getNamedAttr(rock::KernelAttr::getMnemonic(),
                           builder.getUnitAttr()),
      archAttr, numCUAttr, numChipletsAttr};

  // Construct the FuncOp.
  func = func::FuncOp::create(builder.getUnknownLoc(), kernelName, funcType,
                              ArrayRef<NamedAttribute>(kernelAttrs));
  if (!config.disableSplitKForTuning) {
    func->setAttr(rock::EnableSplitKForTuningAttr::getMnemonic(),
                  builder.getUnitAttr());
  }
  module.push_back(func);
  if (func.getName() != kernelName) {
    return failure();
  }
  kernelFunc = func;

  // Construct a new Block.
  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  // Construct a new ConvOp.
  SmallVector<StringAttr, 5> filterLayoutSpec;
  SmallVector<StringAttr, 5> inputLayoutSpec;
  SmallVector<StringAttr, 5> outputLayoutSpec;
  for (auto &key : config.filterLayout)
    filterLayoutSpec.push_back(builder.getStringAttr(StringRef(&key, 1)));
  for (auto &key : config.inputLayout)
    inputLayoutSpec.push_back(builder.getStringAttr(StringRef(&key, 1) + "i"));
  for (auto &key : config.outputLayout)
    outputLayoutSpec.push_back(builder.getStringAttr(StringRef(&key, 1) + "o"));

  std::vector<NamedAttribute> attributes{
      builder.getNamedAttr(
          "filter_layout",
          builder.getArrayAttr(ArrayRef<Attribute>(filterLayoutSpec.begin(),
                                                   filterLayoutSpec.end()))),
      builder.getNamedAttr(
          "input_layout", builder.getArrayAttr(ArrayRef<Attribute>(
                              inputLayoutSpec.begin(), inputLayoutSpec.end()))),
      builder.getNamedAttr(
          "output_layout",
          builder.getArrayAttr(ArrayRef<Attribute>(outputLayoutSpec.begin(),
                                                   outputLayoutSpec.end()))),
  };

  SmallVector<int64_t, 8> paddingArray;
  for (const auto &[left, right] :
       zip(config.paddingLeftDims, config.paddingRightDims)) {
    paddingArray.push_back(left);
    paddingArray.push_back(right);
  }

  attributes.push_back(
      builder.getNamedAttr("padding", builder.getIndexArrayAttr(paddingArray)));

  attributes.push_back(builder.getNamedAttr(
      "strides", builder.getIndexArrayAttr(config.strideDims)));

  attributes.push_back(builder.getNamedAttr(
      "dilations", builder.getIndexArrayAttr(config.dilationDims)));

  // perf_config
  if (!ignoreTuning && !config.perfConfig.empty()) {
    attributes.push_back(builder.getNamedAttr(
        "perf_config", builder.getStringAttr(config.perfConfig)));
  }

  SmallVector<SmallVector<StringRef>, 4> argDimNameRefs;
  argDimNameRefs.reserve(logicalFuncArgTypes.size());
  auto referenceNames = [&](ArrayRef<StringAttr> layout) {
    argDimNameRefs.push_back(llvm::map_to_vector(
        layout, [](StringAttr sa) { return sa.getValue(); }));
  };
  // Build in standard [filter, input, output] order, then apply the same
  // reordering used for logicalFuncArgTypes.
  referenceNames(filterLayoutSpec);
  referenceNames(inputLayoutSpec);
  referenceNames(outputLayoutSpec);
  reorderConvArgsForKernel(config.operation.value(), argDimNameRefs);

  // Expand all function arguments from flat 1D to their logical shapes,
  // except the output arg whose transform is applied after the conv op.
  SmallVector<Value, 4> args;
  expandFlatFunctionArguments(builder, func,
                              ArrayRef(argDimNameRefs).drop_back(),
                              ArrayRef(logicalFuncArgTypes).drop_back(), args);

  // After reordering, the two conv input operands are always args[0] and
  // args[1], and the store destination is args[storeDestIdx] (the last arg).
  //   ConvOp:         (filter=args[0], input=args[1])   -> store to output
  //   ConvBwdDataOp:  (filter=args[0], gradient=args[1]) -> store to input
  //   ConvBwdWeightOp:(input=args[0], gradient=args[1])  -> store to filter
  Type resultType = logicalFuncArgTypes[storeDestIdx];
  Value convResult;
  switch (config.operation.value()) {
  case ConvOpType::Fwd: {
    SmallVector<Value, 2> convArgs = {args[0], args[1]};
    auto convOp = ConvOp::create(builder, builder.getUnknownLoc(), resultType,
                                 convArgs, attributes);
    convResult = convOp.getResult();
  } break;
  case ConvOpType::BwdData: {
    if (!rock::isEveryElementWrittenBwdData(
            config.strideDims, config.dilationDims, config.filterDims)) {
      // Zero-initialize the store destination (input tensor) for backward
      // data convolutions that don't write to every pixel.
      zeroInitArg(builder, func, storeDestIdx);
    }
    SmallVector<Value, 2> convArgs = {args[0], args[1]};
    auto convOp = ConvBwdDataOp::create(builder, builder.getUnknownLoc(),
                                        resultType, convArgs, attributes);
    convResult = convOp.getResult();
  } break;
  case ConvOpType::BwdWeight: {
    bool needsZeroInit = false;
    bool needExtraPad = false;
    if (succeeded(needExtraPadBwdWeight(builder, needExtraPad))) {
      if (!needExtraPad) {
        auto dataType = getInputDataType(builder);
        needsZeroInit = dataType.isF32() || dataType.isF16();
      }
    }
    if (needsZeroInit)
      zeroInitArg(builder, func, storeDestIdx);

    SmallVector<Value, 2> convArgs = {args[0], args[1]};
    auto convOp = ConvBwdWeightOp::create(builder, builder.getUnknownLoc(),
                                          resultType, convArgs, attributes);
    convResult = convOp.getResult();
  } break;
  }

  // Apply the output transform to flatten the conv result, then store to
  // the flat destination argument.
  Value flatResult = flattenOutput(builder, builder.getUnknownLoc(), convResult,
                                   argDimNameRefs[storeDestIdx]);
  Value flatStoreDest = func.getArgument(storeDestIdx);
  Value storedVal = rock::StoreOp::create(
      builder, builder.getUnknownLoc(), resultFlatType, flatResult,
      flatStoreDest, /*resultAlias=*/Value(),
      builder.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set));

  func::ReturnOp::create(builder, builder.getUnknownLoc(), storedVal);
  return success();
}

func::FuncOp ConvGenerator::getKernelFunc() const { return kernelFunc; }
