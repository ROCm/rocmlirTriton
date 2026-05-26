//===- rocmlir-gen.cpp - MLIR Rock Test Generator ------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for rocmlir-gen test generator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/Generator/ConvGenerator.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/RocmDeviceName.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/tosaUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/InitRocMLIRCLOptions.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <limits>
#include <tuple>
#include <unordered_map>

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init(""));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    testFuncName("func-under-test", llvm::cl::desc("Name of func to test"),
                 llvm::cl::init(""));
static llvm::cl::alias aliasTestFuncName("fut",
                                         llvm::cl::aliasopt(testFuncName));

//////////////////////////////////////////////////////////////////////////////////////////////////////
//// Rock Convolution spec

static llvm::cl::opt<rock::KernelType> operation(
    "operation", llvm::cl::desc("Convolution operation,"),
    llvm::cl::values(
        clEnumValN(rock::KernelType::Conv, "conv", "Forward convolution"),
        clEnumValN(rock::KernelType::ConvBwdData, "conv_bwd_data",
                   "Backpropogate convolution data"),
        clEnumValN(rock::KernelType::ConvBwdWeight, "conv_bwd_weight",
                   "Backpropogate convolution weights"),
        clEnumValN(rock::KernelType::Gemm, "gemm", "Matrix multiplication"),
        clEnumValN(rock::KernelType::Attention, "attention",
                   "Attention operation of transformer models"),
        clEnumValN(rock::KernelType::GemmElementwiseGemm, "gemm_gemm",
                   "gemm+elementwise+gemm operation"),
        clEnumValN(rock::KernelType::ConvElementwiseGemm, "conv_gemm",
                   "conv+elementwise+gemm operation")),
    llvm::cl::value_desc("kernel type"),
    llvm::cl::init(rock::KernelType::Conv));

static llvm::cl::opt<std::string> arch(
    "arch",
    llvm::cl::desc("amdgpu architecture, eg: gfx906, gfx908, gfx942, gfx950, "
                   "gfx1100, gfx1200, gfx1250"),
    llvm::cl::value_desc("GFX architecture string"), llvm::cl::init(""));

static llvm::cl::opt<int> num_cu(
    "num_cu",
    llvm::cl::desc("Number of compute units. If omitted, defaults to the "
                   "per-arch minimum returned by rock::getMinNumCU (e.g. "
                   "gfx906=10, gfx908=120, gfx90a=104, gfx942=20, "
                   "gfx950=256, gfx1010/gfx1030=30, gfx1100=2, gfx1200=12, "
                   "gfx1250=256). Any positive value is accepted."),
    llvm::cl::value_desc("compute unit value"), llvm::cl::init(0));

static llvm::cl::opt<int> numChiplets("num_chiplets",
                                      llvm::cl::desc("Number of chiplets"),
                                      llvm::cl::value_desc("chiplets value"),
                                      llvm::cl::init(0));

static llvm::cl::opt<std::string> perfConfig(
    "perf_config", llvm::cl::desc("performance config data used for tuning"),
    llvm::cl::value_desc("Serialized tuning parameters"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    filterLayout("fil_layout", llvm::cl::desc("Filter layout"),
                 llvm::cl::value_desc("layout string"),
                 llvm::cl::init("gkcyx"));

static llvm::cl::opt<std::string>
    inputLayout("in_layout", llvm::cl::desc("Input layout"),
                llvm::cl::value_desc("layout string"), llvm::cl::init("ngchw"));

static llvm::cl::opt<std::string>
    outputLayout("out_layout", llvm::cl::desc("Output layout"),
                 llvm::cl::value_desc("layout string"),
                 llvm::cl::init("ngkhw"));

static llvm::cl::opt<int64_t> groupSize("groupsize",
                                        llvm::cl::desc("Group size"),
                                        llvm::cl::value_desc("dimension value"),
                                        llvm::cl::init(1));
static llvm::cl::alias groupSizeShort("g",
                                      llvm::cl::desc("alias for -groupsize"),
                                      llvm::cl::aliasopt(groupSize));

static llvm::cl::opt<int> convKernelId(
    "kernel_id",
    llvm::cl::desc("When set, emit only the sub-kernel with this index "
                   "(0-based)"),
    llvm::cl::value_desc("index"), llvm::cl::init(-1));

// N
static llvm::cl::opt<int64_t> batchSize("batchsize",
                                        llvm::cl::desc("Batch size"),
                                        llvm::cl::value_desc("dimension value"),
                                        llvm::cl::init(-1));

// C
static llvm::cl::opt<int64_t>
    inputChannel("in_channels", llvm::cl::desc("Input channels"),
                 llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// Hi
static llvm::cl::opt<int64_t>
    inputHeight("in_h", llvm::cl::desc("Input height"),
                llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// Wi
static llvm::cl::opt<int64_t>
    inputWidth("in_w", llvm::cl::desc("Input width"),
               llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// Di
static llvm::cl::opt<int64_t>
    inputDepth("in_d", llvm::cl::desc("Input depth"),
               llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// K
static llvm::cl::opt<int64_t>
    outputChannel("out_channels", llvm::cl::desc("Output channels"),
                  llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// Y
static llvm::cl::opt<int64_t>
    filterWidth("fil_w", llvm::cl::desc("Filter width"),
                llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// X
static llvm::cl::opt<int64_t>
    filterHeight("fil_h", llvm::cl::desc("Filter height"),
                 llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// Z
static llvm::cl::opt<int64_t>
    filterDepth("fil_d", llvm::cl::desc("Filter depth"),
                llvm::cl::value_desc("dimension value"), llvm::cl::init(-1));

// Ho
static llvm::cl::opt<int64_t> outputHeight(
    "out_h", llvm::cl::desc("Output height"),
    llvm::cl::value_desc("ouput dimension value, does not need to set."),
    llvm::cl::init(-1));

// Wo
static llvm::cl::opt<int64_t> outputWidth(
    "out_w", llvm::cl::desc("Output width"),
    llvm::cl::value_desc("ouput dimension value, does not need to set."),
    llvm::cl::init(-1));

// Do
static llvm::cl::opt<int64_t> outputDepth(
    "out_d", llvm::cl::desc("Output depth"),
    llvm::cl::value_desc("ouput dimension value, does not need to set."),
    llvm::cl::init(-1));

// dilation height
static llvm::cl::opt<int>
    dilationHeight("dilation_h", llvm::cl::desc("Dilation height"),
                   llvm::cl::value_desc("attribute value"), llvm::cl::init(1));

// dilation width
static llvm::cl::opt<int> dilationWidth("dilation_w",
                                        llvm::cl::desc("Dilation width"),
                                        llvm::cl::value_desc("attribute value"),
                                        llvm::cl::init(1));

// dilation depth
static llvm::cl::opt<int> dilationDepth("dilation_d",
                                        llvm::cl::desc("Dilation depth"),
                                        llvm::cl::value_desc("attribute value"),
                                        llvm::cl::init(1));

// stride height
static llvm::cl::opt<int> strideHeight("conv_stride_h",
                                       llvm::cl::desc("Stride height"),
                                       llvm::cl::value_desc("attribute value"),
                                       llvm::cl::init(1));

// stride width
static llvm::cl::opt<int> strideWidth("conv_stride_w",
                                      llvm::cl::desc("Stride width"),
                                      llvm::cl::value_desc("attribute value"),
                                      llvm::cl::init(1));

// stride depth
static llvm::cl::opt<int> strideDepth("conv_stride_d",
                                      llvm::cl::desc("Stride depth"),
                                      llvm::cl::value_desc("attribute value"),
                                      llvm::cl::init(1));

// padding height
static llvm::cl::opt<int> paddingHeight("padding_h",
                                        llvm::cl::desc("Padding height"),
                                        llvm::cl::value_desc("attribute value"),
                                        llvm::cl::init(0));

static llvm::cl::opt<int>
    paddingHeightLeft("padding_h_l", llvm::cl::desc("Padding height Left"),
                      llvm::cl::value_desc("attribute value"),
                      llvm::cl::init(0));

static llvm::cl::opt<int>
    paddingHeightRight("padding_h_r", llvm::cl::desc("Padding height Right"),
                       llvm::cl::value_desc("attribute value"),
                       llvm::cl::init(0));
// padding width
static llvm::cl::opt<int> paddingWidth("padding_w",
                                       llvm::cl::desc("Padding width"),
                                       llvm::cl::value_desc("attribute value"),
                                       llvm::cl::init(0));

static llvm::cl::opt<int>
    paddingWidthLeft("padding_w_l", llvm::cl::desc("Padding width Left"),
                     llvm::cl::value_desc("attribute value"),
                     llvm::cl::init(0));

static llvm::cl::opt<int>
    paddingWidthRight("padding_w_r", llvm::cl::desc("Padding width Right"),
                      llvm::cl::value_desc("attribute value"),
                      llvm::cl::init(0));

// padding depth
static llvm::cl::opt<int> paddingDepth("padding_d",
                                       llvm::cl::desc("Padding depth"),
                                       llvm::cl::value_desc("attribute value"),
                                       llvm::cl::init(0));

static llvm::cl::opt<int>
    paddingDepthLeft("padding_d_l", llvm::cl::desc("Padding depth Left"),
                     llvm::cl::value_desc("attribute value"),
                     llvm::cl::init(0));

static llvm::cl::opt<int>
    paddingDepthRight("padding_d_r", llvm::cl::desc("Padding depth Right"),
                      llvm::cl::value_desc("attribute value"),
                      llvm::cl::init(0));

/// Matrix options
static llvm::cl::opt<int64_t> gemmM("m",
                                    llvm::cl::desc("M dimension of gemm()"),
                                    llvm::cl::value_desc("positive integer"),
                                    llvm::cl::init(-1));

static llvm::cl::opt<int64_t> gemmK("k",
                                    llvm::cl::desc("K dimension of gemm()"),
                                    llvm::cl::value_desc("positive integer"),
                                    llvm::cl::init(-1));

static llvm::cl::opt<int64_t> gemmN("n",
                                    llvm::cl::desc("N dimension of gemm()"),
                                    llvm::cl::value_desc("positive integer"),
                                    llvm::cl::init(-1));

static llvm::cl::opt<bool> scaledGemm(
    "scaledGemm",
    llvm::cl::desc("Indicates whether to generate scaling gemm or not"),
    llvm::cl::value_desc("boolean"), llvm::cl::init(false));

/// gemm+elementwise+gemm options
static llvm::cl::opt<int64_t>
    gemmO("gemmO", llvm::cl::desc("N dimension of the second gemm()"),
          llvm::cl::value_desc("positive integer"), llvm::cl::init(-1));

static llvm::cl::opt<bool>
    transposeA("transA",
               llvm::cl::desc("whether matrix A is GxMxK (default) or GxKxM"),
               llvm::cl::init(false));

static llvm::cl::opt<bool>
    transposeB("transB",
               llvm::cl::desc("whether matrix B is GxKxN (default) or GxNxK"),
               llvm::cl::init(false));

static llvm::cl::opt<bool> transposeScaleA(
    "transScaleA",
    llvm::cl::desc("whether matrix A is GxMxK (default) or GxKxM"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> transposeScaleB(
    "transScaleB",
    llvm::cl::desc("whether matrix B is GxNxK (default) or GxKxN"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    transposeC("transC",
               llvm::cl::desc("whether matrix C is GxMxN (default) or GxNxM"),
               llvm::cl::init(false));

static llvm::cl::opt<int>
    quantBlockSize("quantBlockSize",
                   llvm::cl::desc("Block size for block-scaled quantized GEMM"),
                   llvm::cl::value_desc("positive integer"),
                   llvm::cl::init(32));

static llvm::cl::opt<rock::StoreMethod> storeMethod(
    "store-method", llvm::cl::desc("storage method for gemm"),
    llvm::cl::values(
        clEnumValN(rock::StoreMethod::Set, "set", "set results in C (default)"),
        clEnumValN(rock::StoreMethod::AtomicAdd, "atomic_add",
                   "atomically add results to values in matrix C")),
    llvm::cl::init(rock::StoreMethod::Set));

static llvm::cl::opt<std::string>
    filterDataType("fil_dtype",
                   llvm::cl::desc("Data type for filter tensor or matrix A"),
                   llvm::cl::init("f32"));
static llvm::cl::alias filTypeAliasA("a_dtype",
                                     llvm::cl::aliasopt(filterDataType));
static llvm::cl::alias filTypeAliasShortF("tf",
                                          llvm::cl::aliasopt(filterDataType));
static llvm::cl::alias filTypeAliasShortA("ta",
                                          llvm::cl::aliasopt(filterDataType));

static llvm::cl::opt<std::string>
    inputDataType("in_dtype",
                  llvm::cl::desc("Data type for input tensor or matrix B"),
                  llvm::cl::init("f32"));
static llvm::cl::alias inTypeAliasB("b_dtype",
                                    llvm::cl::aliasopt(inputDataType));
static llvm::cl::alias inTypeAliasShortI("ti",
                                         llvm::cl::aliasopt(inputDataType));
static llvm::cl::alias inTypeAliasShortB("tb",
                                         llvm::cl::aliasopt(inputDataType));
static llvm::cl::opt<std::string>
    scaleADataType("scale_a_dtype",
                   llvm::cl::desc("Data type for scale tensor or matrix A"),
                   llvm::cl::init("f8E8M0FNU"));
static llvm::cl::opt<std::string>
    scaleBDataType("scale_b_dtype",
                   llvm::cl::desc("Data type for scale tensor or matrix B"),
                   llvm::cl::init("f8E8M0FNU"));

// Note that this is defaulted to blank so we can implement `-t` easily
// and know if we should use default i8 input -> i32 output behavior.
static llvm::cl::opt<std::string>
    outputDataType("out_dtype",
                   llvm::cl::desc("Data type for output tensor or matrix C"),
                   llvm::cl::init(""));
static llvm::cl::alias outTypeAliasC("c_dtype",
                                     llvm::cl::aliasopt(outputDataType));
static llvm::cl::alias outTypeAliasLongOut("out_datatype",
                                           llvm::cl::aliasopt(outputDataType));
static llvm::cl::alias outTypeAliasShortO("to",
                                          llvm::cl::aliasopt(outputDataType));
static llvm::cl::alias outTypeAliasShortC("tc",
                                          llvm::cl::aliasopt(outputDataType));

// Convenience setter for when you need all the data types the same or when you
// want the default (32-bit output) behavior for i8 or 8-bit floats. Also allows
// [a]_[b] syntax for mixed-type operations.
static llvm::cl::opt<std::string> dataTypeAlias(
    "t",
    llvm::cl::desc("Data type selector. Extends i8 to i32 output and 8-bit "
                   "floats to f32 output"),
    llvm::cl::value_desc("Type or Type_Type for mixed-type kernels."),
    llvm::cl::cb<void, std::string>([](std::string v) {
      StringRef val(v);
      if (val.contains("_")) {
        StringRef filter, input;
        std::tie(filter, input) = val.split("_");
        filterDataType = filter.str();
        inputDataType = input.str();
      } else {
        filterDataType = v;
        inputDataType = v;
      }

      if (outputDataType.getNumOccurrences() == 0 || outputDataType.empty()) {
        if (val == "i8")
          outputDataType = "i32";
        else if (val.starts_with("f8") || val.starts_with("fp8") ||
                 val.starts_with("bf8") || val.starts_with("f4E2M1FN"))
          outputDataType = "f32";
        else if (filterDataType == inputDataType)
          outputDataType = v;
      }
    }));
llvm::cl::alias dataTypeAliasLong("dtype", llvm::cl::aliasopt(dataTypeAlias));

// populate default values
static llvm::cl::opt<bool>
    populateDefaultValues("p", llvm::cl::desc("To populate default values"),
                          llvm::cl::value_desc("To populate default values"),
                          llvm::cl::init(false));

static llvm::cl::opt<bool> emitSplitKSelectionLikelihood(
    "emit-split-k-selection-likelihood",
    llvm::cl::desc(
        "Print SplitK selection likelihood for the specified kernel"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> emitModuleFusabilityForPerfConfig(
    "emit-module-fusibility-for",
    llvm::cl::desc("Print whether module is fusible given a perf config"),
    llvm::cl::init(""));

static llvm::cl::opt<rock::TuningParamSetKind> emitTuningSpace(
    "emit-tuning-space",
    llvm::cl::desc("Print a tuning space for the specified kernel"),
    llvm::cl::values(
        clEnumValN(rock::TuningParamSetKind::Quick, "quick",
                   "Quick tuning space"),
        clEnumValN(rock::TuningParamSetKind::Full, "full",
                   "Full tuning space, excluding known-bad configurations"),
        clEnumValN(
            rock::TuningParamSetKind::Exhaustive, "exhaustive",
            "All tuning space combinations, including inapplicable ones")),
    llvm::cl::value_desc("tuning space kind to emit"),
    llvm::cl::init(rock::TuningParamSetKind::Full));

static llvm::cl::opt<bool> emitTuningKey(
    "emit-tuning-key",
    llvm::cl::desc(
        "Prints out the struct of the problem to be tuned for inspection."),
    llvm::cl::value_desc(
        "String formatted fields of the problem which is going to be tuned."),
    llvm::cl::init(false));

// Attention related args
// ----------------------

static llvm::cl::opt<int64_t>
    numHeadsQ("num_heads_q",
              llvm::cl::desc("number of heads of Q in attention()"),
              llvm::cl::value_desc("positive integer"), llvm::cl::init(1));

static llvm::cl::opt<int64_t>
    numHeadsKV("num_heads_kv",
               llvm::cl::desc("number of heads of K,V in attention()"),
               llvm::cl::value_desc("positive integer"), llvm::cl::init(1));

static llvm::cl::list<int64_t>
    currentSeqLen("current_seq_len",
                  llvm::cl::desc("List of sequence lengths of K and V (related "
                                 "to KV-cache) in attention()"),
                  llvm::cl::value_desc("list of positive integers"),
                  llvm::cl::CommaSeparated);

static llvm::cl::opt<int64_t> sequenceLengthQ(
    "seq_len_q", llvm::cl::desc("sequence length of Q in attention()"),
    llvm::cl::value_desc("positive integer"), llvm::cl::init(-1));

static llvm::cl::opt<int64_t> sequenceLengthK(
    "seq_len_k", llvm::cl::desc("sequence length of K in attention()"),
    llvm::cl::value_desc("positive integer"), llvm::cl::init(-1));

static llvm::cl::opt<int64_t>
    headDimQK("head_dim_qk",
              llvm::cl::desc("head dimension of Q,K in attention()"),
              llvm::cl::value_desc("positive integer"), llvm::cl::init(-1));

static llvm::cl::opt<int64_t>
    headDimV("head_dim_v", llvm::cl::desc("head dimension of v in attention()"),
             llvm::cl::value_desc("positive integer"), llvm::cl::init(-1));

static llvm::cl::opt<bool>
    hasAttnScale("with-attn-scale",
                 llvm::cl::desc("Generate an attention kernel that is using a "
                                "scaling input for the first gemm"),
                 llvm::cl::init(false));

static llvm::cl::opt<bool> hasAttnBias(
    "with-attn-bias",
    llvm::cl::desc("Generate an attention kernel that is using a bias"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> transposeQ(
    "transQ",
    llvm::cl::desc("whether matrix Q of attention op is "
                   "Gxseq_len_qxhead_qk (default) or Gxhead_qkxseq_len_q"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> transposeK(
    "transK",
    llvm::cl::desc("whether matrix K of attention op is "
                   "Gxseq_len_kxhead_qk (default) or Gxheadxseq_len_q"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> transposeV(
    "transV",
    llvm::cl::desc("whether matrix V of attention op is "
                   "Gxseq_len_kxhead_v (default) or Gxhead_vxseq_len_k"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> transposeO(
    "transO",
    llvm::cl::desc("whether matrix O of attention op is "
                   "Gxseq_len_qxhead_v (default) or Gxhead_vxseq_len_q"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    causalMasking("causal",
                  llvm::cl::desc("whether we implement causal masking"),
                  llvm::cl::init(false));

static llvm::cl::list<int64_t>
    prefixOffset("prefix_offset",
                 llvm::cl::desc("List of prefix offsets for prefix causal "
                                "masking (mask when key > query + offset)"),
                 llvm::cl::value_desc("list of positive integers"),
                 llvm::cl::CommaSeparated);

static llvm::cl::opt<int64_t> splitKV(
    "split_kv",
    llvm::cl::desc("Flash decoding enabled if split-kv > 1. Describes "
                   "the number of blocks in the sequenceLengthK dimension."),
    llvm::cl::value_desc("positive integer"), llvm::cl::init(1));

static llvm::cl::opt<bool> returnLSE(
    "return_lse",
    llvm::cl::desc("whether the attention kernel returns LSE (log-sum-exp)"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string>
    softmaxDataType("softmax_dtype",
                    llvm::cl::desc("Data type for softmax (attention)"),
                    llvm::cl::init("f32"));

//////////////////////////////////////////////////////////////////////////
////  Host Generator options
//////////////////////////////////////////////////////////////////////////
////  * Host harness
////    * kernel options
////      * cmd-line def (see above)
////        * gpu gen
////        * cpu gen
////      * user defined (input file)
////    * verifier
////      * cpu gen
////      * gpu gen
////      * compare results
////    * print results
////      * optionally print inputs
////      * optionally print validation results
////    * profiling (TBD)
////      * plumb thru runner (TBD)
//////////////////////////////////////////////////////////////////////////

// generate host harness program.
static llvm::cl::opt<bool>
    genHostHarness("host-harness", llvm::cl::desc("To use host harness"),
                   llvm::cl::value_desc("To use host harness"),
                   llvm::cl::init(false));

static llvm::cl::alias aliasGenHostHarness("ph",
                                           llvm::cl::aliasopt(genHostHarness));

// generate clone harness program.
static llvm::cl::opt<bool> genCloneHarness(
    "clone-harness", llvm::cl::desc("To generate clone harness"),
    llvm::cl::value_desc("To generate clone harness"), llvm::cl::init(false));

// print results
static llvm::cl::opt<bool>
    printResults("print-results", llvm::cl::desc("To print result tensor"),
                 llvm::cl::init(false));
static llvm::cl::alias aliasPrintResults("pr",
                                         llvm::cl::aliasopt(printResults));

static llvm::cl::opt<bool> printInputs("print-inputs",
                                       llvm::cl::desc("To print input tensors"),
                                       llvm::cl::init(false));
static llvm::cl::alias aliasPrintInputs("pi", llvm::cl::aliasopt(printInputs));

static llvm::cl::opt<bool> printValidationResults(
    "print-validation-results",
    llvm::cl::desc("To print result tensor for validation"),
    llvm::cl::init(false));
static llvm::cl::alias
    aliasPrintValidationResults("pvr",
                                llvm::cl::aliasopt(printValidationResults));

// populate host validation logic.
//
// `mlir` (a.k.a. `-pv`, the "fast" CPU verifier) widens narrow floats to f32
// for vectorisation. That trades GPU-faithful rounding for speed and is the
// right default for most tests.
//
// `mlir-strict` (a.k.a. `-pv-strict`) matches the GPU's effective precision
// for attention by choosing the first-GEMM output element type
// structurally:
//   * No pre-softmax scale/bias: the GPU's truncf-store / load-extf
//     round-trip folds away, so the chain effectively runs at f32. The CPU
//     verifier promotes the first GEMM output to f32 to match.
//   * With scale and/or bias: the round-trip cannot fold because narrow
//     arithmetic (e.g. `v_dot2_bf16_bf16` on RDNA) sits between, so the GPU
//     genuinely runs in the narrow type. The CPU verifier keeps the first
//     GEMM output narrow so it rounds between operations the same way.
// Use it when an existing test had to relax thresholds because of CPU/GPU
// dtype divergence (see PR #161 / PrAttentionBF16.toml).
static llvm::cl::opt<std::string> genValidation(
    "verifier",
    llvm::cl::desc("Select verification from: none(default), cpu, mlir, "
                   "mlir-strict, clone"),
    llvm::cl::cb<void, std::string>([](const std::string &v) {
      if (!v.empty())
        genHostHarness = true;
    }),
    llvm::cl::value_desc("Specify host validation logic"), llvm::cl::init(""));

static llvm::cl::opt<bool>
    genCPUValidation("pv", llvm::cl::Hidden, llvm::cl::init(false),
                     llvm::cl::Optional, llvm::cl::cb<void, bool>([](bool v) {
                       if (v) {
                         genValidation = "mlir";
                         genHostHarness = true;
                       }
                     }));

static llvm::cl::opt<bool>
    genMLIRValidation("pv_with_mlir", llvm::cl::Hidden, llvm::cl::init(false),
                      llvm::cl::Optional, llvm::cl::cb<void, bool>([](bool v) {
                        if (v) {
                          genValidation = "mlir";
                          genHostHarness = true;
                        }
                      }));

// Convenience alias for `--verifier=mlir-strict`. The strict CPU verifier
// preserves narrow-float dtypes through the pre-softmax fusion so the CPU
// reference rounds between operations the way the GPU does. Tests that
// previously had to relax thresholds (see PrAttentionBF16.toml GQA + KV
// Cache configs and PR #161) can opt in here to recover tight thresholds.
static llvm::cl::opt<bool>
    genCPUValidationStrict("pv-strict", llvm::cl::Hidden, llvm::cl::init(false),
                           llvm::cl::Optional,
                           llvm::cl::cb<void, bool>([](bool v) {
                             if (v) {
                               genValidation = "mlir-strict";
                               genHostHarness = true;
                             }
                           }));

static llvm::cl::opt<bool>
    genCPUKernel("cpu-kernels", llvm::cl::desc("Generate CPU kernel for test"),
                 llvm::cl::init(false), llvm::cl::Optional,
                 llvm::cl::cb<void, bool>([](bool v) {
                   if (v) {
                     genValidation = "mlir";
                     genHostHarness = true;
                     printResults = true;
                   }
                 }));
static llvm::cl::alias aliasGenCPUKernel("prc",
                                         llvm::cl::aliasopt(genCPUKernel));

static llvm::cl::opt<bool>
    cpuTimers("cpu-timers",
              llvm::cl::desc("Enable CPU timing instrumentation for JIT "
                             "compilation, memory init, GPU kernel, and CPU "
                             "validation"),
              llvm::cl::init(false));

static llvm::cl::opt<bool> pvF64(
    "pv-f64",
    llvm::cl::desc("Run the host CPU attention reference internally in f64 "
                   "for higher precision validation of non-quantized "
                   "attention, especially at long seq_len_k. No effect on "
                   "the i8 attention. This also enables pv-strict together"),
    llvm::cl::init(false), llvm::cl::Optional,
    llvm::cl::cb<void, bool>([](bool v) {
      if (v) {
        genValidation = "mlir-strict";
        genHostHarness = true;
      }
    }));

// Input data spec
static llvm::cl::opt<std::string> randomSeed(
    "rand",
    llvm::cl::desc(
        "A positive integer or zero indicates the seed of random data generator"
        "for convolution inputs, e.g. -rand 1. If not specifed, or 'fixed', "
        "use a fixed nonuniform test pattern. If 'none', use all 1s as the "
        "values. If 0, use time(0) as the seed."),
    llvm::cl::value_desc("seed"), llvm::cl::init("fixed"));

static llvm::cl::opt<std::string> randomDataType(
    "rand_type",
    llvm::cl::desc("To specify data type for random number generator,"
                   "e.g. -rand_type float, -rand_type int (default)."),
    llvm::cl::value_desc("type"), llvm::cl::init("int"));

static llvm::cl::list<int> randomTypeIntForInputs(
    "rand_type_int_for_inputs",
    llvm::cl::desc(
        "To specify int type for random number generator for specific inputs."),
    llvm::cl::value_desc("list of indices"), llvm::cl::CommaSeparated);

static llvm::cl::opt<std::string> randomSide(
    "rand_side",
    llvm::cl::desc(
        "To populate random numbers to a specified tensor: "
        "For conv, -rand_side filter or -rand_side input; "
        "For conv_bwd_data, -rand_side filter or -rand_side output; "
        "For conv_bwd_weight, -rand_side input or -rand_side output. "
        "By default, populate random numbers to both tensors."),
    llvm::cl::value_desc("tensor"), llvm::cl::init("both"));

// float random inputs range
static llvm::cl::opt<int>
    randMin("rand_min", llvm::cl::desc("lower bound for float random input"),
            llvm::cl::value_desc("range"), llvm::cl::init(0));

static llvm::cl::opt<int>
    randMax("rand_max", llvm::cl::desc("upper bound for float random input"),
            llvm::cl::value_desc("range"), llvm::cl::init(1));

// int random inputs range
static llvm::cl::opt<int>
    randMinInt("rand_min_int",
               llvm::cl::desc("lower bound for int random input"),
               llvm::cl::value_desc("range"), llvm::cl::init(-5));

static llvm::cl::opt<int>
    randMaxInt("rand_max_int",
               llvm::cl::desc("upper bound for int random input"),
               llvm::cl::value_desc("range"), llvm::cl::init(5));

// Verification function options
static llvm::cl::opt<float>
    RMSThreshold("RMS_threshold", llvm::cl::desc("Threshold for RMS metric"),
                 llvm::cl::value_desc("error"));

static llvm::cl::opt<float>
    absDiffThreshold("absDiff_threshold",
                     llvm::cl::desc("Threshold for absDiff metric"),
                     llvm::cl::value_desc("error"), llvm::cl::init(100.0f));

static llvm::cl::opt<float>
    relDiffThreshold("relDiff_threshold",
                     llvm::cl::desc("Threshold for relDiff metric"),
                     llvm::cl::value_desc("error"), llvm::cl::init(0.000001f));

// Comparator selection.
// `legacy` is the three-gate RMS/absDiff/relDiff verifier emitted by
// mcpuVerifyFloat. `allclose` is the |a - b| <= atol + rtol*|b| verifier
// emitted by mcpuVerifyFloatAllclose, with per-dtype K-scaled defaults. Default
// stays `legacy` until the ~100 existing tests get migrated in the follow-up PR
// setting -atol or -rtol on the command line implies `allclose`.
enum class ComparatorMode { Legacy, Allclose };
static llvm::cl::opt<ComparatorMode> comparatorMode(
    "comparator",
    llvm::cl::desc(
        "Comparator used by the verifier (orthogonal to --verifier)"),
    llvm::cl::values(
        clEnumValN(ComparatorMode::Legacy, "legacy",
                   "Three-gate RMS/absDiff/relDiff (default)"),
        clEnumValN(ComparatorMode::Allclose, "allclose",
                   "|a - b| <= (atol + rtol*|b|) with per-dtype defaults")),
    llvm::cl::init(ComparatorMode::Legacy));

// Allclose tolerance overrides. Setting either implies --comparator=allclose.
static llvm::cl::opt<float> atolThreshold(
    "atol",
    llvm::cl::desc(
        "Absolute tolerance for allclose comparator "
        "(|a - b| <= atol + rtol*|b|). Setting this flag forces "
        "--comparator=allclose, overriding any other --comparator value, "
        "and replaces the K-scaled atol default."),
    llvm::cl::value_desc("error"));

static llvm::cl::opt<float> rtolThreshold(
    "rtol",
    llvm::cl::desc(
        "Relative tolerance for allclose comparator "
        "(|a - b| <= atol + rtol*|b|). Setting this flag forces "
        "--comparator=allclose, overriding any other --comparator value, "
        "and replaces the per-dtype rtol default."),
    llvm::cl::value_desc("error"));

// A toggle to control what to print in the verification function
enum class VerificationPrintToggle : char {
  Always = 3,
  Failure = 2,
  Summary = 1,
  Off = 0
};

static llvm::cl::opt<VerificationPrintToggle> printVerifyResults(
    "print-verify-results",
    llvm::cl::desc("Choose when to print verbose debug information in the "
                   "verification function:"),
    llvm::cl::values(
        clEnumValN(VerificationPrintToggle::Always, "always",
                   "always print debug info"),
        clEnumValN(VerificationPrintToggle::Failure, "failure",
                   "print elem-wise diff + summary only if the test fails"),
        clEnumValN(VerificationPrintToggle::Summary, "summary",
                   "print summary info only if the test fails"),
        clEnumValN(VerificationPrintToggle::Off, "off",
                   "do not print debug info")),
    llvm::cl::init(VerificationPrintToggle::Summary));
static llvm::cl::alias
    aliasPrintVerifyResults("p_verify", llvm::cl::aliasopt(printVerifyResults));

static llvm::cl::opt<int> deviceNum(
    "device",
    llvm::cl::desc(
        "Device index on which to run the kernel (only with host code)"),
    llvm::cl::value_desc("Between 0 and number of GPUs on system. "
                         "Omission leaves current device intact."));
static llvm::cl::alias deviceShort("dev", llvm::cl::aliasopt(deviceNum));

static llvm::cl::opt<int> kernelRepeats(
    "kernel-repeats",
    llvm::cl::desc("Number of times to repeat the kernel invocation"),
    llvm::cl::value_desc("positive integer"), llvm::cl::init(1));

// TODO[split-K]: remove after integrating with MIGraphX
static llvm::cl::opt<bool> disableSplitKForTuning(
    "disable-split-k-for-tuning",
    llvm::cl::desc("disable split-K GEMM scheme for tuning"),
    llvm::cl::init(false));

////////////////////////////////////////////////////////////////////////////////
////  Struct KernelIF
////  - Detected/capture kernel interface
////////////////////////////////////////////////////////////////////////////////
struct KernelIF {
  func::FuncOp func;
  SmallVector<Type, 8> params;
  SmallVector<int32_t, 2> outIndices;
  SmallVector<Type, 2> resultTypes;

  // CTOR w/ FuncOp
  KernelIF(func::FuncOp _f) : func(_f) {
    size_t argCount = func.getArguments().size();
    for (size_t i = 0; i < argCount; i++) {
      params.push_back(func.getArgument(i).getType());
    }

    // Handle functions that return results (tensor-based)
    if (func.getNumResults() > 0) {
      for (Type resultType : func.getResultTypes()) {
        resultTypes.push_back(resultType);
      }
    } else {
      // Handle functions with output arguments (memref-based)
      llvm::SmallDenseSet<Value> outs;
      auto walker = [&](memref::CopyOp copy) { outs.insert(copy.getTarget()); };
      func.walk(walker);
      for (size_t i = 0; i < argCount; i++) {
        if (outs.contains(func.getArgument(i))) {
          outIndices.push_back(i);
        }
      }
    }
  }
};

// This helper struct defines the argument ordering for
// quantized attention operator.
struct AttentionQuantizedArgIndex {
  static const size_t q = 0;
  static const size_t k = 1;
  static const size_t v = 2;
  static const size_t quantBias = 3;
  static const size_t quantScale = 4;
  static const size_t scale = 5;
  static const size_t bias = 6;
  static const size_t currentSeqLen = 7;
  static const size_t prefixOffset = 8;
  static const size_t lse = 9;
};

struct AttentionArgIndex {
  static const size_t q = 0;
  static const size_t k = 1;
  static const size_t v = 2;
  static const size_t scale = 3;
  static const size_t bias = 4;
  static const size_t currentSeqLen = 5;
  static const size_t prefixOffset = 6;
  static const size_t lse = 7;
};

struct GenParams {
  std::optional<rock::KernelType> operation = std::nullopt;
  SmallVector<Type, 5> types;
  std::optional<const rock::ConvGenerator::Config *> convConfig = std::nullopt;
  StringRef arch;
  StringRef perfConfig;

  /// When true, the CPU attention verifier picks the first-GEMM output
  /// element type to match the GPU's effective precision: f32 when no
  /// pre-softmax scale/bias is present (the GPU's truncf/extf round-trip
  /// folds away), and the original narrow float type otherwise (the GPU
  /// genuinely runs the scale*QK / qk+bias chain in the narrow type).
  /// Selected via `-pv-strict` / `--verifier=mlir-strict`.
  bool strictMode = false;
};

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

static bool isConv(rock::KernelType kernelType) {
  return kernelType == rock::KernelType::Conv ||
         kernelType == rock::KernelType::ConvBwdData ||
         kernelType == rock::KernelType::ConvBwdWeight ||
         kernelType == rock::KernelType::ConvElementwiseGemm;
}

static void correctConvParameters() {
  auto addGToLayout = [](std::string &layoutValue) -> std::string {
    std::string layout;
    if (layoutValue.find('g') == std::string::npos) {
      // Always add 'g' after 'n' when it's missing
      size_t nPos = layoutValue.find('n');
      assert(nPos != std::string::npos);
      layout =
          layoutValue.substr(0, nPos + 1) + "g" + layoutValue.substr(nPos + 1);
    } else {
      layout = layoutValue;
    }
    return layout;
  };

  if (filterLayout.getValue().find('g') == std::string::npos)
    filterLayout = "g" + filterLayout.getValue();
  inputLayout = addGToLayout(inputLayout.getValue());
  outputLayout = addGToLayout(outputLayout.getValue());

  // update old key names.
  std::replace(filterLayout.getValue().begin(), filterLayout.getValue().end(),
               'y', '0');
  std::replace(filterLayout.getValue().begin(), filterLayout.getValue().end(),
               'x', '1');
  std::replace(inputLayout.getValue().begin(), inputLayout.getValue().end(),
               'h', '0');
  std::replace(inputLayout.getValue().begin(), inputLayout.getValue().end(),
               'w', '1');
  std::replace(outputLayout.getValue().begin(), outputLayout.getValue().end(),
               'h', '0');
  std::replace(outputLayout.getValue().begin(), outputLayout.getValue().end(),
               'w', '1');

  auto validatePadding = [](llvm::cl::opt<int> &combined,
                            llvm::cl::opt<int> &left, llvm::cl::opt<int> &right,
                            StringRef name) {
    if (combined.getValue() > 0) {
      int combinedVal = combined.getValue();
      int leftVal = left.getValue();
      int rightVal = right.getValue();
      if (leftVal == 0 && rightVal == 0) {
        left = combinedVal;
        right = combinedVal;
      } else {
        if (leftVal != combinedVal || rightVal != combinedVal) {
          llvm::errs() << "you can't use both " << name << " and (" << name
                       << "_l," << name << "_r).\n";
        }
      }
    }
  };

  validatePadding(paddingHeight, paddingHeightLeft, paddingHeightRight,
                  "padding_h");
  validatePadding(paddingWidth, paddingWidthLeft, paddingWidthRight,
                  "padding_w");
  validatePadding(paddingDepth, paddingDepthLeft, paddingDepthRight,
                  "padding_d");

  // adjust the padding size
  // getOutputDim can give us correct output size
  // output size = input size+ padding size
  // then -(filter size-1) * dilation size -1
  // ,/ stride size and add 1
  auto getOutputDim = [](int64_t inputLen, int64_t filLen, int leftPadLen,
                         int rightPadLen, int strideLen, int dilLen) {
    return (inputLen + leftPadLen + rightPadLen - (filLen - 1) * dilLen - 1) /
               strideLen +
           1;
  };

  int hi = inputHeight.getValue();
  int y = filterHeight.getValue();
  int in_left_pad_h = paddingHeightLeft.getValue();
  int in_right_pad_h = paddingHeightRight.getValue();
  int conv_stride_h = strideHeight.getValue();
  int conv_dilation_h = dilationHeight.getValue();
  int ho = getOutputDim(hi, y, in_left_pad_h, in_right_pad_h, conv_stride_h,
                        conv_dilation_h);
  int hi_minimum = 1 + (y - 1) * conv_dilation_h + (ho - 1) * conv_stride_h;
  int hi_specified = hi + in_left_pad_h + in_right_pad_h;
  // hi_minimum is the mininum number of input elements needed to correctly
  // apply the filter in the h direction, which is a function of the stride and
  // dilation parameters. If the specified input height is less than this value,
  // add extra padding on the right to allow the convolution to execute
  // successfully.
  if (hi_minimum > hi_specified)
    paddingHeightRight = in_right_pad_h + (hi_minimum - hi_specified);

  int wi = inputWidth.getValue();
  int x = filterWidth.getValue();
  int in_left_pad_w = paddingWidthLeft.getValue();
  int in_right_pad_w = paddingWidthRight.getValue();
  int conv_stride_w = strideWidth.getValue();
  int conv_dilation_w = dilationWidth.getValue();
  int wo = getOutputDim(wi, x, in_left_pad_w, in_right_pad_w, conv_stride_w,
                        conv_dilation_w);

  int wi_minimum = 1 + (x - 1) * conv_dilation_w + (wo - 1) * conv_stride_w;
  int wi_specified = wi + in_left_pad_w + in_right_pad_w;
  // wi_minimum is the miminum number of input elements needed to correctly
  // apply the filter in the w direction, which is a function of the stride and
  // dilation parameters. If the specified input height is less than this value,
  // add extra padding on the right to allow the convolution to execute
  // successfully.
  if (wi_minimum > wi_specified)
    paddingWidthRight = in_right_pad_w + (wi_minimum - wi_specified);

  int di = inputDepth.getValue();
  int z = filterDepth.getValue();
  int in_left_pad_d = paddingDepthLeft.getValue();
  int in_right_pad_d = paddingDepthRight.getValue();
  int conv_stride_d = strideDepth.getValue();
  int conv_dilation_d = dilationDepth.getValue();
  int d_o = getOutputDim(di, z, in_left_pad_d, in_right_pad_d, conv_stride_d,
                         conv_dilation_d);

  int di_minimum = 1 + (z - 1) * conv_dilation_d + (d_o - 1) * conv_stride_d;
  int di_specified = di + in_left_pad_d + in_right_pad_d;
  // di_minimum is the miminum number of input elements needed to correctly
  // apply the filter in the d direction, which is a function of the stride and
  // dilation parameters. If the specified input height is less than this value,
  // add extra padding on the right to allow the convolution to execute
  // successfully.
  if (di_minimum > di_specified)
    paddingDepthRight = in_right_pad_d + (di_minimum - di_specified);
}

static void populateDefaults() {
  const bool isGemm = operation == rock::KernelType::Gemm;
  const bool isAttention = operation == rock::KernelType::Attention;
  const bool isGemmElntwiseGemm =
      operation == rock::KernelType::GemmElementwiseGemm;
  const bool isConvElntwiseGemm =
      operation == rock::KernelType::ConvElementwiseGemm;

  // here we treat ConvElementwiseGemm as a convolution as well
  const bool isConv = !(isGemm || isAttention || isGemmElntwiseGemm);
  // Default f32 if we passed no `-t` arguments at all.
  if (outputDataType.empty()) {
    if (filterDataType != inputDataType) {
      llvm::errs() << "Missing output type for mixed input types\n";
      exit(1);
    }

    outputDataType = "f32";
  }
  if (populateDefaultValues) {
    if (isGemm) {
      groupSize = 1;
      gemmM = 1024;
      gemmK = 769;
      gemmN = 512;
    }
    if (isGemmElntwiseGemm) {
      groupSize = 1;
      gemmM = 1024;
      gemmK = 769;
      gemmN = 512;
      gemmO = 769;
    }
    if (isConvElntwiseGemm) {
      gemmO = 769;
    }
    if (isAttention) {
      groupSize = 1;
      sequenceLengthQ = 1024;
      sequenceLengthK = 1024;
      numHeadsQ = 1;
      numHeadsKV = 1;
      headDimQK = 32;
      headDimV = 32;
    }
    if (isConv) {
      groupSize = 1;
      batchSize = 128;
      inputChannel = 8;
      outputChannel = 128;
      inputHeight = 32;
      inputWidth = 32;
      inputDepth = 1;
      filterHeight = 3;
      filterWidth = 3;
      filterDepth = 1;
      dilationHeight = 1;
      dilationWidth = 1;
      dilationDepth = 1;
      strideHeight = 1;
      strideWidth = 1;
      strideDepth = 1;
      paddingHeightLeft = 0;
      paddingHeightRight = 0;
      paddingWidthLeft = 0;
      paddingWidthRight = 0;
      paddingDepthLeft = 0;
      paddingDepthRight = 0;
    }
  }

  if (isConv && outputHeight.getNumOccurrences() == 0) {
    outputHeight = rock::ConvGenerator::outputDim(
        inputHeight.getValue(), filterHeight.getValue(),
        paddingHeightLeft.getValue(), paddingHeightRight.getValue(),
        strideHeight.getValue(), dilationHeight.getValue());
  }
  if (isConv && outputWidth.getNumOccurrences() == 0) {
    outputWidth = rock::ConvGenerator::outputDim(
        inputWidth.getValue(), filterWidth.getValue(),
        paddingWidthLeft.getValue(), paddingWidthRight.getValue(),
        strideWidth.getValue(), dilationWidth.getValue());
  }
  if (isConv && outputDepth.getNumOccurrences() == 0) {
    outputDepth = rock::ConvGenerator::outputDim(
        inputDepth.getValue(), filterDepth.getValue(),
        paddingDepthLeft.getValue(), paddingDepthRight.getValue(),
        strideDepth.getValue(), dilationDepth.getValue());
  }
}

static auto getRequiredArgs(std::optional<rock::KernelType> kernelType) {
  using RequiredArgsType = std::vector<const llvm::cl::opt<int64_t> *>;
  const static RequiredArgsType requiredConvArgs = {
      &groupSize,  &batchSize,     &inputChannel, &inputHeight,
      &inputWidth, &outputChannel, &filterWidth,  &filterHeight};
  switch (kernelType.value()) {
  case rock::KernelType::Gemm: {
    const static RequiredArgsType requiredGemmArgs = {&groupSize, &gemmM,
                                                      &gemmK, &gemmN};
    return requiredGemmArgs;
  }
  case rock::KernelType::GemmElementwiseGemm: {
    const static RequiredArgsType requiredGemmElntwiseGemmArgs = {
        &groupSize, &gemmM, &gemmK, &gemmN, &gemmO};
    return requiredGemmElntwiseGemmArgs;
  }
  case rock::KernelType::Attention: {
    const static RequiredArgsType requiredAttenArgs = {
        &groupSize, &sequenceLengthQ, &sequenceLengthK, &headDimQK, &headDimV};
    return requiredAttenArgs;
  }
  case rock::KernelType::ConvElementwiseGemm: {
    RequiredArgsType requiredConvElntwiseGemmArgs(requiredConvArgs);
    requiredConvElntwiseGemmArgs.push_back(&gemmO);
    return requiredConvElntwiseGemmArgs;
  }
  default: {
    return requiredConvArgs;
  }
  };
}

static LogicalResult detectMissingArguments() {
  const static auto requiredArgs = getRequiredArgs(operation);

  if (arch.getValue().empty()) {
    llvm::errs() << "--arch is not set\n";
    return failure();
  }

  for (auto *arg : requiredArgs) {
    if (arg->getValue() <= 0) {
      llvm::errs() << "Value for: " << arg->ArgStr << " not specified\n";
      return failure();
    }
  }

  if (operation == rock::KernelType::Attention) {
    if (splitKV > 1 && !returnLSE) {
      llvm::errs()
          << "If split-kv > 1 (flash decoding), we need to return LSE\n";
      return failure();
    }
  }

  if (operation == rock::KernelType::Attention ||
      operation == rock::KernelType::GemmElementwiseGemm ||
      operation == rock::KernelType::ConvElementwiseGemm) {
    if (dataTypeAlias.getValue().empty()) {
      llvm::errs() << "Type of the attention/gemm+gemm/conv+gemm operation is "
                      "not specified\n";
      return failure();
    }
  }

  return success();
}

static func::FuncOp makeFuncDecl(ModuleOp module, StringRef funcName,
                                 TypeRange inputs, TypeRange results = {}) {
  func::FuncOp func = module.lookupSymbol<func::FuncOp>(funcName);
  if (!func) {
    OpBuilder builder(module.getContext());
    func = func::FuncOp::create(builder.getUnknownLoc(), funcName,
                                builder.getFunctionType(inputs, results));
    func.setSymVisibilityAttr(builder.getStringAttr("private"));
    module.push_back(func);
  }

  return func;
}

static Value makeNDMemRef(OpBuilder &b, Value var, uint32_t ndim) {
  MLIRContext *context = b.getContext();
  auto oprType = cast<ShapedType>(var.getType());
  if (!oprType.hasStaticShape())
    return Value();

  auto shape = oprType.getShape();
  auto loc = var.getLoc();

  if (shape.size() > ndim) {
    // Collapse last dims
    SmallVector<int64_t, 5> colShape;
    SmallVector<ReassociationExprs, 5> reassocs;
    uint32_t dim = 0;
    for (; dim < ndim - 1; ++dim) {
      colShape.push_back(shape[dim]);
      reassocs.push_back({getAffineDimExpr(dim, context)});
    }

    // Last dim
    uint64_t lastDim = 1;
    SmallVector<AffineExpr, 2> exprs;
    for (; dim < shape.size(); ++dim) {
      lastDim *= shape[dim];
      exprs.push_back(getAffineDimExpr(dim, context));
    }
    colShape.push_back(lastDim);
    reassocs.push_back(exprs);

    auto colType = MemRefType::get(colShape, oprType.getElementType());

    // Emit memref.collapse_shape
    var = memref::CollapseShapeOp::create(b, loc, colType, var, reassocs);
  } else if (!shape.empty() && shape.size() < ndim) {
    // Expand last dims
    SmallVector<int64_t, 5> expShape;
    SmallVector<ReassociationExprs, 5> reassocs;
    uint32_t dim = 0;
    for (; dim < shape.size() - 1; ++dim) {
      expShape.push_back(shape[dim]);
      reassocs.push_back({getAffineDimExpr(dim, context)});
    }

    // Last dim
    expShape.push_back(shape[dim]);
    SmallVector<AffineExpr, 2> exprs;
    for (; dim < ndim; ++dim) {
      expShape.push_back(1);
      exprs.push_back(getAffineDimExpr(dim, context));
    }
    expShape.pop_back();
    reassocs.push_back(exprs);

    auto expType = MemRefType::get(expShape, oprType.getElementType());

    // Emit memref.collapse_shape
    var = memref::ExpandShapeOp::create(b, loc, expType, var, reassocs);
  }

  return var;
}

static std::pair<int64_t, int64_t> getMandNPerBlock(OpBuilder builder,
                                                    const GenParams &params) {
  // Mirrors PopulateParamsGemmGemm::obtainTuningParameters: prefer the
  // user-supplied perfConfig, otherwise fall back to the first quick-tuning
  // entry for this (arch, kernel, dtype).
  assert(params.operation.has_value() && !params.types.empty());
  std::vector<rock::GemmGemmParamsAttr> defaults =
      rock::PopulateParamsGemmGemm::getTuningParameters(
          builder, params.arch, *params.operation, params.types[0]);
  FailureOr<rock::GemmGemmParamsAttr> attnPerfConfig =
      rock::materializeTuningParams<rock::GemmGemmParamsAttr>(
          builder, params.perfConfig, defaults);
  assert(succeeded(attnPerfConfig) &&
         "no quick-tuning entry for this arch / kernel / dtype");
  return {attnPerfConfig->getMPerBlockG0(), attnPerfConfig->getNPerBlockG0()};
}

// Compute the number of valid split-KV entries for each batch-head.
// This determines which splits should have valid results vs -inf.
// Note on the M/N convention: rocMLIR's blockwise attention computes the
// transposed product V * (K * Q^T) rather than the standard (Q * K^T) * V,
// which puts the key-sequence dimension on GEMM0's M axis. The Triton
// attention lowering keeps the standard formulation, so the key-sequence
// dimension is GEMM0's N axis. Split-KV partitions the first GEMM along
// the key-sequence dimension, hence we use gemm0NPerBlock here.
static SmallVector<int32_t> computeValidSplitKV(int64_t nPerBlock) {
  SmallVector<int32_t> validSplitKV;
  for (int64_t i = 0; i < groupSize; ++i) {
    int32_t currSeqLen =
        currentSeqLen.empty() ? (sequenceLengthK - 1) : currentSeqLen[i];
    // For prefix causal masking, the effective sequence length is
    // min(currSeqLen, queryPos + prefixOffset). Since queryPos is 0
    // for seqLenQ=1 (decoding), this becomes min(currSeqLen, prefixOffset).
    // Note: prefixOffset implies causal is enabled, so we handle it first.
    if (!prefixOffset.empty()) {
      // Prefix causal: effectiveSeqLen = prefixOffset (when queryPos=0)
      // If KVCache is also enabled, take min with currentSeqLen
      int32_t prefixEffectiveLen = static_cast<int32_t>(prefixOffset[i]);
      currSeqLen = std::min(currSeqLen, prefixEffectiveLen);
    } else if (causalMasking) {
      // Regular causal: only implemented if sequenceLengthQ <= nPerBlock
      // currSeqLen = min(currSeqLen, n_block * gemm0NPerBlock)
      currSeqLen = 0;
    }
    int32_t numPerBlock = (currSeqLen + nPerBlock) / nPerBlock;
    int32_t itersPerBlock = nPerBlock * llvm::divideCeil(numPerBlock, splitKV);
    int32_t numValidKV = llvm::divideCeil(currSeqLen + 1, itersPerBlock);
    for (int64_t j = 0; j < numHeadsQ; ++j)
      validSplitKV.push_back(numValidKV);
  }
  return validSplitKV;
}

static Value computeFinalAttentionStage(OpBuilder builder, Location loc,
                                        Value resultTensor, Value lseTensor,
                                        SmallVector<int32_t> &validSplitKV);
static func::FuncOp createGPUWrapper(ModuleOp module,
                                     const std::string &funcName,
                                     const SmallVector<KernelIF, 8> &kernels,
                                     const GenParams &params,
                                     ArrayRef<int32_t> outIndices) {
  MLIRContext *context = module.getContext();
  OpBuilder b(context);
  auto loc = kernels[0].func->getLoc();

  // Create gpu wrapper function
  // Convert tensor types to memref types for the wrapper function
  SmallVector<Type, 4> wrapperArgTypes;
  for (Type t : kernels[0].params) {
    if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
      wrapperArgTypes.push_back(
          MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
    } else {
      wrapperArgTypes.push_back(t);
    }
  }
  std::string funcNameGpu = funcName + "_gpu";
  auto gpuWrapperFuncType = b.getFunctionType(wrapperArgTypes, {});
  auto gpuWrapperFunc =
      func::FuncOp::create(loc, StringRef(funcNameGpu), gpuWrapperFuncType);
  module.push_back(gpuWrapperFunc);

  // Emit device selection
  if (deviceNum.getNumOccurrences() > 0) {
    const int32_t priority = 122;
    const StringRef constructorName = "setDeviceCtor";
    auto func = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(constructorName);
    if (!func) {
      func = mlir::LLVM::LLVMFuncOp::create(
          b, module.getLoc(), constructorName,
          mlir::LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                            {}));
      module.push_back(func);

      Block *block = func.addEntryBlock(b);
      b.setInsertionPoint(block, block->begin());
      gpu::SetDefaultDeviceOp::create(
          b, loc,
          arith::ConstantIntOp::create(b, loc, b.getIntegerType(32),
                                       deviceNum.getValue()));
      mlir::LLVM::ReturnOp::create(b, loc, ValueRange{});

      b.setInsertionPointToEnd(module.getBody());
      mlir::LLVM::GlobalCtorsOp::create(
          b, loc, b.getArrayAttr(mlir::SymbolRefAttr::get(func)),
          b.getI32ArrayAttr({priority}),
          b.getArrayAttr(mlir::LLVM::ZeroAttr::get(context)));
    }
  }

  // Emit gpu convolution logic.
  Block *block = gpuWrapperFunc.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  SmallVector<Value, 4> cpuMem;
  SmallVector<Value, 4> gpuMem;
  for (auto pair : llvm::enumerate(kernels[0].params)) {
    Value arg = block->getArgument(pair.index());
    cpuMem.push_back(arg);

    // Emit GPU memory allocation function calls.
    auto gpuAllocOp = gpu::AllocOp::create(
        b, loc, arg.getType(), Type(), /*asyncDependencies=*/ValueRange{},
        /*dynamicSizes=*/ValueRange{}, /*symbolOperands=*/ValueRange{});
    Value gpuAlloc = gpuAllocOp.getResult(0);
    gpuMem.push_back(gpuAlloc);

    // Emit CPU->GPU memcpy function calls.
    gpu::MemcpyOp::create(b, loc, TypeRange{}, ValueRange{gpuAlloc, arg});
  }

  // Emit kernel function call, repeating it if needed.
  // We assume that the repeated atomic add usages in a wrw kernel will not
  // substantially impact performance as the result becomes large
  auto emitWrappedCall = [&kernels, &gpuMem,
                          &outIndices](OpBuilder &b, Location loc,
                                       Value ignoredIv, ValueRange noArgs) {
    for (const auto &kernel : kernels) {
      // Check if kernel expects tensor arguments
      // Use kernel.params which stores the function argument types
      bool expectsTensors =
          !kernel.params.empty() && isa<TensorType>(kernel.params.front());

      if (expectsTensors) {
        SmallVector<Value, 4> tensorArgs;
        for (Value memrefArg : gpuMem) {
          tensorArgs.push_back(rock::getAsTensor(b, loc, memrefArg, true));
        }

        auto callOp = func::CallOp::create(b, loc, kernel.func, tensorArgs);

        // Result should be stored back to the corresponding output memref.
        // Kernel returns (Output, LSE, ...) while args are (..., LSE,
        // Output), so map result i to the (numResults - 1 - i)-th-from-last
        // argument.
        for (auto [resultIdx, result] : llvm::enumerate(callOp.getResults())) {
          int32_t outIdx = outIndices[resultIdx];
          auto outMemrefType = cast<MemRefType>(gpuMem[outIdx].getType());
          Value resultMemref =
              bufferization::ToBufferOp::create(b, loc, outMemrefType, result);
          memref::CopyOp::create(b, loc, resultMemref, gpuMem[outIdx]);
        }
      } else {
        // Legacy memref-based kernel - call directly
        func::CallOp::create(b, loc, kernel.func, gpuMem);
      }
    }
    if (ignoredIv) { // we're creating an actual loop
      scf::YieldOp::create(b, loc);
    }
  };
  if (kernelRepeats > 1) {
    Value zeroOp = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
    Value kernelRepeatsOp =
        b.createOrFold<arith::ConstantIndexOp>(loc, kernelRepeats);
    Value step = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
    scf::ForOp::create(b, loc, zeroOp, kernelRepeatsOp, step,
                       /*initArgs=*/{}, emitWrappedCall);
  } else {
    emitWrappedCall(b, loc, nullptr, {});
  }

  for (auto pair : llvm::enumerate(kernels[0].params)) {
    uint32_t i = pair.index();
    gpu::MemcpyOp::create(b, loc, TypeRange{},
                          ValueRange{cpuMem[i], gpuMem[i]});
    gpu::DeallocOp::create(b, loc, TypeRange{}, ValueRange{gpuMem[i]});
  }
  // hack for split-kv:
  // use LSE and output tensors to compute final attention result
  if (splitKV > 1) {
    int64_t nPerBlock = getMandNPerBlock(b, params).second;

    // TODO: causal masking is not implemented yet
    // typically, causal masking is used in the prefill phase, where split-KV is
    // not typically used
    if (sequenceLengthQ > nPerBlock && causalMasking) {
      llvm::errs() << "Causal masking + split-KV is not supported with "
                      "sequenceLengthQ > nPerBlock (rocmlir-gen limitation)\n";
      exit(1);
    }

    // The split-KV finalization below masks out split slots that did not run.
    // Match GridwiseAttnToBlockwise's Triton lowering, which splits the
    // current_seq_len / key-sequence work over gemm0NPerBlock.
    SmallVector<int32_t> validSplitKV = computeValidSplitKV(nPerBlock);

    // split KV to batch
    Value resultTensor = bufferization::ToTensorOp::create(
        b, loc,
        memref::getTensorTypeFromMemRefType(
            cpuMem[cpuMem.size() - 1].getType()),
        cpuMem[cpuMem.size() - 1], true, false);
    Value lseTensor = bufferization::ToTensorOp::create(
        b, loc,
        memref::getTensorTypeFromMemRefType(
            cpuMem[cpuMem.size() - 2].getType()),
        cpuMem[cpuMem.size() - 2], true, false);
    auto out = computeFinalAttentionStage(b, loc, resultTensor, lseTensor,
                                          validSplitKV);
    Value outMemref = bufferization::ToBufferOp::create(
        b, loc,
        cast<mlir::bufferization::BufferLikeType>(
            cpuMem[cpuMem.size() - 1].getType()),
        out);
    memref::CopyOp::create(b, loc, outMemref, cpuMem[cpuMem.size() - 1]);
  }
  func::ReturnOp::create(b, loc, ValueRange{});
  return gpuWrapperFunc;
}

static llvm::SmallString<32> archChip() {
  RocmDeviceName targetInfo;
  if (failed(targetInfo.parse(arch.getValue()))) {
    llvm::errs() << "Invalid architecture name: " << arch << "\n";
    exit(1);
  }
  return targetInfo.getChip();
}

// Map data type string to MLIR type
static Type typeFromString(StringRef name, MLIRContext *ctx) {
  std::optional<Type> result =
      llvm::StringSwitch<std::optional<Type>>(name)
          .Case("f32", Float32Type::get(ctx))
          .Case("fp32", Float32Type::get(ctx))
          .Case("f16", Float16Type::get(ctx))
          .Case("fp16", Float16Type::get(ctx))
          .Case("bf16", BFloat16Type::get(ctx))
          .Case("i8", IntegerType::get(ctx, 8))
          .Case("i32", IntegerType::get(ctx, 32))
          .Case("f4E2M1FN", Float4E2M1FNType::get(ctx))
          .Case("f8E5M2", Float8E5M2Type::get(ctx))
          .Case("f8E4M3FN", Float8E4M3FNType::get(ctx))
          .Case("f8E5M2FNUZ", Float8E5M2FNUZType::get(ctx))
          .Case("f8E4M3FNUZ", Float8E4M3FNUZType::get(ctx))
          .Case("f8E8M0FNU", Float8E8M0FNUType::get(ctx))
          .Default(std::nullopt);
  if (!result) {
    llvm::errs() << "Unknown data type: " << name << "\n";
    exit(1);
  }
  return *result;
}

// Determine the range and seed for the random data generator
static int getRandomSeed() {
  std::string rseed = randomSeed;
  if (rseed[0] >= '0' and rseed[0] <= '9')
    return std::stoi(rseed);
  return -1;
}

static std::tuple<short, short> getRandomTestData(int idx, bool isRandFloat) {
  short min = 1, max = 1;

  // Map the logical tensor name from -rand_side to the argument index.
  // The kernel arg layout puts the store destination last:
  //   Fwd:       [filter(0), input(1), output(2)]
  //   BwdData:   [filter(0), output(1), input(2)]
  //   BwdWeight: [input(0), output(1), filter(2)]
  // Each string encodes the kernel arg order as [arg0, arg1, arg2] where
  // the characters 'f', 'i', 'o' denote filter, input, output respectively.
  // Finding the -rand_side character in the string gives its arg index.
  StringRef argOrder = "fio"; // default (Fwd / Gemm)
  if (operation.getNumOccurrences() > 0) {
    if (operation.getValue() == rock::KernelType::ConvBwdData)
      argOrder = "foi";
    else if (operation.getValue() == rock::KernelType::ConvBwdWeight)
      argOrder = "iof";
  }
  char side = randomSide.getValue()[0];
  size_t pos = argOrder.find(side);
  int32_t idxSpec = (pos != StringRef::npos) ? static_cast<int32_t>(pos) : -1;

  if (randomSeed != "none" && randomSeed != "fixed") {
    if ((idxSpec >= 0) && (idxSpec != idx)) {
    } else if (isRandFloat) {
      // generate random floats in [rand_min, rand_max)
      min = randMin.getValue();
      max = randMax.getValue();
    } else {
      // generate random integer in [rand_min_int, rand_max_int)
      min = randMinInt.getValue();
      max = randMaxInt.getValue();
    }
  }
  return std::make_tuple(min, max);
}

static llvm::SmallVector<float, 3> getTensorInitPattern(Type elemType,
                                                        int idx) {
  llvm::SmallVector<float, 3> pattern;
  if (randomSeed == "none") {
    float fixedVal = 1.0f;
    bool isRandFloat = (randomDataType == "float");
    if (llvm::is_contained(randomTypeIntForInputs, idx)) {
      isRandFloat = false;
    }
    if (isRandFloat)
      // Clamp the fixed random float by 0.1 to avoid infs in some f16 tests
      fixedVal *= 0.1f;
    pattern = {static_cast<float>(fixedVal)};
  } else if (randomSeed == "fixed") {
    if (elemType.isIntOrIndex())
      pattern = {1.0, -1.0, 2.0};
    // float4E2M1FN can only represent 8 values. Use a small set of values to
    // avoid quantization error
    else if (isa<Float4E2M1FNType>(elemType)) {
      pattern = {1, -4, 0.5, 1.5};
      // Float8E8M0FNU is only supposed to represent values that are in powers
      // of 2 Use a small set of such values to avoid quantization error
    } else if (isa<Float8E8M0FNUType>(elemType)) {
      pattern = {1, 2, 4, 8, 0.5, 0.25, 0.125};
    } else
      pattern = {0.5, -1, 0.75};
  } else {
    llvm_unreachable("We shouldn't be here for random values");
  }
  return pattern;
}

static LogicalResult populateTensorFillLogic(OpBuilder &b, Location loc,
                                             ArrayRef<float> pattern,
                                             Type elemType, Value toFill) {
  // TODO(kdrewnia) Refactor this to create the constant vector up front
  Value constantsVec = rock::createZeroConstantOp(
      b, loc, VectorType::get(pattern.size(), elemType));
  for (auto v : llvm::enumerate(pattern)) {
    Value vOp;
    if (elemType.isIntOrIndex()) {
      vOp = rock::createConstantIntOp(b, loc, elemType, elemType,
                                      static_cast<int64_t>(v.value()));
    } else {
      vOp = rock::createConstantFloatOp(b, loc, elemType, elemType, v.value());
    }
    constantsVec =
        vector::InsertOp::create(b, loc, vOp, constantsVec, v.index())
            .getResult();
  }

  Value toFillFlat = makeNDMemRef(b, toFill, 1);
  MemRefType flatType = cast<MemRefType>(toFillFlat.getType());
  SmallVector<int64_t, 1> lowerBounds;
  SmallVector<int64_t, 1> upperBounds;
  SmallVector<int64_t, 1> steps;
  AffineMap rowMajorMap = AffineMap::getConstantMap(0, b.getContext());
  if (!flatType.getShape().empty()) {
    AffineExpr rowMajor = b.getAffineDimExpr(0);
    rowMajor = rowMajor % b.getAffineConstantExpr(pattern.size());
    rowMajorMap = AffineMap::get(1, 0, {rowMajor}, b.getContext());
    lowerBounds.push_back(0);
    upperBounds.push_back(flatType.getNumElements());
    steps.push_back(1);
  }

  affine::buildAffineLoopNest(
      b, loc, lowerBounds, upperBounds, steps,
      [rowMajorMap, &constantsVec, toFillFlat](OpBuilder &b, Location loc,
                                               ValueRange ivs) {
        auto selectorOp =
            affine::AffineApplyOp::create(b, loc, rowMajorMap, ivs);
        Value toStore = vector::ExtractOp::create(b, loc, constantsVec,
                                                  selectorOp->getResult(0))
                            .getResult();
        memref::StoreOp::create(b, loc, toStore, toFillFlat, ivs);
      });
  return success();
}

static LogicalResult
populateRandomTensorFillLogic(OpBuilder &b, Location loc, ModuleOp module,
                              Type elemType, Value toFill, int idx,
                              std::optional<float> prefillValue) {
  llvm::SmallDenseMap<short, Value> i16vals;
  auto getI16Val = [&](short v) {
    if (i16vals.find(v) == i16vals.end()) {
      auto i16Type = b.getIntegerType(16);
      i16vals.try_emplace(
          v, b.createOrFold<arith::ConstantIntOp>(loc, i16Type, v));
    }
    return i16vals[v];
  };

  Value toFillFlat = makeNDMemRef(b, toFill, 1);
  auto flatType = cast<MemRefType>(toFillFlat.getType());

  bool isRandFloat = (randomDataType == "float");
  if (llvm::is_contained(randomTypeIntForInputs, idx)) {
    isRandFloat = false;
  }
  func::FuncOp randFunc;
  Type i16 = b.getI16Type();
  Type f32 = b.getF32Type();
  if (isRandFloat)
    randFunc = makeFuncDecl(module, "randomFloatValue", {i16, i16}, {f32});
  else
    randFunc = makeFuncDecl(module, "randomIntegerValue", {i16, i16}, {f32});

  short min, max;
  std::tie(min, max) = getRandomTestData(idx, isRandFloat);
  Value minConst = getI16Val(min), maxConst = getI16Val(max);

  SmallVector<int64_t, 1> lowerBounds;
  SmallVector<int64_t, 1> upperBounds;
  SmallVector<int64_t, 1> steps;

  if (!flatType.getShape().empty()) {
    lowerBounds.push_back(0);
    upperBounds.push_back(flatType.getNumElements());
    steps.push_back(1);
  }

  affine::buildAffineLoopNest(
      b, loc, lowerBounds, upperBounds, steps,
      [prefillValue, elemType, randFunc, toFillFlat, minConst,
       maxConst](OpBuilder &b, Location loc, ValueRange ivs) {
        Value randVal;
        if (prefillValue.has_value()) {
          if (elemType.isIntOrIndex()) {
            if (std::isinf(*prefillValue) || std::isnan(*prefillValue) ||
                *prefillValue >
                    static_cast<float>(std::numeric_limits<int64_t>::max()) ||
                *prefillValue <
                    static_cast<float>(std::numeric_limits<int64_t>::min())) {
              llvm::report_fatal_error(
                  "prefill value cannot be cast to int64_t for "
                  "integer element type");
            }
            randVal =
                rock::createConstantIntOp(b, loc, elemType, elemType,
                                          static_cast<int64_t>(*prefillValue));
          } else
            randVal = rock::createConstantFloatOp(b, loc, elemType, elemType,
                                                  *prefillValue);
        } else {
          auto randFloatCall = func::CallOp::create(
              b, loc, randFunc, ValueRange{minConst, maxConst});
          Value randFloat = randFloatCall.getResult(0);
          if (elemType.isIntOrIndex())
            randVal = arith::FPToSIOp::create(b, loc, elemType, randFloat);
          else if (!elemType.isF32())
            randVal = arith::TruncFOp::create(b, loc, elemType, randFloat);
          else
            randVal = randFloat;
        }

        memref::StoreOp::create(b, loc, randVal, toFillFlat, ivs);
      });

  return success();
}

struct ConvTensorDimInfo {
  unsigned nonImg1Dim;
  int64_t nonImg1Len;
  unsigned nonImg2Dim;
  int64_t nonImg2Len;
  unsigned gDim;
  int64_t gLen;
  SmallVector<unsigned, 4> imageDims;
  SmallVector<int64_t, 4> imageLens;
};

/// Given the layout string for some tensor (ex ngc01 or gk012c), the tensor
/// shape of the value whose layout has that form, and the identifiers ('n',
/// 'c', or 'k') for the two non-image dimensions expected in the layout, return
/// the positions and lengths of those two non-image dimensions and the image
/// dimensions (in order).
static ConvTensorDimInfo parseConvTensorLayout(StringRef layout,
                                               ArrayRef<int64_t> shape,
                                               char nonImg1Sym,
                                               char nonImg2Sym) {
  // The two non-image dimensions and the group.
  unsigned nImageDims = shape.size() - 3;
  // Neither value is particularly special, excetpt that I used -2 because -1 is
  // the dynamic dimension indicator and we might need that later.
  SmallVector<unsigned, 4> imageDims(nImageDims, 0xdeadbeef);
  SmallVector<int64_t, 4> imageLens(nImageDims, -2);
  unsigned nonImg1Dim, nonImg2Dim, gDim = 0xdeadbeef;
  int64_t nonImg1Len, nonImg2Len, gLen = -2;

  for (auto [pos, dim, len] : llvm::enumerate(layout, shape)) {
    if (dim == nonImg1Sym) {
      nonImg1Dim = pos;
      nonImg1Len = len;
    } else if (dim == nonImg2Sym) {
      nonImg2Dim = pos;
      nonImg2Len = len;
    } else if (dim >= '0' && dim <= '9') {
      size_t dimIdx = dim - '0';
      if (dimIdx >= nImageDims) {
        llvm::errs() << "Dimension value '" << dimIdx << "' too large\n";
        exit(1);
      }
      imageDims[dimIdx] = pos;
      imageLens[dimIdx] = len;
    } else if (dim == 'g') {
      gDim = pos;
      gLen = len;
    } else {
      llvm::errs() << "Unknown layout key '" << dim << "'\n";
      exit(1);
    }
  }
  return ConvTensorDimInfo{nonImg1Dim, nonImg1Len, nonImg2Dim, nonImg2Len,
                           gDim,       gLen,       imageDims,  imageLens};
}

/// Arrange values/expressions according to a convolution tensor layout.
/// Works with both Value (for runtime indices) and AffineExpr (for indexing
/// maps).
template <typename T>
static SmallVector<T> arrangeByConvLayout(const ConvTensorDimInfo &layout,
                                          T nonImg1, T nonImg2, T g,
                                          ArrayRef<T> image) {
  SmallVector<T> result;
  result.resize_for_overwrite(image.size() + 3);
  result[layout.nonImg1Dim] = nonImg1;
  result[layout.nonImg2Dim] = nonImg2;
  result[layout.gDim] = g;
  for (auto [idx, value] : llvm::zip(layout.imageDims, image))
    result[idx] = value;
  return result;
}

static func::FuncOp getMemcpyFuncDecl(ModuleOp module, const MemRefType srcType,
                                      const MemRefType destType) {
  OpBuilder b(module.getContext());

  assert(srcType.getRank() <= 1 && "Memcopy takes 1-D sources");
  assert(destType.getRank() <= 1 && "Memcopy takes 1-D destinations");

  Type srcElemType = srcType.getElementType();
  Type dstElemType = destType.getElementType();
  // memcpy_<srcElemType>_<dstElemType>_(srcSize|any)
  std::string funcName = "_memcpy_";
  llvm::raw_string_ostream funcNameStr(funcName);
  funcNameStr << srcElemType << "_" << dstElemType << "_";

  int64_t numElements = -1;
  if (srcType.hasStaticShape())
    numElements = srcType.getNumElements();

  if (numElements != -1)
    funcNameStr << numElements;
  else
    funcNameStr << "any";

  if ((numElements == -1 && !destType.hasStaticShape()) ||
      numElements != destType.getNumElements())
    assert(0 && "Called for an uneven memcpy");

  func::FuncOp func = module.lookupSymbol<func::FuncOp>(funcName);
  if (func) // already exists
    return func;

  Location loc = b.getUnknownLoc();

  // clang-format off
  // func _memcpy_<srcElemType>_<dstElemType>_<size> (%arg0 : memref<sizexf32>, %arg1 : memref<sizexf16>) {
  //   %size = memref.dim %arg0(%c0)
  //   scf.for %i0 = %c0 to %size step %c1 {
  //     %2 = load %arg0[%i0] : memref<?xf32>
  //     store %2, %arg1[%i0] : memref<?xf32>
  //   }
  // }
  // clang-format on

  // Emit function definition
  func = func::FuncOp::create(loc, funcName,
                              b.getFunctionType({srcType, destType}, {}));

  module.push_back(func);

  // Create a new block
  Block *block = func.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  auto src = block->getArgument(0);
  auto dst = block->getArgument(1);

  auto insertConversionLogic = [&](OpBuilder &opBuilder, Value loadOp) {
    Value newLoadOp = loadOp;
    if (srcElemType != dstElemType) {
      // insert conversion logic
      auto srcBitWidth = srcElemType.getIntOrFloatBitWidth();
      auto dstBitWidth = dstElemType.getIntOrFloatBitWidth();
      if (srcElemType.isIntOrIndex()) {
        if (dstElemType.isIntOrIndex()) {
          if (srcBitWidth < dstBitWidth)
            newLoadOp =
                arith::ExtSIOp::create(opBuilder, loc, dstElemType, loadOp);
          else
            newLoadOp =
                arith::TruncIOp::create(opBuilder, loc, dstElemType, loadOp);
        } else {
          assert(isa<FloatType>(dstElemType));
          newLoadOp =
              arith::SIToFPOp::create(opBuilder, loc, dstElemType, loadOp);
        }
      } else {
        assert(isa<FloatType>(srcElemType));
        if (dstElemType.isIntOrIndex()) {
          newLoadOp =
              arith::FPToSIOp::create(opBuilder, loc, dstElemType, loadOp);
        } else {
          if (srcBitWidth < dstBitWidth)
            newLoadOp =
                arith::ExtFOp::create(opBuilder, loc, dstElemType, loadOp);
          else
            newLoadOp =
                arith::TruncFOp::create(opBuilder, loc, dstElemType, loadOp);
        }
      }
    }
    return newLoadOp;
  };

  if (srcType.getRank() == 1) {
    auto cst0Op = arith::ConstantIndexOp::create(b, loc, 0);
    auto cst1Op = arith::ConstantIndexOp::create(b, loc, 1);
    auto size = memref::DimOp::create(b, loc, src, cst0Op);

    auto loop0 = scf::ForOp::create(b, loc, cst0Op, size, cst1Op);
    auto bt0 = OpBuilder::atBlockTerminator(loop0.getBody());
    auto iv0 = loop0.getInductionVar();

    Value loadOp = memref::LoadOp::create(bt0, loc, src, ValueRange{iv0});
    loadOp = insertConversionLogic(bt0, loadOp);
    memref::StoreOp::create(bt0, loc, loadOp, dst, ValueRange{iv0});
  } else {
    Value loadOp = memref::LoadOp::create(b, loc, src, ValueRange{});
    loadOp = insertConversionLogic(b, loadOp);
    memref::StoreOp::create(b, loc, loadOp, dst, ValueRange{});
  }

  func::ReturnOp::create(b, loc, ValueRange{});

  return func;
}

static void emitMemcpy(OpBuilder &b, Value src, Value dst) {
  auto module = b.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  auto loc = b.getUnknownLoc();

  Value srcFlat = makeNDMemRef(b, src, 1);
  Value dstFlat = makeNDMemRef(b, dst, 1);
  auto srcFlatType = cast<MemRefType>(srcFlat.getType());
  auto dstFlatType = cast<MemRefType>(dstFlat.getType());

  if (srcFlatType == dstFlatType) {
    memref::CopyOp::create(b, loc, srcFlat, dstFlat);
  } else {
    auto memcpyFunc = getMemcpyFuncDecl(module, srcFlatType, dstFlatType);
    func::CallOp::create(b, loc, memcpyFunc, ValueRange{srcFlat, dstFlat});
  }
}

/// Converts tensor element type to f32 if needed (handles both float and
/// integer types).
static Value ensureFloatIsF32(OpBuilder &b, Location loc, Value tensor,
                              Type floatType) {
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  assert(tensorType && "ensureFloatIsF32 expects tensor type");
  Type elemType = tensorType.getElementType();
  if (elemType.isF32())
    return tensor;

  // Ensure we're not accidentally truncating from larger types
  if (auto floatElemType = dyn_cast<FloatType>(elemType)) {
    assert(floatElemType.getWidth() <= 32 &&
           "ensureFloatIsF32 does not support types larger than f32");
  } else if (auto intElemType = dyn_cast<IntegerType>(elemType)) {
    // Allow up to i32 for i8 GEMM/conv verification.
    assert(intElemType.getWidth() <= 32 &&
           "ensureFloatIsF32 does not support integer types larger than i32");
  }

  // Create a new tensor with f32 elements and convert
  auto f32TensorType = RankedTensorType::get(tensorType.getShape(), floatType);
  Value emptyTensor =
      tensor::EmptyOp::create(b, loc, f32TensorType, ValueRange{});

  // Use linalg.generic to convert each element
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(tensorType.getRank(), b.getContext());
  SmallVector<utils::IteratorType> iteratorTypes(tensorType.getRank(),
                                                 utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      b, loc, f32TensorType, ValueRange{tensor}, ValueRange{emptyTensor},
      ArrayRef<AffineMap>{identityMap, identityMap}, iteratorTypes,
      [elemType](OpBuilder &nestedB, Location nestedLoc, ValueRange args) {
        Value result;
        if (isa<IntegerType>(elemType)) {
          result = arith::SIToFPOp::create(nestedB, nestedLoc,
                                           nestedB.getF32Type(), args[0]);
        } else {
          result = arith::ExtFOp::create(nestedB, nestedLoc,
                                         nestedB.getF32Type(), args[0]);
        }
        linalg::YieldOp::create(nestedB, nestedLoc, result);
      });
  return genericOp.getResult(0);
}

/// Helper function for linalg.generic convolution body (MAC operation) - f32
static void convBodyBuilderF32(OpBuilder &b, Location loc,
                               ValueRange blockArgs) {
  assert(blockArgs.size() == 3 && "convBodyBuilder expects 3 arguments");
  Value inputVal = blockArgs[0];
  Value filterVal = blockArgs[1];
  Value outputVal = blockArgs[2];
  Value mul = arith::MulFOp::create(b, loc, inputVal, filterVal);
  Value add = arith::AddFOp::create(b, loc, outputVal, mul);
  linalg::YieldOp::create(b, loc, add);
}

/// Helper function for linalg.generic convolution body (MAC operation) - i32.
/// Used for i8 inputs to match the GPU's i32 accumulator semantics
/// (e.g. the i8 MFMA instructions accumulate into i32).
static void convBodyBuilderI32(OpBuilder &b, Location loc,
                               ValueRange blockArgs) {
  assert(blockArgs.size() == 3 && "convBodyBuilder expects 3 arguments");
  Value inputVal = blockArgs[0];  // i8
  Value filterVal = blockArgs[1]; // i8
  Value outputVal = blockArgs[2]; // i32
  Type i32Type = b.getIntegerType(32);
  Value inputExt = arith::ExtSIOp::create(b, loc, i32Type, inputVal);
  Value filterExt = arith::ExtSIOp::create(b, loc, i32Type, filterVal);
  Value mul = arith::MulIOp::create(b, loc, inputExt, filterExt);
  Value add = arith::AddIOp::create(b, loc, outputVal, mul);
  linalg::YieldOp::create(b, loc, add);
}

/// Emit a grouped convolution using linalg.generic.
/// Builds indexing maps based on actual tensor layouts - no transposes needed!
/// The layout info tells us where each dimension is in the original tensor.
using ConvBodyBuilder = void (*)(OpBuilder &, Location, ValueRange);
static Value emitConvGeneric(
    OpBuilder &b, Location loc, RankedTensorType resultType, Value input,
    Value filter, Value zero, const ConvTensorDimInfo &inputInfo,
    const ConvTensorDimInfo &filterInfo, const ConvTensorDimInfo &outputInfo,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations,
    ConvBodyBuilder bodyBuilder = convBodyBuilderF32) {
  MLIRContext *ctx = b.getContext();
  int64_t rank = cast<RankedTensorType>(input.getType()).getRank();
  assert(rank >= 3 && "emitConvGeneric expects at least 3 dimensions");
  int64_t dim = rank - 3; // number of spatial dimensions

  // Iteration domain:
  //   parallel:  batch, group, filter (k), oh_0 .. oh_{dim-1}
  //   reduction: channel (c), kh_0 .. kh_{dim-1}
  int64_t totalDims = 4 + 2 * dim;
  SmallVector<AffineExpr> d;
  for (int64_t i = 0; i < totalDims; ++i)
    d.push_back(getAffineDimExpr(i, ctx));

  AffineExpr batch = d[0], group = d[1], filterExpr = d[2];
  AffineExpr channel = d[3 + dim];

  // Build input indexing map based on actual input layout
  // ih_i = oh_i * stride + kh_i * dilation
  SmallVector<AffineExpr> inputImageExprs;
  for (int64_t i = 0; i < dim; ++i)
    inputImageExprs.push_back(d[3 + i] * strides[i] +
                              d[4 + dim + i] * dilations[i]);
  SmallVector<AffineExpr> inputExprs = arrangeByConvLayout(
      inputInfo, batch, channel, group, ArrayRef<AffineExpr>(inputImageExprs));

  // Build filter indexing map based on actual filter layout
  SmallVector<AffineExpr> filterImageExprs;
  for (int64_t i = 0; i < dim; ++i)
    filterImageExprs.push_back(d[4 + dim + i]);
  SmallVector<AffineExpr> filterExprs =
      arrangeByConvLayout(filterInfo, filterExpr, channel, group,
                          ArrayRef<AffineExpr>(filterImageExprs));

  // Build output indexing map based on actual output layout
  SmallVector<AffineExpr> outputImageExprs;
  for (int64_t i = 0; i < dim; ++i)
    outputImageExprs.push_back(d[3 + i]);
  SmallVector<AffineExpr> outputExprs =
      arrangeByConvLayout(outputInfo, batch, filterExpr, group,
                          ArrayRef<AffineExpr>(outputImageExprs));

  SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(totalDims, 0, inputExprs, ctx),
      AffineMap::get(totalDims, 0, filterExprs, ctx),
      AffineMap::get(totalDims, 0, outputExprs, ctx)};

  SmallVector<utils::IteratorType> iteratorTypes(3 + dim,
                                                 utils::IteratorType::parallel);
  iteratorTypes.append(1 + dim, utils::IteratorType::reduction);

  return linalg::GenericOp::create(b, loc, resultType,
                                   ValueRange{input, filter}, zero,
                                   indexingMaps, iteratorTypes, bodyBuilder)
      .getResult(0);
}

/// Emit backward weight convolution using linalg.generic.
/// Builds indexing maps based on actual tensor layouts - no transposes needed!
/// Computes: filter_grad = sum_{n,oh,ow} output_grad * input
static Value emitBwdWeightConvGeneric(
    OpBuilder &b, Location loc, RankedTensorType resultType, Value input,
    Value outputGrad, Value zero, const ConvTensorDimInfo &inputInfo,
    const ConvTensorDimInfo &outputInfo, const ConvTensorDimInfo &filterInfo,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations) {
  MLIRContext *ctx = b.getContext();
  int64_t rank = cast<RankedTensorType>(input.getType()).getRank();
  int64_t dim = rank - 3;

  // Iteration domain for backward weight:
  //   parallel:  group, k, c, kh_0 .. kh_{dim-1}
  //   reduction: n, oh_0 .. oh_{dim-1}
  int64_t totalDims = 4 + 2 * dim;
  SmallVector<AffineExpr> d;
  for (int64_t i = 0; i < totalDims; ++i)
    d.push_back(getAffineDimExpr(i, ctx));

  AffineExpr group = d[0], k = d[1], c = d[2];
  AffineExpr n = d[3 + dim];

  // Build input indexing map based on actual input layout
  // ih_i = oh_i * stride + kh_i * dilation
  SmallVector<AffineExpr> inputImageExprs;
  for (int64_t i = 0; i < dim; ++i)
    inputImageExprs.push_back(d[4 + dim + i] * strides[i] +
                              d[3 + i] * dilations[i]);
  SmallVector<AffineExpr> inputExprs = arrangeByConvLayout(
      inputInfo, n, c, group, ArrayRef<AffineExpr>(inputImageExprs));

  // Build output grad indexing map based on actual output layout
  SmallVector<AffineExpr> outputImageExprs;
  for (int64_t i = 0; i < dim; ++i)
    outputImageExprs.push_back(d[4 + dim + i]);
  SmallVector<AffineExpr> outputGradExprs = arrangeByConvLayout(
      outputInfo, n, k, group, ArrayRef<AffineExpr>(outputImageExprs));

  // Build filter grad (result) indexing map based on actual filter layout
  SmallVector<AffineExpr> filterImageExprs;
  for (int64_t i = 0; i < dim; ++i)
    filterImageExprs.push_back(d[3 + i]);
  SmallVector<AffineExpr> filterGradExprs = arrangeByConvLayout(
      filterInfo, k, c, group, ArrayRef<AffineExpr>(filterImageExprs));

  SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(totalDims, 0, inputExprs, ctx),
      AffineMap::get(totalDims, 0, outputGradExprs, ctx),
      AffineMap::get(totalDims, 0, filterGradExprs, ctx)};

  SmallVector<utils::IteratorType> iteratorTypes(3 + dim,
                                                 utils::IteratorType::parallel);
  iteratorTypes.append(1 + dim, utils::IteratorType::reduction);

  return linalg::GenericOp::create(
             b, loc, resultType, ValueRange{input, outputGrad}, zero,
             indexingMaps, iteratorTypes, convBodyBuilderF32)
      .getResult(0);
}

/// Dilate a tensor by inserting zeros between elements in spatial dimensions.
/// Layout-aware: uses actual spatial dimension positions from layout info.
static Value dilateTensor(OpBuilder &b, Location loc, Value tensor,
                          ArrayRef<int64_t> strides,
                          ArrayRef<unsigned> spatialDimPositions) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  ArrayRef<int64_t> shape = tensorType.getShape();
  int64_t dim = spatialDimPositions.size();

  // Check if dilation is needed (all strides == 1 means no dilation)
  bool needsDilation = !llvm::all_of(strides, [](int64_t s) { return s == 1; });
  if (!needsDilation)
    return tensor;

  // Compute dilated shape at actual spatial positions
  SmallVector<int64_t> dilatedShape(shape.begin(), shape.end());
  for (int64_t i = 0; i < dim; ++i) {
    unsigned pos = spatialDimPositions[i];
    dilatedShape[pos] = (shape[pos] - 1) * strides[i] + 1;
  }

  auto dilatedType =
      RankedTensorType::get(dilatedShape, tensorType.getElementType());
  Value zeroVal = arith::ConstantOp::create(
      b, loc, b.getZeroAttr(tensorType.getElementType()));

  // Use tensor.generate to create the dilated tensor
  // This avoids the non-invertible indexing map issue with linalg.generic
  SmallVector<Value> dynamicSizes; // all static, so empty
  return tensor::GenerateOp::create(
             b, loc, dilatedType, dynamicSizes,
             [&](OpBuilder &nestedB, Location nestedLoc, ValueRange indices) {
               // Check if all spatial indices are divisible by their strides
               // If so, extract from original tensor; otherwise return zero
               Value allDivisible = arith::ConstantOp::create(
                   nestedB, nestedLoc, nestedB.getBoolAttr(true));
               SmallVector<Value> srcIndices(indices.begin(), indices.end());

               for (int64_t i = 0; i < dim; ++i) {
                 unsigned pos = spatialDimPositions[i];
                 Value strideVal = arith::ConstantIndexOp::create(
                     nestedB, nestedLoc, strides[i]);
                 Value rem = arith::RemUIOp::create(nestedB, nestedLoc,
                                                    indices[pos], strideVal);
                 Value zero =
                     arith::ConstantIndexOp::create(nestedB, nestedLoc, 0);
                 Value isDivisible = arith::CmpIOp::create(
                     nestedB, nestedLoc, arith::CmpIPredicate::eq, rem, zero);
                 allDivisible = arith::AndIOp::create(
                     nestedB, nestedLoc, allDivisible, isDivisible);
                 // Compute source index: indices[pos] / stride
                 srcIndices[pos] = arith::DivUIOp::create(
                     nestedB, nestedLoc, indices[pos], strideVal);
               }

               // Extract from original tensor or return zero
               Value elem = tensor::ExtractOp::create(nestedB, nestedLoc,
                                                      tensor, srcIndices);
               Value result = arith::SelectOp::create(
                   nestedB, nestedLoc, allDivisible, elem, zeroVal);
               tensor::YieldOp::create(nestedB, nestedLoc, result);
             })
      .getResult();
}

/// Flip (reverse) the spatial dimensions of a filter tensor.
/// Layout-aware: uses actual spatial dimension positions from layout info.
static Value flipFilterSpatial(OpBuilder &b, Location loc, Value filter,
                               ArrayRef<int64_t> filterSpatialSizes,
                               ArrayRef<unsigned> spatialDimPositions) {
  auto filterType = cast<RankedTensorType>(filter.getType());
  int64_t dim = spatialDimPositions.size();

  // Use tensor.generate to create the flipped filter
  // This avoids the non-invertible indexing map issue with linalg.generic
  SmallVector<Value> dynamicSizes; // all static, so empty
  return tensor::GenerateOp::create(
             b, loc, filterType, dynamicSizes,
             [&](OpBuilder &nestedB, Location nestedLoc, ValueRange indices) {
               SmallVector<Value> srcIndices(indices.begin(), indices.end());
               // Flip only the spatial indices at their actual positions
               for (int64_t i = 0; i < dim; ++i) {
                 unsigned pos = spatialDimPositions[i];
                 Value maxIdx = arith::ConstantIndexOp::create(
                     nestedB, nestedLoc, filterSpatialSizes[i] - 1);
                 Value flipped = arith::SubIOp::create(nestedB, nestedLoc,
                                                       maxIdx, indices[pos]);
                 srcIndices[pos] = flipped;
               }
               Value elem = tensor::ExtractOp::create(nestedB, nestedLoc,
                                                      filter, srcIndices);
               tensor::YieldOp::create(nestedB, nestedLoc, elem);
             })
      .getResult();
}

/// Emit backward data convolution using linalg.generic.
/// Layout-aware: builds indexing maps based on actual tensor layouts.
/// This is a transposed convolution implemented as:
/// 1. Dilate output grad by inserting (stride-1) zeros between elements
/// 2. Pad the dilated output grad
/// 3. Flip the filter spatially
/// 4. Convolve with flipped filter
static Value emitBwdDataConvGeneric(
    OpBuilder &b, Location loc, RankedTensorType resultType, Value outputGrad,
    Value filter, Value zero, const ConvTensorDimInfo &outputInfo,
    const ConvTensorDimInfo &filterInfo, const ConvTensorDimInfo &inputInfo,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations,
    ArrayRef<int64_t> filterSpatialSizes, ArrayRef<int64_t> paddingLeft,
    ArrayRef<int64_t> paddingRight) {
  MLIRContext *ctx = b.getContext();
  int64_t dim = outputInfo.imageDims.size();

  // Get spatial dimension positions from output layout
  SmallVector<unsigned> outputSpatialPos(outputInfo.imageDims.begin(),
                                         outputInfo.imageDims.end());
  SmallVector<unsigned> filterSpatialPos(filterInfo.imageDims.begin(),
                                         filterInfo.imageDims.end());

  // Step 1: Dilate the output gradient at actual spatial positions
  Value dilatedOutputGrad =
      dilateTensor(b, loc, outputGrad, strides, outputSpatialPos);

  // Step 2: Pad/crop the dilated output gradient to match convolution
  // requirements For transposed conv, we need: paddedSize = inputSize +
  // (filterSize-1)*dilation The adjustment from forward padding may require
  // cropping (negative) or padding.
  auto dilatedType = cast<RankedTensorType>(dilatedOutputGrad.getType());
  int64_t tensorRank = dilatedType.getRank();
  ArrayRef<int64_t> inputShape = resultType.getShape();
  SmallVector<int64_t> dilatedShape(dilatedType.getShape());

  // Compute target shape: what the padded tensor needs to be for convolution
  SmallVector<int64_t> targetShape(dilatedShape);
  for (int64_t i = 0; i < dim; ++i) {
    unsigned pos = outputSpatialPos[i];
    int64_t inputSpatialSize = inputShape[inputInfo.imageDims[i]];
    targetShape[pos] =
        inputSpatialSize + (filterSpatialSizes[i] - 1) * dilations[i];
  }

  // Compute pad/crop amounts per side
  // Low side: (filterSize-1)*dilation - paddingLeft
  // High side: targetSize - dilatedSize - lowDelta
  SmallVector<int64_t> lowDelta(dim), highDelta(dim);
  for (int64_t i = 0; i < dim; ++i) {
    unsigned pos = outputSpatialPos[i];
    int64_t filterPad = (filterSpatialSizes[i] - 1) * dilations[i];
    lowDelta[i] = filterPad - paddingLeft[i];
    highDelta[i] = targetShape[pos] - dilatedShape[pos] - lowDelta[i];
  }

  // Apply crop if any delta is negative (extract_slice)
  Value processed = dilatedOutputGrad;
  SmallVector<int64_t> currentShape(dilatedShape);
  bool needsCrop = llvm::any_of(llvm::seq<int64_t>(0, dim), [&](int64_t i) {
    return lowDelta[i] < 0 || highDelta[i] < 0;
  });

  if (needsCrop) {
    SmallVector<OpFoldResult> offsets(tensorRank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes(tensorRank);
    SmallVector<OpFoldResult> unitStrides(tensorRank, b.getIndexAttr(1));

    for (int64_t i = 0; i < dim; ++i) {
      unsigned pos = outputSpatialPos[i];
      int64_t lowCrop = std::max(-lowDelta[i], int64_t(0));
      int64_t highCrop = std::max(-highDelta[i], int64_t(0));
      offsets[pos] = b.getIndexAttr(lowCrop);
      currentShape[pos] -= lowCrop + highCrop;
    }
    for (int64_t i = 0; i < tensorRank; ++i)
      sizes[i] = b.getIndexAttr(currentShape[i]);

    auto croppedType =
        RankedTensorType::get(currentShape, dilatedType.getElementType());
    processed = tensor::ExtractSliceOp::create(b, loc, croppedType, processed,
                                               offsets, sizes, unitStrides);
  }

  // Apply pad if any delta is positive (tensor.pad)
  bool needsPad = llvm::any_of(llvm::seq<int64_t>(0, dim), [&](int64_t i) {
    return lowDelta[i] > 0 || highDelta[i] > 0;
  });

  if (needsPad) {
    SmallVector<OpFoldResult> lowPad(tensorRank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> highPad(tensorRank, b.getIndexAttr(0));
    for (int64_t i = 0; i < dim; ++i) {
      unsigned pos = outputSpatialPos[i];
      lowPad[pos] = b.getIndexAttr(std::max(lowDelta[i], int64_t(0)));
      highPad[pos] = b.getIndexAttr(std::max(highDelta[i], int64_t(0)));
    }
    Value padValue = arith::ConstantOp::create(
        b, loc, b.getZeroAttr(dilatedType.getElementType()));
    auto paddedType =
        RankedTensorType::get(targetShape, dilatedType.getElementType());
    processed = tensor::PadOp::create(b, loc, paddedType, processed, lowPad,
                                      highPad, padValue)
                    .getResult();
  }

  Value paddedOutputGrad = processed;

  // Step 3: Flip the filter spatially at actual spatial positions
  Value flippedFilter =
      flipFilterSpatial(b, loc, filter, filterSpatialSizes, filterSpatialPos);

  // Step 4: Perform convolution with flipped filter using layout-aware indexing
  // Iteration domain: (n, g, c, ih_0..., k, kh_0...)
  int64_t totalDims = 4 + 2 * dim;
  SmallVector<AffineExpr> d;
  for (int64_t i = 0; i < totalDims; ++i)
    d.push_back(getAffineDimExpr(i, ctx));

  AffineExpr n = d[0], group = d[1], c = d[2];
  AffineExpr k = d[3 + dim];

  // Build padded output grad indexing map based on actual output layout
  SmallVector<AffineExpr> outputImageExprs;
  for (int64_t i = 0; i < dim; ++i)
    outputImageExprs.push_back(d[3 + i] + d[4 + dim + i] * dilations[i]);
  SmallVector<AffineExpr> outputGradExprs = arrangeByConvLayout(
      outputInfo, n, k, group, ArrayRef<AffineExpr>(outputImageExprs));

  // Build flipped filter indexing map based on actual filter layout
  SmallVector<AffineExpr> filterImageExprs;
  for (int64_t i = 0; i < dim; ++i)
    filterImageExprs.push_back(d[4 + dim + i]);
  SmallVector<AffineExpr> filterExprs = arrangeByConvLayout(
      filterInfo, k, c, group, ArrayRef<AffineExpr>(filterImageExprs));

  // Build input grad (result) indexing map based on actual input layout
  SmallVector<AffineExpr> inputImageExprs;
  for (int64_t i = 0; i < dim; ++i)
    inputImageExprs.push_back(d[3 + i]);
  SmallVector<AffineExpr> inputGradExprs = arrangeByConvLayout(
      inputInfo, n, c, group, ArrayRef<AffineExpr>(inputImageExprs));

  SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(totalDims, 0, outputGradExprs, ctx),
      AffineMap::get(totalDims, 0, filterExprs, ctx),
      AffineMap::get(totalDims, 0, inputGradExprs, ctx)};

  SmallVector<utils::IteratorType> iteratorTypes(3 + dim,
                                                 utils::IteratorType::parallel);
  iteratorTypes.append(1 + dim, utils::IteratorType::reduction);

  return linalg::GenericOp::create(
             b, loc, resultType, ValueRange{paddedOutputGrad, flippedFilter},
             zero, indexingMaps, iteratorTypes, convBodyBuilderF32)
      .getResult(0);
}

/// Create a tensor-based CPU convolution kernel using linalg.generic.
/// This function handles forward, backward data, and backward weight
/// operations.
static func::FuncOp
createCPUConvWithMLIR(ModuleOp module,
                      const rock::ConvGenerator::Config &genConfig) {
  assert(genConfig.operation.has_value());
  MLIRContext *ctx = module.getContext();
  OpBuilder b(ctx);
  Location loc = module->getLoc();

  // Determine element types
  Type inputElemType = typeFromString(genConfig.inputDataTypeStr, ctx);
  Type filterElemType = typeFromString(genConfig.filterDataTypeStr, ctx);
  Type outputElemType = typeFromString(genConfig.outputDataTypeStr, ctx);

  if (genConfig.inputDataTypeStr == "i8") {
    inputElemType = b.getI8Type();
    filterElemType = b.getI8Type();
    outputElemType = b.getIntegerType(32);
    assert(genConfig.operation.value() == rock::ConvOpType::Fwd);
  }

  // Create flat tensor types for function signature
  int64_t filterElems = computeProduct(genConfig.filterDimension);
  int64_t inputElems = computeProduct(genConfig.inputDimension);
  int64_t outputElems = computeProduct(genConfig.outputDimension);

  auto filterFlatType = RankedTensorType::get({filterElems}, filterElemType);
  auto inputFlatType = RankedTensorType::get({inputElems}, inputElemType);
  auto outputFlatType = RankedTensorType::get({outputElems}, outputElemType);

  rock::ConvGenerator convGenerator(genConfig);
  bool hasWorkspace = false;
  if (failed(convGenerator.hasWorkspace(b, hasWorkspace))) {
    assert(genConfig.operation.value() == rock::ConvOpType::Fwd);
  }

  // Build argument types in standard [filter, input, output, workspace?] order,
  // then reorder so the store destination (result) is last.
  SmallVector<Type, 4> funcArgTypes = {filterFlatType, inputFlatType,
                                       outputFlatType};
  if (hasWorkspace)
    funcArgTypes.push_back(
        RankedTensorType::get({filterElems}, b.getF32Type()));
  rock::reorderConvArgsForKernel(genConfig.operation.value(), funcArgTypes);
  Type resultFlatType = funcArgTypes.back();

  std::string funcName =
      rock::getNameForConvOpType(genConfig.operation.value()).str();

  funcName += "_cpu";

  // Check if function already exists
  if (auto existingFunc = module.lookupSymbol<func::FuncOp>(funcName))
    return existingFunc;

  auto func = func::FuncOp::create(
      b, loc, funcName, b.getFunctionType(funcArgTypes, {resultFlatType}));
  // Mark as CPU verifier so buildHostLoweringPipeline can identify it
  func->setAttr(rock::CpuVerifierAttr::getMnemonic(), b.getUnitAttr());
  module.push_back(func);

  Block *block = func.addEntryBlock();
  b.setInsertionPointToStart(block);

  // Map block args back to semantic names (inverse of
  // reorderConvArgsForKernel).
  Value filterFlat, inputFlat, outputFlat;
  switch (genConfig.operation.value()) {
  case rock::ConvOpType::Fwd:
    filterFlat = block->getArgument(0);
    inputFlat = block->getArgument(1);
    outputFlat = block->getArgument(2);
    break;
  case rock::ConvOpType::BwdData:
    filterFlat = block->getArgument(0);
    outputFlat = block->getArgument(1);
    inputFlat = block->getArgument(2);
    break;
  case rock::ConvOpType::BwdWeight:
    inputFlat = block->getArgument(0);
    outputFlat = block->getArgument(1);
    filterFlat = block->getArgument(hasWorkspace ? 3 : 2);
    break;
  }

  // i8 convolutions use i32 / f32 accumulation.
  bool isI8Conv = genConfig.inputDataTypeStr == "i8";
  Type computeType =
      isI8Conv ? Type(b.getIntegerType(32)) : Type(b.getF32Type());
  size_t nSpatialDims = genConfig.strideDims.size();

  // Helper to expand flat tensor to logical shape
  auto expandToLogicalShape = [&](Value flat,
                                  ArrayRef<int64_t> logicalShape) -> Value {
    auto flatType = cast<RankedTensorType>(flat.getType());
    Type elemType = flatType.getElementType();
    auto logicalType = RankedTensorType::get(logicalShape, elemType);
    ReassociationIndices allDims = llvm::to_vector(
        llvm::iota_range<int64_t>(0, logicalShape.size(), false));
    return tensor::ExpandShapeOp::create(b, loc, logicalType, flat, allDims);
  };

  // Helper to expand flat tensor to logical shape with f32 elements
  auto expandAndConvertToF32 = [&](Value flat,
                                   ArrayRef<int64_t> logicalShape) -> Value {
    Value expanded = expandToLogicalShape(flat, logicalShape);
    return ensureFloatIsF32(b, loc, expanded, b.getF32Type());
  };

  // Expand tensors to logical shapes
  // For i8, keep original element types; for others, convert to f32
  Value filter, input, output;
  if (isI8Conv) {
    filter = expandToLogicalShape(filterFlat, genConfig.filterDimension);
    input = expandToLogicalShape(inputFlat, genConfig.inputDimension);
    output = expandToLogicalShape(outputFlat, genConfig.outputDimension);
  } else {
    filter = expandAndConvertToF32(filterFlat, genConfig.filterDimension);
    input = expandAndConvertToF32(inputFlat, genConfig.inputDimension);
    output = expandAndConvertToF32(outputFlat, genConfig.outputDimension);
  }

  // Parse tensor layouts - these tell us where each dimension is
  ConvTensorDimInfo filterInfo = parseConvTensorLayout(
      genConfig.filterLayout, genConfig.filterDimension, 'k', 'c');
  ConvTensorDimInfo inputInfo = parseConvTensorLayout(
      genConfig.inputLayout, genConfig.inputDimension, 'n', 'c');
  ConvTensorDimInfo outputInfo = parseConvTensorLayout(
      genConfig.outputLayout, genConfig.outputDimension, 'n', 'k');

  // Get tensor types in original layout
  auto inputType = cast<RankedTensorType>(input.getType());
  auto filterType = cast<RankedTensorType>(filter.getType());
  auto outputType = cast<RankedTensorType>(output.getType());

  // Apply padding to input for Forward and BwdWeight
  // Uses actual spatial dimension positions (no transpose needed)
  bool needsPadding =
      (genConfig.operation.value() == rock::ConvOpType::Fwd ||
       genConfig.operation.value() == rock::ConvOpType::BwdWeight) &&
      (!llvm::all_of(genConfig.paddingLeftDims,
                     [](int64_t p) { return p == 0; }) ||
       !llvm::all_of(genConfig.paddingRightDims,
                     [](int64_t p) { return p == 0; }));

  if (needsPadding) {
    // Pad at actual spatial dimension positions from layout
    SmallVector<OpFoldResult> lowPad(inputType.getRank(), b.getIndexAttr(0));
    SmallVector<OpFoldResult> highPad(inputType.getRank(), b.getIndexAttr(0));
    SmallVector<int64_t> newShape(inputType.getShape());

    for (size_t i = 0; i < nSpatialDims; ++i) {
      int64_t lowP = genConfig.paddingLeftDims[i];
      int64_t highP = genConfig.paddingRightDims[i];
      unsigned pos = inputInfo.imageDims[i]; // actual spatial dim position
      lowPad[pos] = b.getIndexAttr(lowP);
      highPad[pos] = b.getIndexAttr(highP);
      newShape[pos] += lowP + highP;
    }

    Type paddedElemType = inputType.getElementType();
    auto paddedType = RankedTensorType::get(newShape, paddedElemType);
    Value padValue =
        arith::ConstantOp::create(b, loc, b.getZeroAttr(paddedElemType));
    input = tensor::PadOp::create(b, loc, paddedType, input, lowPad, highPad,
                                  padValue)
                .getResult();
    inputType = paddedType;
  }

  // Determine result shape in original layout and create zero tensor
  ArrayRef<int64_t> resultShape;
  switch (genConfig.operation.value()) {
  case rock::ConvOpType::Fwd:
    resultShape = outputType.getShape();
    break;
  case rock::ConvOpType::BwdData:
    resultShape = inputType.getShape();
    break;
  case rock::ConvOpType::BwdWeight:
    resultShape = filterType.getShape();
    break;
  }

  auto resultType = RankedTensorType::get(resultShape, computeType);
  // Use tensor.empty + linalg.fill instead of arith.constant to avoid
  // memref copy during bufferization
  Value zeroVal = rock::createZeroConstantOp(b, loc, computeType);
  Value emptyResult = tensor::EmptyOp::create(b, loc, resultType, ValueRange{});
  Value zeroResult =
      linalg::FillOp::create(b, loc, zeroVal, emptyResult).getResult(0);

  // Emit the convolution using linalg.generic with layout-aware indexing
  // No transposes needed - the indexing maps use actual dimension positions!
  // Use i32 body builder for i8 inputs, f32 otherwise.
  Value result;
  ConvBodyBuilder bodyBuilder =
      isI8Conv ? convBodyBuilderI32 : convBodyBuilderF32;
  switch (genConfig.operation.value()) {
  case rock::ConvOpType::Fwd:
    result = emitConvGeneric(
        b, loc, resultType, input, filter, zeroResult, inputInfo, filterInfo,
        outputInfo, genConfig.strideDims, genConfig.dilationDims, bodyBuilder);
    break;
  case rock::ConvOpType::BwdWeight:
    result = emitBwdWeightConvGeneric(
        b, loc, resultType, input, output, zeroResult, inputInfo, outputInfo,
        filterInfo, genConfig.strideDims, genConfig.dilationDims);
    break;
  case rock::ConvOpType::BwdData:
    result = emitBwdDataConvGeneric(
        b, loc, resultType, output, filter, zeroResult, outputInfo, filterInfo,
        inputInfo, genConfig.strideDims, genConfig.dilationDims,
        filterInfo.imageLens, genConfig.paddingLeftDims,
        genConfig.paddingRightDims);
    break;
  }

  // Result is already in original layout (no transpose needed)
  Value resultOrigLayout = result;

  // Convert back to original element type if needed
  auto resultFlatTensorType = cast<RankedTensorType>(resultFlatType);
  ArrayRef<int64_t> finalResultShape =
      cast<RankedTensorType>(resultOrigLayout.getType()).getShape();
  if (resultFlatTensorType.getElementType() != computeType) {
    auto resultOrigType = RankedTensorType::get(
        finalResultShape, resultFlatTensorType.getElementType());
    Value emptyConvert =
        tensor::EmptyOp::create(b, loc, resultOrigType, ValueRange{});
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(finalResultShape.size(), ctx);
    SmallVector<utils::IteratorType> iteratorTypes(
        finalResultShape.size(), utils::IteratorType::parallel);
    auto convertOp = linalg::GenericOp::create(
        b, loc, resultOrigType, ValueRange{resultOrigLayout},
        ValueRange{emptyConvert}, ArrayRef<AffineMap>{identityMap, identityMap},
        iteratorTypes,
        [](OpBuilder &nestedB, Location nestedLoc, ValueRange args) {
          Value src = args[0];
          Type dstType = args[1].getType();
          Value converted;
          if (isa<IntegerType>(dstType)) {
            converted =
                arith::FPToSIOp::create(nestedB, nestedLoc, dstType, src);
          } else {
            converted =
                arith::TruncFOp::create(nestedB, nestedLoc, dstType, src);
          }
          linalg::YieldOp::create(nestedB, nestedLoc, converted);
        });
    resultOrigLayout = convertOp.getResult(0);
  }

  // Collapse to flat 1D tensor
  ArrayRef<int64_t> finalShape =
      cast<RankedTensorType>(resultOrigLayout.getType()).getShape();
  ReassociationIndices allDims =
      llvm::to_vector(llvm::iota_range<int64_t>(0, finalShape.size(), false));
  Value flatResult = tensor::CollapseShapeOp::create(
      b, loc, resultFlatTensorType, resultOrigLayout, allDims);

  func::ReturnOp::create(b, loc, flatResult);
  return func;
}

static func::FuncOp
createCPUConvFunc(ModuleOp module,
                  const rock::ConvGenerator::Config &genConfig) {
  // Delegate to the tensor-based implementation
  return createCPUConvWithMLIR(module, genConfig);
}

static void getGemmTypes(ArrayRef<Type> elemTypes,
                         SmallVectorImpl<Type> &result, bool isCpuVerifier) {
  Type cElemType = elemTypes[2];
  OpBuilder b(elemTypes[0].getContext());
  // i8 GEMM accumulates in i32 on the GPU (e.g. the i8 MFMA instructions
  // accumulate into i32). Mirror that in the CPU verifier so the reference
  // matches the GPU compute precision.
  if (elemTypes[0].isInteger(8) && isCpuVerifier)
    cElemType = IntegerType::get(cElemType.getContext(), 32);

  int64_t quantK = llvm::divideCeil(gemmK, quantBlockSize);
  SmallVector<int64_t> aDims = {groupSize, transposeA ? gemmK : gemmM,
                                transposeA ? gemmM : gemmK},
                       bDims = {groupSize, transposeB ? gemmN : gemmK,
                                transposeB ? gemmK : gemmN},
                       cDims = {groupSize, transposeC ? gemmN : gemmM,
                                transposeC ? gemmM : gemmN},
                       aScale = {groupSize, transposeScaleA ? quantK : gemmM,
                                 transposeScaleA ? gemmM : quantK},
                       bScale = {groupSize, transposeScaleB ? quantK : gemmN,
                                 transposeScaleB ? gemmN : quantK};

  RankedTensorType aType = RankedTensorType::get(aDims, elemTypes[0]),
                   bType = RankedTensorType::get(bDims, elemTypes[1]),
                   cType = RankedTensorType::get(cDims, cElemType),
                   aScaleType = RankedTensorType::get(aScale, elemTypes[3]),
                   bScaleType = RankedTensorType::get(bScale, elemTypes[4]);
  result.push_back(aType);
  result.push_back(bType);
  if (scaledGemm) {
    result.push_back(aScaleType);
    result.push_back(bScaleType);
  }
  result.push_back(cType);
}

static Value normalizeScaleShape(OpBuilder b, Location loc, Value scale,
                                 bool transpose, bool isA) {
  // Initial logical dim order for A or B scale
  SmallVector<StringRef, 3> initDims;
  StringRef dDim = isA ? "m" : "n";
  if (transpose) {
    initDims = {"g", "kScale", dDim};
  } else {
    initDims = {"g", dDim, "kScale"};
  }
  rock::BottomUpTMBuilder transposeScale(
      b, initDims, cast<ShapedType>(scale.getType()).getShape(), loc);

  if (transpose) {
    transposeScale.passThrough({"g", "kScale", dDim}, {0, 1, 2},
                               {"g", "kScale", dDim});
  } else {
    transposeScale.passThrough({"g", dDim, "kScale"});
  }
  auto transposeScaleAttr = transposeScale.get();
  return rock::TransformOp::create(b, loc, scale, transposeScaleAttr);
}

static func::FuncOp createGpuGemmKernel(ModuleOp module,
                                        const GenParams &params) {
  MLIRContext *ctx = module.getContext();
  Location loc = module->getLoc();
  OpBuilder b(ctx);

  // Set arch on module to make compilation pipeline work
  StringAttr archAttr = b.getStringAttr(params.arch);
  if (!module->hasAttr(rock::ArchAttr::getMnemonic()))
    module->setAttr(rock::ArchAttr::getMnemonic(), archAttr);

  SmallVector<Type, 5> argTypes;
  getGemmTypes(params.types, argTypes,
               /*isCpuVerifier=*/false);
  constexpr StringLiteral kernelName("rock_gemm");
  IntegerAttr numCUAttr =
      (num_cu.getNumOccurrences() > 0
           ? b.getI64IntegerAttr(num_cu)
           : b.getI64IntegerAttr(rock::getMinNumCU(archAttr.getValue())));

  IntegerAttr numChipletsAttr =
      (numChiplets.getNumOccurrences() > 0
           ? b.getI64IntegerAttr(numChiplets)
           : b.getI64IntegerAttr(rock::getMaxNumChiplets(archAttr.getValue())));
  SmallVector<NamedAttribute> funcAttrs = {
      b.getNamedAttr(rock::KernelAttr::getMnemonic(), b.getUnitAttr()),
      b.getNamedAttr(rock::ArchAttr::getMnemonic(), archAttr)};

  if (numCUAttr)
    funcAttrs.push_back(
        b.getNamedAttr(rock::NumCUAttr::getMnemonic(), numCUAttr));

  if (numChipletsAttr)
    funcAttrs.push_back(
        b.getNamedAttr(rock::NumChipletsAttr::getMnemonic(), numChipletsAttr));

  constexpr StringLiteral gName = "g", mName = "m", kName = "k", nName = "n";
  SmallVector<SmallVector<StringRef>> allArgNames;
  allArgNames.emplace_back(SmallVector<StringRef>{
      gName, transposeA ? kName : mName, transposeA ? mName : kName});
  allArgNames.emplace_back(SmallVector<StringRef>{
      gName, transposeB ? nName : kName, transposeB ? kName : nName});
  if (scaledGemm) {
    allArgNames.emplace_back(
        SmallVector<StringRef>{gName, transposeScaleA ? kName : mName,
                               transposeScaleA ? mName : kName});
    allArgNames.emplace_back(
        SmallVector<StringRef>{gName, transposeScaleB ? kName : nName,
                               transposeScaleB ? nName : kName});
  }

  // Function takes flattened (a, b, [aScale, bScale], c) as inputs and returns
  // flattened c.
  SmallVector<Type, 5> funcArgTypes;
  SmallVector<Type, 5> funcArgLogicalTypes;
  int cIdx = scaledGemm ? 4 : 2;
  Type cType = argTypes[cIdx];
  Type cFlatType = rock::getFlattenedType(cType);

  funcArgTypes.push_back(rock::getFlattenedType(argTypes[0]));
  funcArgTypes.push_back(rock::getFlattenedType(argTypes[1]));
  funcArgLogicalTypes.push_back(argTypes[0]);
  funcArgLogicalTypes.push_back(argTypes[1]);
  if (scaledGemm) {
    funcArgTypes.push_back(rock::getFlattenedType(argTypes[2]));
    funcArgTypes.push_back(rock::getFlattenedType(argTypes[3]));
    funcArgLogicalTypes.push_back(argTypes[2]);
    funcArgLogicalTypes.push_back(argTypes[3]);
  }

  SmallVector<StringRef> cDimNames = {gName, transposeC ? nName : mName,
                                      transposeC ? mName : nName};
  funcArgTypes.push_back(cFlatType);

  auto func = func::FuncOp::create(b, loc, kernelName,
                                   b.getFunctionType(funcArgTypes, {cFlatType}),
                                   funcAttrs);

  Block *block = func.addEntryBlock();
  b.setInsertionPointToStart(block);

  // Expand flattened arguments to logical shapes with dimension names
  SmallVector<Value, 4> expandedArgs;
  rock::expandFlatFunctionArguments(b, func, allArgNames, funcArgLogicalTypes,
                                    expandedArgs);

  Value aVal = expandedArgs[0], bVal = expandedArgs[1];
  Value aScale = nullptr, bScale = nullptr;

  if (scaledGemm) {
    aScale = normalizeScaleShape(b, loc, expandedArgs[2], transposeScaleA,
                                 /*isA=*/true);
    bScale = normalizeScaleShape(b, loc, expandedArgs[3], transposeScaleB,
                                 /*isA=*/false);
    auto truncScaleToF8 = [&](Value scale) -> Value {
      auto scaleTy = cast<RankedTensorType>(scale.getType());
      if (scaleTy.getElementType().isF32()) {
        auto f8Ty = RankedTensorType::get(scaleTy.getShape(),
                                          Float8E8M0FNUType::get(ctx));
        return arith::TruncFOp::create(b, loc, f8Ty, scale);
      }
      return scale;
    };
    aScale = truncScaleToF8(aScale);
    bScale = truncScaleToF8(bScale);
  }

  // GEMM produces result in logical shape (e.g., tensor<1x64x64xf32>)
  auto gemm = rock::GemmOp::create(
      b, loc, cType, aVal, bVal, aScale, bScale, transposeA, transposeB,
      transposeC, transposeScaleA, transposeScaleB,
      scaledGemm ? b.getI64IntegerAttr(quantBlockSize) : nullptr,
      /*params=*/nullptr);

  if (!params.perfConfig.empty())
    gemm->setAttr("perf_config", b.getStringAttr(params.perfConfig));

  // Apply the output transform to flatten the GEMM result
  Value flatResult = rock::flattenOutput(b, loc, gemm.getResult(), cDimNames);

  // Store the flat result to the C argument
  Value cArg = func.getArgument(cIdx);
  Value storedVal =
      rock::StoreOp::create(b, loc, cFlatType, flatResult, cArg, storeMethod);

  func::ReturnOp::create(b, loc, storedVal);

  if (!disableSplitKForTuning)
    func->setAttr(rock::EnableSplitKForTuningAttr::getMnemonic(),
                  b.getUnitAttr());

  module.push_back(func);
  return func;
}

static void getAttentionTypes(SmallVectorImpl<Type> &result,
                              ArrayRef<Type> elemTypes) {
  SmallVector<int64_t> qDims{groupSize * numHeadsQ, sequenceLengthQ, headDimQK};
  SmallVector<int64_t> transposedQDims{groupSize * numHeadsQ, headDimQK,
                                       sequenceLengthQ};
  SmallVector<int64_t> kDims{groupSize * numHeadsKV, sequenceLengthK,
                             headDimQK};
  SmallVector<int64_t> transposedKDims{groupSize * numHeadsKV, headDimQK,
                                       sequenceLengthK};
  SmallVector<int64_t> vDims{groupSize * numHeadsKV, sequenceLengthK, headDimV};
  SmallVector<int64_t> transposedVDims{groupSize * numHeadsKV, headDimV,
                                       sequenceLengthK};
  SmallVector<int64_t> oDims{groupSize * numHeadsQ * splitKV, sequenceLengthQ,
                             headDimV};
  SmallVector<int64_t> transposedODims{groupSize * numHeadsQ * splitKV,
                                       headDimV, sequenceLengthQ};

  bool isQuantized =
      elemTypes[0] == IntegerType::get(elemTypes[0].getContext(), 8);

  const size_t qIndex =
      isQuantized ? AttentionQuantizedArgIndex::q : AttentionArgIndex::q;
  const size_t kIndex =
      isQuantized ? AttentionQuantizedArgIndex::k : AttentionArgIndex::k;
  const size_t vIndex =
      isQuantized ? AttentionQuantizedArgIndex::v : AttentionArgIndex::v;
  const size_t scaleIndex = isQuantized ? AttentionQuantizedArgIndex::scale
                                        : AttentionArgIndex::scale;
  const size_t biasIndex =
      isQuantized ? AttentionQuantizedArgIndex::bias : AttentionArgIndex::bias;
  const size_t currentSeqLenIndex =
      isQuantized ? AttentionQuantizedArgIndex::currentSeqLen
                  : AttentionArgIndex::currentSeqLen;
  const size_t lseIndex =
      isQuantized ? AttentionQuantizedArgIndex::lse : AttentionArgIndex::lse;

  // output type = bias type
  const size_t outputIndex = biasIndex;

  RankedTensorType qType = RankedTensorType::get(
                       transposeQ ? transposedQDims : qDims, elemTypes[qIndex]),
                   kType = RankedTensorType::get(
                       transposeK ? kDims : transposedKDims, elemTypes[kIndex]),
                   vType = RankedTensorType::get(
                       transposeV ? transposedVDims : vDims, elemTypes[vIndex]);

  result.push_back(qType);
  result.push_back(kType);
  result.push_back(vType);
  if (isQuantized) {
    // quant bias is to be broadcasted
    SmallVector<int64_t> quantBiasDims{1, 1, 1};
    RankedTensorType qbType = RankedTensorType::get(
        quantBiasDims, elemTypes[AttentionQuantizedArgIndex::quantBias]);
    result.push_back(qbType);
    // quant scale is to be broadcasted
    SmallVector<int64_t> quantScaleDims{1, 1, 1};
    RankedTensorType qsType = RankedTensorType::get(
        quantScaleDims, elemTypes[AttentionQuantizedArgIndex::quantScale]);
    result.push_back(qsType);
  }
  if (hasAttnScale) {
    SmallVector<int64_t> scaleDims{groupSize * numHeadsQ, sequenceLengthQ,
                                   sequenceLengthK};
    RankedTensorType sType =
        RankedTensorType::get(scaleDims, elemTypes[scaleIndex]);
    result.push_back(sType);
  }
  if (hasAttnBias) {
    SmallVector<int64_t> biasDims{groupSize * numHeadsQ, sequenceLengthQ,
                                  sequenceLengthK};
    RankedTensorType bType =
        RankedTensorType::get(biasDims, elemTypes[biasIndex]);
    result.push_back(bType);
  }
  if (!currentSeqLen.empty()) {
    SmallVector<int64_t> currentSeqDims{groupSize};
    RankedTensorType currSeqLenType =
        RankedTensorType::get(currentSeqDims, elemTypes[currentSeqLenIndex]);
    result.push_back(currSeqLenType);
  }
  if (!prefixOffset.empty()) {
    SmallVector<int64_t> prefixOffsetDims{groupSize};
    // prefixOffset uses the same i32 type as currentSeqLen
    RankedTensorType prefixOffsetType =
        RankedTensorType::get(prefixOffsetDims, elemTypes[currentSeqLenIndex]);
    result.push_back(prefixOffsetType);
  }
  if (returnLSE) {
    SmallVector<int64_t> lseDims{groupSize * numHeadsQ * splitKV,
                                 sequenceLengthQ};
    RankedTensorType lseType =
        RankedTensorType::get(lseDims, elemTypes[lseIndex]);
    result.push_back(lseType);
  }
  RankedTensorType outType = RankedTensorType::get(
      transposeO ? transposedODims : oDims, elemTypes[outputIndex]);
  result.push_back(outType);
}

static void
getAttentionDimNames(SmallVectorImpl<SmallVector<StringRef>> &result,
                     ArrayRef<Type> elementTypes) {
  result.reserve(elementTypes.size());
  constexpr StringLiteral gName = "g", seqQName = "seq_q", seqKName = "seq_k",
                          headQKName = "head_qk", headVName = "head_v";
  if (transposeQ)
    result.emplace_back(SmallVector<StringRef>{gName, headQKName, seqQName});
  else
    result.emplace_back(SmallVector<StringRef>{gName, seqQName, headQKName});
  if (transposeK)
    result.emplace_back(SmallVector<StringRef>{gName, seqKName, headQKName});
  else
    result.emplace_back(SmallVector<StringRef>{gName, headQKName, seqKName});
  if (transposeV)
    result.emplace_back(SmallVector<StringRef>{gName, headVName, seqKName});
  else
    result.emplace_back(SmallVector<StringRef>{gName, seqKName, headVName});
  bool isQuantized = elementTypes[0].isInteger(8);
  if (isQuantized) {
    result.emplace_back(SmallVector<StringRef>{gName, seqQName, seqKName});
    result.emplace_back(SmallVector<StringRef>{gName, seqQName, seqKName});
  }
  if (hasAttnScale)
    result.emplace_back(SmallVector<StringRef>{gName, seqQName, seqKName});
  if (hasAttnBias)
    result.emplace_back(SmallVector<StringRef>{gName, seqQName, seqKName});
  if (!currentSeqLen.empty())
    result.emplace_back(SmallVector<StringRef>{gName});
  if (!prefixOffset.empty())
    result.emplace_back(SmallVector<StringRef>{gName});
  if (returnLSE)
    result.emplace_back(SmallVector<StringRef>{gName, seqQName});
  if (transposeO)
    result.emplace_back(SmallVector<StringRef>{gName, headVName, seqQName});
  else
    result.emplace_back(SmallVector<StringRef>{gName, seqQName, headVName});
}

static rock::GemmSize
getConvElementwiseGemmTypes(SmallVectorImpl<Type> &result,
                            const rock::ConvGenerator::Config *config,
                            ArrayRef<Type> elemTypes) {
  // determine gemmM and gemmN from convolution sizes
  rock::ConvolutionDims convDims =
      rock::ConvGenerator::getConvolutionDims(config);
  rock::GemmSize gemmSize =
      rock::GemmSize::fromConvolution(rock::ConvOpType::Fwd, convDims);

  SmallVector<int64_t> filterDims(config->filterDimension.begin(),
                                  config->filterDimension.end()),
      inputDims(config->inputDimension.begin(), config->inputDimension.end()),
      cDims = {1, transposeC ? gemmO : gemmSize.m,
               transposeC ? gemmSize.m : gemmO},
      outDims = {1, transposeO ? gemmO : gemmSize.n,
                 transposeO ? gemmSize.n : gemmO};

  RankedTensorType filterType = RankedTensorType::get(filterDims, elemTypes[0]),
                   inputType = RankedTensorType::get(inputDims, elemTypes[1]),
                   cType = RankedTensorType::get(cDims, elemTypes[2]),
                   outType = RankedTensorType::get(outDims, elemTypes[3]);
  result.push_back(filterType);
  result.push_back(inputType);
  result.push_back(cType);
  result.push_back(outType);

  return gemmSize;
}

static void getGemmElementwiseGemmTypes(SmallVectorImpl<Type> &result,
                                        ArrayRef<Type> elemTypes) {
  SmallVector<int64_t> aDims = {groupSize, transposeA ? gemmK : gemmM,
                                transposeA ? gemmM : gemmK},
                       bDims = {groupSize, transposeB ? gemmN : gemmK,
                                transposeB ? gemmK : gemmN},
                       cDims = {groupSize, transposeC ? gemmO : gemmN,
                                transposeC ? gemmN : gemmO},
                       outDims = {groupSize, transposeO ? gemmO : gemmM,
                                  transposeO ? gemmM : gemmO};

  RankedTensorType aType = RankedTensorType::get(aDims, elemTypes[0]),
                   bType = RankedTensorType::get(bDims, elemTypes[1]),
                   cType = RankedTensorType::get(cDims, elemTypes[2]),
                   outType = RankedTensorType::get(outDims, elemTypes[3]);
  result.push_back(aType);
  result.push_back(bType);
  result.push_back(cType);
  result.push_back(outType);
}

static void
getConvElementwiseGemmDimNames(SmallVectorImpl<SmallVector<StringRef>> &result,
                               const rock::ConvGenerator::Config *config,
                               ArrayRef<Type> elementTypes) {

  SmallVector<StringRef> filterLayoutSpec;
  SmallVector<StringRef> inputLayoutSpec;
  for (auto &key : config->filterLayout)
    filterLayoutSpec.push_back(StringRef(&key, 1));
  for (auto &key : config->inputLayout)
    inputLayoutSpec.push_back(StringRef(&key, 1));

  result.reserve(elementTypes.size());
  constexpr StringLiteral gName = "g", m = "m", n = "n", gemmONameStr = "gemmO";

  result.emplace_back(filterLayoutSpec);
  result.emplace_back(inputLayoutSpec);
  if (transposeC)
    result.emplace_back(SmallVector<StringRef>{gName, gemmONameStr, m});
  else
    result.emplace_back(SmallVector<StringRef>{gName, m, gemmONameStr});
  if (transposeO)
    result.emplace_back(SmallVector<StringRef>{gName, gemmONameStr, n});
  else
    result.emplace_back(SmallVector<StringRef>{gName, n, gemmONameStr});
}

static void
getGemmElementwiseGemmDimNames(SmallVectorImpl<SmallVector<StringRef>> &result,
                               ArrayRef<Type> elementTypes) {
  result.reserve(elementTypes.size());
  constexpr StringLiteral gName = "g", m = "m", n = "n", k = "k",
                          gemmONameStr = "gemmO";
  if (transposeA)
    result.emplace_back(SmallVector<StringRef>{gName, k, m});
  else
    result.emplace_back(SmallVector<StringRef>{gName, m, k});
  if (transposeB)
    result.emplace_back(SmallVector<StringRef>{gName, n, k});
  else
    result.emplace_back(SmallVector<StringRef>{gName, k, n});
  if (transposeC)
    result.emplace_back(SmallVector<StringRef>{gName, gemmONameStr, n});
  else
    result.emplace_back(SmallVector<StringRef>{gName, n, gemmONameStr});
  if (transposeO)
    result.emplace_back(SmallVector<StringRef>{gName, gemmONameStr, m});
  else
    result.emplace_back(SmallVector<StringRef>{gName, m, gemmONameStr});
}

static Value addTensorArgToBlock(OpBuilder &builder, Location loc,
                                 Block *preSoftmaxElemwiseBlock,
                                 Value funcArg) {
  ShapedType funcArgType = cast<ShapedType>(funcArg.getType());
  Value funcArgTensor = preSoftmaxElemwiseBlock->addArgument(
      RankedTensorType::get(funcArgType.getShape(),
                            funcArgType.getElementType()),
      loc);
  return funcArgTensor;
}

static Value applyMask(OpBuilder builder, Location loc, Value inputTensor,
                       Value mask, float initValue) {
  auto inpType = cast<RankedTensorType>(inputTensor.getType());
  ArrayRef<int64_t> inpShape = inpType.getShape();

  // create a tensor with a single value and broadcast it
  assert(isa<FloatType>(inpType.getElementType()));
  std::pair<APFloat, llvm::detail::opStatus> floatRes =
      rock::createAPFloat(inpType.getElementType(), initValue);
  APFloat fpVal = floatRes.first;
  auto status = floatRes.second;
  assert(status == APFloat::opOK);

  DenseElementsAttr initValueAttr = DenseFPElementsAttr::get(
      RankedTensorType::get(inpShape, inpType.getElementType()), fpVal);

  Value initVal = tosa::ConstOp::create(builder, loc, initValueAttr.getType(),
                                        initValueAttr);

  // mask is 1 for values we want to set to "initVal"
  auto result = rock::tosa::createOpAndInfer<tosa::SelectOp>(
      builder, loc, inpType.getElementType(), mask, initVal, inputTensor);
  return result;
}

static Value createRange(OpBuilder builder, Location loc, size_t index,
                         const ArrayRef<int64_t> inpShape) {
  assert(index < inpShape.size());

  // create range 0 to inpShape[axis]
  llvm::SmallVector<int32_t> range;
  range.reserve(inpShape[index]);
  for (int i = 0; i < inpShape[index]; i++)
    range.push_back(i);
  DenseElementsAttr rangeAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({inpShape[index]}, builder.getI32Type()), range);
  Value rangeVal =
      tosa::ConstOp::create(builder, loc, rangeAttr.getType(), rangeAttr);

  // reshape
  ImplicitLocOpBuilder implicitBuilder(loc, builder);
  SmallVector<int64_t> newShape;
  newShape.reserve(inpShape.size());
  for (size_t i = 0; i < inpShape.size(); i++)
    newShape.push_back((i == index) ? inpShape[index] : 1);

  auto shapeValue = tosa::getTosaConstShape(implicitBuilder, newShape);
  auto rangeValReshaped = rock::tosa::createOpAndInfer<tosa::ReshapeOp>(
      builder, loc, builder.getI32Type(), rangeVal, shapeValue);

  // broadcast range to inputTensor shape
  auto outType = RankedTensorType::get(inpShape, builder.getI32Type());
  auto rangeBroadcast = rock::tosa::getMulOp(
      builder, loc, rangeValReshaped,
      rock::tosa::getOneTensor(builder, loc, outType), builder.getI32Type());
  return rangeBroadcast;
}

static Value causalMaskingTosa(OpBuilder builder, Location loc,
                               Value inputTensor, float initValue) {
  // create a range for row and column
  auto inpType = cast<RankedTensorType>(inputTensor.getType());
  ArrayRef<int64_t> inpShape = inpType.getShape();

  Value rowRange = createRange(builder, loc, 1, inpShape);
  Value colRange = createRange(builder, loc, 2, inpShape);

  // create mask (diagonal and lower triangular are zero)
  auto mask = rock::tosa::createOpAndInfer<tosa::GreaterOp>(
      builder, loc, builder.getIntegerType(1), colRange, rowRange);

  // apply mask
  Value result = applyMask(builder, loc, inputTensor, mask, initValue);
  return result;
}

static Value prefixOffsetMaskingTosa(OpBuilder builder, Location loc,
                                     Value inputTensor, Value offsetTensor,
                                     float initValue) {
  auto origType = cast<RankedTensorType>(inputTensor.getType());
  ArrayRef<int64_t> origShape = origType.getShape();
  SmallVector<int64_t, 4> newShape = {origShape[0] / numHeadsQ, numHeadsQ,
                                      origShape[1], origShape[2]};
  ImplicitLocOpBuilder implicitBuilder(loc, builder);
  auto newShapeValue = tosa::getTosaConstShape(implicitBuilder, newShape);
  inputTensor = rock::tosa::createOpAndInfer<tosa::ReshapeOp>(
      builder, loc, origType.getElementType(), inputTensor, newShapeValue);

  auto inpType = cast<RankedTensorType>(inputTensor.getType());
  ArrayRef<int64_t> inpShape = inpType.getShape();

  // Create row and column ranges
  Value rowRange = createRange(builder, loc, 2, inpShape);
  Value colRange = createRange(builder, loc, 3, inpShape);

  // Broadcast offset
  auto outType = RankedTensorType::get(inpShape, builder.getI32Type());
  auto offsetBroadcast = rock::tosa::getMulOp(
      builder, loc, offsetTensor,
      rock::tosa::getOneTensor(builder, loc, outType), builder.getI32Type());

  // Compute row + offset
  auto rowPlusOffset = rock::tosa::createOpAndInfer<tosa::AddOp>(
      builder, loc, builder.getI32Type(), rowRange, offsetBroadcast);

  // Create mask: col > row + offset
  auto mask = rock::tosa::createOpAndInfer<tosa::GreaterOp>(
      builder, loc, builder.getIntegerType(1), colRange, rowPlusOffset);

  // Apply mask
  Value result = applyMask(builder, loc, inputTensor, mask, initValue);

  // Reshape result
  auto origShapeValue = tosa::getTosaConstShape(implicitBuilder, origShape);
  auto resultReshaped = rock::tosa::createOpAndInfer<tosa::ReshapeOp>(
      builder, loc, inpType.getElementType(), result, origShapeValue);

  return resultReshaped;
}

static Value maskKVCacheTosa(OpBuilder builder, Location loc, Value inputTensor,
                             Value currentSeqLenVal, float initValue) {
  // inputTensor is [B*NUM_HEADS, SEQ_LEN_Q, SEQ_LEN_KV], we want to reshape to
  // [B, NUM_HEADS, SEQ_LEN_Q, SEQ_LEN_KV]
  auto origType = cast<RankedTensorType>(inputTensor.getType());
  ArrayRef<int64_t> origShape = origType.getShape();
  SmallVector<int64_t, 4> newShape = {origShape[0] / numHeadsQ, numHeadsQ,
                                      origShape[1], origShape[2]};
  ImplicitLocOpBuilder implicitBuilder(loc, builder);
  auto newShapeValue = tosa::getTosaConstShape(implicitBuilder, newShape);
  inputTensor = rock::tosa::createOpAndInfer<tosa::ReshapeOp>(
      builder, loc, origType.getElementType(), inputTensor, newShapeValue);

  auto inpType = cast<RankedTensorType>(inputTensor.getType());
  ArrayRef<int64_t> inpShape = inpType.getShape();

  for (auto v : currentSeqLen)
    assert(v >= 0 && v < inpShape[3]);

  // generate range
  Value rangeBroadcast = createRange(builder, loc, 3, inpShape);

  // zero tensor
  auto outType = RankedTensorType::get(inpShape, builder.getI32Type());
  // broadcast currentSeqLen
  auto currentSeqLenBroadcast = rock::tosa::getMulOp(
      builder, loc, currentSeqLenVal,
      rock::tosa::getOneTensor(builder, loc, outType), builder.getI32Type());
  // create mask
  auto mask = rock::tosa::createOpAndInfer<tosa::GreaterOp>(
      builder, loc, builder.getIntegerType(1), rangeBroadcast,
      currentSeqLenBroadcast);

  Value result = applyMask(builder, loc, inputTensor, mask, initValue);

  // reshape result back to [B*NUM_HEADS, SEQ_LEN_Q, SEQ_LEN_KV]
  auto origShapeValue = tosa::getTosaConstShape(implicitBuilder, origShape);
  auto resultReshaped = rock::tosa::createOpAndInfer<tosa::ReshapeOp>(
      builder, loc, inpType.getElementType(), result, origShapeValue);

  return resultReshaped;
}

static Value broadcastBatchTosa(OpBuilder builder, Location loc,
                                Value inputTensor, int64_t numRepeat) {
  if (numRepeat == 1)
    return inputTensor;

  auto inpType = cast<RankedTensorType>(inputTensor.getType());
  ArrayRef<int64_t> inpShape = inpType.getShape();

  // add one dimension
  SmallVector<ReassociationIndices> reassocIndices = {{0, 1}};
  SmallVector<int64_t> expandedShape = {inpShape[0], 1};
  for (size_t i = 1; i < inpShape.size(); i++) {
    expandedShape.push_back(inpShape[i]);
    reassocIndices.push_back({static_cast<int64_t>(i + 1)});
  }

  auto newType = RankedTensorType::get(expandedShape, inpType.getElementType());
  auto expandedValue = tensor::ExpandShapeOp::create(
      builder, loc, newType, inputTensor, reassocIndices);

  // broadcast
  SmallVector<int64_t, 4> outShape = {inpShape[0], numRepeat};
  for (size_t i = 1; i < inpShape.size(); i++)
    outShape.push_back(inpShape[i]);
  auto outType = RankedTensorType::get(outShape, inpType.getElementType());

  auto mulWithOne =
      rock::tosa::getMulOp(builder, loc, expandedValue,
                           rock::tosa::getOneTensor(builder, loc, outType),
                           inpType.getElementType());
  // collapse
  return tensor::CollapseShapeOp::create(builder, loc, mulWithOne,
                                         reassocIndices);
}

static Value createMaskSplitKV(OpBuilder &builder, Location loc,
                               SmallVector<int64_t> &shape, int64_t index,
                               SmallVector<int32_t> &validSplitKV) {
  assert(static_cast<size_t>(index) < shape.size() &&
         "Index out of bounds for shape");
  assert(static_cast<size_t>(shape[0]) == validSplitKV.size() &&
         "Shape size must match the size of validSplitKV");
  // generate mask for valid resultTensor
  auto rangeTensor = createRange(builder, loc, index, shape);

  // constant tensor
  SmallVector<int64_t> initialShape = {
      static_cast<int64_t>(validSplitKV.size())};
  for (size_t i = 1; i < shape.size(); i++)
    initialShape.push_back(1);

  auto initialType = RankedTensorType::get(initialShape, builder.getI32Type());
  auto denseAttr =
      DenseElementsAttr::get(initialType, ArrayRef<int32_t>(validSplitKV));
  Value initialTensor =
      tosa::ConstOp::create(builder, loc, initialType, denseAttr);

  // Create zero tensor of target shape
  auto outType = RankedTensorType::get(shape, builder.getI32Type());

  // Use tosa.mul to broadcast reshaped [batch,1,1,...] to [batch,D1,D2,...]
  Value validSplitKVTensor = rock::tosa::getMulOp(
      builder, loc, initialTensor,
      rock::tosa::getOneTensor(builder, loc, outType), outType);
  // create mask
  return rock::tosa::createOpAndInfer<tosa::GreaterEqualOp>(
      builder, loc, builder.getIntegerType(1), rangeTensor, validSplitKVTensor);
}

// This function computes the final attention result.
// It is used for flash decoding (split-kv > 1), where the result is computed in
// two stages, this is the second stage.
static Value computeFinalAttentionStage(OpBuilder builder, Location loc,
                                        Value resultTensor, Value lseTensor,
                                        SmallVector<int32_t> &validSplitKV) {
  if (!currentSeqLen.empty())
    assert(validSplitKV.size() == (numHeadsQ * currentSeqLen.size()) &&
           "Number of valid split KV must match current sequence length");
  SmallVector<int64_t> newResultShape;
  SmallVector<int64_t> newResultShapeAfterTranpose = {
      groupSize * numHeadsQ, splitKV, sequenceLengthQ, headDimV};
  if (transposeO) {
    newResultShape = {groupSize * numHeadsQ, splitKV, headDimV,
                      sequenceLengthQ};
  } else {
    newResultShape = newResultShapeAfterTranpose;
  }

  auto elementType = cast<ShapedType>(lseTensor.getType()).getElementType();
  ImplicitLocOpBuilder implicitBuilder(loc, builder);
  auto newResultShapeValue =
      tosa::getTosaConstShape(implicitBuilder, newResultShape);
  resultTensor = rock::tosa::createOpAndInfer<tosa::ReshapeOp>(
      builder, loc, elementType, resultTensor, newResultShapeValue);
  if (transposeO)
    resultTensor =
        rock::tosa::getTransposeOp(builder, loc, resultTensor, {0, 1, 3, 2});

  SmallVector<int64_t> newLseShape = {groupSize * numHeadsQ, splitKV,
                                      sequenceLengthQ, 1};
  auto newLseShapeValue = tosa::getTosaConstShape(implicitBuilder, newLseShape);
  lseTensor = rock::tosa::createOpAndInfer<tosa::ReshapeOp>(
      builder, loc, elementType, lseTensor, newLseShapeValue);

  Value resultTensorMask = createMaskSplitKV(
      builder, loc, newResultShapeAfterTranpose, 1, validSplitKV);
  resultTensor = applyMask(builder, loc, resultTensor, resultTensorMask,
                           -std::numeric_limits<float>::infinity());

  // generate mask for valid lseTensor
  Value lseTensorMask =
      createMaskSplitKV(builder, loc, newLseShape, 1, validSplitKV);
  lseTensor = applyMask(builder, loc, lseTensor, lseTensorMask,
                        -std::numeric_limits<float>::infinity());

  IntegerAttr axisAttr = builder.getI32IntegerAttr(1);
  auto maxSplitKV = rock::tosa::createOpAndInfer<tosa::ReduceMaxOp>(
      builder, loc, elementType, lseTensor, axisAttr);

  auto norm = rock::tosa::createOpAndInfer<tosa::SubOp>(
      builder, loc, elementType, lseTensor, maxSplitKV);
  auto exp = rock::tosa::createOpAndInfer<tosa::ExpOp>(builder, loc,
                                                       elementType, norm);

  auto sumExpNorm = rock::tosa::createOpAndInfer<tosa::ReduceSumOp>(
      builder, loc, elementType, exp, axisAttr);
  auto sumExpNormRecip = rock::tosa::createOpAndInfer<tosa::ReciprocalOp>(
      builder, loc, elementType, sumExpNorm);

  Value outExp =
      rock::tosa::getMulOp(builder, loc, exp, resultTensor, elementType);

  // apply mask to outExp to prevent NaN values
  outExp = applyMask(builder, loc, outExp, resultTensorMask, 0.0f);

  auto outExpNorm = rock::tosa::createOpAndInfer<tosa::ReduceSumOp>(
      builder, loc, elementType, outExp, axisAttr);

  Value finalResult = rock::tosa::getMulOp(builder, loc, outExpNorm,
                                           sumExpNormRecip, elementType);

  // broadcast the result in splitKV dimension
  // we have to do this because both cpu and gpu buffers have the same shape.
  // So, in order to keep the same shape, after reducing splitKV dimension,
  // we have to broadcast the result.
  finalResult = broadcastBatchTosa(builder, loc, finalResult, splitKV);
  if (transposeO)
    finalResult =
        rock::tosa::getTransposeOp(builder, loc, finalResult, {0, 1, 3, 2});

  // convert to one dimensional tensor
  SmallVector<ReassociationIndices> reassocIndices = {{0, 1, 2, 3}};
  finalResult = tensor::CollapseShapeOp::create(builder, loc, finalResult,
                                                reassocIndices);
  return finalResult;
}

static Value broadcastGQATosa(OpBuilder builder, Location loc,
                              Value inputTensor) {
  assert(numHeadsQ % numHeadsKV == 0);

  int64_t numRepeat = numHeadsQ / numHeadsKV;
  return broadcastBatchTosa(builder, loc, inputTensor, numRepeat);
}

// Broadcasts a 1D batch tensor (e.g., currentSeqLen or prefixOffset) to have
// shape [G * numHeadsQ] by adding a numHeadsQ dimension and merging it with G.
static Value broadcastBatchTensorRock(OpBuilder builder, Location loc,
                                      Value inputTensor) {
  ArrayRef<int64_t> inpShape =
      cast<ShapedType>(inputTensor.getType()).getShape();
  SmallVector<StringRef> startNames = {"gemmG"};
  rock::BottomUpTMBuilder addDim(builder, startNames, inpShape);
  addDim.addDim("numHeadsQ", 1, 1);
  addDim.passThrough(ArrayRef<uint32_t>{0}, ArrayRef<uint32_t>{0});
  auto addDimAttr = addDim.get();
  Value matrixAddDim =
      rock::TransformOp::create(builder, loc, inputTensor, addDimAttr);

  auto broadcaster = rock::BottomUpTMBuilder::above(addDim, addDimAttr);
  broadcaster.broadcast({1}, {numHeadsQ});
  broadcaster.passThrough(ArrayRef<uint32_t>{0}, ArrayRef<uint32_t>{0});
  auto broadcasterAttr = broadcaster.get();
  Value tensorBroadcast =
      rock::TransformOp::create(builder, loc, matrixAddDim, broadcasterAttr);

  auto merger = rock::BottomUpTMBuilder::above(broadcaster, broadcasterAttr);
  merger.merge("gemmG", 0, {"gemmG", "numHeadsQ"});
  auto mergerAttr = merger.get();
  return rock::TransformOp::create(builder, loc, tensorBroadcast, mergerAttr);
}

static func::FuncOp createGpuAttentionKernel(ModuleOp module,
                                             const GenParams &params) {
  MLIRContext *ctx = module.getContext();
  Location loc = module->getLoc();
  OpBuilder builder(ctx);

  // Set arch on module to make compilation pipeline work
  StringAttr archAttr = builder.getStringAttr(params.arch);
  if (!module->hasAttr(rock::ArchAttr::getMnemonic()))
    module->setAttr(rock::ArchAttr::getMnemonic(), archAttr);

  SmallVector<Type, 5> argTypes;
  getAttentionTypes(argTypes, params.types);
  bool isQuantized = params.types[0] == IntegerType::get(ctx, 8);
  SmallVector<Type, 5> flatArgTypes =
      llvm::map_to_vector(argTypes, rock::getFlattenedType);

  IntegerAttr numCUAttr =
      (num_cu.getNumOccurrences() > 0 ? builder.getI32IntegerAttr(num_cu)
                                      : nullptr);

  IntegerAttr numChipletsAttr = (numChiplets.getNumOccurrences() > 0
                                     ? builder.getI32IntegerAttr(numChiplets)
                                     : nullptr);

  SmallVector<NamedAttribute, 3> funcAttrs = {
      builder.getNamedAttr(rock::KernelAttr::getMnemonic(),
                           builder.getUnitAttr()),
      builder.getNamedAttr(rock::ArchAttr::getMnemonic(), archAttr)};

  if (numCUAttr)
    funcAttrs.push_back(
        builder.getNamedAttr(rock::NumCUAttr::getMnemonic(), numCUAttr));

  if (numChipletsAttr)
    funcAttrs.push_back(builder.getNamedAttr(
        rock::NumChipletsAttr::getMnemonic(), numChipletsAttr));

  constexpr StringLiteral kernelName("rock_attention");
  SmallVector<Type, 2> resultTypes = {flatArgTypes[flatArgTypes.size() - 1]};
  if (returnLSE) {
    resultTypes.push_back(flatArgTypes[flatArgTypes.size() - 2]);
  }
  auto func = func::FuncOp::create(
      builder, loc, kernelName,
      builder.getFunctionType(flatArgTypes, resultTypes), funcAttrs);

  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  SmallVector<Value> unflattenedArgs;
  SmallVector<SmallVector<StringRef>> allNames;
  getAttentionDimNames(allNames, params.types);

  // Save output dim names/types and trim from expansion (output is last)
  SmallVector<StringRef> outputDimNames = allNames.back();
  Type outputLogicalType = argTypes.back();
  Type outputFlatType = flatArgTypes.back();
  unsigned outputArgIdx = allNames.size() - 1;
  allNames.pop_back();

  // If returnLSE, also save and trim LSE (now last after output was popped)
  SmallVector<StringRef> lseDimNames;
  Type lseLogicalType;
  Type lseFlatType;
  unsigned lseArgIdx = 0;
  if (returnLSE) {
    lseDimNames = allNames.back();
    lseLogicalType = argTypes[allNames.size() - 1];
    lseFlatType = flatArgTypes[allNames.size() - 1];
    lseArgIdx = allNames.size() - 1;
    allNames.pop_back();
  }

  SmallVector<Type> expandArgTypes(argTypes.begin(),
                                   argTypes.begin() + allNames.size());
  rock::expandFlatFunctionArguments(builder, func, allNames, expandArgTypes,
                                    unflattenedArgs);

  Value queries = unflattenedArgs[0];
  Value keys = unflattenedArgs[1];
  Value values = unflattenedArgs[2];

  Value quantBias;
  Value quantScale;
  Value scale;
  Value bias;
  Value currentSeqLenTensor;
  Value prefixOffsetTensor;

  ShapedType qType = cast<ShapedType>(queries.getType());
  Type qkElemType = qType.getElementType();
  ArrayRef<int64_t> qShape = qType.getShape();
  SmallVector<int64_t> qkShape = {qShape[0], sequenceLengthQ, sequenceLengthK};

  SmallVector<Value> elemwiseInputs;
  unsigned optionalArgsCounter = 3;
  if (isQuantized) {
    quantBias = unflattenedArgs[optionalArgsCounter++];
    quantBias = rock::insertBroadcast(builder, loc, quantBias, qkShape);
    elemwiseInputs.push_back(quantBias);
    quantScale = unflattenedArgs[optionalArgsCounter++];
    quantScale = rock::insertBroadcast(builder, loc, quantScale, qkShape);
    elemwiseInputs.push_back(quantScale);
  }
  if (hasAttnScale) {
    scale = unflattenedArgs[optionalArgsCounter++];
    elemwiseInputs.push_back(scale);
  }
  if (hasAttnBias) {
    bias = unflattenedArgs[optionalArgsCounter++];
    elemwiseInputs.push_back(bias);
  }
  if (!currentSeqLen.empty()) {
    currentSeqLenTensor = broadcastBatchTensorRock(
        builder, loc, unflattenedArgs[optionalArgsCounter++]);
  }
  if (!prefixOffset.empty()) {
    prefixOffsetTensor = broadcastBatchTensorRock(
        builder, loc, unflattenedArgs[optionalArgsCounter++]);
  }

  // Prefix causal masking requires causal to be enabled
  bool actualCausal = causalMasking || !prefixOffset.empty();

  auto softmaxType =
      TypeAttr::get(typeFromString(softmaxDataType.getValue(), ctx));
  auto attention = rock::AttentionOp::create(
      builder, loc, outputLogicalType, returnLSE ? lseLogicalType : nullptr,
      queries, keys, values, elemwiseInputs, currentSeqLenTensor,
      prefixOffsetTensor, numHeadsQ, numHeadsKV, transposeQ, transposeK,
      transposeV, transposeO, actualCausal, splitKV, softmaxType,
      /*params0=*/nullptr, /*params1=*/nullptr);
  {
    Block *preSoftmaxElemwiseBlock =
        &attention.getPreSoftmaxBody().emplaceBlock();
    PatternRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(preSoftmaxElemwiseBlock);
    if (isQuantized) {
      qkElemType = IntegerType::get(ctx, 32);
    }
    RankedTensorType qkTensorRefType =
        RankedTensorType::get(qkShape, qkElemType);
    Value qkTensor = preSoftmaxElemwiseBlock->addArgument(qkTensorRefType, loc);
    if (isQuantized) {
      auto qkBlockShape = cast<ShapedType>(qkTensor.getType()).getShape();
      Value quantBiasI8 =
          addTensorArgToBlock(builder, loc, preSoftmaxElemwiseBlock, quantBias);
      Value quantScaleF16 = addTensorArgToBlock(
          builder, loc, preSoftmaxElemwiseBlock, quantScale);
      Value quantBiasI32 = rock::createTypeConversionOp(
          builder, loc, quantBiasI8,
          RankedTensorType::get(qkBlockShape, IntegerType::get(ctx, 32)));
      qkTensor = arith::SubIOp::create(builder, loc, qkTensor, quantBiasI32);
      qkTensor = rock::createTypeConversionOp(
          builder, loc, qkTensor,
          RankedTensorType::get(qkBlockShape, Float16Type::get(ctx)));

      qkTensor = arith::MulFOp::create(builder, loc, qkTensor, quantScaleF16);
    }
    if (hasAttnScale) {
      Value scaleTensor =
          addTensorArgToBlock(builder, loc, preSoftmaxElemwiseBlock, scale);
      qkTensor = arith::MulFOp::create(builder, loc, qkTensor, scaleTensor);
    }
    if (hasAttnBias) {
      Value biasTensor =
          addTensorArgToBlock(builder, loc, preSoftmaxElemwiseBlock, bias);
      qkTensor = arith::AddFOp::create(builder, loc, qkTensor, biasTensor);
    }
    rock::YieldOp::create(builder, loc, qkTensor);
  }

  if (!params.perfConfig.empty())
    attention->setAttr("perf_config", builder.getStringAttr(params.perfConfig));

  // Apply output transform to flatten the attention result, then store
  Value flatResult =
      rock::flattenOutput(builder, loc, attention.getResult(), outputDimNames);
  Value outputArg = func.getArgument(outputArgIdx);
  Value storedOut = rock::StoreOp::create(builder, loc, outputFlatType,
                                          flatResult, outputArg, storeMethod);

  SmallVector<Value> returnOperands = {storedOut};
  if (returnLSE) {
    Value flatLSE =
        rock::flattenOutput(builder, loc, attention.getLse(), lseDimNames);
    Value lseArg = func.getArgument(lseArgIdx);
    Value storedLSE = rock::StoreOp::create(builder, loc, lseFlatType, flatLSE,
                                            lseArg, rock::StoreMethod::Set);
    returnOperands.push_back(storedLSE);
  }

  func::ReturnOp::create(builder, loc, returnOperands);
  module.push_back(func);
  return func;
}
static func::FuncOp
createGpuConvElementwiseGemmKernel(ModuleOp module, const GenParams &params) {
  MLIRContext *ctx = module.getContext();
  Location loc = module->getLoc();
  OpBuilder builder(ctx);

  // Set arch on module to make compilation pipeline work
  StringAttr archAttr = builder.getStringAttr(params.arch);
  if (!module->hasAttr(rock::ArchAttr::getMnemonic()))
    module->setAttr(rock::ArchAttr::getMnemonic(), archAttr);

  const auto *config = params.convConfig.value();
  SmallVector<Type, 5> argTypes;
  rock::GemmSize firstGemmSize =
      getConvElementwiseGemmTypes(argTypes, config, params.types);
  SmallVector<Type, 5> flatArgTypes =
      llvm::map_to_vector(argTypes, rock::getFlattenedType);
  IntegerAttr numCUAttr =
      (num_cu.getNumOccurrences() > 0 ? builder.getI32IntegerAttr(num_cu)
                                      : nullptr);

  IntegerAttr numChipletsAttr = (numChiplets.getNumOccurrences() > 0
                                     ? builder.getI32IntegerAttr(numChiplets)
                                     : nullptr);

  SmallVector<NamedAttribute> funcAttrs = {
      builder.getNamedAttr(rock::KernelAttr::getMnemonic(),
                           builder.getUnitAttr()),
      builder.getNamedAttr(rock::ArchAttr::getMnemonic(), archAttr)};

  if (numCUAttr)
    funcAttrs.push_back(
        builder.getNamedAttr(rock::NumCUAttr::getMnemonic(), numCUAttr));

  if (numChipletsAttr)
    funcAttrs.push_back(builder.getNamedAttr(
        rock::NumChipletsAttr::getMnemonic(), numChipletsAttr));

  constexpr StringLiteral kernelName("rock_conv_gemm");
  auto func = func::FuncOp::create(
      builder, loc, kernelName,
      builder.getFunctionType(flatArgTypes,
                              {flatArgTypes[flatArgTypes.size() - 1]}),
      funcAttrs);

  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  SmallVector<Value> unflattenedArgs;
  SmallVector<SmallVector<StringRef>> allNames;
  getConvElementwiseGemmDimNames(allNames, config, params.types);
  rock::expandFlatFunctionArguments(builder, func, allNames, argTypes,
                                    unflattenedArgs);

  Value filter = unflattenedArgs[0];
  Value input = unflattenedArgs[1];
  Value c = unflattenedArgs[2];
  Value output = unflattenedArgs[3];
  SmallVector<Value> elemwiseInputs;

  SmallVector<int64_t, 8> pad;
  for (const auto &[left, right] :
       zip(config->paddingLeftDims, config->paddingRightDims)) {
    pad.push_back(left);
    pad.push_back(right);
  }
  auto convElntGemm = rock::ConvElementwiseGemmOp::create(
      builder, loc, output.getType(), filter, input, c, elemwiseInputs,
      transposeC, transposeO, builder.getIndexArrayAttr(pad),
      builder.getIndexArrayAttr(config->strideDims),
      builder.getIndexArrayAttr(config->dilationDims),
      /*params0=*/nullptr, /*params1=*/nullptr);
  {
    Block *preSecondGemmBlock =
        &convElntGemm.getPreSecondGemmBody().emplaceBlock();
    PatternRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(preSecondGemmBlock);
    ShapedType aType = cast<ShapedType>(filter.getType());
    ArrayRef<int64_t> aShape = aType.getShape();
    Type abElemType = aType.getElementType();
    RankedTensorType abTensorType = RankedTensorType::get(
        {aShape[0], firstGemmSize.m, firstGemmSize.n}, abElemType);
    Value abTensor = preSecondGemmBlock->addArgument(abTensorType, loc);
    rock::YieldOp::create(builder, loc, abTensor);
  }

  if (!params.perfConfig.empty())
    convElntGemm->setAttr("perf_config",
                          builder.getStringAttr(params.perfConfig));

  // convolution attributes
  SmallVector<StringAttr, 5> filterLayoutSpec;
  SmallVector<StringAttr, 5> inputLayoutSpec;
  for (auto &key : config->filterLayout)
    filterLayoutSpec.push_back(builder.getStringAttr(StringRef(&key, 1)));
  for (auto &key : config->inputLayout)
    inputLayoutSpec.push_back(builder.getStringAttr(StringRef(&key, 1) + "i"));

  convElntGemm->setAttr("filter_layout",
                        builder.getArrayAttr(ArrayRef<Attribute>(
                            filterLayoutSpec.begin(), filterLayoutSpec.end())));
  convElntGemm->setAttr("input_layout",
                        builder.getArrayAttr(ArrayRef<Attribute>(
                            inputLayoutSpec.begin(), inputLayoutSpec.end())));

  // Store the result to the transformed output tensor
  Value storedOut =
      rock::StoreOp::create(builder, loc, flatArgTypes[flatArgTypes.size() - 1],
                            convElntGemm.getResult(), output, storeMethod);

  func::ReturnOp::create(builder, loc, storedOut);

  if (!disableSplitKForTuning)
    func->setAttr(rock::EnableSplitKForTuningAttr::getMnemonic(),
                  builder.getUnitAttr());

  module.push_back(func);
  return func;
}

static func::FuncOp
createGpuGemmElementwiseGemmKernel(ModuleOp module, const GenParams &params) {
  MLIRContext *ctx = module.getContext();
  Location loc = module->getLoc();
  OpBuilder builder(ctx);

  // Set arch on module to make compilation pipeline work
  StringAttr archAttr = builder.getStringAttr(params.arch);
  if (!module->hasAttr(rock::ArchAttr::getMnemonic()))
    module->setAttr(rock::ArchAttr::getMnemonic(), archAttr);

  SmallVector<Type, 5> argTypes;
  getGemmElementwiseGemmTypes(argTypes, params.types);
  SmallVector<Type, 5> flatArgTypes =
      llvm::map_to_vector(argTypes, rock::getFlattenedType);
  IntegerAttr numCUAttr =
      (num_cu.getNumOccurrences() > 0 ? builder.getI32IntegerAttr(num_cu)
                                      : nullptr);

  IntegerAttr numChipletsAttr = (numChiplets.getNumOccurrences() > 0
                                     ? builder.getI32IntegerAttr(numChiplets)
                                     : nullptr);
  SmallVector<NamedAttribute> funcAttrs = {
      builder.getNamedAttr(rock::KernelAttr::getMnemonic(),
                           builder.getUnitAttr()),
      builder.getNamedAttr(rock::ArchAttr::getMnemonic(), archAttr)};

  if (numCUAttr)
    funcAttrs.push_back(
        builder.getNamedAttr(rock::NumCUAttr::getMnemonic(), numCUAttr));

  if (numChipletsAttr)
    funcAttrs.push_back(builder.getNamedAttr(
        rock::NumChipletsAttr::getMnemonic(), numChipletsAttr));

  constexpr StringLiteral kernelName("rock_gemm_gemm");
  auto func = func::FuncOp::create(
      builder, loc, kernelName,
      builder.getFunctionType(flatArgTypes,
                              {flatArgTypes[flatArgTypes.size() - 1]}),
      funcAttrs);

  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  SmallVector<Value> unflattenedArgs;
  SmallVector<SmallVector<StringRef>> allNames;
  getGemmElementwiseGemmDimNames(allNames, params.types);

  // Save output dim names and logical type, then trim from expansion
  SmallVector<StringRef> outputDimNames = allNames.back();
  Type outputLogicalType = argTypes.back();
  Type outputFlatType = flatArgTypes.back();
  allNames.pop_back();
  SmallVector<Type> expandArgTypes(argTypes.begin(), argTypes.end() - 1);
  rock::expandFlatFunctionArguments(builder, func, allNames, expandArgTypes,
                                    unflattenedArgs);

  Value a = unflattenedArgs[0];
  Value b = unflattenedArgs[1];
  Value c = unflattenedArgs[2];
  SmallVector<Value> elemwiseInputs;

  auto gemmElntGemm = rock::GemmElementwiseGemmOp::create(
      builder, loc, outputLogicalType, a, b, c, elemwiseInputs, transposeA,
      transposeB, transposeC, transposeO,
      /*params0=*/nullptr, /*params1=*/nullptr);
  {
    Block *preSecondGemmBlock =
        &gemmElntGemm.getPreSecondGemmBody().emplaceBlock();
    PatternRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(preSecondGemmBlock);
    ShapedType aType = cast<ShapedType>(a.getType());
    ArrayRef<int64_t> aShape = aType.getShape();
    Type abElemType = aType.getElementType();
    RankedTensorType abType =
        RankedTensorType::get({aShape[0], gemmM, gemmN}, abElemType);
    Value abTensor = preSecondGemmBlock->addArgument(abType, loc);
    rock::YieldOp::create(builder, loc, abTensor);
  }

  if (!params.perfConfig.empty())
    gemmElntGemm->setAttr("perf_config",
                          builder.getStringAttr(params.perfConfig));

  // Apply the output transform to flatten the result, then store
  Value flatResult = rock::flattenOutput(builder, loc, gemmElntGemm.getResult(),
                                         outputDimNames);
  Value outputArg = func.getArgument(func.getNumArguments() - 1);
  Value storedOut = rock::StoreOp::create(builder, loc, outputFlatType,
                                          flatResult, outputArg, storeMethod);

  func::ReturnOp::create(builder, loc, {storedOut});
  if (!disableSplitKForTuning)
    func->setAttr(rock::EnableSplitKForTuningAttr::getMnemonic(),
                  builder.getUnitAttr());

  module.push_back(func);
  return func;
}

static func::FuncOp createCpuGemmKernelWithMlir(ModuleOp module,
                                                const GenParams &params) {
  MLIRContext *ctx = module.getContext();
  OpBuilder b(ctx);
  Location loc = module->getLoc();

  auto cpuTypes = params.types;
  SmallVector<Type, 3> argTypes;
  getGemmTypes(cpuTypes, argTypes, /*isCpuVerifier=*/true);

  SmallVector<Type> flatArgTypes;
  for (Type t : argTypes) {
    flatArgTypes.push_back(rock::getFlattenedType(t));
  }

  // Result type is the C tensor (index 4 for scaled gemm, 2 otherwise)
  int cArgIdx = scaledGemm ? 4 : 2;
  Type resultType = flatArgTypes[cArgIdx];

  constexpr llvm::StringLiteral cpuKernName("host_naive_gemm");
  auto func = func::FuncOp::create(
      b, loc, cpuKernName, b.getFunctionType(flatArgTypes, {resultType}));
  // Mark as CPU verifier so buildHostLoweringPipeline can identify it
  func->setAttr(rock::CpuVerifierAttr::getMnemonic(), b.getUnitAttr());
  module.push_back(func);

  Block *block = func.addEntryBlock();
  b.setInsertionPointToStart(block);

  Value aVal = block->getArgument(0), bVal = block->getArgument(1);

  auto floatType = b.getF32Type();

  // Convert to f32 if needed
  aVal = ensureFloatIsF32(b, loc, aVal, floatType);
  bVal = ensureFloatIsF32(b, loc, bVal, floatType);

  Value aScaleVal = nullptr, bScaleVal = nullptr;
  Value cVal;
  if (scaledGemm) {
    aScaleVal = block->getArgument(2);
    bScaleVal = block->getArgument(3);
    aScaleVal = ensureFloatIsF32(b, loc, aScaleVal, floatType);
    bScaleVal = ensureFloatIsF32(b, loc, bScaleVal, floatType);
    cVal = block->getArgument(4);
  } else {
    cVal = block->getArgument(2);
  }
  cVal = ensureFloatIsF32(b, loc, cVal, floatType);

  // Expand flat tensors to logical 3D shapes
  auto expandTensorArg = [&loc, &b](Value arg, Type rawLogicalType) -> Value {
    auto shapedType = cast<ShapedType>(rawLogicalType);
    auto flatType = cast<RankedTensorType>(arg.getType());
    Type elemType = flatType.getElementType();
    // Use f32 element type if it's a float
    if (isa<FloatType>(elemType))
      elemType = Float32Type::get(arg.getContext());
    auto logicalType = RankedTensorType::get(shapedType.getShape(), elemType);
    ArrayRef<int64_t> logicalShape = logicalType.getShape();
    ReassociationIndices allDims = llvm::to_vector(
        llvm::iota_range<int64_t>(0, logicalShape.size(), false));
    return tensor::ExpandShapeOp::create(b, loc, logicalType, arg, allDims);
  };

  AffineExpr g = b.getAffineDimExpr(0), m = b.getAffineDimExpr(1),
             n = b.getAffineDimExpr(2), k = b.getAffineDimExpr(3);
  AffineMap aMap = AffineMap::get(
                4, 0, {g, transposeA ? k : m, transposeA ? m : k}, ctx),
            bMap = AffineMap::get(
                4, 0, {g, transposeB ? n : k, transposeB ? k : n}, ctx),
            cMap = AffineMap::get(
                4, 0, {g, transposeC ? n : m, transposeC ? m : n}, ctx);
  Value aExpVal = expandTensorArg(aVal, argTypes[0]),
        bExpVal = expandTensorArg(bVal, argTypes[1]),
        cExpVal = expandTensorArg(cVal, argTypes[cArgIdx]);

  Value aExpValScaled = nullptr, bExpValScaled = nullptr;

  // Initialize output with zeros using linalg.fill
  auto cExpType = cast<RankedTensorType>(cExpVal.getType());
  Value zeroVal = rock::createZeroConstantOp(b, loc, cExpType.getElementType());
  Value emptyC = tensor::EmptyOp::create(b, loc, cExpType, ValueRange{});
  Value zeroC = linalg::FillOp::create(b, loc, zeroVal, emptyC).getResult(0);

  // Perform GEMM using linalg.generic with tensor semantics
  Value result;
  if (scaledGemm) {
    // Create scaled AffineMaps
    // (g, m, n, k) -> (g, m, k // blockSize)  for A if not transposed
    // (g, m, n, k) -> (g, k // blockSize, n)  for B if not transposed
    auto scaleAffineMap = [&](bool transposeFlag, AffineExpr d) -> AffineMap {
      auto cBlockSize = getAffineConstantExpr(quantBlockSize, b.getContext());
      auto kFloorDiv = k.floorDiv(cBlockSize);
      SmallVector<AffineExpr> resultExprs = {g, transposeFlag ? kFloorDiv : d,
                                             transposeFlag ? d : kFloorDiv};
      return AffineMap::get(4, 0, resultExprs, b.getContext());
    };
    AffineMap aMapScaled = scaleAffineMap(transposeScaleA, m);
    AffineMap bMapScaled = scaleAffineMap(transposeScaleB, n);

    aExpValScaled = expandTensorArg(aScaleVal, argTypes[2]);
    bExpValScaled = expandTensorArg(bScaleVal, argTypes[3]);

    auto genericOp = linalg::GenericOp::create(
        b, loc, cExpType,
        ValueRange{aExpVal, bExpVal, aExpValScaled, bExpValScaled},
        ValueRange{zeroC},
        ArrayRef<AffineMap>{aMap, bMap, aMapScaled, bMapScaled, cMap},
        ArrayRef<utils::IteratorType>{
            utils::IteratorType::parallel, utils::IteratorType::parallel,
            utils::IteratorType::parallel, utils::IteratorType::reduction},
        /*doc=*/"", /*library_call=*/"",
        [](OpBuilder &builder, Location loc, ValueRange elems) {
          Value a = elems[0], bArg = elems[1], aScale = elems[2],
                bScale = elems[3];
          Value c = elems[4];
          Type cType = c.getType();
          if (isa<IntegerType>(cType)) {
            Value aExt = rock::createTypeConversionOp(builder, loc, a, cType);
            Value bExt =
                rock::createTypeConversionOp(builder, loc, bArg, cType);
            Value mul = arith::MulIOp::create(builder, loc, aExt, bExt);
            Value add = arith::AddIOp::create(builder, loc, mul, c);
            linalg::YieldOp::create(builder, loc, add);
          } else {
            a = arith::MulFOp::create(builder, loc, a, aScale);
            bArg = arith::MulFOp::create(builder, loc, bArg, bScale);
            Value mul = arith::MulFOp::create(builder, loc, a, bArg);
            Value add = arith::AddFOp::create(builder, loc, mul, c);
            linalg::YieldOp::create(builder, loc, add);
          }
        });
    result = genericOp.getResult(0);
  } else {
    auto genericOp = linalg::GenericOp::create(
        b, loc, cExpType, ValueRange{aExpVal, bExpVal}, ValueRange{zeroC},
        ArrayRef<AffineMap>{aMap, bMap, cMap},
        ArrayRef<utils::IteratorType>{
            utils::IteratorType::parallel, utils::IteratorType::parallel,
            utils::IteratorType::parallel, utils::IteratorType::reduction},
        /*doc=*/"", /*library_call=*/"",
        [](OpBuilder &builder, Location loc, ValueRange elems) {
          Value a = elems[0], bArg = elems[1];
          Value c = elems[2];
          Type cType = c.getType();
          if (isa<IntegerType>(cType)) {
            Value aExt = rock::createTypeConversionOp(builder, loc, a, cType);
            Value bExt =
                rock::createTypeConversionOp(builder, loc, bArg, cType);
            Value mul = arith::MulIOp::create(builder, loc, aExt, bExt);
            Value add = arith::AddIOp::create(builder, loc, mul, c);
            linalg::YieldOp::create(builder, loc, add);
          } else {
            Value mul = arith::MulFOp::create(builder, loc, a, bArg);
            Value add = arith::AddFOp::create(builder, loc, mul, c);
            linalg::YieldOp::create(builder, loc, add);
          }
        });
    result = genericOp.getResult(0);
  }

  // Collapse back to flat shape for return
  auto resultFlatType = cast<RankedTensorType>(resultType);
  // If element types differ (f32 vs original), we need to convert back
  if (resultFlatType.getElementType() != cExpType.getElementType()) {
    // Convert f32 result back to original type (keep 3D shape, collapse later)
    auto resultExpType = cExpType.cloneWith(cExpType.getShape(),
                                            resultFlatType.getElementType());
    Value emptyResult =
        tensor::EmptyOp::create(b, loc, resultExpType, ValueRange{});
    AffineMap identityMap =
        AffineMap::getMultiDimIdentityMap(cExpType.getRank(), b.getContext());
    SmallVector<utils::IteratorType> iteratorTypes(
        cExpType.getRank(), utils::IteratorType::parallel);
    auto convertOp = linalg::GenericOp::create(
        b, loc, resultExpType, ValueRange{result}, ValueRange{emptyResult},
        ArrayRef<AffineMap>{identityMap, identityMap}, iteratorTypes,
        [](OpBuilder &nestedB, Location nestedLoc, ValueRange args) {
          Value src = args[0];
          Type dstType = args[1].getType();
          Value converted;
          if (isa<IntegerType>(dstType)) {
            converted =
                arith::FPToSIOp::create(nestedB, nestedLoc, dstType, src);
          } else {
            converted =
                arith::TruncFOp::create(nestedB, nestedLoc, dstType, src);
          }
          linalg::YieldOp::create(nestedB, nestedLoc, converted);
        });
    result = convertOp.getResult(0);
  }

  // Collapse to flat 1D tensor
  ArrayRef<int64_t> resultShape =
      cast<RankedTensorType>(result.getType()).getShape();
  ReassociationIndices allDims =
      llvm::to_vector(llvm::iota_range<int64_t>(0, resultShape.size(), false));
  Value flatResult =
      tensor::CollapseShapeOp::create(b, loc, resultFlatType, result, allDims);

  func::ReturnOp::create(b, loc, flatResult);
  return func;
}

static Value squeeze(OpBuilder &builder, Location loc, Value src,
                     size_t squeezeDim) {
  auto origShape = cast<ShapedType>(src.getType()).getShape();
  assert(origShape[squeezeDim] == 1);
  SmallVector<int64_t> newShape;
  newShape.reserve(origShape.size() - 1);

  // Copy all elements except the one at squeezeDim
  for (size_t i = 0; i < origShape.size(); ++i) {
    if (i != squeezeDim) {
      newShape.push_back(origShape[i]);
    }
  }
  ImplicitLocOpBuilder implicitBuilder(loc, builder);
  auto shapeValue = tosa::getTosaConstShape(implicitBuilder, newShape);
  return tosa::ReshapeOp::create(builder, loc, src, shapeValue);
}

static Value convOutToGemmA(OpBuilder &builder, Location loc, Value convOut,
                            rock::GemmSize firstGemmSize) {
  // tensor<bxhxwxc> -> tensor<1x(bxhxw)xc>
  SmallVector<int64_t> newShape = {1, firstGemmSize.n, firstGemmSize.m};
  ImplicitLocOpBuilder implicitBuilder(loc, builder);
  auto shapeValue = tosa::getTosaConstShape(implicitBuilder, newShape);
  return tosa::ReshapeOp::create(builder, loc, convOut, shapeValue);
}

static func::FuncOp
createCpuConvElementwiseGemmKernelWithMlir(ModuleOp module,
                                           const GenParams &params) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);
  Location loc = module->getLoc();

  const auto *config = params.convConfig.value();
  SmallVector<Type, 5> argTypes;
  rock::GemmSize firstGemmSize =
      getConvElementwiseGemmTypes(argTypes, config, params.types);

  // Convert tensor types to memref types for CPU verifier
  SmallVector<Type, 5> flatArgTypes;
  for (Type t : argTypes) {
    Type flatType = rock::getFlattenedType(t);
    if (auto tensorType = dyn_cast<RankedTensorType>(flatType)) {
      flatArgTypes.push_back(
          MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
    } else {
      flatArgTypes.push_back(flatType);
    }
  }

  constexpr llvm::StringLiteral cpuKernName("host_naive_conv_gemm");
  auto func = func::FuncOp::create(builder, loc, cpuKernName,
                                   builder.getFunctionType(flatArgTypes, {}));

  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  auto getTensorForBlockArg = [&builder, &loc, &block,
                               &argTypes](unsigned blockArgIndex,
                                          bool isWritable = false) {
    constexpr bool isRestrict{true};
    Value flatTensor = bufferization::ToTensorOp::create(
        builder, loc,
        memref::getTensorTypeFromMemRefType(
            block->getArgument(blockArgIndex).getType()),
        block->getArgument(blockArgIndex), isRestrict, isWritable);
    ArrayRef<int64_t> origShape =
        cast<ShapedType>(argTypes[blockArgIndex]).getShape();

    Value reshapedTensor;
    ImplicitLocOpBuilder implicitBuilder(loc, builder);
    if (origShape.size() == 2) {
      SmallVector<int64_t, 3> expShape(origShape.size() + 1, 0);
      expShape[0] = 1;
      llvm::copy(origShape, expShape.begin() + 1);
      auto shapeValue = tosa::getTosaConstShape(implicitBuilder, expShape);
      reshapedTensor =
          tosa::ReshapeOp::create(builder, loc, flatTensor, shapeValue);
    } else {
      auto shapeValue = tosa::getTosaConstShape(implicitBuilder, origShape);
      reshapedTensor =
          tosa::ReshapeOp::create(builder, loc, flatTensor, shapeValue);
    }
    return reshapedTensor;
  };

  ConvTensorDimInfo filterInfo = parseConvTensorLayout(config->filterLayout,
                                                       config->filterDimension,
                                                       'k', 'c'),
                    inputInfo = parseConvTensorLayout(
                        config->inputLayout, config->inputDimension, 'n', 'c');

  auto filterTensor = getTensorForBlockArg(0);
  filterTensor = squeeze(builder, loc, filterTensor, filterInfo.gDim);
  int32_t kDim = (filterInfo.nonImg1Dim < filterInfo.gDim)
                     ? filterInfo.nonImg1Dim
                     : filterInfo.nonImg1Dim - 1;
  int32_t cDim = (filterInfo.nonImg2Dim < filterInfo.gDim)
                     ? filterInfo.nonImg2Dim
                     : filterInfo.nonImg2Dim - 1;
  int32_t hDim = (filterInfo.imageDims[0] < filterInfo.gDim)
                     ? filterInfo.imageDims[0]
                     : filterInfo.imageDims[0] - 1;
  int32_t wDim = (filterInfo.imageDims[1] < filterInfo.gDim)
                     ? filterInfo.imageDims[1]
                     : filterInfo.imageDims[1] - 1;
  filterTensor = rock::tosa::getTransposeOp(builder, loc, filterTensor,
                                            {kDim, hDim, wDim, cDim});
  auto inputTensor = getTensorForBlockArg(1);
  inputTensor = squeeze(builder, loc, inputTensor, inputInfo.gDim);
  int32_t nDim = (inputInfo.nonImg1Dim < inputInfo.gDim)
                     ? inputInfo.nonImg1Dim
                     : inputInfo.nonImg1Dim - 1;
  cDim = (inputInfo.nonImg2Dim < inputInfo.gDim) ? inputInfo.nonImg2Dim
                                                 : inputInfo.nonImg2Dim - 1;
  hDim = (inputInfo.imageDims[0] < inputInfo.gDim) ? inputInfo.imageDims[0]
                                                   : inputInfo.imageDims[0] - 1;
  wDim = (inputInfo.imageDims[1] < inputInfo.gDim) ? inputInfo.imageDims[1]
                                                   : inputInfo.imageDims[1] - 1;
  inputTensor = rock::tosa::getTransposeOp(builder, loc, inputTensor,
                                           {nDim, hDim, wDim, cDim});

  auto cTensor = getTensorForBlockArg(2);
  if (transposeC) {
    cTensor = rock::tosa::getTransposeOp(builder, loc, cTensor, {0, 2, 1});
  }
  auto weightZp =
      tosa::createZeroPointTensor(builder, loc, filterTensor.getType(), 0)
          .value();

  Type convOutElemType = params.types[2];
  // accumulate in 32 bit
  Type firstAccType = rock::getAccType(params.types[0], params.types[1]);

  auto biasTy = RankedTensorType::get(
      cast<ShapedType>(filterTensor.getType()).getShape()[0], convOutElemType);
  auto biasTensor = tosa::ConstOp::create(
      builder, loc, biasTy, cast<ElementsAttr>(builder.getZeroAttr(biasTy)));

  SmallVector<int64_t> pads;
  assert(config->paddingLeftDims.size() == config->paddingRightDims.size());
  for (size_t i = 0; i < config->paddingLeftDims.size(); i++) {
    pads.push_back(config->paddingLeftDims[i]);
    pads.push_back(config->paddingRightDims[i]);
  }

  // Floor-mode convolutions may have partial windows that violate TOSA's
  // exact-divisibility-by-stride requirement. Adjust padding/input to fix.
  inputTensor = rock::tosa::adjustConvPadding(
      builder, loc, inputTensor, filterTensor, pads, config->strideDims,
      config->dilationDims);
  auto inputZp =
      tosa::createZeroPointTensor(builder, loc, inputTensor.getType(), 0)
          .value();

  Value convOut = rock::tosa::createOpAndInfer<tosa::Conv2DOp>(
      builder, loc, convOutElemType, inputTensor, filterTensor, biasTensor,
      inputZp, weightZp, builder.getDenseI64ArrayAttr(pads),
      builder.getDenseI64ArrayAttr(config->strideDims),
      builder.getDenseI64ArrayAttr(config->dilationDims), firstAccType);
  // TODO(roctriton): group conv
  // builder.getI64IntegerAttr(groupSize));

  // convert conv output to matmul A matrix
  // tensor<bxhxwxkxf16> -> tensor<1x(b*h*w)xkxf16>
  Value gemmA = convOutToGemmA(builder, loc, convOut, firstGemmSize);
  auto abZp =
      tosa::createZeroPointTensor(builder, loc, gemmA.getType(), 0).value();
  auto cZp =
      tosa::createZeroPointTensor(builder, loc, cTensor.getType(), 0).value();
  Type secondGemmOutElemType = params.types[3];
  // accumulate in 32 bit
  Type secondAccType = rock::getAccType(convOutElemType, params.types[2]);
  auto resultTensorMatMul = rock::tosa::createOpAndInfer<tosa::MatMulOp>(
      builder, loc, secondGemmOutElemType, gemmA, cTensor, abZp, cZp);
  resultTensorMatMul->setAttr("acc_type", TypeAttr::get(secondAccType));
  Value resultTensor = resultTensorMatMul.getResult();

  if (transposeO) {
    resultTensor =
        rock::tosa::getTransposeOp(builder, loc, resultTensor, {0, 2, 1});
  }

  Value output = block->getArguments().back();
  auto outputType = cast<bufferization::BufferLikeType>(output.getType());

  ImplicitLocOpBuilder implicitBuilder(loc, builder);
  auto shapeValue = tosa::getTosaConstShape(
      implicitBuilder, cast<ShapedType>(outputType).getShape());
  auto flatResultTensor =
      tosa::ReshapeOp::create(builder, loc, resultTensor, shapeValue);

  auto flatResultMemref = bufferization::ToBufferOp::create(
      builder, loc, outputType, flatResultTensor);

  memref::CopyOp::create(builder, loc, flatResultMemref, output);

  func::ReturnOp::create(builder, loc);
  module.push_back(func);
  return func;
}

static func::FuncOp
createCpuGemmElementwiseGemmKernelWithMlir(ModuleOp module,
                                           const GenParams &params) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);
  Location loc = module->getLoc();

  SmallVector<Type, 5> argTypes;
  getGemmElementwiseGemmTypes(argTypes, params.types);
  // Convert tensor types to memref types for CPU verifier
  SmallVector<Type, 5> flatArgTypes;
  for (Type t : argTypes) {
    Type flatType = rock::getFlattenedType(t);
    if (auto tensorType = dyn_cast<RankedTensorType>(flatType)) {
      flatArgTypes.push_back(
          MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
    } else {
      flatArgTypes.push_back(flatType);
    }
  }

  constexpr llvm::StringLiteral cpuKernName("host_naive_gemm_gemm");
  auto func = func::FuncOp::create(builder, loc, cpuKernName,
                                   builder.getFunctionType(flatArgTypes, {}));

  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  auto getTensorForBlockArg = [&builder, &loc, &block,
                               &argTypes](unsigned blockArgIndex,
                                          bool isWritable = false) {
    constexpr bool isRestrict{true};
    Value flatTensor = bufferization::ToTensorOp::create(
        builder, loc,
        memref::getTensorTypeFromMemRefType(
            block->getArgument(blockArgIndex).getType()),
        block->getArgument(blockArgIndex), isRestrict, isWritable);
    ArrayRef<int64_t> origShape =
        cast<ShapedType>(argTypes[blockArgIndex]).getShape();

    Value reshapedTensor;
    ImplicitLocOpBuilder implicitBuilder(loc, builder);
    if (origShape.size() == 2) {
      SmallVector<int64_t, 3> expShape(origShape.size() + 1, 0);
      expShape[0] = 1;
      llvm::copy(origShape, expShape.begin() + 1);
      auto shapeValue = tosa::getTosaConstShape(implicitBuilder, expShape);
      reshapedTensor =
          tosa::ReshapeOp::create(builder, loc, flatTensor, shapeValue);
    } else {
      auto shapeValue = tosa::getTosaConstShape(implicitBuilder, origShape);
      reshapedTensor =
          tosa::ReshapeOp::create(builder, loc, flatTensor, shapeValue);
    }
    return reshapedTensor;
  };

  auto aTensor = getTensorForBlockArg(0);
  if (transposeA) {
    aTensor = rock::tosa::getTransposeOp(builder, loc, aTensor, {0, 2, 1});
  }
  auto bTensor = getTensorForBlockArg(1);
  if (transposeB) {
    bTensor = rock::tosa::getTransposeOp(builder, loc, bTensor, {0, 2, 1});
  }
  auto cTensor = getTensorForBlockArg(2);
  if (transposeC) {
    cTensor = rock::tosa::getTransposeOp(builder, loc, cTensor, {0, 2, 1});
  }
  auto aZp =
      tosa::createZeroPointTensor(builder, loc, aTensor.getType(), 0).value();
  auto bZp =
      tosa::createZeroPointTensor(builder, loc, bTensor.getType(), 0).value();

  Type firstGemmOutElemType = params.types[2];
  // accumulate in 32 bit
  Type firstAccType = rock::getAccType(params.types[0], params.types[1]);
  auto abTensorMatMul = rock::tosa::createOpAndInfer<tosa::MatMulOp>(
      builder, loc, firstGemmOutElemType, aTensor, bTensor, aZp, bZp);
  abTensorMatMul->setAttr("acc_type", TypeAttr::get(firstAccType));
  Value abTensor = abTensorMatMul.getResult();

  auto abZp =
      tosa::createZeroPointTensor(builder, loc, abTensor.getType(), 0).value();
  auto cZp =
      tosa::createZeroPointTensor(builder, loc, cTensor.getType(), 0).value();
  Type secondGemmOutElemType = params.types[3];
  // accumulate in 32 bit
  Type secondAccType = rock::getAccType(firstGemmOutElemType, params.types[2]);
  auto resultTensorMatMul = rock::tosa::createOpAndInfer<tosa::MatMulOp>(
      builder, loc, secondGemmOutElemType, abTensor, cTensor, abZp, cZp);
  resultTensorMatMul->setAttr("acc_type", TypeAttr::get(secondAccType));
  Value resultTensor = resultTensorMatMul.getResult();

  if (transposeO) {
    resultTensor =
        rock::tosa::getTransposeOp(builder, loc, resultTensor, {0, 2, 1});
  }

  Value output = block->getArguments().back();
  auto outputType = cast<mlir::bufferization::BufferLikeType>(output.getType());

  ImplicitLocOpBuilder implicitBuilder(loc, builder);
  auto shapeValue = tosa::getTosaConstShape(
      implicitBuilder, cast<ShapedType>(outputType).getShape());
  auto flatResultTensor =
      tosa::ReshapeOp::create(builder, loc, resultTensor, shapeValue);

  auto flatResultMemref = bufferization::ToBufferOp::create(
      builder, loc, outputType, flatResultTensor);

  memref::CopyOp::create(builder, loc, flatResultMemref, output);

  func::ReturnOp::create(builder, loc);
  module.push_back(func);
  return func;
}

static func::FuncOp createCpuAttentionKernelWithMlir(ModuleOp module,
                                                     const GenParams &params) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);
  Location loc = module->getLoc();

  bool isQuantized = params.types[0] == IntegerType::get(ctx, 8);
  // Optionally run the host reference's interior in f64 to get more
  // accurate validation answer when running attention with a large seq len.
  const bool promoteHost = pvF64.getValue() && !isQuantized;
  Type f64Type = Float64Type::get(ctx);
  auto upcastF = [&](Value v) -> Value {
    if (!promoteHost)
      return v;
    Type elem = cast<ShapedType>(v.getType()).getElementType();
    if (!isa<FloatType>(elem) || elem == f64Type)
      return v;
    return rock::tosa::createOpAndInfer<tosa::CastOp>(builder, loc, f64Type, v);
  };
  auto wideF = [&](Type t) -> Type {
    return (promoteHost && isa<FloatType>(t)) ? f64Type : t;
  };

  SmallVector<Type, 5> argTypes;
  getAttentionTypes(argTypes, params.types);
  // Convert tensor types to memref types for CPU verifier
  SmallVector<Type, 5> flatArgTypes;
  for (Type t : argTypes) {
    Type flatType = rock::getFlattenedType(t);
    if (auto tensorType = dyn_cast<RankedTensorType>(flatType)) {
      flatArgTypes.push_back(
          MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
    } else {
      flatArgTypes.push_back(flatType);
    }
  }

  constexpr llvm::StringLiteral cpuKernName("host_naive_attention");
  auto func = func::FuncOp::create(builder, loc, cpuKernName,
                                   builder.getFunctionType(flatArgTypes, {}));

  Block *block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  auto getTensorForBlockArg = [&builder, &loc, &block,
                               &argTypes](unsigned blockArgIndex,
                                          bool isWritable = false) {
    constexpr bool isRestrict{true};
    Value flatTensor = bufferization::ToTensorOp::create(
        builder, loc,
        memref::getTensorTypeFromMemRefType(
            block->getArgument(blockArgIndex).getType()),
        block->getArgument(blockArgIndex), isRestrict, isWritable);
    ArrayRef<int64_t> origShape =
        cast<ShapedType>(argTypes[blockArgIndex]).getShape();

    Value reshapedTensor;
    ImplicitLocOpBuilder implicitBuilder(loc, builder);
    if (origShape.size() == 2) {
      SmallVector<int64_t, 3> expShape(origShape.size() + 1, 0);
      expShape[0] = 1;
      llvm::copy(origShape, expShape.begin() + 1);
      auto shapeValue = tosa::getTosaConstShape(implicitBuilder, expShape);
      reshapedTensor =
          tosa::ReshapeOp::create(builder, loc, flatTensor, shapeValue);
    } else {
      auto shapeValue = tosa::getTosaConstShape(implicitBuilder, origShape);
      reshapedTensor =
          tosa::ReshapeOp::create(builder, loc, flatTensor, shapeValue);
    }
    return reshapedTensor;
  };

  auto queriesTensor = upcastF(getTensorForBlockArg(0));
  if (transposeQ) {
    queriesTensor =
        rock::tosa::getTransposeOp(builder, loc, queriesTensor, {0, 2, 1});
  }
  auto keysTensor = upcastF(getTensorForBlockArg(1));
  if (transposeK) {
    keysTensor =
        rock::tosa::getTransposeOp(builder, loc, keysTensor, {0, 2, 1});
  }
  auto valuesTensor = upcastF(getTensorForBlockArg(2));
  if (transposeV) {
    valuesTensor =
        rock::tosa::getTransposeOp(builder, loc, valuesTensor, {0, 2, 1});
  }
  // GQA
  keysTensor = broadcastGQATosa(builder, loc, keysTensor);
  valuesTensor = broadcastGQATosa(builder, loc, valuesTensor);

  Type firstGemmOutElemType = wideF(params.types[0]);
  if (isQuantized) {
    firstGemmOutElemType = IntegerType::get(ctx, 32);
  } else if (params.strictMode) {
    // Strict mode (`-pv-strict`) must match the GPU's effective precision for
    // the first GEMM. For narrow floats (f16, bf16), the GPU emits a
    // round-to-narrow store of the first-GEMM result followed by a load+extend
    // before softmax. Without any pre-softmax elementwise op, that
    // truncf -> store -> load -> extf round-trip folds away during host/GPU
    // lowering, so the GPU effectively runs the chain at f32. Promote the CPU
    // reference accordingly so it also runs at f32.
    if (auto floatTy = dyn_cast<FloatType>(firstGemmOutElemType);
        floatTy && floatTy.getWidth() < 32 && !hasAttnScale && !hasAttnBias)
      firstGemmOutElemType = builder.getF32Type();
  }
  auto queriesZp =
      tosa::createZeroPointTensor(builder, loc, queriesTensor.getType(), 0)
          .value();
  auto keysZp =
      tosa::createZeroPointTensor(builder, loc, keysTensor.getType(), 0)
          .value();
  // accumulate in 32 bit (or f64 when promoting the host reference)
  Type firstAccType =
      promoteHost ? f64Type
                  : rock::getAccType(firstGemmOutElemType, params.types[1]);
  auto qkTensorMatMul = rock::tosa::createOpAndInfer<tosa::MatMulOp>(
      builder, loc, firstGemmOutElemType, queriesTensor, keysTensor, queriesZp,
      keysZp);
  qkTensorMatMul->setAttr("acc_type", TypeAttr::get(firstAccType));
  Value qkTensor = qkTensorMatMul.getResult();

  // Helper to load and reshape a 1D tensor for masking
  auto loadMaskingTensor = [&](unsigned argIndex) -> Value {
    auto tensorRaw = getTensorForBlockArg(argIndex);
    auto type = cast<RankedTensorType>(tensorRaw.getType());
    ArrayRef<int64_t> shape = type.getShape();
    assert(shape.size() == 1);

    ImplicitLocOpBuilder implicitBuilder(loc, builder);
    auto shapeValue =
        tosa::getTosaConstShape(implicitBuilder, {shape[0], 1, 1, 1});
    return rock::tosa::createOpAndInfer<tosa::ReshapeOp>(
        builder, loc, type.getElementType(), tensorRaw, shapeValue);
  };

  // get currentSeqLenTensor and prefixOffsetTensor
  Value currentSeqLenTensor;
  Value prefixOffsetTensor;
  // Walk through optional arguments to find the correct indices.
  // Argument layout: [q, k, v, (quantBias, quantScale)?, scale?, bias?,
  //                   currentSeqLen?, prefixOffset?, lse?, output]
  unsigned optionalArgIndex = 3; // Start after q, k, v
  if (isQuantized)
    optionalArgIndex += 2; // Skip quantBias, quantScale
  if (hasAttnScale)
    optionalArgIndex++; // Skip scale
  if (hasAttnBias)
    optionalArgIndex++; // Skip bias
  if (!currentSeqLen.empty()) {
    currentSeqLenTensor = loadMaskingTensor(optionalArgIndex++);
  }
  if (!prefixOffset.empty()) {
    prefixOffsetTensor = loadMaskingTensor(optionalArgIndex++);
  }

  unsigned optionalArgsCounter = 3;
  if (isQuantized) {
    auto quantBiasI8 = getTensorForBlockArg(optionalArgsCounter++);
    Value quantBiasI32 = rock::tosa::createOpAndInfer<tosa::CastOp>(
        builder, loc, IntegerType::get(ctx, 32), quantBiasI8);
    qkTensor = rock::tosa::createOpAndInfer<tosa::SubOp>(
        builder, loc, IntegerType::get(ctx, 32), qkTensor, quantBiasI32);
    qkTensor = rock::tosa::createOpAndInfer<tosa::CastOp>(
        builder, loc, Float16Type::get(ctx), qkTensor);
    auto quantScaleF16 = getTensorForBlockArg(optionalArgsCounter++);
    qkTensor = rock::tosa::getMulOp(builder, loc, qkTensor, quantScaleF16,
                                    Float16Type::get(ctx));
  }

  if (hasAttnScale) {
    auto scaleTensor = upcastF(getTensorForBlockArg(optionalArgsCounter++));
    if (!currentSeqLen.empty())
      scaleTensor =
          maskKVCacheTosa(builder, loc, scaleTensor, currentSeqLenTensor, 1.0f);

    // Use prefix offset masking if provided, otherwise standard causal
    if (prefixOffsetTensor)
      scaleTensor = prefixOffsetMaskingTosa(builder, loc, scaleTensor,
                                            prefixOffsetTensor, 1.0f);
    else if (causalMasking)
      scaleTensor = causalMaskingTosa(builder, loc, scaleTensor, 1.0f);

    qkTensor = rock::tosa::getMulOp(
        builder, loc, qkTensor, scaleTensor,
        cast<ShapedType>(scaleTensor.getType()).getElementType());
  }

  if (hasAttnBias) {
    auto biasTensor = upcastF(getTensorForBlockArg(optionalArgsCounter++));
    if (!currentSeqLen.empty())
      biasTensor =
          maskKVCacheTosa(builder, loc, biasTensor, currentSeqLenTensor, 0.0f);

    // Use prefix offset masking if provided, otherwise standard causal
    if (prefixOffsetTensor)
      biasTensor = prefixOffsetMaskingTosa(builder, loc, biasTensor,
                                           prefixOffsetTensor, 0.0f);
    else if (causalMasking)
      biasTensor = causalMaskingTosa(builder, loc, biasTensor, 0.0f);

    qkTensor = rock::tosa::createOpAndInfer<tosa::AddOp>(
        builder, loc, cast<ShapedType>(biasTensor.getType()).getElementType(),
        qkTensor, biasTensor);
  }
  // cast to softmaxType (overridden to f64 when promoting the host reference)
  auto softmaxType = wideF(typeFromString(softmaxDataType.getValue(), ctx));
  qkTensor = rock::tosa::createOpAndInfer<tosa::CastOp>(builder, loc,
                                                        softmaxType, qkTensor);

  // Apply KV-cache masking if currentSeqLen is provided
  if (currentSeqLenTensor) {
    qkTensor = maskKVCacheTosa(builder, loc, qkTensor, currentSeqLenTensor,
                               -std::numeric_limits<float>::infinity());
    optionalArgsCounter++;
  }

  // Apply prefix offset masking if prefixOffset is provided
  if (prefixOffsetTensor) {
    qkTensor =
        prefixOffsetMaskingTosa(builder, loc, qkTensor, prefixOffsetTensor,
                                -std::numeric_limits<float>::infinity());
    optionalArgsCounter++;
  }

  Value lseOut;
  // if split-kv is > 1, we use the LSE to compute the final result.
  // There's no need to verify it, and it's expected to be different
  // cpu vs gpu (sometimes).
  if (returnLSE)
    lseOut = block->getArgument(optionalArgsCounter++);

  // Apply standard causal masking only if prefix offset is not provided.
  if (causalMasking && !prefixOffsetTensor)
    qkTensor = causalMaskingTosa(builder, loc, qkTensor,
                                 -std::numeric_limits<float>::infinity());

  constexpr int64_t reductionAxis = 2;
  auto qkMaxs = rock::tosa::createOpAndInfer<tosa::ReduceMaxOp>(
      builder, loc, softmaxType, qkTensor, reductionAxis);
  auto normalizedQkTensor = rock::tosa::createOpAndInfer<tosa::SubOp>(
      builder, loc, softmaxType, qkTensor, qkMaxs);
  auto expsTensor = rock::tosa::createOpAndInfer<tosa::ExpOp>(
      builder, loc, softmaxType, normalizedQkTensor);
  auto expsSums = rock::tosa::createOpAndInfer<tosa::ReduceSumOp>(
      builder, loc, softmaxType, expsTensor, reductionAxis);
  Type resultOutElementType =
      isQuantized ? Float16Type::get(ctx) : firstGemmOutElemType;
  Value lseTensor;
  // qkMaxs = max x
  // expsSums = sum e^(x-qkMaxs)
  // lse = (log(expsSums) + qkMaxs)
  if (returnLSE) {
    Type lseType = cast<ShapedType>(lseOut.getType()).getElementType();
    // When promoting the host reference, keep LSE in f64 too; we cast back
    // to lseType right before storing into the output memref.
    Type lseComputeType = wideF(lseType);
    Value expsSumsForLSE = rock::tosa::createOpAndInfer<tosa::CastOp>(
        builder, loc, lseComputeType, expsSums);
    Value qkMaxsForLSE = rock::tosa::createOpAndInfer<tosa::CastOp>(
        builder, loc, lseComputeType, qkMaxs);
    lseTensor = rock::tosa::createOpAndInfer<tosa::LogOp>(
        builder, loc, lseComputeType, expsSumsForLSE);
    lseTensor = rock::tosa::createOpAndInfer<tosa::AddOp>(
        builder, loc, lseComputeType, lseTensor, qkMaxsForLSE);
  }

  auto invExpsSums = rock::tosa::createOpAndInfer<tosa::ReciprocalOp>(
      builder, loc, softmaxType, expsSums);

  Value softmaxTensor =
      rock::tosa::getMulOp(builder, loc, expsTensor, invExpsSums, softmaxType);
  softmaxTensor = rock::tosa::createOpAndInfer<tosa::CastOp>(
      builder, loc, resultOutElementType, softmaxTensor);
  auto softmaxZp =
      tosa::createZeroPointTensor(builder, loc, softmaxTensor.getType(), 0)
          .value();
  auto valuesZp =
      tosa::createZeroPointTensor(builder, loc, valuesTensor.getType(), 0)
          .value();

  // accumulate in 32 bit (or f64 when promoting the host reference)
  Type secondAccType = promoteHost ? f64Type
                                   : rock::getAccType(resultOutElementType,
                                                      resultOutElementType);
  auto resultTensorMatMul = rock::tosa::createOpAndInfer<tosa::MatMulOp>(
      builder, loc, resultOutElementType, softmaxTensor, valuesTensor,
      softmaxZp, valuesZp);
  resultTensorMatMul->setAttr("acc_type", TypeAttr::get(secondAccType));
  Value resultTensor = resultTensorMatMul.getResult();

  if (transposeO) {
    resultTensor =
        rock::tosa::getTransposeOp(builder, loc, resultTensor, {0, 2, 1});
  }

  if (splitKV > 1) {
    // Broadcast result and LSE to splitKV dimension (no masking needed -
    // matches GPU's computeFinalAttentionStage which broadcasts combined
    // result to all splits)
    resultTensor = broadcastBatchTosa(builder, loc, resultTensor, splitKV);
    lseTensor = broadcastBatchTosa(builder, loc, lseTensor, splitKV);
  }

  Value output = block->getArguments().back();
  assert(optionalArgsCounter == (block->getNumArguments() - 1) &&
         "All optional args should be consumed by now");

  auto outputType = cast<mlir::bufferization::BufferLikeType>(output.getType());

  // If we promoted intermediate computation to f32 (strict mode, no
  // pre-softmax scale/bias), cast the result back to the original output
  // element type before writing it to the output buffer.
  Type resultElemType =
      cast<ShapedType>(resultTensor.getType()).getElementType();
  Type outputElemType = cast<ShapedType>(outputType).getElementType();
  if (resultElemType != outputElemType) {
    resultTensor = rock::tosa::createOpAndInfer<tosa::CastOp>(
        builder, loc, outputElemType, resultTensor);
  }

  ImplicitLocOpBuilder implicitBuilder(loc, builder);
  // When promoting, downcast the f64 result to the output memref's element
  // type before reshape/buffer-write so the bufferization op sees a matching
  // element type.
  if (promoteHost) {
    Type outElem = cast<ShapedType>(outputType).getElementType();
    if (cast<ShapedType>(resultTensor.getType()).getElementType() != outElem)
      resultTensor = rock::tosa::createOpAndInfer<tosa::CastOp>(
          builder, loc, outElem, resultTensor);
  }
  auto shapeValue = tosa::getTosaConstShape(
      implicitBuilder, cast<ShapedType>(outputType).getShape());
  auto flatResultTensor =
      tosa::ReshapeOp::create(builder, loc, resultTensor, shapeValue);

  auto flatResultMemref = bufferization::ToBufferOp::create(
      builder, loc, outputType, flatResultTensor);

  memref::CopyOp::create(builder, loc, flatResultMemref, output);

  // return LSE (log-sum-exp)
  if (returnLSE) {
    auto lseOutType = cast<bufferization::BufferLikeType>(lseOut.getType());
    if (promoteHost) {
      Type lseElem = cast<ShapedType>(lseOutType).getElementType();
      if (cast<ShapedType>(lseTensor.getType()).getElementType() != lseElem)
        lseTensor = rock::tosa::createOpAndInfer<tosa::CastOp>(
            builder, loc, lseElem, lseTensor);
    }
    auto lseShapeValue = tosa::getTosaConstShape(
        implicitBuilder, cast<ShapedType>(lseOutType).getShape());
    auto flatLseTensor =
        tosa::ReshapeOp::create(builder, loc, lseTensor, lseShapeValue);

    auto flatLseMemref = bufferization::ToBufferOp::create(
        builder, loc, lseOutType, flatLseTensor);

    memref::CopyOp::create(builder, loc, flatLseMemref, lseOut);
  }

  func::ReturnOp::create(builder, loc);
  module.push_back(func);
  return func;
}

static void emitPrintTensor(OpBuilder &b, Value var) {
  auto loc = b.getUnknownLoc();
  auto varType = dyn_cast<MemRefType>(var.getType());
  auto elemType = varType.getElementType();
  auto floatType = b.getF32Type();
  auto int32Type = b.getIntegerType(32);

  // get print func
  std::string printFuncName = "printMemrefF32";
  Value pvar = var;
  Type tensorType = floatType;
  if (elemType.isIntOrIndex()) {
    printFuncName = "printMemrefI32";
    tensorType = int32Type;
    if (elemType != int32Type) {
      // make copy
      auto pvarType = MemRefType::get(varType.getShape(), int32Type);
      pvar = memref::AllocOp::create(b, loc, pvarType);
      emitMemcpy(b, var, pvar);
    }
  } else if (elemType != floatType) {
    // make copy
    auto pvarType = MemRefType::get(varType.getShape(), floatType);
    pvar = memref::AllocOp::create(b, loc, pvarType);
    emitMemcpy(b, var, pvar);
  }

  auto module = b.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  auto unrankedMRType = UnrankedMemRefType::get(tensorType, 0);
  auto printFunc = makeFuncDecl(module, printFuncName, {unrankedMRType});

  // Emit cast + call print
  auto printCast = memref::CastOp::create(b, loc, unrankedMRType, pvar);
  func::CallOp::create(b, loc, printFunc, ValueRange{printCast});

  if (pvar != var) {
    // dealloc pvar
    memref::DeallocOp::create(b, loc, pvar);
  }
}

// Per-dtype "expected rounding error per accumulation step" constants for
// the reduction-aware atol bound:
//   atol_eff = atol + K_eff * sum_error_tolerance<T>
//
// fp16 / bf16 use rocBLAS's general `sum_error_tolerance<T>` from
// `near.hpp`.
//
// fp32 / fp64 are *tighter* than rocBLAS uses. This is because
// rocBLAS uses the loose
// `K * 1e-4` only for K > 10000 or to compare against external libraries (see
// `near.hpp::reduction_requires_near`).
//
// fp8/fp4 are not in rocBLAS's table. We use eps(T) directly as the
// per-accumulation-step bound:
//   - fp8 e4m3: eps = 2^-3 = 0.125
//   - fp8 e5m2: eps = 2^-2 = 0.25
//   - fp4 e2m1: eps = 2^-1 = 0.5
static float sumErrorTolerance(Type t) {
  if (isa<Float16Type>(t))
    return 1.0f / 900.0f;
  if (isa<BFloat16Type>(t))
    return 1.0f / 100.0f;
  if (isa<Float32Type>(t))
    return 1e-6f;
  if (isa<Float64Type>(t))
    return 1e-15f;
  if (isa<Float8E4M3FNType, Float8E4M3FNUZType>(t))
    return 0.125f;
  if (isa<Float8E5M2Type, Float8E5M2FNUZType>(t))
    return 0.25f;
  if (isa<Float4E2M1FNType>(t))
    return 0.5f;
  return 1.0f / 100.0f; // conservative
}

// Trace `value` backward through value-preserving / shape-only ops to find
// a matmul-like op feeding `value`. Returns the matmul op
// (RockGemmWrapperInterface / RockGemmGemmWrapperInterface; the latter
// covers AttentionOp, GemmElementwiseGemmOp and ConvElementwiseGemmOp)
// or nullptr if no such producer exists within a bounded search.
//
// "Value-preserving" here is conservative: rock.transform (pure layout
// change) and any single-result op whose result shape matches the operand
// shape (elementwise add/mul, arith.extf/truncf etc.). For multi-operand
// elementwise ops the trace follows each operand that has the same shape
// as the result, recursing until it finds a matmul or runs out of
// candidates.
//
// This lets us match `rock.gemm -> arith.addf -> rock.reduce` dataflow
// chains (arrow = "feeds into"). Walking backward from the reduce, we
// recover the matmul so the reduce can contribute its axis length to the
// matmul's K_eff for patterns like `reduce_sum(matmul(A, B) + bias)`.
static Operation *traceToMatmulLikeProducer(Value value, unsigned depth = 0) {
  // Realistic gemm -> elementwise -> reduce chains are at most 6 hops;
  // 8 gives a small safety margin.
  constexpr unsigned kMaxDepth = 8;
  if (depth > kMaxDepth)
    return nullptr;
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return nullptr;
  if (isa<rock::RockGemmWrapperInterface, rock::RockGemmGemmWrapperInterface>(
          defOp))
    return defOp;
  if (auto xform = dyn_cast<rock::TransformOp>(defOp))
    return traceToMatmulLikeProducer(xform.getInput(), depth + 1);
  if (defOp->getNumResults() != 1)
    return nullptr;
  auto resTy = dyn_cast<ShapedType>(defOp->getResult(0).getType());
  if (!resTy)
    return nullptr;
  // Follow every operand whose shape matches the result; recurse and return
  // the first matmul-like producer found.
  for (Value operand : defOp->getOperands()) {
    auto opTy = dyn_cast<ShapedType>(operand.getType());
    if (!opTy)
      continue;
    if (opTy.getShape() != resTy.getShape())
      continue;
    if (Operation *found = traceToMatmulLikeProducer(operand, depth + 1))
      return found;
  }
  return nullptr;
}

// Return the narrowest float element type appearing on any 
// RockGemmWrapperInterface and RockGemmGemmWrapperInterface ops in
// `module`, or std::nullopt if none are present. Used as a precision floor
// when the kernel's *output* dtype is wider than the dtype in which the
// computation actually happens (e.g. `arith.extf f16 -> f32` /
// `migraphx.convert` right before the function return). In that case the
// verifier should use the narrower dtype's allclose baseline, since the
// values being compared cannot be more accurate than the narrowest float
// in the dataflow.
static std::optional<Type> scanModuleForNarrowestFloat(ModuleOp module) {
  std::optional<Type> best;
  auto consider = [&](Type t) {
    auto ft = dyn_cast_or_null<FloatType>(t);
    if (!ft)
      return;
    auto bestFt = best.has_value() ? dyn_cast<FloatType>(*best) : FloatType();
    if (!bestFt || ft.getWidth() < bestFt.getWidth())
      best = t;
  };
  module.walk([&](Operation *op) {
    if (auto gemmLike = dyn_cast<rock::RockGemmWrapperInterface>(op)) {
      consider(gemmLike.getAType());
      consider(gemmLike.getBType());
      consider(gemmLike.getCType());
      return;
    }
    if (auto gemmGemm = dyn_cast<rock::RockGemmGemmWrapperInterface>(op)) {
      consider(gemmGemm.getAType());
      consider(gemmGemm.getBType());
      consider(gemmGemm.getCType());
      consider(gemmGemm.getOutType());
      return;
    }
  });
  return best;
}

// Scan an MLIR module for RockGemmWrapperInterface and RockGemmGemmWrapperInterface ops and return the largest
// effective reduction length found, or std::nullopt if no rock reduction
// op is present.
//
// Used as a fallback when the kernel comes from a pre-lowered IR (e.g.
// --clone-harness / --verifier=clone) and so the command-line -operation
// flag is unset. The reduction axis is the source of truth for the
// K-scaled atol bound, but for clone-harness flows it is only available by
// inspecting the IR.
//
// Each matmul-like op's K is *multiplied* by every downstream rock.reduce
// whose input traces back (through rock.transform) to that op. This
// captures patterns like `reduce_sum(matmul(A, B))` where each output
// element accumulates K_gemm * N_reduce products. The multiplicative model
// matches the worst-case fp16/bf16 behaviour observed in rocBLAS-style
// testing for chained reductions.
//
// Conventions follow rock dialect:
//   rock.gemm / rock.conv*: getGemmSize().k via RockGemmWrapperInterface
//   rock.attention: head_dim_qk + seq_len_k (additive model, matches the
//     command-line path in computeReductionK below)
//   rock.gemm_elementwise_gemm / conv_elementwise_gemm: K + N (additive)
static std::optional<int64_t> scanModuleForReductionK(ModuleOp module) {
  // Pass 1: base K_eff per matmul-like op.
  llvm::DenseMap<Operation *, int64_t> baseK;
  module.walk([&](Operation *op) {
    if (auto gemmLike = dyn_cast<rock::RockGemmWrapperInterface>(op)) {
      baseK[op] = gemmLike.getGemmSize().k;
      return;
    }
    // rock.attention -- checked before the generic GemmGemm fallback because
    // AttentionOp implements RockGemmGemmWrapperInterface but its operand
    // shapes encode K differently (head_dim_qk + seq_len_k via the
    // q/kTransposed attributes), not GemmGemmSize.k + GemmGemmSize.n.
    if (auto attn = dyn_cast<rock::AttentionOp>(op)) {
      // queries: [G x] seq_q x head_qk (or transposed)
      // keys:    [G x] head_qk x seq_k (or transposed)
      auto qTy = dyn_cast<ShapedType>(attn.getQueries().getType());
      auto kTy = dyn_cast<ShapedType>(attn.getKeys().getType());
      if (!qTy || !kTy || qTy.getRank() < 2 || kTy.getRank() < 2)
        return;
      ArrayRef<int64_t> qShape = qTy.getShape();
      ArrayRef<int64_t> kShape = kTy.getShape();
      int64_t qLast = qShape[qShape.size() - 1];
      int64_t qPenult = qShape[qShape.size() - 2];
      int64_t kLast = kShape[kShape.size() - 1];
      int64_t kPenult = kShape[kShape.size() - 2];
      int64_t headDimQK = attn.getQTransposed() ? qPenult : qLast;
      int64_t seqLenK = attn.getKTransposed() ? kPenult : kLast;
      if (headDimQK > 0 && seqLenK > 0)
        baseK[op] = headDimQK + seqLenK;
      return;
    }
    if (auto gemmGemm = dyn_cast<rock::RockGemmGemmWrapperInterface>(op)) {
      auto sz = gemmGemm.getGemmGemmSize();
      baseK[op] = sz.k + sz.n;
      return;
    }
  });

  // Pass 2: multiply each matmul-like op's K by the extents of any
  // downstream rock.reduce ops feeding off it.
  llvm::DenseMap<Operation *, int64_t> kEff(baseK.begin(), baseK.end());
  module.walk([&](rock::ReduceOp reduceOp) {
    Operation *producer = traceToMatmulLikeProducer(reduceOp.getIn());
    if (!producer)
      return;
    auto it = kEff.find(producer);
    if (it == kEff.end())
      return;
    auto inTy = dyn_cast<ShapedType>(reduceOp.getIn().getType());
    if (!inTy)
      return;
    int64_t axis = reduceOp.getAxis().getSExtValue();
    if (axis < 0 || axis >= inTy.getRank())
      return;
    int64_t axisExtent = inTy.getShape()[axis];
    if (axisExtent > 0)
      it->second *= axisExtent;
  });

  int64_t best = 0;
  for (auto &kv : kEff)
    best = std::max(best, kv.second);
  if (best == 0)
    return std::nullopt;
  return best;
}

// Effective reduction length for the operation under test. For GEMM this is
// K; for attention it is head_dim_qk + seq_len_k (two cascaded reductions);
// for conv it is Cin * filter_volume (the im2col K). For element-wise ops it
// is 1. Returns 0 if shape is unknown; caller should treat as "skip scaling".
//
// This mirrors how rocBLAS's testing_gemm scales `near_check` tolerance:
//   tol = K * sum_error_tolerance<T>
//   https://github.com/ROCm/rocBLAS/blob/develop/clients/include/blas3/testing_gemm.hpp
//
// When the command-line `-operation` flag is set, K is read from the gen
// params (covers `rocmlir-gen -operation gemm/conv/attention ...`). When it
// is not (clone-harness / pre-lowered IR flows), `module` is scanned for
// rock reduction ops and the largest reduction length found is used.
static int64_t computeReductionK(const GenParams &genParams, ModuleOp module) {
  if (!genParams.operation.has_value()) {
    if (auto scanned = scanModuleForReductionK(module))
      return *scanned;
    return 1;
  }
  // Helper: convolution im2col K = Cin * product(filter spatial dims).
  // filterDimension is laid out per filterLayout; multiplying everything
  // except K (output channels) and G (group) is equivalent to that product.
  auto convK = [&]() -> int64_t {
    if (genParams.convConfig.has_value()) {
      int64_t k = 1;
      const auto &dims = (*genParams.convConfig)->filterDimension;
      const auto &layout = (*genParams.convConfig)->filterLayout;
      assert(dims.size() == layout.size());
      for (auto [d, l] : llvm::zip(dims, layout)) {
        if (l == 'k' || l == 'g')
          continue;
        k *= d;
      }
      return k;
    }
    return inputChannel * filterHeight * filterWidth;
  };
  switch (*genParams.operation) {
  case rock::KernelType::Gemm:
    return gemmK;
  case rock::KernelType::Attention:
    // Conservative additive model: errors from the QK^T reduction (over
    // head_dim_qk) and the softmax-weighted-V reduction (over seq_len_k)
    // are roughly independent and accumulate.
    return headDimQK + sequenceLengthK;
  case rock::KernelType::Conv:
  case rock::KernelType::ConvBwdData:
  case rock::KernelType::ConvBwdWeight:
    return convK();
  case rock::KernelType::GemmElementwiseGemm:
    // Fused (A.B).C: two cascaded reductions of length K and N. Additive
    // model, same as attention.
    return gemmK + gemmN;
  case rock::KernelType::ConvElementwiseGemm:
    // Fused (Conv(A,B)).C: first reduction is the conv im2col K, second
    // is gemmN (the output channels of the conv become the K of the
    // second GEMM).
    return convK() + gemmN;
  default:
    return 1;
  }
}

static func::FuncOp createVerifierFunc(const GenParams &genParams,
                                       ModuleOp module, const KernelIF &kernel,
                                       MemRefType testType, MemRefType valType,
                                       std::string funcName) {
  func::FuncOp func = module.lookupSymbol<func::FuncOp>(funcName);
  if (func) // already exists
    return func;

  OpBuilder b(module.getContext());
  auto loc = b.getUnknownLoc();
  auto floatType = b.getF32Type();
  auto charType = b.getIntegerType(8);

  // Emit verify_results function call
  func = func::FuncOp::create(loc, funcName,
                              b.getFunctionType({testType, valType}, {}));
  module.push_back(func);

  // Emit verification logic.
  // Create a new block
  Block *block = func.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  // obtain function arguments
  // arg0: test result
  // arg1: validation result
  auto test = block->getArgument(0);
  auto val = block->getArgument(1);
  // obtain element type
  auto testElemType = testType.getElementType();
  auto valElemType = valType.getElementType();

  // Flatten the arguments to 1D for passing to the verification function
  // %test_flat = memref.collapse_shape %arg0 ...
  // %val_flat = memref.collapse_shape %arg1 ...
  Value testFlat = makeNDMemRef(b, test, 1);
  Value valFlat = makeNDMemRef(b, val, 1);
  auto valFlatType = cast<MemRefType>(valFlat.getType());
  // Emit constants for thresholds

  // clang-format off
  // %cst = arith.constant 9.99999974E-6 : f32
  // %cst_0 = arith.constant 1.000000e+02 : f32
  // %cst_1 = arith.constant 1.000000e+02 : f32
  // clang-format on

  auto getF32Val = [&](float val) -> Value {
    llvm::APFloat apVal(val);
    return arith::ConstantFloatOp::create(b, loc, floatType, apVal);
  };
  // Thresholds for different metrics
  // RMS: 0.00003f for all data types
  // absDiff: 100.0f for all data types, i.e. the maxAbsDiff metric is disabled
  // relDiff 100.0f for f16, i.e. maxRelDiff metric is disabled for f16
  // datatypes
  //         0.000001f for other data types
  char printDebug = static_cast<char>(printVerifyResults.getValue());

  auto printDebugVal =
      arith::ConstantIntOp::create(b, loc, charType, printDebug);

  // Comparator selection. The default stays `legacy` to preserve behavior of
  // the ~100 existing test files until the follow-up migration PR. Setting
  // -atol or -rtol on the command line implies allclose.
  bool useAllclose = (comparatorMode == ComparatorMode::Allclose) ||
                     atolThreshold.getNumOccurrences() ||
                     rtolThreshold.getNumOccurrences();

  // obtain function name of the verifier wrapper
  std::string verifyFuncName = "mcpuVerify";
  if (isa<FloatType>(valElemType)) {
    // f16, bf16, fp8, bf8 will be converted to f32 by wrapper.
    verifyFuncName += "Float";
    if (useAllclose)
      verifyFuncName += "Allclose";
  } else if (valElemType.isInteger(8) || valElemType.isInteger(32) ||
             valElemType.isInteger(64)) {
    verifyFuncName +=
        "Int" + std::to_string(testElemType.getIntOrFloatBitWidth()) + "Int" +
        std::to_string(valElemType.getIntOrFloatBitWidth());
  } else {
    llvm_unreachable("There's a valElemType not accounted for");
  }

  auto mr1DUnkTestType =
      MemRefType::get({mlir::ShapedType::kDynamic}, testElemType);
  auto mr1DUnkValType =
      MemRefType::get({mlir::ShapedType::kDynamic}, valElemType);
  auto mr1DUnkF32Type =
      MemRefType::get({mlir::ShapedType::kDynamic}, floatType);

  bool isTestAndValSameType =
      (testElemType.isIntOrIndex() || testElemType.isF32());

  Value testResult, valResult;       // Values passed to the verify function
  Value testResultNew, valResultNew; // Values used for type conversion
  if (!isTestAndValSameType) {
    // When gpu kernel output data type = f16 | bf16, type conversions
    // are required before calling the verify function

    // Cast test result to the same type as valid result

    // clang-format off
    // %0 = memref.alloc() : memref<802816xf32>
    // call @_memcpy_f16_f32_802816(%test_flat, %0) : (memref<802816xf16>, memref<802816xf32>) -> ()
    // %5 = memref.cast %0 : memref<802816xf32> to memref<?x?x?x?x?xf32>
    // clang-format on

    auto f32FlatType = MemRefType::get(valFlatType.getShape(), floatType);
    testResultNew = memref::AllocOp::create(b, loc, f32FlatType);
    emitMemcpy(b, testFlat, testResultNew);
    testResult = memref::CastOp::create(b, loc, mr1DUnkF32Type, testResultNew);
    mr1DUnkTestType = mr1DUnkF32Type;
    if (!valElemType.isF32()) {
      valResultNew = memref::AllocOp::create(b, loc, f32FlatType);
      emitMemcpy(b, valFlat, valResultNew);
      valResult = memref::CastOp::create(b, loc, mr1DUnkF32Type, valResultNew);
      mr1DUnkValType = mr1DUnkF32Type;
    } else {
      valResult = memref::CastOp::create(b, loc, mr1DUnkValType, valFlat);
    }

    // Cast valid result down to the same type as test result and cast back
    //   For f16 and bf16 datatypes, gpu hardware outputs f32 results, which are
    //   truncated to f16/bf16 before returning from the gpu kernel
    //   To make the comparison fair, the truncation step is added manually to
    //   the validation results.

    // clang-format off
    // affine.for %arg2 = 0 to 802816 {
    //   %7 = memref.load %arg1[%arg2] : memref<802816xf32>
    //   %8 = arith.truncf %7 : f32 to f16
    //   %9 = arith.extf %8 : f16 to f32
    //   memref.store %9, %arg1[%arg2] : memref<802816xf32>
    // }
    // clang-format on

    SmallVector<int64_t, 1> lowerBounds(1, 0);
    SmallVector<int64_t, 1> upperBounds = {valFlatType.getNumElements()};
    SmallVector<int64_t, 1> steps(1, 1);

    if (testElemType != valElemType) {
      affine::buildAffineLoopNest(
          b, loc, lowerBounds, upperBounds, steps,
          [valFlat, testElemType, valElemType](OpBuilder &b, Location loc,
                                               ValueRange ivs) {
            Value valOrig = affine::AffineLoadOp::create(b, loc, valFlat, ivs);
            Value valTruncated =
                arith::TruncFOp::create(b, loc, testElemType, valOrig);
            Value valExt =
                arith::ExtFOp::create(b, loc, valElemType, valTruncated);
            affine::AffineStoreOp::create(b, loc, valExt, valFlat, ivs);
          });
    }
  } else {
    testResult = memref::CastOp::create(b, loc, mr1DUnkTestType, testFlat);
    valResult = memref::CastOp::create(b, loc, mr1DUnkValType, valFlat);
  }

  // Prepare the validation result for the verify function
  // Declare and call the wrapper verify function
  func::FuncOp verifyFuncDecl;

  if (isa<FloatType>(testElemType)) {
    Type boolType = b.getIntegerType(1);
    bool isFP32 = isa<Float32Type>(testElemType);
    auto isFP32Val = arith::ConstantIntOp::create(b, loc, boolType, isFP32);

    if (useAllclose) {
      // Per-dtype (atol, rtol) baselines for K=1 (element-wise) kernels.
      //
      // fp16/bf16/fp32/fp64 mirror PyTorch's _DTYPE_PRECISIONS:
      //   https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py
      //
      // fp8/fp4 are not in PyTorch's table, so we use the per-dtype eps from
      // the IEEE/OCP/AMD format definitions as rtol, and JAX's
      // jax._src.public_test_util._default_tolerance value as atol:
      //   - fp8 e4m3 eps = 2^-3 = 0.125; matches hipBLASLt's "F8 tolerance =
      //     0.125" (ROCm/hipBLASLt PR #674) and NVIDIA TransformerEngine's
      //     PR #501 table.
      //   - fp8 e5m2 eps = 2^-2 = 0.25;  matches hipBLASLt's "B8 tolerance =
      //     0.25" (same PR #674).
      //   - fp4 e2m1 eps = 2^-1 = 0.5;   single mantissa bit, no upstream
      //     citation, but consistent with the per-dtype-eps rule.
      //   - atol = 1e-1 for fp8, 1e0 for fp4: from JAX's default_tolerance
      //     table at jax-ml/jax:jax/_src/public_test_util.py. JAX uses one
      //     scalar per dtype, applied as both atol and rtol; we use it as
      //     atol only since rtol is already covered by the eps row.
      auto allcloseBaseline = [&](Type t) -> std::pair<float, float> {
        if (isa<Float16Type>(t))
          return {1e-5f, 1e-3f};
        if (isa<BFloat16Type>(t))
          return {1e-5f, 1.6e-2f};
        if (isa<Float32Type>(t))
          return {1e-5f, 1.3e-6f};
        if (isa<Float64Type>(t))
          return {1e-7f, 1e-7f};
        // e4m3 variants: 3-bit mantissa, eps = 2^-3 = 0.125.
        if (isa<Float8E4M3FNType, Float8E4M3FNUZType>(t))
          return {1e-1f, 0.125f};
        // e5m2 variants: 2-bit mantissa, eps = 2^-2 = 0.25.
        if (isa<Float8E5M2Type, Float8E5M2FNUZType>(t))
          return {1e-1f, 0.25f};
        // fp4 e2m1: 1-bit mantissa, eps = 2^-1 = 0.5.
        if (isa<Float4E2M1FNType>(t))
          return {1e0f, 0.5f};
        return {1e-2f, 1e-2f};
      };
      // If a matmul-like op in the module uses a narrower float dtype than
      // the kernel's output (e.g. f16 GEMM up-cast to f32 via arith.extf or
      // migraphx.convert right before return), the values being compared
      // cannot be more accurate than that narrower dtype. Take the
      // narrower of {output dtype, narrowest matmul dtype} as the baseline
      // -- this is what hipBLASLt's `unit_check` vs `norm_check` selection
      // is implicitly tracking via the "F8/B8 output" rule.
      Type baselineType = testElemType;
      if (auto narrowest = scanModuleForNarrowestFloat(module)) {
        auto outFt = dyn_cast<FloatType>(testElemType);
        auto narrowFt = cast<FloatType>(*narrowest);
        if (outFt && narrowFt.getWidth() < outFt.getWidth())
          baselineType = *narrowest;
      }
      auto [baseAtol, baseRtol] = allcloseBaseline(baselineType);

      // Reduction-aware atol bound: atol_eff = baseAtol + K * sumErrTol.
      // Matches rocBLAS's `tol = K * sum_error_tolerance<T>` in
      //   clients/include/blas3/testing_gemm.hpp
      // K_eff per op is computed by computeReductionK (see above). For
      // clone-harness / pre-lowered IR flows where -operation is unset, K
      // is scanned from rock.gemm / rock.conv* / rock.attention in the
      // module.
      int64_t kEff = computeReductionK(genParams, module);
      float defaultAtol =
          baseAtol + static_cast<float>(kEff) * sumErrorTolerance(baselineType);

      float atolValue = atolThreshold.getNumOccurrences()
                            ? atolThreshold.getValue()
                            : defaultAtol;
      float rtolValue = rtolThreshold.getNumOccurrences()
                            ? rtolThreshold.getValue()
                            : baseRtol;
      Value atolVal = getF32Val(atolValue);
      Value rtolVal = getF32Val(rtolValue);

      verifyFuncDecl = makeFuncDecl(module, verifyFuncName,
                                    {mr1DUnkTestType, mr1DUnkValType, floatType,
                                     floatType, charType, boolType});
      func::CallOp::create(b, loc, verifyFuncDecl,
                           ValueRange{testResult, valResult, atolVal, rtolVal,
                                      printDebugVal, isFP32Val});
    } else {
      constexpr float defaultRMSThreshold(0.00003f);
      constexpr float defaultRMSThresholdFP16(0.001f);
      float RMSThresholdValue = isa<Float16Type, BFloat16Type>(testElemType)
                                    ? defaultRMSThresholdFP16
                                    : defaultRMSThreshold;
      if (RMSThreshold)
        RMSThresholdValue = RMSThreshold.getValue();
      Value thr_RMS = getF32Val(RMSThresholdValue);
      Value thr_absDiff = getF32Val(absDiffThreshold.getValue());
      Value thr_relDiff = getF32Val(relDiffThreshold.getValue());
      if (isa<Float16Type, BFloat16Type>(testElemType))
        thr_relDiff = getF32Val(100.0f);

      verifyFuncDecl = makeFuncDecl(module, verifyFuncName,
                                    {mr1DUnkTestType, mr1DUnkValType, floatType,
                                     floatType, floatType, charType, boolType});
      func::CallOp::create(b, loc, verifyFuncDecl,
                           ValueRange{testResult, valResult, thr_RMS,
                                      thr_absDiff, thr_relDiff, printDebugVal,
                                      isFP32Val});
    }
  } else {
    verifyFuncDecl = makeFuncDecl(module, verifyFuncName,
                                  {mr1DUnkTestType, mr1DUnkValType, charType});
    func::CallOp::create(b, loc, verifyFuncDecl,
                         ValueRange{testResult, valResult, printDebugVal});
  }

  if (!isTestAndValSameType) {
    // Deallocate the buffer for f32 version of the test results
    memref::DeallocOp::create(b, loc, testResultNew);
    if (!valElemType.isF32())
      memref::DeallocOp::create(b, loc, valResultNew);
  }

  func::ReturnOp::create(b, loc, ValueRange{});

  return func;
}

/// Helper to call a tensor-based function with memref arguments.
/// Converts memrefs to tensors, calls the function, and copies results back.
static void callTensorFuncWithMemrefs(OpBuilder &b, Location loc,
                                      func::FuncOp callee,
                                      SmallVectorImpl<Value> &memrefArgs,
                                      ArrayRef<int32_t> outputIndices) {
  SmallVector<Value, 8> tensorArgs;
  for (auto [idx, memrefArg] : llvm::enumerate(memrefArgs)) {
    bool isWritable = llvm::is_contained(outputIndices, idx);
    tensorArgs.push_back(rock::getAsTensor(b, loc, memrefArg, isWritable));
  }
  auto callOp = func::CallOp::create(b, loc, callee, tensorArgs);
  for (auto [resultIdx, result] : llvm::enumerate(callOp.getResults())) {
    if (resultIdx < outputIndices.size()) {
      int32_t outIdx = outputIndices[resultIdx];
      auto outMemrefType = cast<MemRefType>(memrefArgs[outIdx].getType());
      Value resultMemref =
          bufferization::ToBufferOp::create(b, loc, outMemrefType, result);
      memref::CopyOp::create(b, loc, resultMemref, memrefArgs[outIdx]);
    }
  }
}

/// Call a cpu_host function with the appropriate subset of valVars.
///
/// The cpu_host function was created from the original function before the
/// kernel pipeline ran. rock-insert-output-stores appends output argument(s)
/// to the end of the kernel's signature, so the kernel will have more args
/// than cpu_host. The first cpuHostFunc.getNumArguments() args of the kernel
/// always match cpu_host's args in order.
static void callCpuHostWithMemrefs(OpBuilder &b, Location loc,
                                   func::FuncOp cpuHostFunc,
                                   SmallVectorImpl<Value> &valVars,
                                   ArrayRef<int32_t> outIndices) {
  size_t numCpuHostArgs = cpuHostFunc.getNumArguments();
  SmallVector<Value, 8> tensorArgs;
  for (size_t i = 0; i < numCpuHostArgs; ++i) {
    tensorArgs.push_back(rock::getAsTensor(b, loc, valVars[i], false));
  }

  auto callOp = func::CallOp::create(b, loc, cpuHostFunc, tensorArgs);

  for (auto [resultIdx, result] : llvm::enumerate(callOp.getResults())) {
    if (resultIdx < outIndices.size()) {
      int32_t outIdx = outIndices[resultIdx];
      auto outMemrefType = cast<MemRefType>(valVars[outIdx].getType());
      Value resultMemref =
          bufferization::ToBufferOp::create(b, loc, outMemrefType, result);
      memref::CopyOp::create(b, loc, resultMemref, valVars[outIdx]);
    }
  }
}

static void insertValidationCalls(const GenParams &genParams, OpBuilder &b,
                                  ModuleOp module,
                                  SmallVectorImpl<Value> &valVars,
                                  SmallVectorImpl<Value> &localVars,
                                  ArrayRef<int32_t> outIndices, Operation *func,
                                  KernelIF &root0) {
  auto validationType = genValidation.getValue();
  auto loc = b.getUnknownLoc();

  if (validationType !=
      "clone") { // --verifier=cpp or --verifier=mlir (-pv / -pv_with_mlir map
                 // to mlir)    // Emit call to host_<conv>
    if (genParams.operation == rock::KernelType::ConvElementwiseGemm) {
      if (validationType == "cpp") {
        llvm::errs()
            << "External conv elementwise gemm validator is not available\n";
        exit(1);
      }
      if (groupSize != 1) {
        llvm::errs()
            << "Group convolution not supported for conv+gemm in rocmlir-gen\n";
        exit(1);
      }
      auto cpuConvElementwiseGemmFunc =
          createCpuConvElementwiseGemmKernelWithMlir(module, genParams);
      func::CallOp::create(b, loc, cpuConvElementwiseGemmFunc, valVars);
    } else if (genParams.convConfig.has_value()) {
      const auto &genConfig = **genParams.convConfig;
      auto cpuConvFunc = createCPUConvWithMLIR(module, genConfig);
      callTensorFuncWithMemrefs(b, loc, cpuConvFunc, valVars, outIndices);
    } else if (genParams.operation == rock::KernelType::Gemm) {
      // Emit call to host gemm
      if (validationType == "cpp") {
        llvm::errs() << "External gemm validator is not available\n";
        exit(1);
      }
      // Start CPU timer
      if (cpuTimers) {
        auto cpuTimerStartFunc = makeFuncDecl(module, "cpuTimerStart", {});
        func::CallOp::create(b, loc, cpuTimerStartFunc, ValueRange{});
      }

      auto cpuGemmFunc = createCpuGemmKernelWithMlir(module, genParams);
      callTensorFuncWithMemrefs(b, loc, cpuGemmFunc, valVars, outIndices);

      // Stop CPU timer and print elapsed time
      if (cpuTimers) {
        auto cpuTimerStopFunc = makeFuncDecl(module, "cpuTimerStop", {});
        func::CallOp::create(b, loc, cpuTimerStopFunc, ValueRange{});
      }
    } else if (genParams.operation == rock::KernelType::Attention) {
      if (validationType == "cpp") {
        llvm::errs() << "External attention validator is not available\n";
        exit(1);
      }
      auto cpuAttentionFunc =
          createCpuAttentionKernelWithMlir(module, genParams);
      func::CallOp::create(b, loc, cpuAttentionFunc, valVars);
    } else if (genParams.operation == rock::KernelType::GemmElementwiseGemm) {
      if (validationType == "cpp") {
        llvm::errs()
            << "External gemm elementwise gemm validator is not available\n";
        exit(1);
      }
      auto cpuGemmElementwiseGemmFunc =
          createCpuGemmElementwiseGemmKernelWithMlir(module, genParams);
      func::CallOp::create(b, loc, cpuGemmElementwiseGemmFunc, valVars);
    } else {
      llvm::errs()
          << "Validation generation requested, but no operation specified\n";
      exit(1);
    }
  } else {
    // The _cpu_host function was created by --clone-harness and lowered by the
    // host pipeline. It provides the CPU reference implementation for
    // validation. Look it up by naming convention: <kernel_name>_cpu_host.
    std::string cpuHostName = root0.func.getName().str() + "_cpu_host";
    auto cpuHostFunc = module.lookupSymbol<func::FuncOp>(cpuHostName);
    if (!cpuHostFunc) {
      llvm::errs() << "Clone validation requires a " << cpuHostName
                   << " function in the module.\n";
      exit(1);
    }
    callCpuHostWithMemrefs(b, loc, cpuHostFunc, valVars, outIndices);
  }

  // Emit call to verifier
  for (int32_t outIdx : outIndices) {
    Value testResult = localVars[outIdx];
    Value valResult = valVars[outIdx];
    auto testType = dyn_cast<MemRefType>(testResult.getType());
    auto valType = dyn_cast<MemRefType>(valResult.getType());
    std::string funcName =
        root0.func.getName().str() + "_verify" + std::to_string(outIdx);
    auto verifierFunc = createVerifierFunc(genParams, module, root0, testType,
                                           valType, funcName);

    func::CallOp::create(b, loc, verifierFunc,
                         ValueRange{testResult, valResult});
  }
}

// Check if a given argument index needs prefill initialization.
// Returns the prefill value if initialization is needed, or std::nullopt
// otherwise. This covers:
//   - Output args when splitK is used (prefilled with 0.0)
//   - Any kernel arg with a rock.prefill attribute (uses the attribute's
//     stored init value, e.g., backward weight atomic convolutions)
static FailureOr<std::optional<float>>
getPrefillValue(size_t argIdx, ArrayRef<int32_t> outIndices, bool isSplitK,
                const SmallVector<KernelIF, 8> &kernels) {
  // Check for an explicit rock.prefill attribute first as it carries the
  // correct identity element for the store method (0 for atomic_add,
  // -inf for atomic_max, etc.) and must take precedence over the generic
  // splitK zero-init default.
  for (const auto &kernel : kernels) {
    func::FuncOp func = kernel.func;
    if (argIdx >= func.getNumArguments())
      continue;

    auto initAttr = func.getArgAttr(argIdx, rock::PrefillAttr::getMnemonic());
    if (!initAttr)
      continue;

    // Verify the prefill attribute type matches the kernel argument's element
    // type (float attr for float args, integer attr for integer args).
    Type argElemType = getElementTypeOrSelf(func.getArgument(argIdx).getType());
    if (isa<FloatAttr>(initAttr) && !isa<FloatType>(argElemType)) {
      llvm::errs() << "error: rock.prefill has float attribute but arg "
                   << argIdx << " has non-float element type " << argElemType
                   << "\n";
      return failure();
    }
    if (isa<IntegerAttr>(initAttr) && !isa<IntegerType>(argElemType)) {
      llvm::errs() << "error: rock.prefill has integer attribute but arg "
                   << argIdx << " has non-integer element type " << argElemType
                   << "\n";
      return failure();
    }

    float prefillVal;
    if (auto floatAttr = dyn_cast<FloatAttr>(initAttr))
      prefillVal = static_cast<float>(floatAttr.getValueAsDouble());
    else if (auto intAttr = dyn_cast<IntegerAttr>(initAttr))
      prefillVal = static_cast<float>(intAttr.getInt());
    else {
      llvm::errs() << "error: unsupported rock.prefill attribute type on arg "
                   << argIdx << "\n";
      return failure();
    }

    // Warn if a non-zero prefill (e.g. -inf for atomic_max) is combined with
    // splitK, since splitK normally expects zero-init for atomic_add
    // accumulation.
    if (isSplitK && prefillVal != 0.0f) {
      llvm::errs() << "warning: rock.prefill value " << prefillVal << " on arg "
                   << argIdx << " is non-zero but isSplitK is true; "
                   << "this may indicate incompatible store methods\n";
    }

    return std::optional<float>(prefillVal);
  }

  // Fall back: splitK outputs without an explicit prefill need zero init.
  if (llvm::is_contained(outIndices, argIdx) && isSplitK)
    return std::optional<float>(0.0f);

  return std::optional<float>{};
}

static LogicalResult populateHostHarnessLogic(
    ModuleOp module, const SmallVector<KernelIF, 8> &kernels,
    const SmallVector<KernelIF, 8> &roots, const GenParams &genParams) {
  MLIRContext *context = module.getContext();
  OpBuilder b(context);
  Location loc = b.getUnknownLoc();

  // Construct main function.
  auto func = func::FuncOp::create(loc, "main", b.getFunctionType({}, {}));
  module.push_back(func);
  Block *block = func.addEntryBlock();
  b.setInsertionPoint(block, block->begin());

  // Timer to measure JIT compilation time (time from library load to main
  // start)
  if (cpuTimers) {
    auto programStartFunc = makeFuncDecl(module, "programStart", {});
    func::CallOp::create(b, loc, programStartFunc, ValueRange{});
  }

  auto floatType = b.getF32Type();
  auto validationType = genValidation.getValue();

  // Create all local variables for each kernel param
  // - assumes all kernels read the same memrefs
  if (roots.size() > 1) {
    // TODO: verify that all parameter lists match
  }
  auto root0 = *roots.begin();
  bool isCPUKernel = !root0.func->hasAttr(rock::KernelAttr::getMnemonic());
  bool hasValidation = !validationType.empty() && !genCPUKernel.getValue();
  bool hasCloneValidation = hasValidation && (validationType == "clone");
  // `--verifier clone` builds a host harness that allocates one buffer per
  // kernel argument and feeds the kernel's tensor results back through the
  // trailing args. That contract is only well-defined if the kernel is
  // at the rock IR level, so make sure we at least have one rock op in the IR.
  if (hasCloneValidation) {
    for (KernelIF kernel : kernels) {
      bool hasRockOp = false;
      kernel.func.walk([&](Operation *op) {
        if (isa_and_nonnull<rock::RockDialect>(op->getDialect())) {
          hasRockOp = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!hasRockOp) {
        kernel.func.emitError()
            << "--verifier=clone cannot build a host harness around a "
               "kernel that is not at the rock level; run the "
               "kernel pipeline first (e.g. `rocmlir-driver "
               "-kernel-pipeline=highlevel`)";
        return failure();
      }
    }
  }
  bool isRandom = (randomSeed != "fixed" && randomSeed != "none");
  bool isSplitK = (genParams.perfConfig.empty())
                      ? false
                      : rock::isSplitKRequested(
                            StringAttr::get(context, genParams.perfConfig));

  if (isRandom) {
    auto seedFunc = makeFuncDecl(module, "seedRandomValues", {b.getI32Type()});
    int seed = getRandomSeed();
    Value seedConst =
        arith::ConstantIntOp::create(b, loc, b.getI32Type(), seed);
    func::CallOp::create(b, loc, seedFunc, seedConst);
  }

  bool isAttention = false;
  SmallVector<int32_t, 2> outIndices;
  SmallVector<int32_t, 2> allOutIndices;
  if (genParams.operation.has_value()) {
    switch (genParams.operation.value()) {
    case rock::KernelType::Conv:
    case rock::KernelType::ConvBwdData:
    case rock::KernelType::ConvBwdWeight:
      outIndices.push_back(2);
      break;
    case rock::KernelType::Gemm:
      outIndices.push_back(scaledGemm ? 4 : 2);
      break;
    case rock::KernelType::GemmElementwiseGemm:
      outIndices.push_back(3);
      break;
    case rock::KernelType::ConvElementwiseGemm:
      outIndices.push_back(3);
      break;
    case rock::KernelType::Attention:
      isAttention = true;
      int32_t optionalArgsCounter{3};
      bool isQuantized = genParams.types[0] == b.getI8Type();
      if (isQuantized)
        optionalArgsCounter += 2;
      if (hasAttnScale)
        ++optionalArgsCounter;
      if (hasAttnBias)
        ++optionalArgsCounter;
      if (!currentSeqLen.empty())
        ++optionalArgsCounter;
      if (!prefixOffset.empty())
        ++optionalArgsCounter;
      if (returnLSE) {
        int32_t lseArgIdx = optionalArgsCounter;
        ++optionalArgsCounter;
        outIndices.push_back(optionalArgsCounter);
        allOutIndices.push_back(optionalArgsCounter);
        allOutIndices.push_back(lseArgIdx);
        // Only verify LSE when splitKV == 1; with splitKV > 1, the LSE
        // is an intermediate used to compute the final result.
        if (splitKV == 1)
          outIndices.push_back(lseArgIdx);
      } else {
        outIndices.push_back(optionalArgsCounter);
      }
    }
  } else {
    outIndices = root0.outIndices;
  }

  SmallVector<Value, 5> localVars;
  SmallVector<Value, 5> valVars;
  // Calculate expected indices for currentSeqLen and prefixOffset tensors.
  // The layout is: ..., currentSeqLen?, prefixOffset?, LSE?, Output
  // We need to count backwards from the end.
  int64_t offsetFromEnd = 1; // Output is always last
  if (returnLSE)
    ++offsetFromEnd;
  const int64_t expectedPrefixOffsetIdx =
      !prefixOffset.empty() ? (root0.params.size() - offsetFromEnd - 1) : -1;
  if (!prefixOffset.empty())
    ++offsetFromEnd;
  const int64_t expectedCurrSeqLenIdx =
      !currentSeqLen.empty() ? (root0.params.size() - offsetFromEnd - 1) : -1;

  // Timer for memory initialization
  func::FuncOp initTimerStopFunc;
  if (cpuTimers) {
    auto initTimerStartFunc = makeFuncDecl(module, "initTimerStart", {});
    initTimerStopFunc = makeFuncDecl(module, "initTimerStop", {});
    func::CallOp::create(b, loc, initTimerStartFunc, ValueRange{});
  }

  for (auto [idx, paramType] : llvm::enumerate(root0.params)) {
    auto paramShapedType = dyn_cast<ShapedType>(paramType);
    assert(paramShapedType &&
           "currently only supports shaped types (memref or tensor)");
    Type elemType = paramShapedType.getElementType();
    auto paramMRType = MemRefType::get(paramShapedType.getShape(), elemType);
    bool isSmallFloat =
        isa<FloatType>(elemType) && elemType.getIntOrFloatBitWidth() < 32;
    if (isCPUKernel) { // -prc
      if (genParams.operation.has_value()) {
        if (idx < genParams.types.size())
          elemType = genParams.types[idx];
        // The CPU verifier for i8 GEMM/conv accumulates in i32 to match the
        // GPU's MFMA semantics; allocate the output buffer as i32 as well.
        if (isa<IntegerType>(elemType) && llvm::is_contained(outIndices, idx))
          elemType = b.getIntegerType(32);
        paramMRType = MemRefType::get(paramShapedType.getShape(), elemType);
      }
    }
    auto lvar = memref::AllocOp::create(b, loc, paramMRType);
    localVars.push_back(lvar);

    // Helper to fill a memref with i32 values from a list
    auto fillWithI32Values = [&](auto &values) {
      for (auto pair : llvm::enumerate(values)) {
        Value index = arith::ConstantIndexOp::create(b, loc, pair.index());
        Value value =
            arith::ConstantIntOp::create(b, loc, b.getI32Type(), pair.value());
        memref::StoreOp::create(b, loc, value, lvar, ValueRange{index});
      }
    };

    if (!currentSeqLen.empty() && isAttention &&
        static_cast<int64_t>(idx) == expectedCurrSeqLenIdx) {
      fillWithI32Values(currentSeqLen);
    } else if (!prefixOffset.empty() && isAttention &&
               static_cast<int64_t>(idx) == expectedPrefixOffsetIdx) {
      fillWithI32Values(prefixOffset);
    } else if (!isRandom) {
      auto prefillResult = getPrefillValue(idx, outIndices, isSplitK, kernels);
      if (failed(prefillResult))
        return failure();
      auto prefill = *prefillResult;
      SmallVector<float> tensorPattern = getTensorInitPattern(elemType, idx);
      auto initPattern = prefill ? SmallVector<float>{*prefill} : tensorPattern;

      if (failed(populateTensorFillLogic(b, loc, initPattern, elemType, lvar)))
        return failure();
    } else {
      auto prefillResult = getPrefillValue(idx, outIndices, isSplitK, kernels);
      if (failed(prefillResult))
        return failure();
      if (failed(populateRandomTensorFillLogic(b, loc, module, elemType, lvar,
                                               idx, *prefillResult)))
        return failure();
    }

    if (hasValidation || (isCPUKernel && isSmallFloat)) {
      // Emit validation var
      Type valElemType = floatType;
      if (genParams.operation.has_value() && isa<IntegerType>(elemType)) {
        valElemType = elemType;
        if (llvm::is_contained(outIndices, idx))
          valElemType = b.getIntegerType(32);
      } else if ((genValidation == "clone") || elemType.isInteger(8) ||
                 elemType.isInteger(32)) {
        valElemType = elemType;
      } else if (isSmallFloat && genParams.operation.has_value()) {
        valElemType = elemType;
      }

      auto valType = MemRefType::get(paramMRType.getShape(), valElemType);
      auto vvar = memref::AllocOp::create(b, loc, valType);
      valVars.push_back(vvar);

      emitMemcpy(b, lvar, vvar);
    }
  }

  // Stop memory initialization timer
  if (cpuTimers) {
    func::CallOp::create(b, loc, initTimerStopFunc, ValueRange{});
  }

  // capture result index
  if (outIndices.empty()) {
    size_t numResults = std::max<size_t>(root0.resultTypes.size(), 1);
    assert(localVars.size() >= numResults &&
           "fewer localVars than kernel results");
    for (size_t i = localVars.size() - numResults; i < localVars.size(); ++i) {
      outIndices.push_back(i);
    }
  }
  if (allOutIndices.empty())
    allOutIndices = outIndices;

  // Helper to call a function with appropriate type conversions
  // Handles both tensor-based (new) and memref-based (legacy) kernel interfaces
  // If willBeWrapped is true, the call will be redirected to a GPU wrapper that
  // expects memref arguments and handles tensor conversion internally.
  auto callFuncWithConversion = [&](func::FuncOp callee,
                                    SmallVectorImpl<Value> &memrefArgs,
                                    ArrayRef<int32_t> outputIndices,
                                    bool willBeWrapped = false) {
    // Check if the function expects tensor arguments by looking at first arg
    bool expectsTensors = !willBeWrapped &&
                          !callee.getArgumentTypes().empty() &&
                          isa<TensorType>(callee.getArgumentTypes().front());

    if (expectsTensors) {
      // Convert memrefs to tensors for the call
      SmallVector<Value, 8> tensorArgs;
      for (auto [idx, memrefArg] : llvm::enumerate(memrefArgs)) {
        bool isWritable = llvm::is_contained(outputIndices, idx);
        tensorArgs.push_back(rock::getAsTensor(b, loc, memrefArg, isWritable));
      }

      // Call the function with tensor arguments
      auto callOp = func::CallOp::create(b, loc, callee, tensorArgs);

      // If the function returns results, use them directly instead of copying
      for (auto [resultIdx, result] : llvm::enumerate(callOp.getResults())) {
        if (resultIdx < outputIndices.size()) {
          int32_t outIdx = outputIndices[resultIdx];
          // Convert result tensor to memref
          auto outMemrefType = cast<MemRefType>(memrefArgs[outIdx].getType());
          Value resultMemref =
              bufferization::ToBufferOp::create(b, loc, outMemrefType, result);
          memrefArgs[outIdx] = resultMemref;
        }
      }
    } else if (willBeWrapped) {
      // Call will be redirected to GPU wrapper which expects memrefs
      // Create call with explicit memref types (callee signature is
      // tensor-based)
      func::CallOp::create(b, loc, callee.getSymName(), TypeRange{},
                           memrefArgs);
    } else {
      // Legacy memref-based interface - call directly
      func::CallOp::create(b, loc, callee, memrefArgs);
    }
  };

  // Timer for GPU kernel execution
  func::FuncOp gpuTimerStartFunc, gpuTimerStopFunc;
  if (cpuTimers) {
    gpuTimerStartFunc = makeFuncDecl(module, "gpuTimerStart", {});
    gpuTimerStopFunc = makeFuncDecl(module, "gpuTimerStop", {});
  }

  // Call the roots.
  for (auto &root : roots) {
    // Is the root also a kernel?
    bool rootKernel =
        std::find_if(kernels.begin(), kernels.end(), [&](const KernelIF &k) {
          return k.func == root.func;
        }) != kernels.end();
    if (rootKernel) {
      // Start GPU timer before kernel execution
      if (cpuTimers) {
        func::CallOp::create(b, loc, gpuTimerStartFunc, ValueRange{});
      }

      // rootKernel calls will be redirected to GPU wrapper, which expects
      // memrefs
      callFuncWithConversion(root.func, localVars, outIndices,
                             /*willBeWrapped=*/true);

      // Stop GPU timer after kernel execution
      if (cpuTimers) {
        func::CallOp::create(b, loc, gpuTimerStopFunc, ValueRange{});
      }
    } else if (!valVars.empty()) {
      callFuncWithConversion(root.func, valVars, outIndices);
      if (!root.func->hasAttr(rock::KernelAttr::getMnemonic())) {
        printValidationResults = true;
        printResults = false;
      }
    } else {
      callFuncWithConversion(root.func, localVars, outIndices);
      if (!root.func->hasAttr(rock::KernelAttr::getMnemonic())) {
        printValidationResults = false;
        printResults = true;
      }
    }
    // Clone-style validation wants to validate each root function.
    // Non-clone validation validates at end;  the roots are related kernels.
    if (hasCloneValidation)
      insertValidationCalls(genParams, b, module, valVars, localVars,
                            outIndices, root.func, root0);
  }

  // Run validation
  if (hasValidation && !hasCloneValidation)
    insertValidationCalls(genParams, b, module, valVars, localVars, outIndices,
                          func, root0);
  // Print and cleanup validation vars
  for (auto &vvar : valVars) {
    // print vvar
    for (int32_t outIdx : outIndices) {
      if (printValidationResults.getValue() && vvar == valVars[outIdx]) {
        emitPrintTensor(b, vvar);
      }
    }
  }

  // Print and cleanup
  for (auto &lvar : localVars) {
    // print lvar
    for (int32_t outIdx : outIndices) {
      bool printp = printInputs.getValue();
      if (lvar == localVars[outIdx])
        printp = printResults.getValue();
      if (printp)
        emitPrintTensor(b, lvar);
    }
  }

  for (auto &vvar : valVars) {
    memref::DeallocOp::create(b, loc, vvar);
  }
  for (auto &lvar : localVars) {
    memref::DeallocOp::create(b, loc, lvar);
  }

  func::ReturnOp::create(b, loc, ValueRange{});

  // Set of kernels
  llvm::SmallSetVector<func::FuncOp, 4> kernelsSet;
  std::string kernelBaseName =
      (genParams.convConfig.has_value())
          ? genParams.convConfig.value()->kernelBaseName
          : root0.func.getName().str();
  for (auto &kernel : kernels) {
    if (kernel.func->hasAttr(rock::KernelAttr::getMnemonic())) {
      kernelsSet.insert(kernel.func);
    }
  }
  func::FuncOp gpuWrapperFunc;
  if (!kernelsSet.empty())
    gpuWrapperFunc = createGPUWrapper(module, kernelBaseName, kernels,
                                      genParams, allOutIndices);
  // Redirect calls to kernel functions to point at wrapped functions.
  func.walk([&](CallOpInterface callOp) -> WalkResult {
    // If the callee matches a wrapped function, update the call.
    Operation *callable = callOp.resolveCallable();
    if (callable) {
      func::FuncOp fop = dyn_cast<func::FuncOp>(*callable);
      if (kernelsSet.contains(fop)) {
        if (fop != root0.func) {
          callOp->erase();
          return WalkResult::advance();
        }
        callOp->setAttr("callee", FlatSymbolRefAttr::get(
                                      context, gpuWrapperFunc.getSymName()));
      }
    }
    return WalkResult::advance();
  });

  return success();
}

static OwningOpRef<ModuleOp> readTestFile(std::string inputFilenameStr,
                                          bool &hasUserKernel,
                                          MLIRContext *context) {
  std::string errorMessage;

  // Set up the input file.
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // Parse the input file.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, context);
  if (!module) {
    llvm::errs() << "Parse host harness " << inputFilename << " failed.\n";
    exit(1);
  }

  if (!perfConfig.empty()) {
    WalkResult findGemmOp =
        module->walk([&](rock::RockGemmWrapperInterface gemmOp) -> WalkResult {
          OpBuilder b(gemmOp.getContext());
          gemmOp->setAttr("perf_config", b.getStringAttr(perfConfig));
          return WalkResult::interrupt();
        });
    if (!findGemmOp.wasInterrupted()) {
      llvm::errs() << "Cannot find a Gemm kernel for perf_config\n";
      exit(1);
    }
  }

  module->walk([&](func::FuncOp func) -> WalkResult {
    if (func->hasAttr(rock::KernelAttr::getMnemonic())) {
      hasUserKernel = true;
    }
    return WalkResult::advance();
  });

  return module;
}

static void generateKernel(MLIRContext *context, GenParams &genParams,
                           ModuleOp module) {
  OpBuilder builder(context);
  static rock::ConvGenerator convGenerator; // genParams keeps pointer to config

  const bool isGemm = operation == rock::KernelType::Gemm;
  const bool isAttention = operation == rock::KernelType::Attention;
  const bool isGemmElntwiseGemm =
      operation == rock::KernelType::GemmElementwiseGemm;
  const bool isConvElntwiseGemm =
      operation == rock::KernelType::ConvElementwiseGemm;

  // ConvElementwiseGemm is treated as convolution
  const bool isConv = !(isGemm || isAttention || isGemmElntwiseGemm);

  if (failed(detectMissingArguments())) {
    exit(1);
  }

  RocmDeviceName targetInfo;
  if (failed(targetInfo.parse(arch.getValue()))) {
    llvm::errs() << "Invalid architecture name: " << arch << "\n";
    exit(1);
  }
  std::string triple = targetInfo.getTriple().str();
  std::string chip = targetInfo.getChip().str();
  std::string chipFeatures = targetInfo.getFeaturesForBackend();
  SmallString<64> canonicalArch;
  targetInfo.getFullName(canonicalArch);
  arch = canonicalArch.str().str();

  LogicalResult status = success();
  Type filterElemType = typeFromString(filterDataType.getValue(), context);
  Type inputElemType = typeFromString(inputDataType.getValue(), context);
  // for regular convolution it does filter * input = output
  // for bwd data convolution it does filter * output = input
  // for the bwd weight convolution it does output * input = filter
  // therefore need to remap data types accordingly before calculating
  // features
  if (operation == rock::KernelType::ConvBwdData) {
    // for the bwd data, input and output are flipped
    inputElemType = typeFromString(outputDataType.getValue(), context);
  } else if (operation == rock::KernelType::ConvBwdWeight) {
    filterElemType = typeFromString(outputDataType.getValue(), context);
  }
  genParams.operation = operation;
  genParams.arch = arch;
  genParams.perfConfig = perfConfig;
  if (isGemm) {
    for (const auto &arg :
         {filterDataType.getValue(), inputDataType.getValue(),
          outputDataType.getValue(), scaleADataType.getValue(),
          scaleBDataType.getValue()})
      genParams.types.push_back(typeFromString(arg, context));
    genParams.convConfig = std::nullopt;
    (void)createGpuGemmKernel(module, genParams);
  } else if (isGemmElntwiseGemm) {
    constexpr size_t numArgs{4};
    // Note: In the current implementation, all operands have the same type.
    // This behaviour enforced by `-t`. See, detectMissingArguments()
    auto elemType = typeFromString(inputDataType.getValue(), context);
    for (size_t argIdx{0}; argIdx < numArgs; ++argIdx) {
      genParams.types.push_back(elemType);
    }
    genParams.convConfig = std::nullopt;
    (void)createGpuGemmElementwiseGemmKernel(module, genParams);
  } else if (isAttention) {
    auto elemType = typeFromString(inputDataType.getValue(), context);
    // We only support first-gemm i8 version of attention
    // This will be changed when we support both gemms of i8.
    if (elemType == IntegerType::get(context, 8)) {
      constexpr size_t maxNumArgs{10};
      genParams.types.resize(maxNumArgs);
      genParams.types[AttentionQuantizedArgIndex::q] =
          IntegerType::get(context, 8);
      genParams.types[AttentionQuantizedArgIndex::k] =
          IntegerType::get(context, 8);
      genParams.types[AttentionQuantizedArgIndex::v] =
          Float16Type::get(context);
      genParams.types[AttentionQuantizedArgIndex::quantBias] =
          IntegerType::get(context, 8);
      genParams.types[AttentionQuantizedArgIndex::quantScale] =
          Float16Type::get(context);
      genParams.types[AttentionQuantizedArgIndex::scale] =
          Float16Type::get(context);
      genParams.types[AttentionQuantizedArgIndex::bias] =
          Float16Type::get(context);
      genParams.types[AttentionQuantizedArgIndex::currentSeqLen] =
          IntegerType::get(context, 32);
      genParams.types[AttentionQuantizedArgIndex::prefixOffset] =
          IntegerType::get(context, 32);
      genParams.types[AttentionQuantizedArgIndex::lse] =
          Float16Type::get(context);
    } else {
      constexpr size_t maxNumArgs{5};
      // Note: In the current implementation, all operands have the same type.
      // This behaviour enforced by `-t`. See, detectMissingArguments()
      for (size_t argIdx{0}; argIdx < maxNumArgs; ++argIdx) {
        genParams.types.push_back(elemType);
      }
      // extra operand: currentSeqLen (i32)
      genParams.types.push_back(IntegerType::get(context, 32));
      // extra operand: prefixOffset (i32)
      genParams.types.push_back(IntegerType::get(context, 32));
      // extra operand: LSE (log-sum-exp)
      genParams.types.push_back(elemType);
    }
    genParams.convConfig = std::nullopt;
    (void)createGpuAttentionKernel(module, genParams);
  } else {
    int nDims = filterLayout.getValue().size() - 3; // +++pf: magic number.
    SmallVector<int, 4> dilations;
    SmallVector<int, 4> strides;
    SmallVector<int, 4> paddingLeft;
    SmallVector<int, 4> paddingRight;

    // +++pf: needs generalising, coupled with command-line options.
    dilations.push_back(dilationHeight.getValue());
    strides.push_back(strideHeight.getValue());
    paddingLeft.push_back(paddingHeightLeft.getValue());
    paddingRight.push_back(paddingHeightRight.getValue());

    if (nDims > 1) {
      dilations.push_back(dilationWidth.getValue());
      strides.push_back(strideWidth.getValue());
      paddingLeft.push_back(paddingWidthLeft.getValue());
      paddingRight.push_back(paddingWidthRight.getValue());
    }
    if (nDims > 2) {
      dilations.push_back(dilationDepth.getValue());
      strides.push_back(strideDepth.getValue());
      paddingLeft.push_back(paddingDepthLeft.getValue());
      paddingRight.push_back(paddingDepthRight.getValue());
    }

    convGenerator = rock::ConvGenerator(
        arch, chip, disableSplitKForTuning, triple, chipFeatures,
        perfConfig.getValue(),
        num_cu.getNumOccurrences() ? std::optional<int>(num_cu.getValue())
                                   : std::nullopt,
        numChiplets.getNumOccurrences()
            ? std::optional<int>(numChiplets.getValue())
            : std::nullopt,
        rock::convOpTypeFromKernelType(operation.getValue()),
        filterDataType.getValue(), inputDataType.getValue(),
        outputDataType.getValue(), dilations, strides, paddingLeft,
        paddingRight, filterLayout.getValue(), inputLayout.getValue(),
        outputLayout.getValue());

    SmallVector<int64_t> inDims{inputHeight, inputWidth};
    if (nDims > 2) {
      if (inputDepth < 1)
        inputDepth = 1;
      inDims.push_back(inputDepth);
    }
    SmallVector<int64_t> outDims{outputHeight, outputWidth};
    if (nDims > 2) {
      if (outputDepth < 1)
        outputDepth = 1;
      outDims.push_back(outputDepth);
    }
    SmallVector<int64_t> filDims{filterHeight, filterWidth};
    if (nDims > 2) {
      if (filterDepth < 1)
        filterDepth = 1;
      filDims.push_back(filterDepth);
    }

    status =
        convGenerator.parseConvDims(batchSize, groupSize, inputChannel, inDims,
                                    outputChannel, outDims, filDims);
    if (failed(status)) {
      llvm::errs() << "Could not parse convolution dimensions\n";
      exit(1);
    }

    if (convKernelId.getNumOccurrences() > 0)
      convGenerator.setKernelId(convKernelId.getValue());

    if (!isConvElntwiseGemm) {
      genParams.types.push_back(convGenerator.getFilterDataType(builder));
      genParams.types.push_back(convGenerator.getInputDataType(builder));
      genParams.types.push_back(convGenerator.getOutputDataType(builder));
      // Reorder types to match the kernel arg ordering (store dest last).
      if (operation.getValue() == rock::KernelType::ConvBwdData)
        std::swap(genParams.types[1], genParams.types[2]);
      else if (operation.getValue() == rock::KernelType::ConvBwdWeight)
        std::rotate(genParams.types.begin(), genParams.types.begin() + 1,
                    genParams.types.end());
    }
    genParams.convConfig = &convGenerator.getConfig();
  }

  // TODO: Extract isApplicable check to be its own component
  if (isConv && failed(convGenerator.isApplicable())) {
    llvm::errs() << "Convolution configuration does not have valid dimension\n";
    exit(1);
  }

  if (genParams.convConfig.has_value()) {
    const auto &genConfig = **genParams.convConfig;

    // Set arch on module to make compilation pipeline work (same as GEMM)
    StringAttr archAttr = builder.getStringAttr(genConfig.arch);
    if (!module->hasAttr(rock::ArchAttr::getMnemonic()))
      module->setAttr(rock::ArchAttr::getMnemonic(), archAttr);

    if (isConvElntwiseGemm) {
      constexpr size_t numArgs{4};
      // Note: In the current implementation, all operands have the same type.
      // This behaviour enforced by `-t`. See, detectMissingArguments()
      auto elemType = typeFromString(inputDataType.getValue(), context);
      for (size_t argIdx{0}; argIdx < numArgs; ++argIdx) {
        genParams.types.push_back(elemType);
      }
      (void)createGpuConvElementwiseGemmKernel(module, genParams);
    } else if (genCPUKernel.getValue()) {
      (void)createCPUConvFunc(module, genConfig);
    } else {
      // Populate the module.
      int kernelStart = genConfig.kernelId;
      int kernelCount = 0;
      if (failed(convGenerator.getKernelCount(builder, kernelCount))) {
        llvm::errs() << "Getting kernel count failed.\n";
        exit(1);
      }
      if (kernelStart < 0) {
        kernelStart = 0;
      } else {
        kernelCount = kernelStart + 1;
      }
      // generate all sub-kernels, and get corresponding gemmId
      std::string kernelBaseName = genConfig.kernelBaseName;
      for (int i = kernelStart; i < kernelCount; ++i) {
        convGenerator.setKernelName(kernelBaseName + "_" + std::to_string(i));
        if (failed(convGenerator.genConvModule(module, i))) {
          llvm::errs() << "Module population failed.\n";
          exit(1);
        }
      }
      convGenerator.setKernelName(kernelBaseName);
    }
  }
}

static void populateCloneHarnessLogic(ModuleOp module) {
  if (arch.getValue().empty()) {
    llvm::errs() << "--arch is not set\n";
    exit(1);
  }
  func::FuncOp originalFunc = module.lookupSymbol<func::FuncOp>(testFuncName);
  assert(originalFunc && "does -fut point to the wrong function?");

  MLIRContext *context = module.getContext();
  OpBuilder b(context);

  originalFunc->removeAttr(rock::KernelAttr::getMnemonic());
  StringAttr archAttr = b.getStringAttr(arch);
  if (originalFunc->hasAttr(rock::ArchAttr::getMnemonic()))
    originalFunc->setAttr(rock::ArchAttr::getMnemonic(), archAttr);

  // Clone the function to create the GPU kernel version before renaming.
  auto *cloneFunc = originalFunc->clone();
  auto cloneFuncOp = dyn_cast<func::FuncOp>(cloneFunc);
  cloneFuncOp->setAttr(rock::KernelAttr::getMnemonic(), b.getUnitAttr());

  // Rename the original (CPU) copy to <name>_cpu_host.
  originalFunc.setSymName(testFuncName + "_cpu_host");

  // Add clone directly to top-level module (flat structure).
  module.push_back(cloneFuncOp);

  // Set arch on top-level module.
  module->setAttr(rock::ArchAttr::getMnemonic(), archAttr);
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerRocMLIRDialects(registry);
  // Parse pass names in main to ensure static initialization completed.
  mlir::registerMLIRCLOptions();
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  // LLVM dialect is temporary for the freeze trick.
  context.loadDialect<
      rock::RockDialect, func::FuncDialect, scf::SCFDialect,
      affine::AffineDialect, memref::MemRefDialect, math::MathDialect,
      arith::ArithDialect, vector::VectorDialect, gpu::GPUDialect,
      linalg::LinalgDialect, bufferization::BufferizationDialect,
      tosa::TosaDialect, mlir::LLVM::LLVMDialect>();

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR Rock Dialect host generation\n");

  amdgpu::Chipset chipset;
  if (!arch.getValue().empty()) {
    FailureOr<amdgpu::Chipset> maybeChipset =
        amdgpu::Chipset::parse(archChip());
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(&context),
                "Invalid chipset name: " + archChip());
      exit(1);
    }
    chipset = *maybeChipset;
    bool archPrefersOCP = amdgpu::hasOcpFp8(chipset);
    auto canonicaliseF8Type = [&](std::string name) -> std::string {
      if (name == "fp8")
        return archPrefersOCP ? "f8E4M3FN" : "f8E4M3FNUZ";
      if (name == "bf8")
        return archPrefersOCP ? "f8E5M2" : "f8E5M2FNUZ";
      return name;
    };

    filterDataType = canonicaliseF8Type(filterDataType);
    inputDataType = canonicaliseF8Type(inputDataType);
    outputDataType = canonicaliseF8Type(outputDataType);
  }

  if (isConv(operation))
    correctConvParameters();

  populateDefaults();

  bool hasUserKernel = !testFuncName.empty();

  OwningOpRef<ModuleOp> module;
  GenParams genParams;
  genParams.strictMode = (genValidation.getValue() == "mlir-strict");

  if (!inputFilename.empty()) {
    module = readTestFile(inputFilename.getValue(), hasUserKernel, &context);
  } else {
    if (genValidation == "clone") {
      llvm::errs()
          << "Clone validation is not compatible with kernel generation.\n";
      exit(1);
    }
    module = ModuleOp::create(UnknownLoc::get(&context));
  }

  if (kernelRepeats.getNumOccurrences() > 0 && !genCPUValidation &&
      !genHostHarness) {
    llvm::errs()
        << "--kernel-repeats is only supported with host harness (-ph) or "
           "CPU validation (-pv).\n";
    return EXIT_FAILURE;
  }

  if (cpuTimers && !genHostHarness) {
    llvm::errs() << "--cpu-timers requires host harness generation\n";
    return EXIT_FAILURE;
  }

  // `-pv-f64` does not apply to i8 (quantized) attention.
  if (pvF64.getValue() && inputDataType.getValue() == "i8") {
    llvm::errs() << "-pv-f64 is not supported for i8 (quantized) attention\n";
    return EXIT_FAILURE;
  }

  if (genCloneHarness.getValue()) {
    populateCloneHarnessLogic(*module);
  } else if (!hasUserKernel) {
    generateKernel(&context, genParams, *module);
  }

  if (emitSplitKSelectionLikelihood) {
    module->walk([](rock::RockGemmWrapperInterface gemmOp) {
      // TODO: use rock::isSplitKFaster when reimplemented
      const auto likelihood = RocmlirSplitKSelectionLikelihood::never;
      switch (likelihood) {
      case RocmlirSplitKSelectionLikelihood::always: {
        llvm::outs() << "always\n";
        break;
      }
      case RocmlirSplitKSelectionLikelihood::maybe: {
        llvm::outs() << "maybe\n";
        break;
      }
      case RocmlirSplitKSelectionLikelihood::never: {
        llvm::outs() << "never\n";
        break;
      }
      }
    });
    return 0;
  }

  if (!emitModuleFusabilityForPerfConfig.empty()) {
    llvm::outs() << "fusible:"
                 << rock::isModuleFusible(module.get(),
                                          emitModuleFusabilityForPerfConfig)
                 << "\n";
    return 0;
  }

  if (emitTuningSpace.getNumOccurrences() > 0) {
    rock::TuningParamSpaceSettings settings;
    std::unique_ptr<rock::TuningParamSet> tunableParams(
        rock::createTunableParamSpace(*module, emitTuningSpace, settings));
    SmallString<64> perfConfigStr;
    for (auto param : tunableParams->tuningRange) {
      param.getPerfConfigStr(perfConfigStr);
      llvm::outs() << perfConfigStr << "\n";
      perfConfigStr.clear();
    }
    return 0;
  }

  if (emitTuningKey) {
    SmallString<2048> tuningKey;
    if (failed(rock::getTuningProblemStr(*module, tuningKey))) {
      llvm::errs() << "Failed to get tuning key for module: " << *module
                   << "\n";
      return EXIT_FAILURE;
    }
    llvm::outs() << tuningKey << "\n";
    return 0;
  }

  SmallVector<KernelIF, 8> kernels;
  SmallVector<KernelIF, 8> rootIFs;

  if (testFuncName.empty()) {
    // Compute set of call-graph root nodes;  they're the ones we need to
    // call from main().  Start with all nodes, then erase the ones that
    // have edges to them.  Use SetVector because we want to preserve the
    // order to match an older implementation.
    CallGraph cg(*module);
    SetVector<CallGraphNode *> roots(cg.begin(), cg.end());
    for (auto &node : roots) {
      for (auto &edge : *node)
        roots.remove(edge.getTarget());
      func::FuncOp func =
          dyn_cast<func::FuncOp>(node->getCallableRegion()->getParentOp());
      if (func->getParentOp() && func->getParentOp()->getParentOp())
        roots.remove(node);
    }

    for (auto *node : roots) {
      func::FuncOp func =
          dyn_cast<func::FuncOp>(node->getCallableRegion()->getParentOp());
      rootIFs.emplace_back(func);
    }
    module->walk([&](func::FuncOp func) -> WalkResult {
      if (func->hasAttr(rock::KernelAttr::getMnemonic())) {
        kernels.emplace_back(func);
      }
      return WalkResult::advance();
    });
  } else if (!genCloneHarness.getValue()) {
    auto func = module->lookupSymbol<func::FuncOp>(testFuncName);
    assert(func && "does -fut point to the wrong function?");
    kernels.emplace_back(func); // +++pf: should it be a kernel?
    rootIFs.emplace_back(func);
  }

  // populate host logic.
  if (genHostHarness.getValue()) {
    if (failed(
            populateHostHarnessLogic(*module, kernels, rootIFs, genParams))) {
      llvm::errs() << "Host logic populated failed.\n";
      exit(1);
    }
  }

  // Set up the output file.
  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  module->print(output->os());
  output->keep();
  return 0;
}
