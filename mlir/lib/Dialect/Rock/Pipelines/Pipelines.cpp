//===- Pipelines.cpp - Create Rock compilation pipelines ---------------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// =============================================================================
//
// This interface adds the Rock compilation pipeline for various flows but
// keeping a unified ordering of the pipeline.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Conversion/ArithToAMDGPU/ArithToAMDGPU.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Conversion/CPU/Passes.h"
#include "mlir/Conversion/RocMLIRPasses.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/Dialect/Tosa/IR/TargetEnv.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/include/TritonAMDGPUToLLVM/Passes.h"
#include "amd/include/TritonAMDGPUTransforms/Passes.h"

// Triton includes (for backend pipeline)
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetSelect.h"
#include <optional>

using namespace mlir;
using namespace mlir::triton;

// Based on make_ttir() in
// @triton//:third_party/amd/backend/compiler.py
static void makeTTIR(mlir::OpPassManager *pm, StringRef arch) {
  pm->addPass(mlir::createInlinerPass());

  if (!rock::supportsTDM(arch)) {
    pm->addPass(mlir::triton::createTritonRewriteTensorDescriptorToPointer());
  }
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::triton::createTritonCombineOps());
  // --- rocmlirTriton pass ----
  // Must run after combine-ops, which folds chained addptrs into the single
  // offset this pass re-materializes and turns select+load into a masked load,
  // and before CSE and LICM, which respectively deduplicate the narrowed
  // address computations and hoist any load narrowing made loop-invariant.
  pm->addPass(rock::createRockNarrowRedundantLoadsPass());
  // --- rocmlirTriton pass ----
  pm->addPass(mlir::triton::createTritonReorderBroadcast());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mlir::triton::createTritonLoopUnroll());
}

// Reject malformed `TritonOptions` knob values before the pipeline is
// constructed.
static void validateTritonOptionsKnobs(const rock::TritonOptions &options) {
  auto reject = [](StringRef field, int64_t value) {
    llvm::report_fatal_error(Twine("invalid `--pass-pipeline=triton{") + field +
                                 "=" + Twine(value) + "}`; expected " +
                                 Twine(rock::kKnobDefault) +
                                 " (arch default), 0 (off), or 1 (on)",
                             /*GenCrashDiag=*/false);
  };
  const std::pair<StringRef, int64_t> boolKnobs[] = {
      {"useAsyncCopy", options.useAsyncCopy},
      {"useBlockPingpong", options.useBlockPingpong},
      {"useInThreadTranspose", options.useInThreadTranspose},
      {"useBufferOps", options.useBufferOps},
      {"useBufferAtomics", options.useBufferAtomics},
      {"bufferOpsAnalyzeSmallTensorRange",
       options.bufferOpsAnalyzeSmallTensorRange},
      {"useReductionLayout", options.useReductionLayout},
      {"useOptimizeEpilogue", options.useOptimizeEpilogue},
  };
  for (auto [name, value] : boolKnobs) {
    if (!rock::isValidKnobBoolean(value))
      reject(name, value);
  }
}

static bool isPingpongScheduleEnabled(StringRef arch, bool useAsyncCopy,
                                      int64_t useBlockPingpongOverride) {
  if (useBlockPingpongOverride != rock::kKnobDefault)
    return useBlockPingpongOverride;
  return arch.starts_with("gfx942") ||
         (arch.starts_with("gfx950") && useAsyncCopy);
}

static bool isInThreadTransposeEnabled(StringRef arch,
                                       int64_t useInThreadTransposeOverride) {
  if (useInThreadTransposeOverride != rock::kKnobDefault)
    return useInThreadTransposeOverride;
  return arch.starts_with("gfx942") || arch.starts_with("gfx110") ||
         arch.starts_with("gfx115") || arch.starts_with("gfx117") ||
         arch.starts_with("gfx120");
}

static bool isAsyncCopyEnabled(StringRef arch, int64_t useAsyncCopyOverride) {
  if (useAsyncCopyOverride != rock::kKnobDefault)
    return useAsyncCopyOverride;
  return arch.starts_with("gfx950") || arch.starts_with("gfx1250");
}

static bool isBufferOpsEnabled(int64_t useBufferOpsOverride) {
  if (useBufferOpsOverride != rock::kKnobDefault)
    return useBufferOpsOverride;
  return true;
}

static bool isBufferAtomicsEnabled(int64_t useBufferAtomicsOverride) {
  if (useBufferAtomicsOverride != rock::kKnobDefault)
    return useBufferAtomicsOverride;
  return true;
}

static bool isBufferOpsAnalyzeSmallTensorRangeEnabled(
    int64_t analyzeSmallTensorRangeOverride) {
  if (analyzeSmallTensorRangeOverride != rock::kKnobDefault)
    return analyzeSmallTensorRangeOverride;
  return false;
}
// Based on make_ttgir() in
// @triton//:third_party/amd/backend/compiler.py
static void makeTTGIR(mlir::OpPassManager *pm, int threadPerWarp,
                      const rock::TritonOptions &options) {
  pm->addPass(mlir::triton::createConvertTritonToTritonGPU(
      {"hip:" + options.arch, options.numWarps, threadPerWarp,
       options.numCTAs}));
  pm->addPass(mlir::triton::gpu::createTritonGPUCoalesce());
  pm->addPass(mlir::triton::gpu::createTritonGPUF32DotTC({false}));
  pm->addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mlir::triton::gpu::createTritonGPUOptimizeThreadLocality());
  pm->addPass(mlir::createTritonAMDGPUAccelerateMatmul(
      {options.arch, options.matrixInstrNonkdim, options.kpack}));
  // --- rocmlirTriton pass ----
  // Must run after accelerate-matmul (consumes the accelerator dot) and before
  // remove-layout-conversions (folds away the convert_layout ops it inserts).
  pm->addPass(rock::createRockSetMatmulOutputTransposePass());
  // --- rocmlirTriton pass ----

  pm->addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  if (options.useOptimizeEpilogue != 0)
    pm->addPass(mlir::createTritonAMDGPUOptimizeEpilogue());
  pm->addPass(mlir::triton::amdgpu::createTritonAMDGPUOptimizeDotOperands(
      {options.arch}));
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUHoistLayoutConversions());
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUSinkLayoutConversions());

  pm->addPass(mlir::triton::gpu::createTritonGPUFuseNestedLoops());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createCanonicalizerPass());

  bool useAsyncCopy = isAsyncCopyEnabled(options.arch, options.useAsyncCopy);
  bool useBlockPingpong = isPingpongScheduleEnabled(options.arch, useAsyncCopy,
                                                    options.useBlockPingpong);

  pm->addPass(mlir::createTritonAMDGPUOptimizeDescriptorEncoding());
  pm->addPass(mlir::createTritonAMDGPUScheduleLoops({options.numStages}));
  pm->addPass(
      mlir::createTritonAMDGPUPipeline({useAsyncCopy, useBlockPingpong}));
  if (useAsyncCopy) {
    pm->addPass(mlir::createTritonAMDGPUCoalesceAsyncCopy({options.arch}));
  }
  pm->addPass(mlir::createTritonAMDGPUConvertToTensorOps());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mlir::triton::gpu::createTritonGPUReduceDataDuplication());
  if (isInThreadTransposeEnabled(options.arch, options.useInThreadTranspose)) {
    pm->addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonAMDGPUInThreadTranspose());
    pm->addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  }
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUMoveUpPrologueLoads());
  if (useBlockPingpong && options.numStages > 1) {
    pm->addPass(mlir::createTritonAMDGPUBlockPingpong({options.numStages}));
  }

  bool useBufferOps = isBufferOpsEnabled(options.useBufferOps);
  if (useBufferOps) {
    pm->addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonAMDGPUCanonicalizePointers());
    pm->addPass(mlir::createCanonicalizerPass());
    pm->addPass(mlir::createTritonAMDGPUConvertToBufferOps(
        {options.arch, isBufferAtomicsEnabled(options.useBufferAtomics),
         isBufferOpsAnalyzeSmallTensorRangeEnabled(
             options.bufferOpsAnalyzeSmallTensorRange)}));
    pm->addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonAMDGPUOptimizeBufferOpPtr());
  }

  pm->addPass(mlir::createTritonAMDFoldTrueCmpI());
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUPrepareIfCombining());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  if (useBufferOps) {
    // Run after CSE so matching assume and loop-bound expressions share SSA,
    // letting range analysis prove both non-negative.
    pm->addPass(mlir::createTritonAMDGPUAnnotateBufferOpSplitSafety());
  }
  pm->addPass(mlir::createSymbolDCEPass());
  // TODO(roctriton): Implement options like this.
  // if (options.instrumentationMode == "fpsan") {
  //   pm->addPass(mlir::createTritonAMDGPUFPSanitizer());
  // }
}

// Based on make_llir() in
// @triton//:third_party/amd/backend/compiler.py
//
// NOTE: make_llir is divided into two parts in our project:
// 1. makeLLIR (the function below)
// 2. TritonToHsaco (in TritonToHsaco.cpp)
// See the comment at the bottom of this function for more details.
static void makeLLIR(mlir::OpPassManager *pm,
                     const rock::TritonOptions &options) {
  const std::string &arch = options.arch;

  pm->addPass(mlir::createTritonAMDGPUUpdateAsyncWaitCount({arch}));
  pm->addPass(mlir::triton::AMD::createConvertWarpPipelinePass(arch));
  // Redistribute the layout of the reduction dimension to reduce register
  // pressure. Always scheduled, but the `useReductionLayout`
  // actually controls whether it runs.
  rock::RockSetReductionLayoutPassOptions reductionLayoutOpts;
  reductionLayoutOpts.useReductionLayout = options.useReductionLayout;
  pm->addPass(rock::createRockSetReductionLayoutPass(reductionLayoutOpts));
  pm->addPass(mlir::createSCFToControlFlowPass());

  // TODO: do we need this?
  // pm->addPass(gluon::createGluonInline());
  pm->addPass(mlir::createConvertIndexToLLVMPass());

  pm->addPass(mlir::triton::createAllocateAMDGPUSharedMemoryPass(arch));
  pm->addPass(mlir::triton::gpu::createTritonGPUGlobalScratchAllocationPass());
  // Upstream calls this pass twice, between
  // HIPBackend.instrumentation.patch("ttgpuir_to_llvmir", ...).
  // Because we do not implement the instrumentation thing (see
  // docs/bump_triton_version.md Step 6), a single call is sufficient.

  // `ftz` controls the denorm flushing behaviour of exp2:
  //  - ftz = 1: exp2 becomes the raw `v_exp_f32`, which flushes input and
  //    output denorms regardless of the kernel's `denormal-fp-math-f32`.
  //  - ftz = 0: exp2 becomes `llvm.exp2.f32`, and the LLVM backend adds the
  //    scaling sequence that preserves denorms unless the kernel attribute
  //    already allows flushing them.
  //
  // Upstream hardwires this to 1 (`__HIP_FTZ`, compiler.py) because it has no
  // user-facing denormal knob. rocmlirTriton does: `denormal-fp-math-f32` is
  // set from `BackendOptions::allowFlushDenorm` in TritonToHsaco. Hardwiring
  // `ftz` to 1 here would flush denorms anyway and silently defeat that, so it
  // follows `TritonOptions::allowFlushDenorm` instead. Those are two separate
  // options on two separate pipelines; both have to be set to get IEEE
  // denormals end to end.
  bool ftz = options.allowFlushDenorm;
  pm->addPass(mlir::triton::createConvertTritonAMDGPUToLLVMPass(arch, ftz));
  pm->addPass(
      mlir::triton::AMD::createTritonAMDGPUConvertWarpSpecializeToLLVMPass(
          arch));
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());

  // Note: translateTritonGPUToLLVMIR adds line info with LLVMDIScopePass.
  pm->addPass(mlir::createConvertControlFlowToLLVMPass());
  pm->addPass(mlir::createArithToLLVMConversionPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());

  // TODO: add_di_scope

  pm->addPass(mlir::triton::createConvertBuiltinFuncToLLVMPass(arch, ftz));
  pm->addPass(mlir::createReconcileUnrealizedCastsPass());

  // --- rocmlirTriton pass ----
  // Must run after convert-triton-amdgpu-to-llvm, which emits the buffer
  // accesses, and after reconcile-unrealized-casts, since the analyses walk
  // plain LLVM offset arithmetic and stop at a cast. Canonicalize and CSE
  // after, so the offset math of an erased access leaves with it.
  if (isBufferOpsEnabled(options.useBufferOps)) {
    pm->addNestedPass<mlir::LLVM::LLVMFuncOp>(
        rock::createRockFoldOobBufferOpsPass());
    pm->addPass(mlir::createCanonicalizerPass());
    pm->addPass(mlir::createCSEPass());
  }
  // --- rocmlirTriton pass ----

  // IMPORTANT:
  //
  // make_llir here has this comment:
  // # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
  // and keeps lowering the IR to LLVM.
  // We have the rest of the lowering in TritonToHsaco.cpp
  //
  // The reason for doing this is to keep Pipelines as a
  // lowering pipeline for MLIR only, leaving the LLVM lowering to
  // TritonToHsaco.cpp.
}

//===- Consolidate the Rock Pipelines here ---------------------===//

void rock::buildHighlevelPipeline(OpPassManager &pm,
                                  const rock::HighlevelOptions &options) {
  bool noRock = options.disableRock;

  pm.addPass(rock::createRockFlattenTosaFuncArgsPass());

  auto &funcPm = pm.nest<func::FuncOp>();

  // TOSA conversion to rock
  if (!noRock) {
    // convert tosa.conv2d/matmul to rock.conv
    /* rocmlir-opt --tosa-to-tensor --tosa-to-rock --rock-view-to-transform
     */
    funcPm.addPass(createTosaToTensorPass());
    funcPm.addPass(createTosaToRockPass());
    funcPm.addPass(rock::createRockViewToTransformPass());
    funcPm.addPass(rock::createRockDetectFlashDecodingPass());
  }

  funcPm.addPass(createRocmlirCustomTosaDecomposePass());

  // rocmlirPromoteSoftmaxPrecisionPass will only run on the CPU path. At this
  // point in the GPU path, we have already converted to rock.attention ops.
  funcPm.addPass(createRocmlirPromoteSoftmaxPrecisionPass());
  if (noRock)
    funcPm.addPass(createRocmlirCustomTosaToLinalgPass());

  if (!noRock) {
    rock::RockTosaToElementwisePassOptions elementwiseOpts;
    elementwiseOpts.disableFastMath = options.disableFastMath;
    funcPm.addPass(rock::createRockTosaToElementwisePass(elementwiseOpts));
  }
  // use tosa conversion pipeline
  // (see mlir/lib/Conversion/TosaToLinalg/TosaToLinalgPass.cpp)
  TosaToLinalgOptions tosaToLinalgOptions;
  TosaToLinalgNamedOptions tosaToLinalgNamedOptions;
  // pass std::nullopt as validation options to avoid running tosa-validate
  // pass
  tosa::addTosaToLinalgPasses(pm, tosaToLinalgOptions, tosaToLinalgNamedOptions,
                              /*validationOptions=*/std::nullopt,
                              /*attachTargetOptions*/ std::nullopt);

  // for tosa control flow
  /* rocmlir-opt --tosa-to-tensor --tosa-to-scf --tosa-to-arith
   */
  auto &funcPm2 = pm.nest<func::FuncOp>();
  funcPm2.addPass(createTosaToTensorPass());
  funcPm2.addPass(createTosaToSCFPass());
  funcPm2.addPass(createTosaToArithPass());

  // linalg tensor opts
  /* rocmlir-opt --linalg-fuse-elementwise-ops --linalg-fold-unit-extent-dims
   */
  funcPm2.addPass(createLinalgElementwiseOpFusionPass());
  funcPm2.addPass(createLinalgFoldUnitExtentDimsPass());
  funcPm2.addPass(rock::createRockViewToTransformPass());
  funcPm2.addPass(rock::createRockFoldBroadcastPass());
  funcPm2.addPass(createCanonicalizerPass());

  pm.addPass(createConvertTensorToLinalgPass());
  if (!noRock) {
    pm.nest<func::FuncOp>().addPass(
        rock::createRockSortDimensionsMemoryLayoutPass());
    pm.addPass(rock::createRockInsertOutputStoresPass());
  }
}

void rock::buildKernelPipeline(OpPassManager &pm,
                               const rock::KernelOptions &options) {
  // rock lowering (tuning, global to block)
  /* rocmlir-opt
   *   --rock-affix-params
   *   --rock-lower-reduce
   *   --rock-regularize-output
   *   --rock-regularize-inter-gemm-fusion
   *   --rock-conv-to-gemm
   *   --rock-fusion-splitk-regularization
   *   --rock-gemm-to-gridwise
   *   --rock-attn-to-gridwise
   *   --rock-gridwise-attn-to-blockwise
   *   --rock-gridwise-gemm-to-blockwise
   *   --rock-insert-output-fusion-loads
   *   --rock-regularize-input
   *   --rock-lower-loads
   *   --rock-lower-stores
   */

  // We must use pm.nest<func::FuncOp>() inside the lambdas instead of a
  // single shared funcPm, because DCE needs to run at the module level
  // to see function callers. Running it at function level causes it to
  // incorrectly remove the host function, which breaks the IR.
  auto addWithDCE = [&pm](std::unique_ptr<Pass> pass) {
    pm.nest<func::FuncOp>().addPass(std::move(pass));
    pm.addPass(createRemoveDeadValuesPass());
  };
  auto addWithCSE = [&pm](std::unique_ptr<Pass> pass) {
    pm.nest<func::FuncOp>().addPass(std::move(pass));
    pm.addPass(createCSEPass());
  };
  // Cannot be merged with addWithDCE: that helper runs the pass in a new
  // func nest and the DCE at module level, whereas here both the pass and the
  // DCE runs on the caller existing func nest.
  auto addWithFuncDCE = [](OpPassManager &funcPm, std::unique_ptr<Pass> pass) {
    funcPm.addPass(std::move(pass));
    funcPm.addPass(createRemoveDeadValuesPass());
  };
  addWithDCE(rock::createRockAffixTuningParametersPass());
  addWithDCE(rock::createRockLowerReducePass());
  addWithDCE(rock::createRockRegularizeOutputPass());
  addWithDCE(rock::createRockRegularizeInterGemmFusionPass());
  addWithDCE(rock::createRockConvToGemmPass());
  addWithDCE(rock::createRockFusionSplitkRegularizationPass());
  addWithDCE(rock::createRockGemmToGridwisePass());
  addWithDCE(rock::createRockAttnToGridwisePass());

  // Must run after AttnToGridwise and before GridwiseGemmToBlockwise.
  addWithDCE(rock::createRockDecomposeNonPow2TilesPass());

  addWithDCE(rock::createRockGridwiseAttnToBlockwisePass());
  addWithDCE(rock::createRockGridwiseGemmToBlockwisePass());
  // Must run after ToBlockwise passes, which build the K loop and the
  // accumulator this pass decomposes, and before FuseSiblingLoops, so that it
  // sees one blockwise_gemm per loop.
  addWithDCE(rock::createRockDecomposeNonPow2KPass());
  // Must run after GridwiseGemmToBlockwise and RockDecomposeNonPow2K, but
  // before InsertOutputFusionLoads. CSE after deduplicates the now-co-located
  // shared operand loads.
  addWithCSE(rock::createRockFuseSiblingLoopsPass());
  addWithDCE(rock::createRockInsertOutputFusionLoadsPass());
  addWithCSE(rock::createRockRegularizeInputPass());
  addWithDCE(rock::createRockLowerLoadsPass());
  addWithDCE(rock::createRockLowerStoresPass());

  // Must run after lower-stores (needs the rock.blockwise_store) and before
  // lower-blockwise-to-ptr (which lowers it away).
  addWithDCE(rock::createRockAddTritonMetadataPass());

  // Rewrite the math ops Triton has no conversion pattern for. This has to come
  // before math-extend-to-supported-types, which would otherwise promote a bf16
  // or fp8 tanh to f32 and leave it indistinguishable from one the user asked
  // for at f32. The two get different lowerings, so the distinction matters.
  {
    rock::RockLegalizeMathForTritonPassOptions legalizeMathOpts;
    legalizeMathOpts.disableFastMath = options.disableFastMath;
    addWithDCE(rock::createRockLegalizeMathForTritonPass(legalizeMathOpts));
  }

  // Legalize math ops on narrow floats before they are converted to integer
  // storage types. LLVM math intrinsics do not accept fp8/fp4 operands.
  {
    math::MathExtendToSupportedTypesOptions mathExtendOptions;
    // TODO: Gfx1250 has support for some bf16 math operations. Therefore this
    // pass should pass through those supported opertions for bf16 on gfx1250.
    mathExtendOptions.extraTypeStrs = {"f16"};
    mathExtendOptions.targetTypeStr = "f32";
    addWithDCE(math::createMathExtendToSupportedTypes(mathExtendOptions));
  }

  // Must run after lower-stores to catch redundant casts that cannot be
  // flagged earlier due to loads/stores that sit between truncf/extf pairs,
  // and before RockToTTIR, whose clamp fold only matches a minnumf/maxnumf
  // pair that already carries `nnan`.
  if (!options.disableFastMath)
    addWithDCE(rock::createRockAllowFastMathFlagsPass());

  // This pass converts unsupported float types to int8 and wraps fusion ops
  // with arith.bitcast (preserving original f8/f4 types inside the wrapper).
  addWithDCE(rock::createRockLegalizeFloatTypesPass());

  // Serialize and erase host functions BEFORE any func-level pass that
  // changes the kernel signature (e.g. RockToTTIRPass sets return to void).
  // Must use a new nest<func::FuncOp>() so these passes go into a separate
  // adaptor that runs AFTER SerializeHostFuncs.
  pm.addPass(rock::createRockSerializeHostFuncsPass());

  // Emulate arith ops on narrow floats (f4/f8) by promoting to f32: LLVM has
  // no native arithmetic for these types. The pass wraps affected arith ops
  // with extf/truncf; ext/trunc/bitcast/select/constant stay legal so tt.dot
  // still sees the original narrow-float tensors. Must run BEFORE arith-expand
  // so the f4 extf/truncf it introduces are expanded to bitwise integer ops.
  {
    arith::ArithEmulateUnsupportedFloatsOptions emulateOpts;
    emulateOpts.sourceTypeStrs = {"f4E2M1FN",   "f8E4M3FN", "f8E4M3FNUZ",
                                  "f8E5M2FNUZ", "f8E5M2",   "f8E8M0FNU"};
    emulateOpts.targetTypeStr = "f32";
    pm.addPass(arith::createArithEmulateUnsupportedFloats(emulateOpts));
  }

  // Expand f8E8M0FNU and f4E2M1FN truncf/extf to bitwise integer ops.
  // Must run AFTER SerializeHostFuncs (arith-expand would corrupt linalg body
  // regions in host code) and BEFORE RockToTTIR, because LLVM lowering cannot
  // handle fptrunc/fpext with these narrow float types. LegalizeFloatTypes
  // converts GEMM operand tensors to i8 but fusion ops inside are wrapped
  // with arith.bitcast, so scalar extf/truncf still use f4/f8 types.
  {
    arith::ArithExpandOpsPassOptions expandOpts;
    expandOpts.includeF8E8M0 = true;
    expandOpts.includeF4E2M1 = true;
    // Anything this pass expands is gone before RockToTTIR runs, so Triton's
    // elementwise lowering never sees it. Keep floating-point and integer
    // min/max ops intact unconditionally so Triton can lower them through the
    // LLVM min/max intrinsics instead of inheriting compare/select chains.
    expandOpts.includeMinMaxF = false;
    expandOpts.includeMinMaxI = false;
    pm.addPass(arith::createArithExpandOpsPass(expandOpts));
  }

  auto &funcPm2 = pm.nest<func::FuncOp>();
  funcPm2.addPass(rock::createRockAnalyzeMemoryUsePass());
  funcPm2.addPass(rock::createRockLowerBlockwiseToPtrPass());
  funcPm2.addPass(rock::createRockPreserveMaskedLoadSemanticsPass());
  // Must run BEFORE TransformsToPointerArith: it simplifies the rock.transform
  // chains feeding TransformsToPtrOp by collapsing contiguous merges.
  // CollapseContiguousMerges builds the collapsed chain fresh and rewires onto
  // it, leaving the original chain dead. DCE it so TransformsToPointerArith
  // only sees the collapsed chain.
  addWithFuncDCE(funcPm2, rock::createRockCollapseContiguousMergesPass());
  addWithFuncDCE(funcPm2, rock::createRockIncrementalPointerArithPass());
  funcPm2.addPass(rock::createRockTransformsToPointerArithPass());
  // Clean up dead transform chains left after TransformsToPointerArith
  funcPm2.addPass(createCanonicalizerPass());

  rock::RockToTTIRPassOptions rockToTTIROpts;
  rockToTTIROpts.disableFastMath = options.disableFastMath;
  funcPm2.addPass(rock::createRockToTTIRPass(rockToTTIROpts));
  // A second run, for the float ops RockToTTIR synthesizes itself (the
  // reduction combiners), which did not exist yet at the run above. Ops that
  // run already flagged are left alone, so the repeat costs nothing.
  if (!options.disableFastMath)
    addWithFuncDCE(funcPm2, rock::createRockAllowFastMathFlagsPass());
  // RockTensorToTritonPtrPass operates on ModuleOp (converts func.func to
  // tt.func)
  pm.addPass(rock::createRockTensorToTritonPtrPass());
  // After this point, function is triton::FuncOp
  auto &ttFuncPm = pm.nest<triton::FuncOp>();
  ttFuncPm.addPass(createCanonicalizerPass());
  ttFuncPm.addPass(createCSEPass());
}

void rock::buildTritonPipeline(OpPassManager &pm,
                               const rock::TritonOptions &options) {
  validateTritonOptionsKnobs(options);

  std::string arch = options.arch;
  int threadPerWarp = rock::getWaveSize(arch);

  makeTTIR(&pm, arch);
  makeTTGIR(&pm, threadPerWarp, options);

  // Run MLIR passes to convert TritonGPU -> LLVM dialect
  makeLLIR(&pm, options);
}

// Build host code lowering pipeline (func + GPU ops -> LLVM)
// Follows the pattern from mlir-hal/lib/Dialect/MHAL/Pipelines/Pipelines.cpp
// (rocMLIR)
void rock::buildHostLoweringPipeline(mlir::OpPassManager &pm,
                                     StringRef dumpCpuSchedules) {
  // CPU optimization phase.

  // Rewrite linalg.generic convolutions in CPU-verifier funcs into a
  // single 8-D fused linalg.generic whose input map inlines the im2col
  // gather. The follow-up FusedConvToMatmulSchedule (inside
  // LowerCpuVerifierPass) tiles the convolution-only dims away, leaving
  // a 3-D matmul that the existing TilingSchedule and
  // VectorizationSchedule can target.
  pm.addPass(cpu::createCpuConvToGemmPass());

  // This transforms the function body but keeps tensor types at boundaries.
  // The pass internally skips verifier functions that involve non-TT float
  // types (f8E8M0FNU, f4E2M1FN) used by scaled GEMMs, because those conflict
  // with ConvertNarrowTypeSignatures / EmulateNarrowTypes passes below.
  cpu::CpuLowerVerifierPassOptions cpuOpts;
  cpuOpts.dumpSchedulesPath = dumpCpuSchedules.str();
  cpuOpts.phase = cpu::CPU_PHASE_OPTIMIZE;
  pm.addPass(cpu::createCpuLowerVerifierPass(cpuOpts));

  // Bufferize tensor ops to memref ops for the whole module.
  // This handles both the CPU verifier functions and their call sites,
  // eliminating the need for manual call site updates.
  bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));

  // Lower FP8 extf/truncf to memref-based table lookups. Must run after the
  // CPU optimization phase above: its VectorizationSchedule vectorizes the
  // verifier matmul into a named contraction, which it cannot do once the fp8
  // extf has been rewritten into a memref.
  // This pass emits memref IR (memref.get_global / memref.load), and we expect
  // this should only happen when the IR is bufferized. Otherwise, this pass
  // introduces memref ops while the surrounding IR is still tensor-based.
  // Thus, it makes more sense to run this pass after bufferization.
  pm.addPass(createEmulateFp8ExtTruncPass());

  // Lower to LLVM phase (after bufferization)
  cpuOpts.phase = cpu::CPU_PHASE_LOWERTOLLVM;
  pm.addPass(cpu::createCpuLowerVerifierPass(cpuOpts));

  // Lower linalg to loops (for operations like linalg.fill in -pv mode)
  pm.addPass(createConvertLinalgToLoopsPass());

  // Expand f8E8M0FNU/f4E2M1FN extf/truncf to bitwise ops first, so that
  // arith operations using these types are lowered before type conversion.
  arith::ArithExpandOpsPassOptions expandOpts;
  expandOpts.includeF8E8M0 = true;
  expandOpts.includeF4E2M1 = true;
  // includeFlushDenormals stays default (false): it only legalizes the
  // `arith.flush_denormals` op (which this pipeline never produces) and is
  // unrelated to BackendOptions::allowFlushDenorm.
  // This is the CPU reference used to verify GPU results, so keep min/max on
  // the compare/select expansion.
  expandOpts.includeMinMaxF = true;
  expandOpts.includeMinMaxI = true;
  pm.addPass(arith::createArithExpandOpsPass(expandOpts));

  // Emulate sub-byte types (f4E2M1FN -> i4 -> packed i8) after ArithExpandOps
  // has lowered the arithmetic.  Split into two passes:
  //   1. Convert function signatures (memref<Nxi4> args -> memref<N/2xi8>)
  //   2. Rewrite loads/stores/allocs using upstream narrow-type emulation
  // The split avoids a crash in the upstream patterns that call
  // extract_strided_metadata on the original (pre-conversion) block argument.
  // NOTE: ExpandStridedMetadata must run AFTER narrow-type emulation, because
  // it converts memref.expand_shape into multi-dim memref.reinterpret_cast
  // that the upstream narrow-type patterns cannot handle (rank-1 only).
  // The upstream ConvertMemRefExpandShape pattern handles expand_shape on
  // sub-byte types as a no-op (since memrefs are linearized by emulation).
  pm.addNestedPass<func::FuncOp>(
      rock::createRockConvertNarrowTypeSignaturesPass());
  pm.addNestedPass<func::FuncOp>(rock::createRockEmulateNarrowTypesPass());

  // Expand strided metadata (handles memref.expand_shape, etc.)
  // Must run after narrow-type emulation so sub-byte expand_shape ops are
  // already handled, and before lower-affine since it generates affine.apply.
  pm.addPass(memref::createExpandStridedMetadataPass());

  // Lower affine to standard arithmetic.  Must be after ExpandStridedMetadata
  // and the narrow-type emulation passes, both of which generate affine.apply.
  pm.addPass(createLowerAffinePass());

  // Lower SCF to control flow (must be after lower-affine, which creates
  // scf.for from affine.for)
  pm.addPass(createSCFToControlFlowPass());

  // Make GPU operations async - required by GpuToLLVMConversionPass patterns
  pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());

  // Lower remaining operations to LLVM (order follows MHAL pipeline)
  pm.addPass(createConvertControlFlowToLLVMPass());

  // Lower math ops before ArithToLLVM: MathToLLVM lowers ops with LLVM
  // intrinsic equivalents, MathToLibm handles the rest (e.g., math.erf) by
  // emitting libm calls.  MathToLibm can introduce arith ops (extf/truncf for
  // non-f32 types), so ArithToLLVM must run after to catch them all.
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());

  pm.addPass(createArithToLLVMConversionPass());

  // TODO(rocmlirTriton): add createConvertVectorToLLVMPass() if any lowering
  // path produces vector dialect ops on the host side (the old runner pipeline
  // had it here).

  // Lower memref operations to LLVM BEFORE GPU conversion (per MHAL pattern)
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());

  // Convert GPU operations to runtime calls
  GpuToLLVMConversionPassOptions gpuOpts;
  gpuOpts.kernelBarePtrCallConv = true; // Use kernel bare ptr, not host
  pm.addPass(createGpuToLLVMConversionPass(gpuOpts));

  // Lower any remaining func operations to LLVM (including external
  // declarations)
  pm.addPass(createConvertFuncToLLVMPass());

  // Cleanup
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

// Build GPU lowering pipeline
void rock::buildBackendPipeline(OpPassManager &pm,
                                const rock::BackendOptions &options) {
  std::string arch = options.chip;

  // Validate LDS usage against the hardware limit, convert dynamic shared
  // memory to static LDS allocation, and strip unused Triton workspace
  // arguments from the kernel signature.  Runs before TritonToHsaco so the
  // static LDS size is baked into the kernel descriptor, and before any
  // downstream consumer of the kernel argument list (e.g. RockEmitGpuBinaryPass
  // in the host lowering pipeline) sees the trimmed signature.
  pm.addPass(rock::createResolveKernelLaunchParamsPass());

  // Annotate LLVM IR for efficient AMDGPU codegen (GEP inbounds, alias
  // scopes, invariant loads, atomic metadata).
  RockPrepareLLVMPassOptions prepareLLVMOpts;
  prepareLLVMOpts.allowFlushDenorm = options.allowFlushDenorm;
  pm.addNestedPass<LLVM::LLVMFuncOp>(
      rock::createRockPrepareLLVMPass(prepareLLVMOpts));

  // Optionally generate the HSACO binary
  if (options.compile) {
    // Add the TritonToHsaco pass to convert LLVM dialect to HSACO binary
    // This implements the functionality from Triton's compiler.py:
    // - make_llir() lines 358-449: LLVM-IR (MLIR) -> LLVM-IR (LLVM)
    // - make_amdgcn() lines 452-473: LLVM -> AMDGCN assembly
    // - make_hsaco() lines 476-488: AMDGCN assembly -> HSACO binary
    rock::TritonToHsacoPassOptions hsacoOpts;
    hsacoOpts.triple = options.triple;
    hsacoOpts.arch = arch;
    hsacoOpts.features = options.features;
    hsacoOpts.optLevel = options.optLevel;
    hsacoOpts.numWarps = options.numWarps;
    hsacoOpts.numCTAs = options.numCTAs;
    hsacoOpts.wavesPerEU = options.wavesPerEU;
    hsacoOpts.enableFpFusion = options.enableFpFusion;
    hsacoOpts.allowFlushDenorm = options.allowFlushDenorm;
    hsacoOpts.llvmFnAttrs = options.llvmFnAttrs;
    hsacoOpts.useExpertScheduling = options.useExpertScheduling;
    pm.addPass(rock::createTritonToHsacoPass(hsacoOpts));
  }

  // Emit gpu.binary from HSACO, restore host functions (main, wrapper) if
  // serialized during RockTensorToTritonPtrPass, and convert func.call @kernel
  // to gpu.launch_func when applicable.
  rock::RockEmitGpuBinaryPassOptions emitGpuBinaryOpts;
  emitGpuBinaryOpts.triple = options.triple;
  emitGpuBinaryOpts.arch = options.chip;
  emitGpuBinaryOpts.features = options.features;
  emitGpuBinaryOpts.optLevel = options.optLevel;
  pm.addPass(rock::createRockEmitGpuBinaryPass(emitGpuBinaryOpts));
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void rock::registerPipelines() {
  PassPipelineRegistration<rock::HighlevelOptions>(
      "rock-highlevel-pipeline",
      " representations and algorithms for sparse tensors.",
      buildHighlevelPipeline);
  PassPipelineRegistration<rock::KernelOptions>(
      "rock-kernel-pipeline",
      " representations and algorithms for sparse tensors.",
      buildKernelPipeline);
  PassPipelineRegistration<rock::TritonOptions>(
      "rock-triton-pipeline", "Convert Triton IR to TritonGPU IR.",
      buildTritonPipeline);
  PassPipelineRegistration<rock::BackendOptions>(
      "rock-backend-pipeline",
      "GPU compilation: lower Triton LLVM-dialect kernels to HSACO binary.",
      buildBackendPipeline);
}
