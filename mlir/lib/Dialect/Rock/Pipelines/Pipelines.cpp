//===- Pipelines.cpp - Create Rock compilation pipelines ---------------===//
//
// Copyright 2021 The MLIR Authors.
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

#include "mlir/Conversion/RocMLIRPasses.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Conversion/CPU/Passes.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Passes.h"
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

#include "amd/include/TritonAMDGPUToLLVM/Passes.h"
#include "amd/include/TritonAMDGPUTransforms/Passes.h"

// Triton includes (for backend pipeline)
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/TargetSelect.h"
#include <optional>

using namespace mlir;
using namespace mlir::triton;

// Based on make_ttir() in
// @triton//:third_party/amd/backend/compiler.py
static void makeTTIR(mlir::OpPassManager *pm, StringRef arch) {
  pm->addPass(mlir::createInlinerPass());

  if (rock::supportsTDM(arch)) {
    pm->addPass(mlir::triton::createTritonRewriteTensorDescriptorToPointer());
  }
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::triton::createTritonCombineOps());
  pm->addPass(mlir::triton::createTritonReorderBroadcast());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mlir::triton::createTritonLoopUnroll());
}

static bool isPingpongScheduleEnabled(StringRef arch, bool useAsyncCopy) {
  return arch.starts_with("gfx942") ||
         (arch.starts_with("gfx950") && useAsyncCopy);
}

static bool isInThreadTransposeEnabled(StringRef arch) {
  return arch.starts_with("gfx942");
}

static bool isAsyncCopyEnabled(StringRef arch) {
  return arch.starts_with("gfx950") || arch.starts_with("gfx1250");
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
  pm->addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  // TODO ROCm Check if we want to compare MI100 and greater
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

  // TODO(ROCm) Modify when corresponding run time flags are introduced.
  std::string scheduleHint = "none";

  bool useAsyncCopy = isAsyncCopyEnabled(options.arch);
  bool useBlockPingpong = isPingpongScheduleEnabled(options.arch, useAsyncCopy);

  pm->addPass(mlir::createTritonAMDGPUOptimizeDescriptorEncoding());
  pm->addPass(mlir::createTritonAMDGPUScheduleLoops({options.numStages}));
  pm->addPass(
      mlir::createTritonAMDGPUPipeline({useAsyncCopy, useBlockPingpong}));
  if (useAsyncCopy) {
    pm->addPass(mlir::createTritonAMDGPUCoalesceAsyncCopy({options.arch}));
  }
  pm->addPass(mlir::createTritonAMDGPUConvertToTensorOps());
  pm->addPass(mlir::createCanonicalizerPass());
  if (scheduleHint != "none") {
    pm->addPass(mlir::triton::createTritonAMDGPUInsertInstructionSchedHintsPass(
        {scheduleHint}));
  }
  pm->addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mlir::triton::gpu::createTritonGPUReduceDataDuplication());
  if (isInThreadTransposeEnabled(options.arch)) {
    pm->addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonAMDGPUInThreadTranspose());
    pm->addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  }
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUMoveUpPrologueLoads());
  if (useBlockPingpong && options.numStages > 1) {
    pm->addPass(mlir::createTritonAMDGPUBlockPingpong({options.numStages}));
  }

  // TODO(rocmlirTriton): if knobs.amd.use_buffer_ops
    pm->addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonAMDGPUCanonicalizePointers());
    pm->addPass(mlir::createCanonicalizerPass());
    pm->addPass(mlir::createTritonAMDGPUConvertToBufferOps(
        {options.arch, /*allowBufferAtomics*/true,
        /*analyzeSmallTensorOfst*/false}));
    pm->addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonAMDGPUOptimizeBufferOpPtr());

  pm->addPass(mlir::createTritonAMDFoldTrueCmpI());
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUPrepareIfCombining());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
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
static void makeLLIR(mlir::OpPassManager *pm, const std::string &arch,
                     int numStages) {
  pm->addPass(mlir::createTritonAMDGPUUpdateAsyncWaitCount({arch}));
  pm->addPass(mlir::triton::AMD::createConvertWarpPipelinePass(arch));
  pm->addPass(mlir::createSCFToControlFlowPass());

  // TODO: do we need this?
  // pm->addPass(gluon::createGluonInline());
  pm->addPass(mlir::createConvertIndexToLLVMPass());

  pm->addPass(mlir::triton::createAllocateAMDGPUSharedMemory());
  pm->addPass(mlir::triton::gpu::createTritonGPUGlobalScratchAllocationPass());
  // Upstream calls this pass twice, between
  // HIPBackend.instrumentation.patch("ttgpuir_to_llvmir", ...).
  // Because we do not implement the instrumentation thing (see
  // docs/bump_triton_version.md Step 6), a single call is sufficient.

  // ## __HIP_FTZ is used to control the denorm flushing behavior of exp2 op as
  // follows:
  // ## 1. If __HIP_FTZ = 1, exp2 flushes denorms in input and output regardless
  // ##    of the value of kernel arg `allow_flush_denorm`.
  // ## 2. If __HIP_FTZ = 0, whether exp2 flushes denorms in input and output
  // ##    depends on the value of kernel arg `allow_flush_denorm`.
  // ## 3. __HIP_FTZ is default to 1 and not exposed as a kernel argument.
  // ##    For now it is used as a controller for developers only.
  pm->addPass(
      mlir::triton::createConvertTritonAMDGPUToLLVMPass(arch, /*ftz=*/true));
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
  if (/*(instruction_sched_variant=="none") == */ /* DISABLES CODE */
      (false)) {
    pm->addPass(mlir::triton::createTritonAMDGPULowerInstructionSchedHintsPass(
        arch, numStages));
  }

  // TODO: add_di_scope

  pm->addPass(
      mlir::triton::createConvertBuiltinFuncToLLVMPass(arch, /*ftz=*/true));

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
    funcPm.addPass(rock::createRockTosaToElementwisePass());
  }
  // use tosa conversion pipeline
  // (see mlir/lib/Conversion/TosaToLinalg/TosaToLinalgPass.cpp)
  TosaToLinalgOptions tosaToLinalgOptions;
  TosaToLinalgNamedOptions tosaToLinalgNamedOptions;
  // pass std::nullopt as validation options to avoid running tosa-validate
  // pass
  tosa::addTosaToLinalgPasses(pm, tosaToLinalgOptions, tosaToLinalgNamedOptions,
                              /*validationOptions=*/std::nullopt, /*attachTargetOptions*/std::nullopt);

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
  /* rocmlir-opt --rock-affix-params --rock-conv-to-gemm
   *   --rock-fold-broadcast --rock-affix-params --rock-gemm-to-gridwise
   *   --rock-regularize --rock-gridwise-gemm-to-blockwise
   * --rock-blockwise-load-tile-to-threadwise
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

  addWithDCE(rock::createRockAffixTuningParametersPass());
  addWithDCE(rock::createRockLowerReducePass());
  addWithDCE(rock::createRockRegularizeOutputPass());
  addWithDCE(rock::createRockRegularizeInterGemmFusionPass());
  addWithDCE(rock::createRockConvToGemmPass());
  addWithDCE(rock::createRockFusionSplitkRegularizationPass());
  addWithDCE(rock::createRockGemmToGridwisePass());
  addWithDCE(rock::createRockGridwiseAttnToBlockwisePass());
  addWithDCE(rock::createRockGridwiseGemmToBlockwisePass());
  addWithDCE(rock::createRockInsertOutputFusionLoadsPass());
  addWithCSE(rock::createRockRegularizeInputPass());
  addWithDCE(rock::createRockLowerLoadsPass());
  addWithDCE(rock::createRockLowerStoresPass());

  // This pass converts unsupported float types to int8 and wraps fusion ops
  // with arith.bitcast (preserving original f8/f4 types inside the wrapper).
  addWithDCE(rock::createRockLegalizeFloatTypesPass());

  // Serialize and erase host functions BEFORE any func-level pass that
  // changes the kernel signature (e.g. RockToTTIRPass sets return to void).
  // Must use a new nest<func::FuncOp>() so these passes go into a separate
  // adaptor that runs AFTER SerializeHostFuncs.
  pm.addPass(rock::createRockSerializeHostFuncsPass());

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
    pm.addPass(arith::createArithExpandOpsPass(expandOpts));
  }

  auto &funcPm2 = pm.nest<func::FuncOp>();
  funcPm2.addPass(rock::createRockLowerBlockwiseToPtrPass());
  funcPm2.addPass(rock::createRockMaskNonZeroPreservingFusionsPass());
  funcPm2.addPass(rock::createRockTransformsToPointerArithPass());
  // Clean up dead transform chains left after TransformsToPointerArith
  funcPm2.addPass(createCanonicalizerPass());

  funcPm2.addPass(rock::createRockToTTIRPass());
  // RockFuncToTritonFuncPass operates on ModuleOp (converts func.func to
  // tt.func)
  pm.addPass(rock::createRockFuncToTritonFuncPass());
  // After this point, function is triton::FuncOp
  auto &ttFuncPm = pm.nest<triton::FuncOp>();
  ttFuncPm.addPass(createCanonicalizerPass());
  ttFuncPm.addPass(createCSEPass());
}

void rock::buildTritonPipeline(OpPassManager &pm,
                               const rock::TritonOptions &options) {
  std::string arch = options.arch;
  int threadPerWarp = rock::getWaveSize(arch);

  makeTTIR(&pm, arch);
  makeTTGIR(&pm, threadPerWarp, options);

  // Run MLIR passes to convert TritonGPU -> LLVM dialect
  makeLLIR(&pm, arch, options.numStages);
}

// Build host code lowering pipeline (func + GPU ops -> LLVM)
// Follows the pattern from mlir-hal/lib/Dialect/MHAL/Pipelines/Pipelines.cpp
// (rocMLIR)
void rock::buildHostLoweringPipeline(mlir::OpPassManager &pm,
                                     const rock::BackendOptions &options) {
  // Lower FP8 extf/truncf to memref-based table lookups. Must run BEFORE
  // OneShotBufferize / CpuLowerVerifier below — otherwise stray
  // arith.extf/truncf on fp8 element types crash bufferization with
  // unsupported builtin.unrealized_conversion_cast.
  pm.addPass(createEmulateFp8ExtTruncPass());

  // CPU optimization phase.
  // This transforms the function body but keeps tensor types at boundaries.
  // The pass internally skips verifier functions that involve non-TT float
  // types (f8E8M0FNU, f4E2M1FN) used by scaled GEMMs, because those conflict
  // with ConvertNarrowTypeSignatures / EmulateNarrowTypes passes below.
  cpu::CpuLowerVerifierPassOptions cpuOpts;
  cpuOpts.dumpSchedulesPath = options.dumpCpuSchedules;
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

// Buidling GPU lowering pipeline
void rock::buildBackendPipeline(OpPassManager &pm,
                                const rock::BackendOptions &options) {
  std::string arch = options.chip;

  // Validate LDS usage against the hardware limit, convert dynamic shared
  // memory to static LDS allocation, and strip unused Triton workspace
  // arguments from the kernel signature.  Runs before TritonToHsaco so the
  // static LDS size is baked into the kernel descriptor, and before any
  // downstream consumer of the kernel argument list (e.g. RestoreHostCode in
  // the host lowering pipeline) sees the trimmed signature.
  pm.addPass(rock::createResolveKernelLaunchParamsPass());

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
    pm.addPass(rock::createTritonToHsacoPass(hsacoOpts));
  }

  // Restore host functions (main, wrapper) that were stored during
  // RockFuncToTritonFuncPass. This converts func.call @kernel to gpu.launch_func.
  rock::RockRestoreHostCodePassOptions restoreOpts;
  restoreOpts.triple = options.triple;
  restoreOpts.arch = options.chip;
  restoreOpts.features = options.features;
  restoreOpts.optLevel = options.optLevel;
  pm.addPass(rock::createRockRestoreHostCodePass(restoreOpts));
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
