//===- rocmlir-driver.cpp - MLIR Rock Dialect Driver ----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for rocmlir-driver.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/RocMLIRPasses.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MIGraphX/Pipeline/Pipeline.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/RocmDeviceName.h"
#include "mlir/Dialect/Rock/utility/compileUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/InitRocMLIRCLOptions.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/InitRocMLIRPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/utils/DetachReattach.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <unordered_map>

using namespace llvm;
using namespace mlir;

// Exit codes: 0 = success, EXIT_FAILURE (from <cstdlib>) = real failure,
// EXIT_NOT_APPLICABLE = rock.not_applicable marker set (config refused, not a
// bug). parameterSweeps.py keys on this contract: it treats 0 as PASS, 2 as
// NOT_APPLICABLE, and anything else as FAIL — so EXIT_FAILURE just needs to
// be non-zero and distinct from EXIT_NOT_APPLICABLE.
#define EXIT_NOT_APPLICABLE 2
static_assert(EXIT_FAILURE != 0 && EXIT_FAILURE != EXIT_NOT_APPLICABLE,
              "rocmlir-driver exit-code contract: EXIT_FAILURE must be "
              "non-zero and distinct from EXIT_NOT_APPLICABLE "
              "(parameterSweeps.py keys on this)");

static cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                          llvm::cl::desc("<input file>"),
                                          llvm::cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<std::string>
    kernelPipeline("kernel-pipeline",
                   cl::desc("rocmlir-driver kernel pipeline list"),
                   cl::value_desc("comma separated list of rock pipelines: "
                                  "migraphx,highlevel,gpu,binary or full"),
                   cl::init(""));

static cl::opt<std::string>
    hostPipeline("host-pipeline", cl::desc("rocmlir-driver host pipeline list"),
                 cl::value_desc("comma separated list of rock pipelines: "
                                "migraphx,highlevel,backend or full"),
                 cl::init(""));

static cl::opt<bool> legacyRockPipeline("c", cl::Hidden, cl::init(false),
                                        cl::Optional,
                                        cl::cb<void, bool>([](bool v) {
                                          if (v) {
                                            kernelPipeline.setValue("full");
                                            hostPipeline.setValue("backend");
                                          }
                                        }));

static cl::opt<bool> disableVerifyPasses(
    "disable-verify-passes", cl::init(false),
    cl::desc("Have the pass manager(s) not run verification after each pass"));

static cl::opt<bool> dumpPipelines(
    "dump-pipelines", cl::init(false),
    cl::desc("Print out a textual form of the requested pipelines"));

cl::opt<std::string>
    dumpCpuSchedules("dump-cpu-schedules", cl::init(""), cl::value_desc("path"),
                     cl::desc("Dump CPU verifier IR and transform schedules to "
                              "the specified directory"));

/////////////////////////////////////////////////////////////////////////////
//// Backend target spec
static cl::opt<int> gpuOpt("gO",
                           cl::desc("Optimization level for GPU compilation"),
                           cl::value_desc("Integer from 0 to 3"), cl::init(3));

static cl::opt<bool> barePointers(
    "bare-ptr-memref-kernels",
    cl::desc("Use bare pointers to represent memrefs when calling kernels"),
    cl::init(true));

static cl::opt<std::string> arch("arch", cl::desc("target architecture"),
                                 cl::value_desc("Target GPU architecture"),
                                 cl::init(""));

/////////////////////////////////////////////////////////////////////////////
//// Knobs that mirror Triton's `knobs.amd.*`
enum class TritonKnob { Auto = -1, Off = 0, On = 1 };

static cl::ValuesClass tritonKnobValues() {
  return cl::values(
      clEnumValN(TritonKnob::Auto, "auto",
                 "use the default (per-arch where applicable)"),
      clEnumValN(TritonKnob::Off, "false", "force off"),
      clEnumValN(TritonKnob::On, "true", "force on"));
}

static cl::opt<TritonKnob>
    useAsyncCopy("use-async-copy",
                 cl::desc("Override async-copy schedule:"),
                 tritonKnobValues(), cl::init(TritonKnob::Auto));

static cl::opt<TritonKnob>
    useBlockPingpong("use-block-pingpong",
                     cl::desc("Override block-pingpong schedule:"),
                     tritonKnobValues(), cl::init(TritonKnob::Auto));

static cl::opt<TritonKnob>
    useInThreadTranspose("use-in-thread-transpose",
                         cl::desc("Override in-thread-transpose pass:"),
                         tritonKnobValues(), cl::init(TritonKnob::Auto));

static cl::opt<TritonKnob> useBufferOps(
    "use-buffer-ops",
    cl::desc("Override use-buffer-ops (canonicalize-pointers / "
             "convert-to-buffer-ops / optimize-buffer-op-ptr):"),
    tritonKnobValues(), cl::init(TritonKnob::Auto));

static cl::opt<TritonKnob> useBufferAtomics(
    "use-buffer-atomics",
    cl::desc("Override buffer atomics in convert-to-buffer-ops "
             "(no effect when --use-buffer-ops is off):"),
    tritonKnobValues(), cl::init(TritonKnob::Auto));

static cl::opt<TritonKnob> bufferOpsAnalyzeSmallTensorRange(
    "buffer-ops-analyze-small-tensor-range",
    cl::desc("Override small-tensor range analysis in "
             "convert-to-buffer-ops (no effect when --use-buffer-ops "
             "is off):"),
    tritonKnobValues(), cl::init(TritonKnob::Auto));

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

static LogicalResult
parsePipeline(StringRef pipeline, llvm::SmallDenseSet<StringRef> &pipelineSet,
              llvm::SmallDenseSet<StringRef> &pipelineOptions,
              llvm::SmallDenseSet<StringRef> &fullOptions) {
  SmallVector<StringRef, 8> tokens;
  pipeline.split(tokens, ',');
  for (auto str : tokens) {
    auto opt = str.trim();
    if (opt.empty()) {
    } else if (opt == "full") {
      pipelineSet = fullOptions;
    } else if (pipelineOptions.contains(opt)) {
      pipelineSet.insert(opt);
    } else {
      auto opts = llvm::join(pipelineOptions, ",");
      llvm::errs() << "Invalid pipeline: " << opt << "\n"
                   << "   Valid options: " << opts << " or full\n";
      return failure();
    }
  }

  return success();
}

// Detach functions matching `detachPredicate`, run a pipeline on the
// remaining functions, then reattach. Skips the pipeline entirely if
// no target functions remain after detaching.
static LogicalResult
runWithDetach(ModuleOp module, StringRef pipelineName,
              mlir::function_ref<bool(func::FuncOp)> detachPredicate,
              mlir::function_ref<void(PassManager &)> buildPipeline) {
  DetachedFuncs detached = detachFuncs(module, detachPredicate);

  bool hasTargetFuncs =
      llvm::any_of(module.getOps<func::FuncOp>(),
                   [](func::FuncOp f) { return !f.isDeclaration(); });

  PassManager pm(module->getName(), PassManager::Nesting::Implicit);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  applyDefaultTimingPassManagerCLOptions(pm);
  pm.enableVerifier(!disableVerifyPasses);
  buildPipeline(pm);

  if (dumpPipelines) {
    llvm::errs() << pipelineName << " pipeline:\n";
    pm.printAsTextualPipeline(llvm::errs());
    llvm::errs() << "\n";
  }

  if (!hasTargetFuncs) {
    reattachFuncs(module, detached);
    return success();
  }

  LogicalResult result = pm.run(module);
  reattachFuncs(module, detached);
  return result;
}

static LogicalResult
runKernelPipeline(StringRef arch, ModuleOp m,
                  llvm::SmallDenseSet<StringRef> &kernelPipelineSet) {
  PassManager pm(m->getName(), PassManager::Nesting::Implicit);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  applyDefaultTimingPassManagerCLOptions(pm);
  pm.enableVerifier(!disableVerifyPasses);
  bool needArch = kernelPipelineSet.contains("binary");
  RocmDeviceName devName;
  if (arch.empty() && needArch) {
    llvm::errs()
        << "Architecture not specified for this pipeline, but one is required\n"
        << "Use --arch or set arch\n";
    return failure();
  }
  if (failed(devName.parse(arch)) && needArch) {
    llvm::errs() << "Invalid architecture: " << arch << "\n";
    return failure();
  }

  rock::TritonOptions tritonOpts;
  tritonOpts.arch = devName.getChip().str();
  rock::BackendOptions backendOpts;
  backendOpts.triple = devName.getTriple().str();
  backendOpts.chip = devName.getChip().str();
  backendOpts.features = devName.getFeaturesForBackend();
  // Set up the lowering pipeline which goes down to ELF Binary
  int optLevel = gpuOpt.getValue();
  if (optLevel < 0 || optLevel > 3) {
    llvm::errs() << "Invalid GPU optimization level: " << optLevel << "\n";
    return failure();
  }
  backendOpts.optLevel = optLevel;

  // TODO(roctriton): add common params to RockTuningParamAttrInterface
  OpBuilder builder(m.getContext());
  auto fillCompilationRes =
      m.walk([&](mlir::rock::RockGemmWrapperInterface op) -> WalkResult {
        auto populateParamsPtr = std::make_unique<rock::PopulateParams>();
        auto maybeGemmParams =
            populateParamsPtr->obtainTuningParameters(builder, op);
        if (failed(maybeGemmParams)) {
          llvm::errs() << "Failed to obtain perfConfig\n";
          return WalkResult::interrupt();
        }

        if (failed(fillCompilationConfigs(maybeGemmParams.value(), tritonOpts,
                                          backendOpts))) {
          llvm::errs() << "Failed to process perfConfig: "
                       << maybeGemmParams.value() << "\n";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (fillCompilationRes.wasInterrupted()) {
    return failure();
  }
  auto fillCompilationResGemmGemm =
      m.walk([&](mlir::rock::RockGemmGemmWrapperInterface op) -> WalkResult {
        auto maybeGemmGemmParams =
            rock::PopulateParamsGemmGemm::obtainTuningParameters(builder, op);
        if (failed(maybeGemmGemmParams)) {
          llvm::errs() << "Failed to obtain perfConfig\n";
          return WalkResult::interrupt();
        }
        if (failed(fillCompilationConfigs(maybeGemmGemmParams.value(),
                                          tritonOpts, backendOpts))) {
          llvm::errs() << "Failed to process perfConfig: "
                       << maybeGemmGemmParams.value() << "\n";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (fillCompilationResGemmGemm.wasInterrupted()) {
    return failure();
  }

  // Apply Triton knobs after perf-config has been processed, so
  // an explicit knob set via CLI flag wins.
  tritonOpts.useAsyncCopy = static_cast<int>(useAsyncCopy.getValue());
  tritonOpts.useBlockPingpong = static_cast<int>(useBlockPingpong.getValue());
  tritonOpts.useInThreadTranspose =
      static_cast<int>(useInThreadTranspose.getValue());
  tritonOpts.useBufferOps = static_cast<int>(useBufferOps.getValue());
  tritonOpts.useBufferAtomics = static_cast<int>(useBufferAtomics.getValue());
  tritonOpts.bufferOpsAnalyzeSmallTensorRange =
      static_cast<int>(bufferOpsAnalyzeSmallTensorRange.getValue());

  // Set up lowering pipeline.
  if (kernelPipelineSet.contains("gpu")) {
    // Set up the default lowering pipeline which goes down to GPU dialect.
    rock::KernelOptions opts;
    opts.arch = arch.str();
    rock::buildKernelPipeline(pm, opts);
  }
  if (kernelPipelineSet.contains("triton")) {
    rock::buildTritonPipeline(pm, tritonOpts);
  }
  if (kernelPipelineSet.contains("binary")) {
    rock::buildBackendPipeline(pm, backendOpts);
  }

  if (dumpPipelines) {
    llvm::errs() << "Kernel pipeline:\n";
    pm.printAsTextualPipeline(llvm::errs());
    llvm::errs() << "\n";
  }

  // Skip the kernel-backend pipeline on CPU-only / kernel-less modules
  // Mirrors the kernel/host gating that Phases 1-2 get from `runWithDetach`
  bool hasKernel = llvm::any_of(m.getOps<func::FuncOp>(), [](func::FuncOp f) {
    return f->hasAttr(rock::KernelAttr::getMnemonic());
  });
  if (!hasKernel)
    return success();

  return pm.run(m);
}

static LogicalResult runMLIRPasses(ModuleOp &module,
                                   mlir::PassPipelineCLParser &passPipeline) {

  // Canonicalize arch name
  if (!arch.empty()) {
    RocmDeviceName devName;
    if (failed(devName.parse(arch))) {
      llvm::errs() << "Unknown value for --arch " << arch << "\n";
      return failure();
    }
    SmallString<64> canonicalArch;
    devName.getFullName(canonicalArch);
    arch = canonicalArch.str().str();
  }

  llvm::SmallDenseSet<StringRef> kernelPipelineOptions{
      "migraphx", "highlevel", "gpu", "binary", "triton"};
  llvm::SmallDenseSet<StringRef> kernelFullPipeline{"gpu", "triton", "binary"};
  llvm::SmallDenseSet<StringRef> kernelPipelineSet;
  std::string kernelPipelineStr = kernelPipeline.getValue();
  if (failed(parsePipeline(kernelPipelineStr, kernelPipelineSet,
                           kernelPipelineOptions, kernelFullPipeline))) {
    return failure();
  }
  llvm::SmallDenseSet<StringRef> hostPipelineOptions{"migraphx", "highlevel",
                                                     "backend"};
  llvm::SmallDenseSet<StringRef> hostPipelineSet;
  std::string hostPipelineStr = hostPipeline.getValue();
  if (failed(parsePipeline(hostPipelineStr, hostPipelineSet,
                           hostPipelineOptions, hostPipelineOptions))) {
    return failure();
  }

  auto isKernel = [](func::FuncOp f) {
    return f->hasAttr(rock::KernelAttr::getMnemonic());
  };
  auto isHost = [&](func::FuncOp f) { return !isKernel(f); };

  // Phase 1: MIGraphX lowering (host and kernel independently)
  if (hostPipelineSet.contains("migraphx")) {
    if (failed(runWithDetach(
            module, "Host MIGraphX", isKernel,
            [](PassManager &pm) { migraphx::addMIGraphXPipeline(pm); })))
      return failure();
  }
  if (kernelPipelineSet.contains("migraphx")) {
    if (failed(runWithDetach(
            module, "Kernel MIGraphX", isHost,
            [](PassManager &pm) { migraphx::addMIGraphXPipeline(pm); })))
      return failure();
  }

  // Phase 2: Highlevel (host and kernel independently)
  if (hostPipelineSet.contains("highlevel")) {
    rock::HighlevelOptions opts;
    opts.disableRock = true;
    if (failed(runWithDetach(
            module, "Host Highlevel", isKernel,
            [&](PassManager &pm) { rock::buildHighlevelPipeline(pm, opts); })))
      return failure();
  }
  if (kernelPipelineSet.contains("highlevel")) {
    if (failed(runWithDetach(
            module, "Kernel Highlevel", isHost,
            [](PassManager &pm) { rock::buildHighlevelPipeline(pm); })))
      return failure();
  }

  // Phase 3: GPU / Triton / Backend (kernel pipeline only).
  //
  // This runs before the host backend (Phase 4) when both are requested
  // (e.g. via `-c`).  buildBackendPipeline produces `gpu.binary` and rewrites
  // `func.call @kernel` -> `gpu.launch_func`; if the host were lowered first,
  // those calls would already be `llvm.call @kernel` and the kernel-launch
  // rewrite would silently fail to find them, leaving an unlinked module.
  bool needsKernelBackend = kernelPipelineSet.contains("gpu") ||
                            kernelPipelineSet.contains("triton") ||
                            kernelPipelineSet.contains("binary");
  if (needsKernelBackend) {
    StringRef onlyArch = arch;
    if (onlyArch.empty()) {
      if (module->hasAttrOfType<StringAttr>(rock::ArchAttr::getMnemonic())) {
        onlyArch =
            module->getAttrOfType<StringAttr>(rock::ArchAttr::getMnemonic())
                .getValue();
      }
    }
    if (failed(runKernelPipeline(onlyArch, module, kernelPipelineSet)))
      return failure();
  }

  // Phase 4: Host backend lowering (func + memref + GPU ops -> LLVM).
  //
  // Runs AFTER kernel compilation so the host module already contains
  // `gpu.launch_func` (created by RockEmitGpuBinaryPass in
  // buildBackendPipeline); gpu-to-llvm at the end of this pipeline then
  // translates those into HIP runtime calls.  Safe to run standalone too: with
  // no kernel pipeline, there is no `gpu.launch_func` and gpu-to-llvm is a
  // no-op.
  if (hostPipelineSet.contains("backend")) {
    if (failed(runWithDetach(
            module, "Host Backend", isKernel, [&](PassManager &pm) {
              rock::buildHostLoweringPipeline(pm, dumpCpuSchedules.getValue());
            })))
      return failure();
  }

  // Custom pipeline fallback (when no named pipelines are requested)
  if (kernelPipelineSet.empty() && hostPipelineSet.empty()) {
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    applyDefaultTimingPassManagerCLOptions(pm);
    pm.enableVerifier(!disableVerifyPasses);
    auto errorHandler = [&](const Twine &msg) {
      emitError(UnknownLoc::get(module.getContext())) << msg;
      return failure();
    };

    if (failed(passPipeline.addToPipeline(pm, errorHandler)))
      return failure();

    if (dumpPipelines) {
      llvm::errs() << "Custom pipeline:\n";
      pm.printAsTextualPipeline(llvm::errs());
      llvm::errs() << "\n";
      if (module.getBody()->empty())
        return success();
    }
    if (failed(pm.run(module)))
      return failure();
  }

  return success();
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerRocMLIRDialects(registry);
  mlir::registerRocMLIRPasses();
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerMLIRCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR Rock Dialect driver\n");

  // Create context after ParseCommandLineOptions, otherwise the context
  // will be created without the command line flags.
  MLIRContext context(registry);
  context.loadDialect<rock::RockDialect, func::FuncDialect, scf::SCFDialect,
                      affine::AffineDialect, memref::MemRefDialect,
                      math::MathDialect, arith::ArithDialect, gpu::GPUDialect,
                      bufferization::BufferizationDialect>();
  OpBuilder builder(&context);
  ModuleOp module;

  std::string errorMessage;
  SourceMgr sourceMgr;
  OwningOpRef<ModuleOp> moduleRef;

  // Set up the input file.
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    exit(EXIT_FAILURE);
  }

  // Parse the input file.
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  moduleRef = parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!moduleRef) {
    llvm::errs() << "Parse host harness " << inputFilename << " failed.\n";
    exit(EXIT_FAILURE);
  }
  module = moduleRef.get();

  // Run MLIR passes with passed in tuning parameters. If a rock pass
  // determined the (kernel x perf-config x hw) combination is structurally
  // inapplicable it will have signalled pass failure AND set the
  // `rock.not_applicable` marker on the module (see RockAttrDefs.td and
  // ResolveKernelLaunchParamsPass for the canonical example). Distinguish
  // that from a real lowering bug via a dedicated exit code so callers
  // (parameterSweeps.py, tuning frontends, ...) can classify the failure
  // without having to scrape stderr.
  if (failed(runMLIRPasses(module, passPipeline))) {
    if (module->hasAttr(rock::NotApplicableAttr::getMnemonic())) {
      llvm::errs() << "Lowering not applicable.\n";
      exit(EXIT_NOT_APPLICABLE);
    }
    llvm::errs() << "Lowering failed.\n";
    exit(EXIT_FAILURE);
  }

  // Set up the output file.
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(EXIT_FAILURE);
  }

  module.print(output->os());
  output->keep();
  return 0;
}
