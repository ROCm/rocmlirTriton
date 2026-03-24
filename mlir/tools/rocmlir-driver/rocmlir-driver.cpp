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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <unordered_map>

using namespace llvm;
using namespace mlir;

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
                                "migraphx,highlevel or full"),
                 cl::init(""));

static cl::opt<bool> legacyRockPipeline("c", cl::Hidden, cl::init(false),
                                        cl::Optional,
                                        cl::cb<void, bool>([](bool v) {
                                          if (v) {
                                            kernelPipeline.setValue("full");
                                          }
                                        }));

static cl::opt<bool> disableVerifyPasses(
    "disable-verify-passes", cl::init(false),
    cl::desc("Have the pass manager(s) not run verification after each pass"));

static cl::opt<bool> dumpPipelines(
    "dump-pipelines", cl::init(false),
    cl::desc("Print out a textual form of the requested pipelines"));

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

//===----------------------------------------------------------------------===//
// Function detach/reattach helpers for pipeline isolation
//===----------------------------------------------------------------------===//

struct DetachedFuncs {
  struct Entry {
    Operation *realFunc;
    func::FuncOp stub;
  };
  SmallVector<Entry> entries;
};

// Physically remove functions matching `shouldDetach` from the module,
// leaving private declaration stubs behind so that symbol references
// (e.g. func.call) remain valid during pass execution.
static DetachedFuncs
detachFuncs(ModuleOp module,
            mlir::function_ref<bool(func::FuncOp)> shouldDetach) {
  DetachedFuncs detached;

  for (auto funcOp :
       llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
    if (!shouldDetach(funcOp))
      continue;

    OpBuilder stubBuilder(funcOp);
    auto stub =
        func::FuncOp::create(stubBuilder, funcOp.getLoc(), funcOp.getName(),
                             funcOp.getFunctionType());
    stub.setVisibility(SymbolTable::Visibility::Private);

    funcOp->remove();
    detached.entries.push_back({funcOp, stub});
  }

  return detached;
}

// Re-insert previously detached functions into the module, replacing
// their stubs.
static void reattachFuncs(ModuleOp module, DetachedFuncs &detached) {
  for (auto &entry : detached.entries) {
    auto *realFunc = entry.realFunc;
    auto stub = entry.stub;

    mlir::FunctionType stubType = stub.getFunctionType();
    mlir::FunctionType realType =
        cast<func::FuncOp>(realFunc).getFunctionType();
    assert(stubType == realType &&
           "detached function signature changed during pipeline execution; "
           "no callers should exist to fixup");

    stub->getBlock()->getOperations().insert(stub->getIterator(), realFunc);
    stub.erase();
  }
  detached.entries.clear();
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

    // Return success in dump-pipelines if the module is empty
    // Otherwise rocmlir-driver will return failure when we run
    // certain passes like triton-to-hsaco (which fail on empty modules)
    if (m.getBody()->empty())
      return success();
  }
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
  llvm::SmallDenseSet<StringRef> hostPipelineOptions{"migraphx", "highlevel"};
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
            [](PassManager &pm) { migraphx::addHighLevelPipeline(pm); })))
      return failure();
  }
  if (kernelPipelineSet.contains("migraphx")) {
    if (failed(runWithDetach(
            module, "Kernel MIGraphX", isHost,
            [](PassManager &pm) { migraphx::addHighLevelPipeline(pm); })))
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

  // Phase 3: GPU / Triton / Backend (kernel pipeline only)
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

  // Custom pipeline fallback (when no named pipelines are requested)
  if (kernelPipelineSet.empty() && hostPipelineSet.empty()) {
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
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
    exit(1);
  }

  // Parse the input file.
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  moduleRef = parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!moduleRef) {
    llvm::errs() << "Parse host harness " << inputFilename << " failed.\n";
    exit(1);
  }
  module = moduleRef.get();

  // Run MLIR passes with passed in tuning parameters
  if (failed(runMLIRPasses(module, passPipeline))) {
    llvm::errs() << "Lowering failed.\n";
    exit(1);
  }

  // Set up the output file.
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  module.print(output->os());
  output->keep();
  return 0;
}
