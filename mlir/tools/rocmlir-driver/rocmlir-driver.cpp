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
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
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

static cl::opt<std::string> kernelPipeline(
    "kernel-pipeline", cl::desc("rocmlir-driver kernel pipeline list"),
    cl::value_desc("comma separated list of rock pipelines: "
                   "migraphx,highlevel,gpu,rocdl,binary or full"),
    cl::init(""));

static cl::opt<std::string>
    hostPipeline("host-pipeline", cl::desc("rocmlir-driver host pipeline list"),
                 cl::value_desc("comma separated list of rock pipelines: "
                                "migraphx,highlevel,runner or full"),
                 cl::init(""));

static cl::opt<bool> legacyRockPipeline("c", cl::Hidden, cl::init(false),
                                        cl::Optional,
                                        cl::cb<void, bool>([](bool v) {
                                          if (v) {
                                            kernelPipeline.setValue("full");
                                            hostPipeline.setValue("runner");
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

static cl::opt<bool> hostAsyncCoroutines(
    "host-async-coroutines",
    cl::desc("Use coroutines when lowering async ops to LLVM"),
    // FIXME: This should be true to match upstream
    cl::init(false));

static cl::opt<std::string> targets("targets", cl::desc("list of target"),
                                    cl::init(""));

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
// Module isolation helpers for buildBufferizePipeline
//===----------------------------------------------------------------------===//

/// Move functions matching `shouldHide` into a temporary nested module,
/// leaving declaration stubs behind. Returns the nested module.
static ModuleOp
isolateFuncs(ModuleOp module,
             mlir::function_ref<bool(func::FuncOp)> shouldHide) {
  OpBuilder builder(module.getBody(), module.getBody()->end());
  auto hiddenModule =
      ModuleOp::create(builder, module.getLoc(), "__rock_hidden");

  for (auto funcOp :
       llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
    if (!shouldHide(funcOp))
      continue;

    // Create a declaration stub with matching name + type.
    OpBuilder stubBuilder(funcOp);
    auto stub = func::FuncOp::create(stubBuilder, funcOp.getLoc(),
                                     funcOp.getName(),
                                     funcOp.getFunctionType());
    // Declarations must be private (MLIR verifier rejects public decls).
    stub.setVisibility(SymbolTable::Visibility::Private);
    stub->setAttr("__rock_stub", builder.getUnitAttr());

    // Move the real function into the nested module.
    funcOp->moveBefore(hiddenModule.getBody(), hiddenModule.getBody()->end());
  }

  return hiddenModule;
}

/// Fix up call sites in `module` when a restored function has additional
/// input arguments compared to the stub it replaces.
///
/// InsertOutputStoresPass appends output tensor arguments to the kernel.
/// For each new argument, this inserts a `tensor.empty` at the call site.
static void fixupCallSites(ModuleOp module, StringRef funcName,
                           mlir::FunctionType oldType,
                           mlir::FunctionType newType) {
  unsigned oldNumInputs = oldType.getNumInputs();
  unsigned newNumInputs = newType.getNumInputs();

  if (oldNumInputs == newNumInputs)
    return; // No signature change

  assert(newNumInputs > oldNumInputs &&
         "restored function has fewer arguments than stub");

  module.walk([&](func::CallOp callOp) {
    if (callOp.getCallee() != funcName)
      return;

    OpBuilder builder(callOp);
    SmallVector<mlir::Value> newOperands(callOp.getOperands());

    // Add tensor.empty for each new output argument.
    for (unsigned i = oldNumInputs; i < newNumInputs; ++i) {
      mlir::Type argType = newType.getInput(i);
      auto tensorType = cast<RankedTensorType>(argType);
      mlir::Value empty = tensor::EmptyOp::create(
          builder, callOp.getLoc(), tensorType, ValueRange{});
      newOperands.push_back(empty);
    }

    // Create replacement call with updated operands.
    auto newCall = func::CallOp::create(
        builder, callOp.getLoc(), funcName, newType.getResults(),
        static_cast<ValueRange>(newOperands));
    callOp.replaceAllUsesWith(newCall.getResults());
    callOp.erase();
  });
}

/// Move functions back from the nested module and erase stubs.
static void restoreFuncs(ModuleOp module, ModuleOp hiddenModule) {
  // Build a map from function name -> real function in the hidden module.
  llvm::StringMap<func::FuncOp> hiddenFuncs;
  for (auto funcOp : hiddenModule.getOps<func::FuncOp>())
    hiddenFuncs[funcOp.getName()] = funcOp;

  // Replace each stub with the corresponding real function.
  SmallVector<func::FuncOp> stubs;
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    if (funcOp->hasAttr("__rock_stub"))
      stubs.push_back(funcOp);
  }
  for (auto stub : stubs) {
    auto it = hiddenFuncs.find(stub.getName());
    assert(it != hiddenFuncs.end() && "stub without matching hidden func");
    func::FuncOp realFunc = it->second;

    // Fix up call sites if the kernel's signature changed.
    mlir::FunctionType oldType = stub.getFunctionType();
    mlir::FunctionType newType = realFunc.getFunctionType();
    if (oldType != newType)
      fixupCallSites(module, stub.getName(), oldType, newType);

    // Move the real function to just before the stub, preserving order.
    realFunc->moveBefore(stub);
    stub.erase();
  }

  // The hidden module should now be empty.
  assert(hiddenModule.getBody()->empty() &&
         "hidden module still has functions after restore");
  hiddenModule.erase();
}

/// Run buildBufferizePipeline with temporary module isolation.
///
/// When the module contains both kernel and host functions, kernel
/// functions are moved into a temporary nested module. The pipeline
/// runs on either the outer module (host: disableRock=true) or the
/// nested module (kernel: disableRock=false).
///
/// When only one kind of function is present, no isolation is needed.
static LogicalResult
runBufferizePipelineWithIsolation(ModuleOp module,
                                  const rock::BufferizeOptions &opts = {}) {
  auto isKernel = [](func::FuncOp f) {
    return f->hasAttr(rock::KernelAttr::getMnemonic());
  };

  bool hasKernels =
      llvm::any_of(module.getOps<func::FuncOp>(), isKernel);
  bool hasHosts = llvm::any_of(module.getOps<func::FuncOp>(),
                               [&](func::FuncOp f) { return !isKernel(f); });

  // If only one kind of function, no isolation needed — run directly.
  if (!hasKernels || !hasHosts) {
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    pm.enableVerifier(!disableVerifyPasses);
    rock::buildBufferizePipeline(pm, opts);
    if (dumpPipelines) {
      llvm::errs() << "Highlevel pipeline:\n";
      pm.printAsTextualPipeline(llvm::errs());
      llvm::errs() << "\n";
    }
    return pm.run(module);
  }

  // Both kernel and host functions present: isolate kernels.
  ModuleOp kernelModule = isolateFuncs(module, isKernel);

  // disableRock=true  (host pipeline)   → outer module (hosts + stubs)
  // disableRock=false (kernel pipeline) → nested module (kernels only)
  ModuleOp target = opts.disableRock ? module : kernelModule;

  PassManager pm(target->getName(), PassManager::Nesting::Implicit);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  pm.enableVerifier(!disableVerifyPasses);
  rock::buildBufferizePipeline(pm, opts);

  if (dumpPipelines) {
    llvm::errs() << (opts.disableRock ? "Host" : "Kernel")
                 << " highlevel pipeline:\n";
    pm.printAsTextualPipeline(llvm::errs());
    llvm::errs() << "\n";
  }

  LogicalResult result = pm.run(target);

  // Always restore, even on failure, to leave the module consistent.
  restoreFuncs(module, kernelModule);

  return result;
}

static LogicalResult runHostHighLevelPipeline(ModuleOp m) {
  rock::BufferizeOptions opts;
  opts.disableRock = true;
  return runBufferizePipelineWithIsolation(m, opts);
}

static LogicalResult
runKernelPipeline(StringRef arch, ModuleOp m,
                  llvm::SmallDenseSet<StringRef> &kernelPipelineSet) {
  PassManager pm(m->getName(), PassManager::Nesting::Implicit);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  pm.enableVerifier(!disableVerifyPasses);
  bool needArch = kernelPipelineSet.contains("rocdl") ||
                  kernelPipelineSet.contains("binary");
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

  if (kernelPipelineSet.contains("migraphx")) {
    migraphx::addHighLevelPipeline(pm);
  }

  if (kernelPipelineSet.contains("highlevel")) {
    rock::buildBufferizePipeline(pm);
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
  bool isRocdlOnly = kernelPipelineSet.contains("rocdl") &&
                     !kernelPipelineSet.contains("binary");
  backendOpts.compile = !isRocdlOnly;

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
  if (kernelPipelineSet.contains("binary") || isRocdlOnly) {

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

  llvm::SmallVector<std::string, 4> targetList;
  StringRef targetsStr = targets.getValue();
  SmallVector<StringRef, 4> tokens;
  targetsStr.split(tokens, ',');
  for (auto str : tokens) {
    auto target = str.trim();
    if (!target.empty()) {
      RocmDeviceName targetDevName;
      if (failed(targetDevName.parse(target))) {
        llvm::errs() << "Invalid target " << target << " in --targets\n";
        return failure();
      }
      SmallString<64> canonicalTarget;
      targetDevName.getFullName(canonicalTarget);
      targetList.push_back(canonicalTarget.str().str());
    }
  }

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
      "migraphx", "highlevel", "gpu", "rocdl", "binary", "triton"};
  llvm::SmallDenseSet<StringRef> kernelFullPipeline{"gpu", "triton", "binary"};
  llvm::SmallDenseSet<StringRef> kernelPipelineSet;
  std::string kernelPipelineStr = kernelPipeline.getValue();
  if (failed(parsePipeline(kernelPipelineStr, kernelPipelineSet,
                           kernelPipelineOptions, kernelFullPipeline))) {
    return failure();
  }
  llvm::SmallDenseSet<StringRef> hostPipelineOptions{"migraphx", "highlevel",
                                                     "runner"};
  llvm::SmallDenseSet<StringRef> hostPipelineSet;
  std::string hostPipelineStr = hostPipeline.getValue();
  if (failed(parsePipeline(hostPipelineStr, hostPipelineSet,
                           hostPipelineOptions, hostPipelineOptions))) {
    return failure();
  }

  if (hostPipelineSet.contains("migraphx")) {
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    pm.enableVerifier(!disableVerifyPasses);
    migraphx::addHighLevelPipeline(pm);
    if (failed(pm.run(module))) {
      return failure();
    }
  }

  StringRef onlyArch;
  if (!targetList.empty())
    onlyArch = targetList.front();
  else
    onlyArch = arch;

  StringRef targetArch = onlyArch;
  // Right now we need to update the target architecture used when we
  // are running the kernel pipeline, or if we are running the highlevel host
  // pipeline.
  bool needsTargetArchUpdate =
      !kernelPipelineSet.empty() || hostPipelineSet.contains("highlevel");
  if (needsTargetArchUpdate) {
    // Determine arch from module attribute or command line.
    if (onlyArch.empty()) {
      if (module->hasAttrOfType<StringAttr>(rock::ArchAttr::getMnemonic())) {
        onlyArch =
            module->getAttrOfType<StringAttr>(rock::ArchAttr::getMnemonic())
                .getValue();
      }
    }
    targetArch = onlyArch;

    LogicalResult kernelResult =
        runKernelPipeline(onlyArch, module, kernelPipelineSet);

    // Run host high-level pipeline if specified
    if (hostPipelineSet.contains("highlevel"))
      kernelResult = runHostHighLevelPipeline(module);

    if (failed(kernelResult))
      return kernelResult;
  } else {
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    pm.enableVerifier(!disableVerifyPasses);
    auto errorHandler = [&](const Twine &msg) {
      emitError(UnknownLoc::get(module.getContext())) << msg;
      return failure();
    };

    // Use lowering pipeline specified at command line.
    if (failed(passPipeline.addToPipeline(pm, errorHandler))) {
      return failure();
    }
    if (dumpPipelines) {
      llvm::errs() << "Custom pipeline:\n";
      pm.printAsTextualPipeline(llvm::errs());
      llvm::errs() << "\n";
      if (module.getBody()->empty())
        return success();
    }
    if (failed(pm.run(module))) {
      return failure();
    }
  }

  // Run Bufferization on the top module
  if (isHighLevel && hasKernels) {
    rock::BufferizeOptions opts;
    opts.disableRock = true;
    if (failed(runBufferizePipelineWithIsolation(module, opts)))
      return failure();
  }

  // Run MHAL generation on the top module
  /*
  if (hostPipelineSet.contains("mhal")) {
    PassManager pm(module.getContext());
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    pm.enableVerifier(!disableVerifyPasses);
    mhal::buildPackagePipeline(pm);
    if (dumpPipelines) {
      llvm::errs() << "MHAL package pipeline:\n";
      pm.printAsTextualPipeline(llvm::errs());
      llvm::errs() << "\n";
    }
    if (failed(pm.run(module))) {
      return failure();
    }
  }*/

  // Run host code lowering that makes the result of this operation accetable
  // to mlir-runner. Explicitly aborts in the case of multiple mhal
  // targets to prevent confusing behavior.
  /*if (hostPipelineSet.contains("runner")) {
    if (targetList.size() > 1) {
      llvm::errs() << "Expected at most one mhal target when compling from "
                      "within rocmlir-driver\n";
      return failure();
    }
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    pm.enableVerifier(!disableVerifyPasses);
    mhal::RunnerOptions runnerOptions;
    runnerOptions.barePtrMemrefs = barePointers.getValue();
    runnerOptions.enableCoroutines = hostAsyncCoroutines.getValue();
    SmallVector<std::string, 4> targetTypes{"GPU"};
    SmallVector<std::string, 4> targetArchs;
    targetArchs.push_back(targetArch.str());
    runnerOptions.targetTypes = targetTypes;
    runnerOptions.targetArchs = targetArchs;
    mhal::buildRunnerPipeline(pm, runnerOptions);
    if (dumpPipelines) {
      llvm::errs() << "Host runner pipeline:\n";
      pm.printAsTextualPipeline(llvm::errs());
      llvm::errs() << "\n";
    }
    if (failed(pm.run(module)))
      return failure();
  }*/

  // Clean up
  module->walk(
      [&](LLVM::LLVMFuncOp func) { func->removeAttr("xmodel.targets"); });
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
