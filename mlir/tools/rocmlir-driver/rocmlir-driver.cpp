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

static cl::opt<std::string> pipeline(
    "pipeline", cl::desc("rocmlir-driver pipeline list"),
    cl::value_desc("comma separated list of rock pipelines: "
                   "migraphx,highlevel,gpu,rocdl,binary or full"),
    cl::init(""));

static cl::opt<bool> legacyRockPipeline("c", cl::Hidden, cl::init(false),
                                        cl::Optional,
                                        cl::cb<void, bool>([](bool v) {
                                          if (v) {
                                            pipeline.setValue("full");
                                          }
                                        }));

static cl::opt<bool> verifyPasses(
    "verify-passes", cl::init(false),
    cl::desc("Have the pass manager(s) run verification after each pass"));

static cl::opt<bool> dumpPipelines(
    "dump-pipelines", cl::init(false),
    cl::desc("Print out a textual form of the requested pipelines"));

/////////////////////////////////////////////////////////////////////////////
//// Backend target spec
static cl::opt<int> gpuOpt("gO",
                           cl::desc("Optimization level for GPU compilation"),
                           cl::value_desc("Integer from 0 to 3"), cl::init(3));


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

static LogicalResult
runPipeline(StringRef arch, ModuleOp m,
            llvm::SmallDenseSet<StringRef> &pipelineSet) {
  PassManager pm(m->getName(), PassManager::Nesting::Implicit);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  pm.enableVerifier(verifyPasses);
  bool needArch = pipelineSet.contains("rocdl") ||
                  pipelineSet.contains("binary");
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

  if (pipelineSet.contains("migraphx")) {
    migraphx::addHighLevelPipeline(pm);
  }

  if (pipelineSet.contains("highlevel")) {
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
  bool isRocdlOnly = pipelineSet.contains("rocdl") &&
                     !pipelineSet.contains("binary");
  backendOpts.compile = !isRocdlOnly;

  // TODO(roctriton): add common params to RockTuningParamAttrInterface
  auto fillCompilationRes =
      m.walk([&](mlir::rock::RockGemmWrapperInterface op) -> WalkResult {
        if (auto perfConfigAttr =
                op->template getAttrOfType<StringAttr>("perf_config")) {
          if (failed(fillCompilationConfigs(perfConfigAttr, tritonOpts,
                                            backendOpts))) {
            llvm::errs() << "Failed to process perfConfig: " << optLevel
                         << "\n";
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
  if (fillCompilationRes.wasInterrupted()) {
    return failure();
  }
  auto fillCompilationResGemmGemm =
      m.walk([&](mlir::rock::RockGemmGemmWrapperInterface op) -> WalkResult {
        if (auto perfConfigAttr =
                op->template getAttrOfType<StringAttr>("perf_config")) {
          if (failed(fillCompilationConfigs(perfConfigAttr, tritonOpts,
                                            backendOpts))) {
            llvm::errs() << "Failed to process perfConfig for gemm_gemm op\n";
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
  if (fillCompilationResGemmGemm.wasInterrupted()) {
    return failure();
  }

  // Set up lowering pipeline.
  if (pipelineSet.contains("gpu")) {
    // Set up the default lowering pipeline which goes down to GPU dialect.
    rock::KernelOptions opts;
    opts.arch = arch.str();
    rock::buildKernelPipeline(pm, opts);
  }
  if (pipelineSet.contains("triton")) {

    rock::buildTritonPipeline(pm, tritonOpts);
  }
  if (pipelineSet.contains("binary") || isRocdlOnly) {

    rock::buildBackendPipeline(pm, backendOpts);
  }

  if (dumpPipelines) {
    llvm::errs() << "Pipeline:\n";
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

  llvm::SmallDenseSet<StringRef> pipelineOptions{
      "migraphx", "highlevel", "gpu", "rocdl", "binary", "triton"};
  llvm::SmallDenseSet<StringRef> fullPipeline{"gpu", "triton", "binary"};
  llvm::SmallDenseSet<StringRef> pipelineSet;
  std::string pipelineStr = pipeline.getValue();
  if (failed(parsePipeline(pipelineStr, pipelineSet,
                           pipelineOptions, fullPipeline))) {
    return failure();
  }

  StringRef onlyArch;
  if (!targetList.empty())
    onlyArch = targetList.front();
  else
    onlyArch = arch;

  bool hasKernels = false;
  if (!pipelineSet.empty()) {
    LogicalResult pipelineResult = success();
    // If sub-modules exist with rock.arch specified and in set
    // of targetChips, run pipeline on those sub-modules
    module->walk([&](ModuleOp subModule) {
      auto archAttr = subModule->getAttrOfType<StringAttr>(
          rock::ArchAttr::getMnemonic());
      hasKernels |= (bool)archAttr;
      if (archAttr && llvm::find(targetList, archAttr.getValue())) {
        pipelineResult = runPipeline(archAttr.getValue(), subModule,
                                     pipelineSet);
      }
    });
    if (!hasKernels) {
      // If no sub-modules, run pipeline on top-level module
      if (onlyArch.empty()) {
        if (module->hasAttrOfType<StringAttr>(rock::ArchAttr::getMnemonic())) {
          onlyArch =
              module->getAttrOfType<StringAttr>(rock::ArchAttr::getMnemonic())
                  .getValue();
        }
      }
      pipelineResult = runPipeline(onlyArch, module, pipelineSet);
    }
    if (failed(pipelineResult))
      return pipelineResult;
  } else {
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    pm.enableVerifier(verifyPasses);
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
