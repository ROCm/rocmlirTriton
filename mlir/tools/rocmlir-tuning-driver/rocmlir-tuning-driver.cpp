//===- rocmlir-tuning-driver.cpp - rocMLIR tuning driver -------------===//
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//
// Part of the rocMLIR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a wrapper script that reads in a MLIR file containing a rocMLIR
// kernel and tunes it. It will run the kernel with all applicable perf configs
// and report the execution time for each perf config. It is a very intentially
// specific program designed to eliminate JIT overhead, process spawn overhead
// and the like.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/RocmDeviceName.h"
#include "mlir/Dialect/Rock/utility/compileUtils.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/InitRocMLIRCLOptions.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/InitRocMLIRPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <thread>

// Utilities to allocate buffers
#include "../utils/performance/common/benchmarkUtils.h"
#include "CacheFlush.h"

#include <hip/hip_runtime.h>

#if !defined(_HIP_CLANG_ONLY__)
// GCC complains if we don't do this
template <std::size_t n, typename... Ts,
          typename std::enable_if<n == sizeof...(Ts)>::type * = nullptr>
void pArgs(const std::tuple<Ts...> &, void *) {}

template <std::size_t n, typename... Ts,
          typename std::enable_if<n != sizeof...(Ts)>::type * = nullptr>
void pArgs(const std::tuple<Ts...> &formals, void **_vargs) {
  using T = typename std::tuple_element<n, std::tuple<Ts...>>::type;

  static_assert(!std::is_reference<T>{},
                "A __global__ function cannot have a reference as one of its "
                "arguments.");
  _vargs[n] =
      const_cast<void *>(reinterpret_cast<const void *>(&std::get<n>(formals)));
  return pArgs<n + 1>(formals, _vargs);
}
#endif

// Needs to go second lest we get compiler issues
#include <hip/hip_ext.h>

using namespace mlir;
using namespace rocmlir::tuningdriver;

// Mirrors _launch() from external/triton/third_party/amd/backend/driver.c
// (lines 603-646). Simplified: gridY/gridZ always 1, blockSize pre-computed,
// launch_cooperative_grid always 0. Returns LogicalResult instead of void.
// Note: hipEventRecord is handled by callers, not by this function.
// Lives here, in the only consumer, so the rest of the Rock libraries stay
// free of any HIP runtime dependency.
static LogicalResult launchKernel(hipFunction_t function, uint32_t gridX,
                                  uint32_t blockSize, uint32_t shared_memory,
                                  uint32_t num_ctas, hipStream_t stream,
                                  void **params) {
  if (gridX == 0)
    return success();
  if (num_ctas > 1) {
    // Note: driver.c checks hipSymbolTable.hipDrvLaunchKernelEx here because
    // it loads HIP symbols via dlsym. We link directly, so no check needed.
    hipLaunchAttribute attributes[2];
    // Attribute0: Cluster dimensions
    attributes[0].id = static_cast<hipLaunchAttributeID>(4);
    int *cluster_dims = reinterpret_cast<int *>(attributes[0].val.pad);
    cluster_dims[0] = num_ctas;
    cluster_dims[1] = 1;
    cluster_dims[2] = 1;
    // Attribute1: Cooperative launch
    attributes[1].id = hipLaunchAttributeCooperative;
    attributes[1].val.cooperative = 0;

    HIP_LAUNCH_CONFIG config = {
        gridX * num_ctas, 1,      1,            // Grid size
        blockSize,        1,      1,            // Block size
        shared_memory,    stream, attributes, 2 // Number of attributes
    };
    hipError_t status = hipDrvLaunchKernelEx(&config, function, params, 0);
    if (status != hipSuccess)
      return failure();
  } else {
    hipError_t status =
        hipModuleLaunchKernel(function, gridX, 1, 1, blockSize, 1, 1,
                              shared_memory, stream, params, nullptr);
    if (status != hipSuccess)
      return failure();
  }
  return success();
}

static llvm::cl::opt<std::string> inputFilename{
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-")};

static llvm::cl::opt<rock::TuningParamSetKind> tuningSpaceKind(
    "tuning-space", llvm::cl::desc("Tuning space to use for this run"),
    llvm::cl::values(
        clEnumValN(rock::TuningParamSetKind::Quick, "quick",
                   "Quick tuning space"),
        clEnumValN(rock::TuningParamSetKind::Full, "full",
                   "Full tuning space, excluding known-bad configurations"),
        clEnumValN(rock::TuningParamSetKind::Exhaustive, "exhaustive",
                   "All tuning space combinations, even inapplicable ones")),
    llvm::cl::value_desc("tuning space to use"),
    llvm::cl::init(rock::TuningParamSetKind::Full));

static llvm::cl::opt<unsigned> rep(
    "rep",
    llvm::cl::desc("Target benchmark time in milliseconds. The number of "
                   "measured iterations is derived from this budget and the "
                   "estimated per-launch runtime (Triton do_bench style)."),
    llvm::cl::value_desc("benchmark milliseconds"), llvm::cl::init(100));

static llvm::cl::opt<unsigned> warmup(
    "warmup",
    llvm::cl::desc("Target warmup time in milliseconds. The number of warmup "
                   "iterations is derived from this budget and the estimated "
                   "per-launch runtime (Triton do_bench style)."),
    llvm::cl::value_desc("warmup milliseconds"), llvm::cl::init(25));

static llvm::cl::opt<bool>
    useMedian("use-median",
              llvm::cl::desc("Use median of runs instead of mean for timing "
                             "(overrides trim-percent)"),
              llvm::cl::init(false));

static llvm::cl::opt<unsigned> trimPercent(
    "trim-percent",
    llvm::cl::desc("Percentage to trim from top and bottom of results"),
    llvm::cl::value_desc("trim percentage [0, 50)"), llvm::cl::init(10));

static llvm::cl::opt<unsigned> sleepUs(
    "sleep-us",
    llvm::cl::desc("Microseconds to sleep between runs to avoid throttling"),
    llvm::cl::value_desc("microseconds to sleep"), llvm::cl::init(1000));

static llvm::cl::opt<bool> showStats(
    "show-stats",
    llvm::cl::desc(
        "Print detailed stats (min, max, median, stddev, cv) in JSON format."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> showAllMeasurements(
    "show-all-measurements",
    llvm::cl::desc("Print all individual timing measurements in JSON format."),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> benchmarkConfig(
    "benchmark-config",
    llvm::cl::desc(
        "Run benchmark with specific perf config only (skip tuning)"),
    llvm::cl::value_desc("perf config string"), llvm::cl::init(""));

static llvm::cl::opt<unsigned> numCompileThreads(
    "num-compile-threads",
    llvm::cl::desc("Number of parallel compilation threads (0 = auto)"),
    llvm::cl::value_desc("thread count"), llvm::cl::init(0));

static llvm::cl::opt<bool> flushLastLevelCache(
    "flush-last-level-cache",
    llvm::cl::desc(
        "Size the cache-flush buffer to the architecture's last-level cache "
        "(e.g. AMD Infinity Cache) instead of the per-XCD L2 cache size "
        "reported by the HIP runtime. Defaults to the L2 cache size."),
    llvm::cl::init(false));

static llvm::cl::opt<unsigned> perfConfigTimeout(
    "perf-config-timeout",
    llvm::cl::desc(
        "Per-perf-config compilation timeout in seconds. 0 (default) compiles "
        "in-process using worker threads. When > 0, each config is compiled in "
        "a separate rocmlir-driver process that is killed if it exceeds this "
        "budget; the timed-out config is skipped and reported as N/A (tuning "
        "continues with the remaining configs for the problem)."),
    llvm::cl::value_desc("seconds"), llvm::cl::init(0));

// Ripped out of JitRunner.cpp
static OwningOpRef<ModuleOp> parseMLIRInput(StringRef filename,
                                            MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, context);
}

static benchmark::DataType getDataType(Type inputType) {
  if (inputType.isF32()) {
    return benchmark::DataType::F32;
  } else if (inputType.isInteger(32)) {
    return benchmark::DataType::I32;
  } else if (inputType.isF16()) {
    return benchmark::DataType::F16;
  } else if (inputType.isBF16()) {
    return benchmark::DataType::BF16;
  } else if (inputType.isInteger(8)) {
    return benchmark::DataType::I8;
  } else if (isa<Float8E4M3FNUZType, Float8E4M3FNType, Float8E5M2Type,
                 Float8E5M2FNUZType>(inputType)) {
    return benchmark::DataType::F8;
  } else if (isa<Float8E8M0FNUType>(inputType)) {
    return benchmark::DataType::F8E8M0FNU;
  } else if (isa<Float4E2M1FNType>(inputType)) {
    return benchmark::DataType::F4;
  } else {
    llvm::errs() << "Unknown data type: " << inputType << "\n";
    llvm_unreachable("Kernels only accept ints or floats");
  }
}

// intentionally leaky macro
#define HIPCHECK(expr)                                                         \
  do {                                                                         \
    hipError_t _status = (expr);                                               \
    if (hipSuccess != _status) {                                               \
      llvm::errs() << "HIP error at " << __FILE__ << ":" << __LINE__ << " - "  \
                   << hipGetErrorString(_status) << "\n";                      \
      return failure();                                                        \
    }                                                                          \
  } while (0)

static double computeMedian(const std::vector<double> &values) {
  if (values.empty())
    return 0.0;

  assert(std::is_sorted(values.begin(), values.end()) &&
         "values must be sorted");

  size_t n = values.size();
  if (n % 2 == 0) {
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
  }
  // else
  return values[n / 2];
}

static double computeMean(const std::vector<double> &values) {
  if (values.empty())
    return 0.0;

  double sum = 0.0;
  for (double value : values) {
    sum += value;
  }

  return sum / values.size();
}

static double computeStdDev(const std::vector<double> &values, double mean) {
  if (values.size() < 2)
    return 0.0;

  double sumSquares = 0.0;
  for (float val : values) {
    double diff = val - mean;
    sumSquares += diff * diff;
  }

  return std::sqrt(sumSquares / values.size());
}

static std::vector<double> trimValues(const std::vector<double> &values,
                                      unsigned trimPct) {
  if (values.empty() || trimPct == 0)
    return values;

  if (trimPct >= 50)
    return {};

  assert(std::is_sorted(values.begin(), values.end()) &&
         "values must be sorted");

  size_t trimCount = values.size() * trimPct / 100;
  size_t startIdx = trimCount;
  size_t endIdx = values.size() - trimCount;

  return std::vector<double>(values.begin() + startIdx,
                             values.begin() + endIdx);
}

struct BenchmarkParams {
  unsigned warmupMs;
  unsigned repMs;
  bool useMedian;
  unsigned trimPercent;
  unsigned sleepUs;
  bool showStats;
  bool showAllMeasurements;
  rock::TuningParamSetKind tuningSpaceKind;
  const unsigned numCompileThreads;
  std::string benchmarkConfig;
  bool flushLastLevelCache;
  unsigned perfConfigTimeoutSec;
};

enum class CompilationStatus {
  NotApplicable,     // Config not applicable for this kernel
  CompilationFailed, // Config applicable but compilation failed
  TimedOut,          // Compilation exceeded the per-config timeout
  Success            // Successfully compiled
};

struct CompilationResult {
  SmallString<64> perfConfig;
  CompilationStatus status = CompilationStatus::NotApplicable;
  std::string hsacoBinary; // Single HSACO binary containing all kernels
  SmallVector<rock::KernelInfo>
      kernels; // Info for each kernel (name, block/grid sizes)
};

// Create a fresh MLIRContext with all rocMLIR dialects registered.
// A new context is created for each compilation to avoid memory accumulation
// from MLIR's BumpPtrAllocator (which never frees individual allocations).
// Each compilation stores large artifacts (HSACO binaries as StringAttrs,
// LLVM types, etc.) in the context, so reusing a context across many
// compilations causes unbounded memory growth.
//
// Error diagnostics emitted while using this context are buffered into
// `bufferedDiags` rather than printed eagerly. This lets the caller decide,
// after the fact, whether a failure represents a real bug (flush) or an
// expected "not applicable" outcome (drop). `bufferedDiags` must outlive the
// returned context.
static std::unique_ptr<MLIRContext>
createCompilationContext(SmallVector<std::string> &bufferedDiags) {
  DialectRegistry registry;
  registerRocMLIRDialects(registry);
  auto ctx = std::make_unique<MLIRContext>(registry);
  // Consume *all* diagnostics so they don't pollute tuning output during the
  // parallel sweep. Errors are additionally buffered into `bufferedDiags` so
  // the caller can surface them on real failures; warnings/remarks/notes are
  // intentionally dropped (matches the long-standing behavior of this tool).
  ctx->getDiagEngine().registerHandler([&bufferedDiags](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error) {
      std::string buf;
      // `os` holds a reference to `buf`; destroy it (flushing any pending
      // internal buffer into `buf`) before moving `buf` into `bufferedDiags`.
      {
        llvm::raw_string_ostream os(buf);
        os << "Diagnostic error: " << diag << "\n";
        for (auto &note : diag.getNotes())
          os << "  note: " << note << "\n";
      }
      bufferedDiags.push_back(std::move(buf));
    }
    return success();
  });
  return ctx;
}

// Path to the rocmlir-driver binary, which sits next to this executable.
static std::string getRocmlirDriverPath() {
  std::string exePath = llvm::sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&getRocmlirDriverPath));
  SmallString<128> driverPath(llvm::sys::path::parent_path(exePath));
  llvm::sys::path::append(driverPath, "rocmlir-driver");
  return std::string(driverPath);
}

// Compile a single perf config by invoking rocmlir-driver as a subprocess with
// a wall-clock timeout (used when --perf-config-timeout > 0). A config whose
// compilation exceeds `timeoutSec` is killed and reported as TimedOut, which is
// non-fatal: the sweep skips it and keeps tuning the rest. Process isolation is
// what makes the timeout robust: the compile (LLVM backend + in-process LLD)
// cannot be safely interrupted on a worker thread, but a child process can be
// killed cleanly.
static CompilationResult
compileConfigViaSubprocess(StringRef perfConfig, StringRef driverPath,
                           StringRef inputPath, StringRef archName,
                           unsigned timeoutSec, std::mutex &outputMutex,
                           std::atomic<bool> &compilationFailed) {
  CompilationResult result;
  result.perfConfig = perfConfig;

  auto fail = [&](const llvm::Twine &header) {
    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::errs() << header << "\n";
    result.status = CompilationStatus::CompilationFailed;
    compilationFailed.store(true, std::memory_order_relaxed);
  };

  // Per-config output file holding the compiled module.
  SmallString<128> outputPath;
  if (llvm::sys::fs::createTemporaryFile("rocmlir-tuning-out", "mlir",
                                         outputPath)) {
    fail("Failed to create temp output file for config: " + perfConfig);
    return result;
  }
  llvm::FileRemover outputRemover(outputPath);

  std::string archArg = ("--arch=" + archName).str();
  std::string perfConfigArg = ("--perf-config=" + perfConfig).str();
  SmallVector<StringRef, 8> args = {
      driverPath, inputPath,     "--kernel-pipeline=gpu,triton,binary",
      archArg,    perfConfigArg, "-o",
      outputPath};

  // Discard the child's stdout and stderr (empty path => /dev/null): stdout
  // would corrupt the results stream, stderr would spam per-config diagnostics.
  // The exit code already tells us what happened.
  std::optional<StringRef> redirects[3] = {std::nullopt, StringRef(""),
                                           StringRef("")};

  std::string execErr;
  bool execFailed = false;
  auto start = std::chrono::steady_clock::now();
  int rc = llvm::sys::ExecuteAndWait(driverPath, args, /*Env=*/std::nullopt,
                                     redirects, /*SecondsToWait=*/timeoutSec,
                                     /*MemoryLimit=*/0, &execErr, &execFailed);
  double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();

  if (execFailed || rc == -1) {
    fail("Failed to launch rocmlir-driver for config: " + perfConfig + " (" +
         execErr + ")");
    return result;
  }

  // ExecuteAndWait returns -2 for both a timeout-kill and a crash. Disambiguate
  // using the measured wall time: at/over budget => timeout. A timeout is a
  // non-fatal skip reported as N/A, so stay silent here (tuningRunner.py parses
  // stdout/stderr and must not see extra noise).
  if (rc == -2) {
    if (elapsed >= static_cast<double>(timeoutSec))
      result.status = CompilationStatus::TimedOut;
    else
      fail("rocmlir-driver crashed for config: " + perfConfig);
    return result;
  }

  if (rc == rock::kExitNotApplicable) {
    result.status = CompilationStatus::NotApplicable;
    return result;
  }

  if (rc != 0) {
    fail("rocmlir-driver failed (exit " + llvm::Twine(rc) +
         ") for config: " + perfConfig);
    return result;
  }

  // Success: parse the compiled module and pull out the HSACO + kernel info,
  // mirroring the in-process path.
  SmallVector<std::string> bufferedDiags;
  auto ctx = createCompilationContext(bufferedDiags);
  OwningOpRef<ModuleOp> compiled =
      parseSourceFile<ModuleOp>(StringRef(outputPath), ctx.get());
  if (!compiled || !*compiled) {
    fail("Failed to parse compiled module for config: " + perfConfig);
    return result;
  }

  SmallVector<rock::KernelInfo> localKernels;
  if (failed(rock::collectKernelInfo(compiled.get(), localKernels)) ||
      localKernels.empty()) {
    fail("Failed to collect kernel info for config: " + perfConfig);
    return result;
  }

  auto hsacoAttr = compiled.get()->getAttrOfType<StringAttr>("triton.hsaco");
  if (!hsacoAttr) {
    fail("No triton.hsaco found for config: " + perfConfig);
    return result;
  }

  result.hsacoBinary = hsacoAttr.getValue().str();
  result.kernels = std::move(localKernels);
  result.status = CompilationStatus::Success;
  return result;
}

static LogicalResult
measureKernel(unsigned iterations, hipStream_t stream,
              const std::vector<hipFunction_t> &functions,
              ArrayRef<uint32_t> blockSizes, ArrayRef<uint32_t> gridSizes,
              ArrayRef<uint32_t> numCTAsList, std::vector<void *> &argPointers,
              std::vector<double> &measurements, bool useLastLevelCacheSize) {
  // Pre-allocate one event pair per iteration so we can record them all in a
  // tight loop and synchronize only once at the end. This matches Triton's
  // do_bench, which minimizes host-side overhead between launches (no
  // per-iteration synchronization).
  std::vector<hipEvent_t> startEvents(iterations, nullptr);
  std::vector<hipEvent_t> stopEvents(iterations, nullptr);
  llvm::scope_exit eventCleanup([&]() {
    for (hipEvent_t event : startEvents) {
      if (event)
        (void)hipEventDestroy(event);
    }
    for (hipEvent_t event : stopEvents) {
      if (event)
        (void)hipEventDestroy(event);
    }
  });
  for (unsigned iter = 0; iter < iterations; ++iter) {
    HIPCHECK(hipEventCreate(&startEvents[iter]));
    HIPCHECK(hipEventCreate(&stopEvents[iter]));
  }

  // Record all iterations back-to-back. Each measurement brackets the full
  // kernel chain (one start before, one stop after), matching how Triton times
  // the whole callable rather than each kernel individually.
  for (unsigned iter = 0; iter < iterations; ++iter) {
    if (failed(flushInstructionCache(stream))) {
      return failure();
    }
    if (failed(flushCache(stream, useLastLevelCacheSize))) {
      return failure();
    }

    HIPCHECK(hipEventRecord(startEvents[iter], stream));
    for (auto [func, blockSize, gridSize, numCTAs] :
         llvm::zip(functions, blockSizes, gridSizes, numCTAsList)) {
      if (failed(launchKernel(func, gridSize, blockSize,
                              /*shared_memory=*/0, numCTAs, stream,
                              argPointers.data())))
        return failure();
    }
    HIPCHECK(hipEventRecord(stopEvents[iter], stream));
  }

  // Single synchronization after all iterations have been queued.
  HIPCHECK(hipStreamSynchronize(stream));

  for (unsigned iter = 0; iter < iterations; ++iter) {
    float currentMilliseconds = 0.0;
    HIPCHECK(hipEventElapsedTime(&currentMilliseconds, startEvents[iter],
                                 stopEvents[iter]));
    measurements.push_back(static_cast<double>(currentMilliseconds));
  }

  return success();
}

// In order to match rocprof, returns time in nanoseconds
static FailureOr<double> benchmarkKernels(const CompilationResult &result,
                                          ArrayRef<void *> hostBuffers,
                                          MutableArrayRef<void *> gpuBuffers,
                                          ArrayRef<size_t> bufferSizes,
                                          const BenchmarkParams &params) {
  const auto &hsacoBinary = result.hsacoBinary;
  const auto &kernels = result.kernels;

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));
  llvm::scope_exit streamCleanup([&]() {
    hipError_t destroyStatus = hipStreamDestroy(stream);
    if (destroyStatus != hipSuccess) {
      llvm::errs() << "HIP error in hipStreamDestroy: "
                   << hipGetErrorString(destroyStatus) << "\n";
    }
  });

  // Initialize device buffers
  for (size_t i = 0; i < bufferSizes.size(); i++) {
    HIPCHECK(hipMemcpyAsync(gpuBuffers[i], hostBuffers[i], bufferSizes[i],
                            hipMemcpyHostToDevice, stream));
  }

  // HIP wants an array of pointers to each argument.
  // ResolveKernelLaunchParams has already stripped unused workspace args from
  // the kernel signature, so gpuBuffers maps 1:1 to kernel arguments.
  std::vector<void *> argPointers;
  for (void *&item : gpuBuffers) {
    argPointers.push_back(reinterpret_cast<void *>(&item));
  }

  // Load ONE module from the single HSACO binary (contains all kernels)
  hipModule_t module = nullptr;
  std::vector<hipFunction_t> functions;
  llvm::scope_exit moduleCleanup([&]() {
    if (module) {
      hipError_t status = hipModuleUnload(module);
      if (status != hipSuccess) {
        llvm::errs() << "HIP error in hipModuleUnload: "
                     << hipGetErrorString(status) << "\n";
      }
    }
  });

  // Load the single HSACO binary as a HIP module
  HIPCHECK(hipModuleLoadData(&module, hsacoBinary.data()));

  // Get each kernel function from the module by name
  for (const rock::KernelInfo &kernel : kernels) {
    hipFunction_t func;
    HIPCHECK(hipModuleGetFunction(&func, module, kernel.name.c_str()));
    functions.push_back(func);
  }

  // Extract block, grid sizes and numCTAs from kernel info.
  // Shared memory is statically baked into the binary by
  // ResolveKernelLaunchParams.
  SmallVector<uint32_t> blockSizes, gridSizes, numCTAsList;
  for (const rock::KernelInfo &kernel : kernels) {
    blockSizes.push_back(static_cast<uint32_t>(kernel.blockSize));
    gridSizes.push_back(static_cast<uint32_t>(kernel.gridSize));
    numCTAsList.push_back(static_cast<uint32_t>(kernel.clusterSize));
  }

  // Sleep guard to avoid GPU throttling
  llvm::scope_exit sleepGuard([&params] {
    if (params.sleepUs > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(params.sleepUs));
    }
  });

  // Estimate the per-launch runtime so we can size warmup/benchmark iteration
  // counts from the requested time budgets (Triton do_bench style). We time a
  // handful of launches (flushing caches between them) using a single event
  // pair.
  constexpr unsigned estimateRuns = 5;
  double estimateMs = 0.0;
  {
    hipEvent_t startEvent, stopEvent;
    HIPCHECK(hipEventCreate(&startEvent));
    HIPCHECK(hipEventCreate(&stopEvent));

    HIPCHECK(hipEventRecord(startEvent, stream));
    for (unsigned iter = 0; iter < estimateRuns; ++iter) {
      // The cache flushes are inside the timed window here (unlike the actual
      // measurement loop, which flushes before recording the start event). This
      // is intentional, to match Triton's do_bench, whose estimate loop also
      // includes the cache clear.
      if (failed(flushInstructionCache(stream)))
        return failure();
      if (failed(flushCache(stream, params.flushLastLevelCache)))
        return failure();
      for (auto [func, blockSize, gridSize, numCTAs] :
           llvm::zip(functions, blockSizes, gridSizes, numCTAsList)) {
        if (failed(launchKernel(func, gridSize, blockSize,
                                /*shared_memory=*/0, numCTAs, stream,
                                argPointers.data())))
          return failure();
      }
    }
    HIPCHECK(hipEventRecord(stopEvent, stream));
    HIPCHECK(hipStreamSynchronize(stream));

    float elapsedMs = 0.0;
    HIPCHECK(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    HIPCHECK(hipEventDestroy(stopEvent));
    HIPCHECK(hipEventDestroy(startEvent));

    estimateMs = static_cast<double>(elapsedMs) / estimateRuns;
    // hipEventElapsedTime can return tiny/negative values for very fast kernels
    // due to GPU clock precision. Clamp to the documented ~1 microsecond
    // resolution (0.001 ms) to avoid divide-by-zero / overflow below.
    constexpr double minMeasurableMs = 0.001;
    if (estimateMs < minMeasurableMs)
      estimateMs = minMeasurableMs;
  }

  // Derive iteration counts from the time budgets, like Triton's do_bench.
  unsigned nWarmup = std::max<unsigned>(
      1, static_cast<unsigned>(params.warmupMs / estimateMs));
  unsigned iterations =
      std::max<unsigned>(1, static_cast<unsigned>(params.repMs / estimateMs));

  // Warm-up (untimed): just run the kernel chain nWarmup times.
  for (unsigned iter = 0; iter < nWarmup; ++iter) {
    for (auto [func, blockSize, gridSize, numCTAs] :
         llvm::zip(functions, blockSizes, gridSizes, numCTAsList)) {
      if (failed(launchKernel(func, gridSize, blockSize,
                              /*shared_memory=*/0, numCTAs, stream,
                              argPointers.data())))
        return failure();
    }
  }
  HIPCHECK(hipStreamSynchronize(stream));

  // Measure runs
  std::vector<double> measurements;

  if (failed(measureKernel(iterations, stream, functions, blockSizes, gridSizes,
                           numCTAsList, argPointers, measurements,
                           params.flushLastLevelCache)))
    return failure();

  if (params.showAllMeasurements) {
    llvm::outs() << "[";
    for (size_t i = 0; i < measurements.size(); ++i) {
      if (i > 0)
        llvm::outs() << ",";
      llvm::outs() << measurements[i];
    }
    llvm::outs() << "]\t";
  }

  std::sort(measurements.begin(), measurements.end());

  if (params.showStats) {
    if (measurements.size() > 1) {
      float median = computeMedian(measurements);
      float min = measurements.front();
      float max = measurements.back();
      float mean = computeMean(measurements);
      float stdDev = computeStdDev(measurements, mean);
      float coefficientOfVariation = (mean > 0) ? (stdDev / mean * 100) : 0;
      llvm::outs() << "{\"min\":" << min << ",\"median\":" << median
                   << ",\"max\":" << max << ",\"stddev\":" << stdDev
                   << ",\"cv\":" << coefficientOfVariation << "}\t";
    }
  }

  auto msToNs = [](double ms) { return 1e6 * ms; };
  if (params.useMedian)
    return msToNs(computeMedian(measurements));
  else
    return msToNs(computeMean(trimValues(measurements, params.trimPercent)));
}

static int toKernelOrder(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr); intAttr)
    return intAttr.getInt();
  return -1;
}

static LogicalResult extractFuncOps(ModuleOp op,
                                    SmallVectorImpl<func::FuncOp> &kernels) {
  if (!op->hasAttr(rock::ArchAttr::getMnemonic())) {
    return op->emitOpError("no architecture set, set arch on the input module");
  }
  op.walk([&kernels](func::FuncOp f) {
    Attribute kernel = f->getAttr(rock::KernelAttr::getMnemonic());
    if (!kernel)
      return;
    kernels.push_back(f);
  });

  std::sort(kernels.begin(), kernels.end(),
            [](const func::FuncOp &a, const func::FuncOp &b) {
              int kernelA =
                  toKernelOrder(a->getAttr(rock::KernelAttr::getMnemonic()));
              int kernelB =
                  toKernelOrder(b->getAttr(rock::KernelAttr::getMnemonic()));
              return kernelA < kernelB;
            });
  return success();
}

static bool doesModuleHaveFusions(ModuleOp module) {
  WalkResult result = module.walk([](Operation *op) {
    // Check for fusion op or rock.reduce (standalone fusion ops)
    if (rock::isFusionOp(op) || isa<rock::ReduceOp>(op)) {
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

static LogicalResult runTuningLoop(ModuleOp source) {
  // Verify prerequisites
  SmallVector<func::FuncOp> funcs;
  if (failed(extractFuncOps(source, funcs)))
    return failure();

  SmallVector<size_t> bufferLengths;
  ArrayRef<Type> argTypes = funcs[0].getArgumentTypes();
  for (Type argType : argTypes) {
    auto shapedTy = dyn_cast<ShapedType>(argType);
    if (!shapedTy) {
      return funcs[0].emitOpError("all kernel inputs must be shaped types");
    }
    if (!shapedTy.hasStaticShape()) {
      return funcs[0].emitOpError(
          "all kernel arguments must have static shape");
    }
    int64_t sizeInBits =
        shapedTy.getNumElements() * shapedTy.getElementTypeBitWidth();
    bufferLengths.push_back(llvm::divideCeil(sizeInBits, 8));
  }

  // 2. Set up compilation options (shared across all threads)
  rock::KernelOptions kernelOpts;

  RocmDeviceName deviceName;
  StringRef archName =
      source->getAttrOfType<StringAttr>(rock::ArchAttr::getMnemonic())
          .getValue();
  if (failed(deviceName.parse(archName)))
    return source->emitOpError("could not parse arch name: " + archName);
  kernelOpts.arch = deviceName.getChip().str();

  // 3. Initialize host buffers and allocate device buffers
  std::vector<void *> hostBuffers;
  std::vector<void *> gpuBuffers;
  llvm::scope_exit bufferCleanup([&]() {
    for (void *buffer : hostBuffers)
      free(buffer);
    for (void *buffer : gpuBuffers) {
      // hipFree does not allow nullptrs, so make sure to check for it first
      if (!buffer)
        continue;
      hipError_t status = hipFree(buffer);
      if (status != hipSuccess) {
        llvm::errs() << "HIP error in hipFree(buffer): "
                     << hipGetErrorString(status) << "\n";
      }
    }
    if (failed(cleanupCacheFlushArtifacts())) {
      llvm::errs() << "Failed to cleanup cache flush artifacts\n";
    }
  });
  assert(argTypes.size() == bufferLengths.size() &&
         "number of arguments and buffer lengths must match");
  for (auto [argType, bufferLength] : llvm::zip(argTypes, bufferLengths)) {
    benchmark::DataType type = getDataType(getElementTypeOrSelf(argType));
    void *hostBuffer = benchmark::allocAndFill(type, bufferLength);
    void *gpuBuffer = nullptr;
    hipError_t hipStatus = hipMalloc(&gpuBuffer, bufferLength);
    if (hipStatus != hipSuccess) {
      free(hostBuffer);
      llvm::errs() << "HIP error in hipMalloc(gpuBuffer): "
                   << hipGetErrorString(hipStatus) << "\n";
      return failure();
    }
    hostBuffers.push_back(hostBuffer);
    gpuBuffers.push_back(gpuBuffer);
  }

  // 4. Multi-iteration tuning loop
  SmallString<64> bestConfigOverall;
  float bestTimeOverall = std::numeric_limits<float>::max();

  // NOTE: Compilation (PassManager::run()) resets the cl opts, so we have to
  // save the values.
  const BenchmarkParams benchmarkParams = {warmup,
                                           rep,
                                           useMedian,
                                           trimPercent,
                                           sleepUs,
                                           showStats,
                                           showAllMeasurements,
                                           tuningSpaceKind,
                                           numCompileThreads,
                                           benchmarkConfig,
                                           flushLastLevelCache,
                                           perfConfigTimeout};

  unsigned numTuningIterations =
      rock::getNumberOfIterations(benchmarkParams.tuningSpaceKind);
  if (!benchmarkParams.benchmarkConfig.empty() && numTuningIterations != 1) {
    llvm::errs() << "benchmarking should do a single tuning iteration\n";
    return failure();
  }

  // Main iteration loop - wraps config generation, compilation, AND
  // benchmarking
  for (unsigned iterIdx = 0; iterIdx < numTuningIterations; ++iterIdx) {
    // PHASE 1: Collect perf configs for this iteration
    std::vector<SmallString<64>> configs;

    if (!benchmarkParams.benchmarkConfig.empty()) {
      // Benchmark mode - just one config
      configs.emplace_back(benchmarkParams.benchmarkConfig);
    } else {
      // Tuning mode - get configs from tuning space
      rock::TuningParamSpaceSettings settings{iterIdx, bestConfigOverall};
      std::unique_ptr<rock::TuningParamSet> tuningSpace(
          rock::createTunableParamSpace(source, benchmarkParams.tuningSpaceKind,
                                        settings));

      if (tuningSpace->tuningRange.empty()) {
        llvm::errs() << "Tuning range is empty for iteration " << iterIdx
                     << "\n";
        return failure();
      }

      for (rock::RockTuningParamAttrInterface tuningAttr :
           tuningSpace->tuningRange) {
        SmallString<64> perfConfig;
        tuningAttr.getPerfConfigStr(perfConfig);
        configs.push_back(perfConfig);
      }
    }

    // Determine number of parallel threads
    unsigned numThreads = (benchmarkParams.numCompileThreads > 0)
                              ? benchmarkParams.numCompileThreads
                              : std::thread::hardware_concurrency();
    if (numThreads == 0)
      numThreads = 4; // fallback

    // Don't create more threads than configs to compile
    numThreads = std::min(numThreads, static_cast<unsigned>(configs.size()));

    // Serialize source module once (shared by all threads for parsing)
    std::string sourceModuleStr;
    {
      llvm::raw_string_ostream sourceOs(sourceModuleStr);
      source->print(sourceOs);
    }

    // In subprocess mode (--perf-config-timeout > 0), materialize the shared
    // source module to a temp file once; each rocmlir-driver invocation reads
    // it and injects its own perf config via --perf-config.
    SmallString<128> sharedInputPath;
    std::optional<llvm::FileRemover> sharedInputRemover;
    std::string driverPath;
    if (benchmarkParams.perfConfigTimeoutSec > 0) {
      driverPath = getRocmlirDriverPath();
      int inputFd = -1;
      if (llvm::sys::fs::createTemporaryFile("rocmlir-tuning-in", "mlir",
                                             inputFd, sharedInputPath)) {
        llvm::errs() << "Failed to create temp input file for tuning\n";
        return failure();
      }
      sharedInputRemover.emplace(sharedInputPath);
      llvm::raw_fd_ostream inputOs(inputFd, /*shouldClose=*/true);
      inputOs << sourceModuleStr;
    }

    // PHASE 2: Parallel compilation phase.
    // Each compilation creates a fresh MLIRContext that is destroyed when the
    // compilation finishes. This prevents unbounded memory growth from MLIR's
    // BumpPtrAllocator (which never frees individual allocations). Each
    // compilation stores large artifacts (HSACO binaries as StringAttrs,
    // LLVM types, etc.) in the context, so reusing a context across many
    // compilations causes the process to consume ever-increasing RAM.
    std::vector<CompilationResult> compilationResults(configs.size());
    std::mutex outputMutex; // For thread-safe console output
    std::atomic<bool> compilationFailed{
        false}; // Flag to signal early termination

    // Compile a single config with a fresh MLIRContext
    auto compileConfig = [&](size_t idx) -> CompilationResult {
      CompilationResult result;
      result.perfConfig = configs[idx];

      // Create a fresh context for this compilation. It will be destroyed
      // when this lambda returns, freeing all accumulated MLIR data.
      // `bufferedDiags` collects error diagnostics so we can suppress them
      // for expected "not applicable" failures and surface them otherwise.
      SmallVector<std::string> bufferedDiags;
      auto ctx = createCompilationContext(bufferedDiags);
      // Report a real failure: print the per-path header followed by any
      // buffered diagnostics atomically under outputMutex, mark the result
      // as failed, and signal early termination to peer workers.
      auto reportFailure = [&](const llvm::Twine &header) {
        {
          std::lock_guard<std::mutex> lock(outputMutex);
          llvm::errs() << header << "\n";
          for (auto &msg : bufferedDiags)
            llvm::errs() << msg;
        }
        result.status = CompilationStatus::CompilationFailed;
        compilationFailed.store(true, std::memory_order_relaxed);
      };
      OwningOpRef<ModuleOp> sourceModule =
          parseSourceString<ModuleOp>(sourceModuleStr, ctx.get());
      if (!sourceModule || !*sourceModule) {
        reportFailure("Failed to parse source module for config: " +
                      llvm::Twine(result.perfConfig));
        return result;
      }

      // Helper to copy IR with perf config set
      auto copyIR = [&](ModuleOp src,
                        StringAttr attr) -> OwningOpRef<ModuleOp> {
        OwningOpRef<ModuleOp> copy = cast<ModuleOp>(src->clone());
        copy->walk([&attr](rock::RockGemmWrapperInterface op) {
          op->setAttr("perf_config", attr);
        });
        copy->walk([&attr](rock::RockGemmGemmWrapperInterface op) {
          op->setAttr("perf_config", attr);
        });
        return copy;
      };

      if (doesModuleHaveFusions(sourceModule.get()) &&
          !rock::isModuleFusible(sourceModule.get(), result.perfConfig)) {
        result.status = CompilationStatus::NotApplicable;
        return result;
      }

      // Pipeline: a single PassManager runs the full kernel + Triton + backend
      // lowering. Failures are classified post-hoc by checking whether any
      // rock pass set the `rock.not_applicable` marker on the module.
      PassManager pm(sourceModule.get()->getName(),
                     PassManager::Nesting::Implicit);

      rock::BackendOptions backendOpts;
      backendOpts.triple = deviceName.getTriple().str();
      backendOpts.chip = deviceName.getChip().str();
      std::string backendFeatures = deviceName.getFeaturesForBackend();
      backendOpts.features = backendFeatures;
      backendOpts.optLevel = 3;

      rock::TritonOptions tritonOpts;
      tritonOpts.arch = backendOpts.chip;

      StringAttr perfConfigStrAttr =
          StringAttr::get(ctx.get(), result.perfConfig);
      // Parse perfConfig (handles both GemmParamsAttr and GemmGemmParamsAttr)
      if (failed(fillCompilationConfigs(ctx.get(), result.perfConfig,
                                        tritonOpts, backendOpts))) {
        reportFailure("Failed to parse perfConfig for config: " +
                      llvm::Twine(result.perfConfig));
        return result;
      }

      rock::buildKernelPipeline(pm, kernelOpts);
      rock::buildTritonPipeline(pm, tritonOpts);
      rock::buildBackendPipeline(pm, backendOpts);

      OwningOpRef<ModuleOp> sourceCopy =
          copyIR(sourceModule.get(), perfConfigStrAttr);
      if (failed(pm.run(sourceCopy.get()))) {
        if (sourceCopy.get()->hasAttr(rock::NotApplicableAttr::getMnemonic())) {
          result.status = CompilationStatus::NotApplicable;
        } else {
          reportFailure("Compilation pipeline failed for config: " +
                        llvm::Twine(result.perfConfig));
        }
        return result;
      }

      // Collect kernel info from the compiled module (block/grid sizes,
      // argument types).
      SmallVector<rock::KernelInfo> localKernels;
      if (failed(rock::collectKernelInfo(sourceCopy.get(), localKernels))) {
        reportFailure("Failed to collect kernel info for config: " +
                      llvm::Twine(result.perfConfig));
        return result;
      }

      if (localKernels.empty()) {
        reportFailure("No kernels found for config: " +
                      llvm::Twine(result.perfConfig));
        return result;
      }

      // Get the HSACO binary from the compiled module
      auto hsacoAttr =
          sourceCopy.get()->getAttrOfType<StringAttr>("triton.hsaco");
      if (!hsacoAttr) {
        reportFailure("No triton.hsaco found for config: " +
                      llvm::Twine(result.perfConfig));
        return result;
      }

      // Store the HSACO binary and kernel info in the result.
      // This copies the data out of the context before it's destroyed.
      result.hsacoBinary = hsacoAttr.getValue().str();
      result.kernels = std::move(localKernels);

      result.status = CompilationStatus::Success;
      return result;
      // ctx and all MLIR data destroyed here, freeing accumulated memory.
    };

    // Launch parallel compilation tasks with dynamic work stealing.
    // Note: We use atomic counter instead of static partitioning because
    // compilation times vary dramatically between configs (NotApplicable is
    // fast, full compilation is slow). Dynamic work stealing provides better
    // load balancing by allowing fast threads to pick up more work.
    {
      std::atomic<size_t> nextIdx{0};

      auto worker = [&]() {
        while (true) {
          if (compilationFailed.load(std::memory_order_relaxed))
            break;

          size_t idx = nextIdx.fetch_add(1, std::memory_order_relaxed);
          if (idx >= configs.size())
            break;

          if (benchmarkParams.perfConfigTimeoutSec > 0)
            compilationResults[idx] = compileConfigViaSubprocess(
                configs[idx], driverPath, sharedInputPath, archName,
                benchmarkParams.perfConfigTimeoutSec, outputMutex,
                compilationFailed);
          else
            compilationResults[idx] = compileConfig(idx);
        }
      };

      std::vector<std::thread> threads;
      threads.reserve(numThreads);
      for (unsigned i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker);
      }

      for (auto &t : threads) {
        t.join();
      }
    }

    // Check if any compilation failed and terminate early
    if (compilationFailed.load(std::memory_order_relaxed)) {
      llvm::errs()
          << "Compilation failed for one or more configs. Terminating.\n";
      return failure();
    }

    int64_t validResults = 0;
    // Sequential benchmarking phase (must be sequential for accurate timing)
    // Note: Due to early exit on compilation failures, only NotApplicable,
    // TimedOut, and Success statuses are possible here. Timed-out configs are
    // reported identically to not-applicable ones (N/A).
    for (const auto &result : compilationResults) {
      llvm::outs() << result.perfConfig << "\t";

      if (result.status == CompilationStatus::NotApplicable ||
          result.status == CompilationStatus::TimedOut) {
        llvm::outs() << "N/A\n";
        continue;
      }

      assert(result.status == CompilationStatus::Success &&
             "Unexpected compilation status in benchmarking phase");

      FailureOr<double> timing = benchmarkKernels(result, hostBuffers, gpuBuffers,
                                                   bufferLengths, benchmarkParams);

      if (failed(timing)) {
        llvm::errs() << "Kernel execution failed\n";
        return failure();
      }
      llvm::outs() << timing.value() << "\n";

      validResults++;
      // Find best config
      if (rock::needToUpdateBest(benchmarkParams.tuningSpaceKind)) {
        if (timing.value() < bestTimeOverall) {
          bestTimeOverall = timing.value();
          bestConfigOverall = result.perfConfig;
        }
      }
    }

    if (validResults == 0) {
      llvm::errs() << "No valid configurations found\n";
      return failure();
    }
  } // End of iteration loop

  return success();
}
#undef HIPCHECK

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::registerMLIRCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "rocMLIR tuning driver");

  if (trimPercent >= 50) {
    llvm::errs() << "trim-percent must be less than 50 to avoid trimming all "
                    "measurements\n";
    return EXIT_FAILURE;
  }

  DialectRegistry registry;
  registerRocMLIRDialects(registry);
  registerRocMLIRPasses();

  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> source = parseMLIRInput(inputFilename, &ctx);
  if (!source) {
    llvm::errs() << "Could not parse input IR\n";
    return EXIT_FAILURE;
  }

  ModuleOp module;
  WalkResult findModule = source->walk([&](func::FuncOp op) -> WalkResult {
    FailureOr<StringAttr> mayBeArch = rock::getArch(op);
    if (succeeded(mayBeArch)) {
      module = op->getParentOfType<ModuleOp>();
      module->setAttr(rock::ArchAttr::getMnemonic(), mayBeArch.value());
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!findModule.wasInterrupted()) {
    source->emitOpError(
        "no architecture set, set arch on the input module or func");
    llvm::errs() << "Tuning loop failed\n";
    return EXIT_FAILURE;
  }

  if (failed(runTuningLoop(module))) {
    llvm::errs() << "Tuning loop failed\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
