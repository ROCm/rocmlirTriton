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
#include "mlir/Dialect/Rock/utility/tritonUtils.h"
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
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <mutex>
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

static llvm::cl::opt<unsigned> numIterations(
    "num-iterations",
    llvm::cl::desc("Number of times to run each kernel for averaging"),
    llvm::cl::value_desc("number of runs"), llvm::cl::init(100));

static llvm::cl::opt<unsigned> warmupIterations(
    "warmup-iterations", llvm::cl::desc("Number of warmup runs"),
    llvm::cl::value_desc("number of warmup runs"), llvm::cl::init(10));

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
        "Print detailed stats (min, max, median, stddev, cv) in JSON format. "
        "In case of small kernels print total_cpu_time and number of "
        "iterations."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> showAllMeasurements(
    "show-all-measurements",
    llvm::cl::desc(
        "Print all individual timing measurements in JSON format. In case of "
        "small kernels print total_cpu_time and number of iterations."),
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

// Cache-flush knobs. See AIROCMLIR-858: the per-iteration cache state changes
// the timing distribution (and the GPU clock ramp), so we expose each flush
// independently and surface the chosen mode in the produced tuning CSV.
static llvm::cl::opt<bool> flushICache(
    "flush-icache",
    llvm::cl::desc("Flush the instruction cache before every measured "
                   "iteration. Disable to measure warm-icache behaviour."),
    llvm::cl::init(true));

static llvm::cl::opt<rocmlir::tuningdriver::L2FlushLevel> flushL2Level(
    "flush-l2",
    llvm::cl::desc("L2 cache flush strategy applied before every measured "
                   "iteration."),
    llvm::cl::values(
        clEnumValN(rocmlir::tuningdriver::L2FlushLevel::All, "all",
                   "Flush the whole L2 (memset a > L2-sized scratch buffer)"),
        clEnumValN(rocmlir::tuningdriver::L2FlushLevel::None, "none",
                   "Do not flush the L2; lets the GPU run with a warm cache"),
        clEnumValN(rocmlir::tuningdriver::L2FlushLevel::Weights, "weights",
                   "Only flush the kernel's weight buffers (heuristic: all "
                   "kernel argument buffers except the last)")),
    llvm::cl::init(rocmlir::tuningdriver::L2FlushLevel::All));

// Timing method (AIROCMLIR-858). Historically the driver auto-selects between
// a CPU-batch timer (small kernels, where per-iter GPU events dominate the
// runtime) and a per-iter GPU event timer (large kernels). The ``--timer``
// flag lets the user force one of those paths, e.g. to time small kernels
// with GPU events for thermal-ramp studies.
enum class TimerKind { Auto, Cpu, Gpu };

static llvm::cl::opt<TimerKind> timer(
    "timer",
    llvm::cl::desc("How to time each measured iteration."),
    llvm::cl::values(
        clEnumValN(TimerKind::Auto, "auto",
                   "Auto-select: CPU batch timer for kernels with average "
                   "warmup runtime < 1ms, GPU event timer otherwise (default, "
                   "matches historical behaviour)"),
        clEnumValN(TimerKind::Cpu, "cpu",
                   "Always use a single CPU timer wrapping all iterations"),
        clEnumValN(TimerKind::Gpu, "gpu",
                   "Always use per-iteration GPU event timers")),
    llvm::cl::init(TimerKind::Auto));

// Per-iteration CPU timing (AIROCMLIR-858 #2). When the CPU timer is in use
// (small kernels under ``--timer=auto`` or any kernel under ``--timer=cpu``)
// the historical behaviour is to wrap *all* iterations in a single
// ``steady_clock`` pair and report the per-iter average. That hides the
// per-iter distribution that the thermal-ramp study needs. With this flag
// each iteration gets its own CPU timer (plus an inline ``hipStreamSynchronize``
// so the timer captures one iteration's worth of GPU work).
static llvm::cl::opt<bool> perIterCpuTiming(
    "per-iter-cpu-timing",
    llvm::cl::desc(
        "Time each iteration individually when the CPU timer is in use, "
        "instead of wrapping the whole batch in one timer pair. Adds a "
        "per-iteration ``hipStreamSynchronize`` so the resulting samples "
        "include the full per-iter cost (kernel + flushes). Defaults to "
        "false, matching the historical small-kernel CPU batch timer."),
    llvm::cl::init(false));

// Ripped out of JitRunner.cpp
static OwningOpRef<ModuleOp> parseMLIRInput(StringRef inputFilename,
                                            MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
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
  unsigned numIterations;
  unsigned warmupIterations;
  bool useMedian;
  unsigned trimPercent;
  unsigned sleepUs;
  bool showStats;
  bool showAllMeasurements;
  rock::TuningParamSetKind tuningSpaceKind;
  const unsigned numCompileThreads;
  std::string benchmarkConfig;
  bool flushICache;
  rocmlir::tuningdriver::L2FlushLevel flushL2Level;
  TimerKind timer;
  bool perIterCpuTiming;
};

// Bundles per-iteration cache-flush controls so the measurement loops don't
// need a long argument list. The weight-buffer ArrayRefs are unused unless
// ``l2Level == Weights``.
struct FlushPolicy {
  bool flushICache;
  rocmlir::tuningdriver::L2FlushLevel l2Level;
  llvm::ArrayRef<void *> weightBuffers;
  llvm::ArrayRef<size_t> weightBufferSizes;
};

enum class CompilationStatus {
  NotApplicable,     // Config not applicable for this kernel
  CompilationFailed, // Config applicable but compilation failed
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

static LogicalResult measureSmallKernel(
    unsigned iterations, hipStream_t stream,
    const std::vector<hipFunction_t> &functions, ArrayRef<uint32_t> blockSizes,
    ArrayRef<uint32_t> gridSizes, ArrayRef<uint32_t> numCTAsList,
    std::vector<void *> &argPointers, std::vector<double> &measurements,
    double &smallKernelCpuMs, bool benchmarkMode,
    const FlushPolicy &flushPolicy, bool perIterCpuTiming) {
  // Two CPU-timed paths:
  //   * Batch (default, historical): one ``steady_clock`` pair wraps all
  //     iterations. ``measurements`` receives a single per-iter average.
  //   * Per-iter (AIROCMLIR-858 #2): one ``steady_clock`` pair per
  //     iteration, with an inline ``hipStreamSynchronize`` to capture this
  //     iteration's GPU work. ``measurements`` receives ``iterations``
  //     samples in temporal order. The added sync makes per-iter samples
  //     noisier for sub-100us kernels, which is why this is opt-in.
  if (perIterCpuTiming) {
    measurements.reserve(measurements.size() + iterations);
    auto iterationStart = std::chrono::steady_clock::now();
    for (unsigned iter = 0; iter < iterations; ++iter) {
      auto iterTimerStart = std::chrono::steady_clock::now();
      // Flushes live inside the timed window so each per-iter sample
      // captures the full cost of starting a fresh iteration (mirrors
      // the batch path's accounting).
      if (!benchmarkMode) {
        if (flushPolicy.flushICache && failed(flushInstructionCache(stream))) {
          return failure();
        }
        if (failed(flushL2Cache(stream, flushPolicy.l2Level,
                                flushPolicy.weightBuffers,
                                flushPolicy.weightBufferSizes))) {
          return failure();
        }
      }
      for (auto [func, blockSize, gridSize, numCTAs] :
           llvm::zip(functions, blockSizes, gridSizes, numCTAsList)) {
        if (failed(rock::launchKernel(func, gridSize, blockSize,
                                      /*shared_memory=*/0, numCTAs, stream,
                                      argPointers.data())))
          return failure();
      }
      HIPCHECK(hipStreamSynchronize(stream));
      auto iterTimerEnd = std::chrono::steady_clock::now();
      measurements.push_back(
          std::chrono::duration<double, std::milli>(iterTimerEnd -
                                                    iterTimerStart)
              .count());
    }
    smallKernelCpuMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - iterationStart)
                           .count();
    return success();
  }

  // Batch CPU timer (historical small-kernel path).
  auto iterationStart = std::chrono::steady_clock::now();
  for (unsigned iter = 0; iter < iterations; ++iter) {
    // Do not flush caches in benchmark mode, as we do not want to
    // time the cache flush (it's okay if we are running in tuning mode).
    if (!benchmarkMode) {
      if (flushPolicy.flushICache && failed(flushInstructionCache(stream))) {
        return failure();
      }
      if (failed(flushL2Cache(stream, flushPolicy.l2Level,
                              flushPolicy.weightBuffers,
                              flushPolicy.weightBufferSizes))) {
        return failure();
      }
    }
    for (auto [func, blockSize, gridSize, numCTAs] :
         llvm::zip(functions, blockSizes, gridSizes, numCTAsList)) {
      if (failed(rock::launchKernel(func, gridSize, blockSize,
                                    /*shared_memory=*/0, numCTAs, stream,
                                    argPointers.data())))
        return failure();
    }
  }

  HIPCHECK(hipStreamSynchronize(stream));
  smallKernelCpuMs = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - iterationStart)
                         .count();
  measurements.push_back(smallKernelCpuMs / iterations);
  return success();
}

static LogicalResult measureLargeKernel(
    unsigned iterations, hipStream_t stream,
    const std::vector<hipFunction_t> &functions, ArrayRef<uint32_t> blockSizes,
    ArrayRef<uint32_t> gridSizes, ArrayRef<uint32_t> numCTAsList,
    std::vector<void *> &argPointers, std::vector<double> &measurements,
    const FlushPolicy &flushPolicy) {
  // Measure runs normally.
  for (unsigned iter = 0; iter < iterations; ++iter) {
    if (flushPolicy.flushICache && failed(flushInstructionCache(stream))) {
      return failure();
    }
    if (failed(flushL2Cache(stream, flushPolicy.l2Level,
                            flushPolicy.weightBuffers,
                            flushPolicy.weightBufferSizes))) {
      return failure();
    }

    double totalMilliseconds = 0.0;

    for (auto [func, blockSize, gridSize, numCTAs] :
         llvm::zip(functions, blockSizes, gridSizes, numCTAsList)) {
      hipEvent_t startEvent, stopEvent;
      HIPCHECK(hipEventCreate(&startEvent));
      HIPCHECK(hipEventCreate(&stopEvent));

      HIPCHECK(hipEventRecord(startEvent, stream));
      if (failed(rock::launchKernel(func, gridSize, blockSize,
                                    /*shared_memory=*/0, numCTAs, stream,
                                    argPointers.data())))
        return failure();
      HIPCHECK(hipEventRecord(stopEvent, stream));
      HIPCHECK(hipEventSynchronize(stopEvent));

      float currentMilliseconds = 0.0;
      HIPCHECK(
          hipEventElapsedTime(&currentMilliseconds, startEvent, stopEvent));

      HIPCHECK(hipEventDestroy(stopEvent));
      HIPCHECK(hipEventDestroy(startEvent));

      totalMilliseconds += static_cast<double>(currentMilliseconds);
    }

    measurements.push_back(totalMilliseconds);
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

  bool benchmarkMode = !params.benchmarkConfig.empty();
  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));
  auto streamCleanup = llvm::make_scope_exit([&]() {
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
  auto moduleCleanup = llvm::make_scope_exit([&]() {
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
  auto sleepGuard = llvm::make_scope_exit([&params] {
    if (params.sleepUs > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(params.sleepUs));
    }
  });

  bool isSmallKernel = false;
  unsigned iterations = params.numIterations;
  // Per-iteration warmup samples. Each entry is the summed time across all
  // kernels for one warmup pass (mirrors how ``totalMillisecondsWarmup``
  // used to be accumulated). Collected unconditionally so the value is
  // available to the unified ``--show-all-measurements`` JSON below
  // (AIROCMLIR-858).
  std::vector<double> warmupMeasurements;

  if (params.warmupIterations > 0) {
    // Warmup run. We measure the warmup to get an estimate of the kernel
    // runtime. We will use this estimate to determine if the kernel is small or
    // not.
    warmupMeasurements.reserve(params.warmupIterations);
    for (unsigned iter = 0; iter < params.warmupIterations; ++iter) {
      double thisWarmupMs = 0.0;
      for (auto [func, blockSize, gridSize, numCTAs] :
           llvm::zip(functions, blockSizes, gridSizes, numCTAsList)) {
        hipEvent_t startEvent, stopEvent;
        HIPCHECK(hipEventCreate(&startEvent));
        HIPCHECK(hipEventCreate(&stopEvent));

        HIPCHECK(hipEventRecord(startEvent, stream));
        if (failed(rock::launchKernel(func, gridSize, blockSize,
                                      /*shared_memory=*/0, numCTAs, stream,
                                      argPointers.data())))
          return failure();
        HIPCHECK(hipEventRecord(stopEvent, stream));

        HIPCHECK(hipStreamSynchronize(stream));

        float currentMilliseconds = 0.0;
        HIPCHECK(
            hipEventElapsedTime(&currentMilliseconds, startEvent, stopEvent));

        HIPCHECK(hipEventDestroy(stopEvent));
        HIPCHECK(hipEventDestroy(startEvent));

        // hipEventElapsedTime seemingly can return negative values for fast
        // kernels due to GPU clock precision issues. This is extremely relevant
        // when we have a small number of warmup iterations (e.g., 1) for small
        // kernels. Clamp to the documented resolution of ~1 microsecond
        // (0.001 ms) if this is the case.
        if (currentMilliseconds < 0.0f) {
          constexpr float minMeasurableMs = 0.001f;
          currentMilliseconds = minMeasurableMs;
        }

        thisWarmupMs += static_cast<double>(currentMilliseconds);
      }
      warmupMeasurements.push_back(thisWarmupMs);
    }
    double totalMillisecondsWarmup = 0.0;
    for (double v : warmupMeasurements)
      totalMillisecondsWarmup += v;
    totalMillisecondsWarmup /= warmupMeasurements.size();
    assert(totalMillisecondsWarmup >= 0.0f &&
           "totalMillisecondsWarmup must be greater than 0");

    // We want to get at least 1ms of kernel execution time
    // (counting all iterations), so increase the number of iterations
    // if necessary.
    constexpr float minTotalMilliseconds = 1.0f;
    iterations = std::max<unsigned>(
        iterations, static_cast<unsigned>(std::ceil(minTotalMilliseconds /
                                                    totalMillisecondsWarmup)));

    // Depending on the runtime of the kernel,
    // we will use a different approach to measure the runs.
    // We consider a kernel to be small if a single iteration takes less than
    // 1ms to run.
    constexpr float smallKernelThreshold = 1.0f;
    isSmallKernel = totalMillisecondsWarmup < smallKernelThreshold;
  }

  // Resolve the effective timer kind. ``--timer=auto`` (the default) keeps
  // the historical small-vs-large heuristic; ``--timer=cpu``/``gpu`` forces
  // the corresponding path so the user can run a thermal-ramp study with a
  // consistent timer across all kernels.
  bool useCpuTimer;
  switch (params.timer) {
  case TimerKind::Cpu:
    useCpuTimer = true;
    break;
  case TimerKind::Gpu:
    useCpuTimer = false;
    break;
  case TimerKind::Auto:
    useCpuTimer = isSmallKernel;
    break;
  }

  // Measure runs
  std::vector<double> measurements;
  double smallKernelCpuMs = 0.0;

  // Build the per-iteration cache-flush policy. For ``--flush-l2=weights`` we
  // hand the L2 flusher every kernel arg except the last one (assumed to be
  // the primary output by rocMLIR convention). When the kernel only has a
  // single argument, leave the weight slice empty so the L2 flush is a no-op
  // for that level.
  llvm::ArrayRef<void *> weightBuffers;
  llvm::ArrayRef<size_t> weightBufferSizes;
  if (gpuBuffers.size() > 1) {
    weightBuffers =
        llvm::ArrayRef<void *>(gpuBuffers.data(), gpuBuffers.size() - 1);
    weightBufferSizes =
        llvm::ArrayRef<size_t>(bufferSizes.data(), bufferSizes.size() - 1);
  }
  const FlushPolicy flushPolicy{params.flushICache, params.flushL2Level,
                                weightBuffers, weightBufferSizes};

  if (useCpuTimer) {
    if (failed(measureSmallKernel(iterations, stream, functions, blockSizes,
                                  gridSizes, numCTAsList, argPointers,
                                  measurements, smallKernelCpuMs, benchmarkMode,
                                  flushPolicy, params.perIterCpuTiming)))
      return failure();
  } else {
    if (failed(measureLargeKernel(iterations, stream, functions, blockSizes,
                                  gridSizes, numCTAsList, argPointers,
                                  measurements, flushPolicy)))
      return failure();
  }

  // True when this run produced a single per-iter-average sample from one
  // batch CPU timer (the historical small-kernel path). Used by the
  // ``--show-stats`` and JSON blocks below since ``isSmallKernel`` is no
  // longer a sufficient proxy after ``--timer`` / ``--per-iter-cpu-timing``.
  const bool isBatchedCpuTimer = useCpuTimer && !params.perIterCpuTiming;

  if (params.showAllMeasurements) {
    // Unified JSON shape (AIROCMLIR-858):
    //   {
    //     "timing_method": "cpu" | "gpu",
    //     "repetitions_per_batch": N,    // 1 for GPU events; ``iterations``
    //                                    // for the CPU batch timer
    //     "warmup": [<ms>, ...],         // per-iter warmup samples
    //     "measurements": [<ms>, ...]    // per-batch measurements (one
    //                                    // entry per batch; for the CPU
    //                                    // timer that's a single entry
    //                                    // equal to the per-iter average)
    //   }
    // Emitted BEFORE the sort below so the array preserves temporal order.
    // Only the batched CPU timer collapses multiple iterations into one
    // sample (``repetitions_per_batch == iterations``); per-iter CPU and
    // per-iter GPU paths each produce one sample per iteration.
    const unsigned repetitionsPerBatch = isBatchedCpuTimer ? iterations : 1u;
    llvm::outs() << "{\"timing_method\":\""
                 << (useCpuTimer ? "cpu" : "gpu")
                 << "\",\"repetitions_per_batch\":" << repetitionsPerBatch
                 << ",\"warmup\":[";
    for (size_t i = 0; i < warmupMeasurements.size(); ++i) {
      if (i > 0)
        llvm::outs() << ",";
      llvm::outs() << warmupMeasurements[i];
    }
    llvm::outs() << "],\"measurements\":[";
    for (size_t i = 0; i < measurements.size(); ++i) {
      if (i > 0)
        llvm::outs() << ",";
      llvm::outs() << measurements[i];
    }
    llvm::outs() << "]}\t";
  }

  // NOTE: this sort is purely local to the median/trim-mean aggregate
  // returned to the runner. The on-disk array emitted above for
  // ``--show-all-measurements`` is in temporal order (AIROCMLIR-858 #11).
  std::sort(measurements.begin(), measurements.end());

  if (params.showStats) {
    // We cannot show the rest of the stats when the batched CPU timer was
    // used because there is only one measurement (the per-iter average), so
    // min/max/stddev are undefined. ``isBatchedCpuTimer`` (rather than the
    // old ``isSmallKernel``) is the correct condition after
    // ``--timer``/``--per-iter-cpu-timing`` are wired in.
    if (isBatchedCpuTimer) {
      llvm::outs() << "{\"total_cpu_time\":" << smallKernelCpuMs
                   << ",\"iterations\":" << iterations << "}\t";
    }
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
  auto bufferCleanup = llvm::make_scope_exit([&]() {
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
  const BenchmarkParams benchmarkParams = {numIterations,
                                           warmupIterations,
                                           useMedian,
                                           trimPercent,
                                           sleepUs,
                                           showStats,
                                           showAllMeasurements,
                                           tuningSpaceKind,
                                           numCompileThreads,
                                           benchmarkConfig,
                                           flushICache,
                                           flushL2Level,
                                           timer,
                                           perIterCpuTiming};

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
      backendOpts.suppressDiagnostic = true;

      rock::TritonOptions tritonOpts;
      tritonOpts.arch = backendOpts.chip;

      StringAttr perfConfigStrAttr =
          StringAttr::get(ctx.get(), result.perfConfig);
      Attribute perfConfigAttr = rock::GemmParamsAttr::get(perfConfigStrAttr);
      if (!perfConfigAttr)
        perfConfigAttr = rock::GemmGemmParamsAttr::get(perfConfigStrAttr);
      // Parse perfConfig
      if (failed(fillCompilationConfigs(perfConfigAttr, tritonOpts,
                                        backendOpts))) {
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
    // Note: Due to early exit on compilation failures, only NotApplicable and
    // Success statuses are possible here.
    for (const auto &result : compilationResults) {
      llvm::outs() << result.perfConfig << "\t";

      if (result.status == CompilationStatus::NotApplicable) {
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
