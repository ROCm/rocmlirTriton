//===- rocmlir-tuning-driver.cpp - rocMLIR tuning driver -------------===//
//
// Copyright Advanced Micro Devices, Inc.
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
#include "mlir/Dialect/Rock/Tuning/TuningSearch.h"
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <cerrno>
#include <csignal>
#include <cstring>
#endif

// Utilities to allocate buffers
#include "../utils/performance/common/benchmarkUtils.h"
#include "ArtifactIO.h"
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

#define HIPCHECK_WITH_CONTEXT(expr, context)                                   \
  do {                                                                         \
    hipError_t _status = (expr);                                               \
    if (hipSuccess != _status) {                                               \
      llvm::errs() << __FILE__ << ":" << __LINE__ << ": HIP error in "         \
                   << #expr << ": " << hipGetErrorString(_status) << context   \
                   << "\n";                                                    \
      return failure();                                                        \
    }                                                                          \
  } while (0)

#define HIPCHECK(expr) HIPCHECK_WITH_CONTEXT(expr, "")

// Mirrors _launch() from external/triton/third_party/amd/backend/driver.c
// (lines 603-646). Simplified: gridY/gridZ always 1, blockSize pre-computed,
// launch_cooperative_grid always 0. Returns LogicalResult instead of void.
// Note: hipEventRecord is handled by callers, not by this function.
// Do not move this to tritonUtils.cpp: that utility library is linked into the
// Rock library set, which must stay free of HIP runtime dependencies. Keeping
// the wrapper in this tool confines HIP linking to the tuning driver.
static LogicalResult launchKernel(hipFunction_t function, uint32_t gridX,
                                  uint32_t blockSize, uint32_t shared_memory,
                                  uint32_t num_ctas, hipStream_t stream,
                                  void **params) {
  if (gridX == 0)
    return success();
  if (num_ctas > 1) {
    // ResolveKernelLaunchParams has already verified that the expanded launch
    // dimensions fit in the dispatch packet.
    uint32_t gridBlocks = gridX * num_ctas;

    // Note: driver.c checks hipSymbolTable.hipDrvLaunchKernelEx here because
    // it loads HIP symbols via dlsym. We link directly, so no check needed.
    // Zero-init so the unused bytes of the 64-byte hipLaunchAttributeValue
    // union are well-defined rather than indeterminate padding.
    hipLaunchAttribute attributes[2] = {};
    // Attribute0: Cluster dimensions. HIP's hipLaunchAttributeID enum does not
    // expose this attribute by name (it mirrors CUDA's
    // CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION == 4), so use the raw value, as
    // upstream driver.c does.
    constexpr auto kHipLaunchAttributeClusterDimension =
        static_cast<hipLaunchAttributeID>(4);
    attributes[0].id = kHipLaunchAttributeClusterDimension;
    int *cluster_dims = reinterpret_cast<int *>(attributes[0].val.pad);
    cluster_dims[0] = num_ctas;
    cluster_dims[1] = 1;
    cluster_dims[2] = 1;
    // Attribute1: Cooperative launch
    attributes[1].id = hipLaunchAttributeCooperative;
    attributes[1].val.cooperative = 0;

    HIP_LAUNCH_CONFIG config = {
        gridBlocks,    1,      1,            // Grid size
        blockSize,     1,      1,            // Block size
        shared_memory, stream, attributes, 2 // Number of attributes
    };
    HIPCHECK_WITH_CONTEXT(hipDrvLaunchKernelEx(&config, function, params, 0),
                          " (grid=" << gridBlocks << "x1x1, block=" << blockSize
                                    << "x1x1, shared-memory=" << shared_memory
                                    << " bytes, num-ctas=" << num_ctas << ")");
  } else {
    HIPCHECK_WITH_CONTEXT(hipModuleLaunchKernel(function, gridX, 1, 1,
                                                blockSize, 1, 1, shared_memory,
                                                stream, params, nullptr),
                          " (grid=" << gridX << "x1x1, block=" << blockSize
                                    << "x1x1, shared-memory=" << shared_memory
                                    << " bytes, num-ctas=" << num_ctas << ")");
  }
  return success();
}

static llvm::cl::opt<std::string> inputFilename{
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-")};

static llvm::cl::opt<rock::SearchStrategyKind> tuningSpaceKind(
    "tuning-space", llvm::cl::desc("Tuning space to use for this run"),
    llvm::cl::values(
        clEnumValN(rock::SearchStrategyKind::Quick, "quick",
                   "Quick tuning space"),
        clEnumValN(rock::SearchStrategyKind::Full, "full",
                   "Full tuning space, excluding known-bad configurations"),
        clEnumValN(rock::SearchStrategyKind::Exhaustive, "exhaustive",
                   "Full tuning space widened along a few axes, such as "
                   "numWaves and the K/block tiles"),
        clEnumValN(rock::SearchStrategyKind::LFBO, "lfbo",
                   "Search the exhaustive space with a surrogate model, "
                   "benchmarking only the configs it rates promising")),
    llvm::cl::value_desc("tuning space to use"),
    llvm::cl::init(rock::SearchStrategyKind::Full));

static llvm::cl::opt<rock::LFBOEffort> lfboEffort(
    "lfbo-effort",
    llvm::cl::desc("How much benchmarking the 'lfbo' tuning space may spend "
                   "looking for a fast config. The fixed spaces are as large "
                   "as they are and ignore this."),
    llvm::cl::values(
        clEnumValN(rock::LFBOEffort::Quick, "quick",
                   "A short search, at the risk of settling for a slower "
                   "config than the space holds"),
        clEnumValN(rock::LFBOEffort::Full, "full",
                   "Search until the search stops making progress")),
    llvm::cl::value_desc("search budget"),
    llvm::cl::init(rock::LFBOEffort::Full));

static llvm::cl::opt<std::string> lfboTrace(
    "lfbo-trace",
    llvm::cl::desc("Append a JSON record of every search iteration to this "
                   "file, one object per line: what was measured, what was "
                   "proposed and what it cost. Only the 'lfbo' tuning space "
                   "searches in iterations, so only it writes anything. See "
                   "mlir/utils/performance/analysis/plotLFBOTrace.py."),
    llvm::cl::value_desc("trace file"), llvm::cl::init(""));

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

static llvm::cl::opt<unsigned> gpuRunTimeout(
    "gpu-run-timeout",
    llvm::cl::desc(
        "Per-perf-config GPU-run timeout in seconds. 0 (default) disables the "
        "timeout. This does not include compilation; use --perf-config-timeout "
        "for that. When > 0, timed stream synchronization bounds the "
        "wall-clock "
        "time spent benchmarking each config; a run that exceeds this budget "
        "is "
        "presumed hung and the process is force-exited with a distinct exit "
        "code "
        "(rock::kExitGpuTimeout). tuningRunner.py maps it to a 'gpu timed out' "
        "state and advances to the next problem config."),
    llvm::cl::value_desc("seconds"), llvm::cl::init(0));

static llvm::cl::opt<std::string> compileOnlyDir(
    "compile-only",
    llvm::cl::desc(
        "Compile every perf config and write a kernel bundle "
        "(manifest + compressed HSACO blobs) into <dir> instead of "
        "benchmarking. Performs no GPU work. <dir> is the per-problem "
        "directory; the orchestrator owns index.json."),
    llvm::cl::value_desc("dir"), llvm::cl::init(""));

static llvm::cl::opt<std::string> benchmarkArtifactsDir(
    "benchmark-artifacts",
    llvm::cl::desc(
        "Benchmark the configs in the kernel bundle at <dir> "
        "(written by --compile-only) without recompiling. Decompresses "
        "each HSACO just-in-time. Ignored with a warning if --compile-only "
        "is also given."),
    llvm::cl::value_desc("dir"), llvm::cl::init(""));

// ------------------------------------------------------------
// Two-stage topK tuning options
// ------------------------------------------------------------
// See docs/adaptive_tuning_budget.md for more details.
static llvm::cl::opt<unsigned> twoStageTopK(
    "two-stage-topk",
    llvm::cl::desc(
        "Enable two-stage tuning (enabled when > 0): every config is "
        "benchmarked at a \"cheap\" budget, then we re-benchmark the top K "
        "configs "
        "at the full budget (--rep/--warmup) from their "
        "already-compiled binaries (no recompilation) to pick the winner. "
        "The cheap budget uses an adaptive measurement budget, which runs a "
        "dynamic number of iterations (instead of a fixed amount of time, like "
        "the full budget)."),
    llvm::cl::value_desc("topK size"), llvm::cl::init(0));

static llvm::cl::opt<unsigned> coarseRepIters(
    "coarse-rep-iters",
    llvm::cl::desc("Max number of iterations to measure in the coarse pass."),
    llvm::cl::value_desc("coarse benchmark iterations"), llvm::cl::init(200));

static llvm::cl::opt<unsigned> coarseMinRepIters(
    "coarse-min-rep-iters",
    llvm::cl::desc("Min number of iterations to measure in the coarse pass."),
    llvm::cl::value_desc("minimum coarse measurement iterations"),
    llvm::cl::init(32));

static llvm::cl::opt<double> coarseRelSemTarget(
    "coarse-rel-sem-target",
    llvm::cl::desc("The relative standard error of the mean (SEM) target. "
                   "The coarse pass will stop measuring when the SEM is less "
                   "than this target."),
    llvm::cl::value_desc("relative SEM fraction"), llvm::cl::init(0.005));

static llvm::cl::opt<unsigned> coarseChunkIters(
    "coarse-chunk-iters",
    llvm::cl::desc(
        "Coarse-pass measurement chunk size. "
        "The relative SEM is recomputed after each "
        "chunk of this many measured iterations, i.e. AdaTune-style "
        "micro-batching (AdaTune uses batches of ~50). Smaller chunks check "
        "more often (finer granularity) at the cost of more synchronizations."),
    llvm::cl::value_desc("iterations per SEM check"), llvm::cl::init(32));

static llvm::cl::opt<unsigned> coarseWarmupIters(
    "coarse-warmup-iters",
    llvm::cl::desc(
        "Coarse-pass warmup iteration count. "
        "The actual warmup is "
        "max(this, --coarse-warmup-floor-ms worth of iterations), capped at "
        "the number of warmup iterations --warmup affords for the kernel being "
        "measured, so the coarse pass never warms up longer than the precise "
        "pass it feeds."),
    llvm::cl::value_desc("coarse warmup iterations"), llvm::cl::init(50));

static llvm::cl::opt<unsigned> coarseWarmupFloorMs(
    "coarse-warmup-floor-ms",
    llvm::cl::desc("Minimum coarse-pass warmup time in milliseconds. Must not "
                   "exceed --warmup, which caps the coarse warmup."),
    llvm::cl::value_desc("coarse warmup floor milliseconds"),
    llvm::cl::init(5));

// ------------------------------------------------------------
// end of two-stage topK tuning options
// ------------------------------------------------------------

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
  } else if (inputType.isInteger(4)) {
    return benchmark::DataType::I4;
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

using SteadyTimePoint = std::chrono::steady_clock::time_point;

static std::optional<SteadyTimePoint> makeTimeoutDeadline(unsigned timeoutSec) {
  if (timeoutSec == 0)
    return std::nullopt;
  return std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
}

static bool hasTimedOut(const std::optional<SteadyTimePoint> &deadline) {
  return deadline && std::chrono::steady_clock::now() >= *deadline;
}

static LogicalResult synchronizeStreamWithTimeout(
    hipStream_t stream, const std::optional<SteadyTimePoint> &gpuRunDeadline,
    unsigned timeoutSec, StringRef perfConfig, StringRef phase) {
  // Preserve the historical behavior when no GPU-run timeout was requested:
  // block until all previously queued work on this stream completes.
  if (!gpuRunDeadline) {
    HIPCHECK(hipStreamSynchronize(stream));
    return success();
  }

  // With a timeout enabled, avoid hipStreamSynchronize because it can block
  // forever when a kernel wedges. Instead, poll the same stream with
  // hipStreamQuery.
  while (true) {
    hipError_t status = hipStreamQuery(stream);
    if (status == hipSuccess)
      return success();

    // hipErrorNotReady is the normal "still running" result for an incomplete
    // stream. Any other status is a real HIP failure rather than a timeout.
    if (status != hipErrorNotReady) {
      llvm::errs() << "HIP error while synchronizing stream during " << phase
                   << " for config: " << perfConfig << " - "
                   << hipGetErrorString(status) << "\n";
      return failure();
    }

    // Once the per-config budget is exceeded, treat the current GPU run as
    // hung. We intentionally terminate the process instead of returning
    // failure: after a presumed GPU hang, the in-process HIP context is not
    // trustworthy.
    if (hasTimedOut(gpuRunDeadline)) {
      llvm::errs() << "GPU run timed out after " << timeoutSec << "s during "
                   << phase << " for config: " << perfConfig
                   << " (kernel presumed hung)\n";
      llvm::errs().flush();
      std::_Exit(rock::kExitGpuTimeout);
    }

    // Keep the polling interval short enough to notice hangs promptly while
    // avoiding a hot spin on long-running but healthy kernels.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

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

// Compute relative standard error of the mean (SEM):
//
//   SEM = s / (sqrt(N) * mean),
//
// where s is the standard deviation (divides by N-1, as the benchmarking
// literature does; cf. Georges et al. OOPSLA'07 and Hoefler & Belli SC'15).
//
// This measures how precise our current mean estimate is, and it shrinks
// ~1/sqrt(N) as we take more samples: so we use this as the quantity to
// threshold when deciding we have measured a config enough to rank it.
//
// It differs from the coefficient of variation (s/mean), which measures a
// kernel's intrinsic jitter and does NOT shrink with N. Returns +infinity when
// it cannot be computed yet (fewer than two samples, or a non-positive mean) so
// callers keep measuring.
static double computeRelSem(const std::vector<double> &values) {
  size_t n = values.size();
  if (n < 2)
    return std::numeric_limits<double>::infinity();

  double mean = computeMean(values);
  if (!(mean > 0.0))
    return std::numeric_limits<double>::infinity();

  double sumSquares = 0.0;
  for (double val : values) {
    double diff = val - mean;
    sumSquares += diff * diff;
  }
  double sampleStdDev = std::sqrt(sumSquares / (n - 1));
  return sampleStdDev / (std::sqrt(static_cast<double>(n)) * mean);
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
  rock::SearchStrategyKind tuningSpaceKind;
  unsigned numCompileThreads;
  std::string benchmarkConfig;
  bool flushLastLevelCache;
  unsigned perfConfigTimeoutSec;
  unsigned gpuRunTimeoutSec;
  std::string compileOnlyDir;
  std::string lfboTracePath;
  rock::LFBOEffort lfboEffort = rock::LFBOEffort::Full;

  // Two-stage tuning. twoStageTopK == 0 disables it.
  unsigned twoStageTopK;

  // Active per-call iteration overrides. When non-zero they take precedence
  // over the *Ms budgets above: warmupIters/repIters fix the iteration counts
  // directly instead of deriving them from time. 0 (default) => derive from ms,
  // i.e. the original time-budget behavior. Used to make the coarse pass
  // hardware-independent.
  unsigned warmupIters = 0;
  unsigned repIters = 0;
  // Wall-clock floor under warmupIters, so a fast GPU still gets enough warmup
  // time for DVFS/clock ramp to complete. Only consulted when warmupIters > 0;
  // warmupMs stays the precise warmup budget, which caps the warmup the coarse
  // pass is allowed to spend.
  unsigned warmupFloorMs = 0;
  // Active adaptive-stop overrides. When relSemTarget > 0 the measurement loop
  // runs in chunks of chunkIters and stops once the relative SEM falls below
  // relSemTarget, floored at minRepIters and capped at repIters. relSemTarget
  // == 0 => fixed-iteration measurement (the precise pass leaves it 0). These
  // mirror the warmupIters/repIters override pattern: runBenchmarkPhase sets
  // them from the CoarseConfig for the coarse pass.
  double relSemTarget = 0.0;
  unsigned chunkIters = 0;
  unsigned minRepIters = 0;
};

// Coarse-pass configuration (two-stage tuning only).
struct CoarseConfig {
  unsigned repIters;
  unsigned minRepIters;
  double relSemTarget;
  unsigned chunkIters;
  unsigned warmupIters;
  unsigned warmupFloorMs;
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

  // Capture diagnostics without letting concurrent child processes interleave
  // on the tuning driver's stderr. The file is only read for a fatal child
  // failure; expected NotApplicable and TimedOut results remain silent.
  SmallString<128> stderrPath;
  if (llvm::sys::fs::createTemporaryFile("rocmlir-tuning-err", "log",
                                         stderrPath)) {
    fail("Failed to create temp stderr file for config: " + perfConfig);
    return result;
  }
  llvm::FileRemover stderrRemover(stderrPath);

  std::string archArg = ("--arch=" + archName).str();
  std::string perfConfigArg = ("--perf-config=" + perfConfig).str();
  SmallVector<StringRef, 8> args = {
      driverPath, inputPath,     "--kernel-pipeline=gpu,triton,binary",
      archArg,    perfConfigArg, "-o",
      outputPath};

  // Discard stdout (empty path => /dev/null) because it would corrupt the
  // results stream. Capture stderr per child so fatal diagnostics can be
  // reported without interleaving output from concurrent compilations.
  std::optional<StringRef> redirects[3] = {std::nullopt, StringRef(""),
                                           StringRef(stderrPath)};

  // Launch the child without waiting, then enforce the wall-clock budget from
  // this thread. We deliberately do not use sys::ExecuteAndWait's
  // SecondsToWait: its timeout is implemented with alarm()/SIGALRM (see
  // llvm/lib/Support/Unix/Program.inc), which are process-global and therefore
  // unsafe when many worker threads compile concurrently. The LLVM source
  // documents this with FIXMEs, and concurrent alarm() calls cancel each other
  // so the per-config timeout never reliably fires. Instead we follow the
  // ExecuteNoWait + Wait-poll idiom from ProgramTest.cpp
  // (TestExecuteNoWaitTimeoutPolling): poll with a non-blocking Wait and
  // Polling=true so transient EINTR cannot make LLVM kill the child, then kill
  // the child ourselves once the deadline passes. Each thread only ever waits
  // on its own PID, so this is race-free across workers.
  std::string execErr;
  bool execFailed = false;
  llvm::sys::ProcessInfo procInfo = llvm::sys::ExecuteNoWait(
      driverPath, args, /*Env=*/std::nullopt, redirects,
      /*MemoryLimit=*/0, &execErr, &execFailed);
  if (execFailed || procInfo.Pid == llvm::sys::ProcessInfo::InvalidPid) {
    fail("Failed to launch rocmlir-driver for config: " + perfConfig + " (" +
         execErr + ")");
    return result;
  }

  auto killAndReap = [&]() -> std::string {
#ifdef _WIN32
    // A finite Wait terminates the child with TerminateProcess if it is still
    // running after the timeout.
    const std::optional<unsigned> secondsToWait = 1;
#else
    if (kill(procInfo.Pid, SIGKILL) != 0 && errno != ESRCH)
      return std::string("failed to kill child: ") + std::strerror(errno);
    const std::optional<unsigned> secondsToWait = std::nullopt;
#endif

    std::string reapErr;
    llvm::sys::ProcessInfo reapResult =
        llvm::sys::Wait(procInfo, secondsToWait, &reapErr);
    if (reapResult.Pid != procInfo.Pid) {
      std::string message = "failed to reap child";
      if (!reapErr.empty())
        message += ": " + reapErr;
      return message;
    }
    return {};
  };

  const auto deadline = makeTimeoutDeadline(timeoutSec);
  llvm::sys::ProcessInfo waitResult;
  bool timedOut = false;
  while (true) {
    std::string waitErr;
    // SecondsToWait=0 => non-blocking poll (WNOHANG). Returns Pid==0 while the
    // child is still running, Pid==procInfo.Pid (with ReturnCode) once it
    // exits.
    waitResult = llvm::sys::Wait(procInfo, /*SecondsToWait=*/0, &waitErr,
                                 /*ProcStat=*/nullptr, /*Polling=*/true);
    if (waitResult.Pid == procInfo.Pid)
      break; // Child finished; waitResult.ReturnCode holds its exit status.
    if (waitResult.Pid != 0) {
      // Neither "still running" (0) nor "our child" => a genuine wait error.
      std::string message =
          ("Error waiting for rocmlir-driver for config: " + perfConfig + " (" +
           waitErr + ")")
              .str();
      if (std::string cleanupErr = killAndReap(); !cleanupErr.empty())
        message += "; " + cleanupErr;
      fail(message);
      return result;
    }
    if (hasTimedOut(deadline)) {
      timedOut = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // Budget exceeded: terminate and reap. rocmlir-driver links in-process (no
  // grandchildren to orphan), so cleanup returns promptly. A timeout is a
  // non-fatal skip reported as N/A, so stay silent here (tuningRunner.py parses
  // stdout/stderr and must not see extra noise).
  if (timedOut) {
    if (std::string cleanupErr = killAndReap(); !cleanupErr.empty()) {
      fail("Failed to clean up timed-out rocmlir-driver for config: " +
           perfConfig + " (" + cleanupErr + ")");
      return result;
    }

    result.status = CompilationStatus::TimedOut;
    return result;
  }

  int rc = waitResult.ReturnCode;

  if (rc == rock::kExitNotApplicable) {
    result.status = CompilationStatus::NotApplicable;
    return result;
  }

  if (rc != 0) {
    std::string message = ("rocmlir-driver failed (exit " + llvm::Twine(rc) +
                           ") for config: " + perfConfig)
                              .str();
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> stderrBuffer =
        llvm::MemoryBuffer::getFile(stderrPath);
    if (!stderrBuffer) {
      message += "\nFailed to read child diagnostics: " +
                 stderrBuffer.getError().message();
    } else if (StringRef diagnostics = stderrBuffer.get()->getBuffer().trim();
               !diagnostics.empty()) {
      message += "\n" + diagnostics.str();
    }
    fail(message);
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
              std::vector<double> &measurements, bool useLastLevelCacheSize,
              const std::optional<SteadyTimePoint> &gpuRunDeadline,
              unsigned timeoutSec, StringRef perfConfig) {
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
  if (failed(synchronizeStreamWithTimeout(stream, gpuRunDeadline, timeoutSec,
                                          perfConfig, "measurement")))
    return failure();

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

  // Apply one user-specified GPU-run deadline across all stream
  // synchronization for this perf config. Compilation is not included; this
  // begins after the config has compiled and the HIP stream/module/launch
  // metadata are ready.
  std::optional<SteadyTimePoint> gpuRunDeadline =
      makeTimeoutDeadline(params.gpuRunTimeoutSec);

  // Finish setup work queued before benchmarking (currently the H2D copies)
  // using the same timeout deadline so no stream synchronization can hang
  // indefinitely.
  if (failed(synchronizeStreamWithTimeout(stream, gpuRunDeadline,
                                          params.gpuRunTimeoutSec,
                                          result.perfConfig, "setup")))
    return failure();

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
    if (failed(synchronizeStreamWithTimeout(
            stream, gpuRunDeadline, params.gpuRunTimeoutSec, result.perfConfig,
            "runtime estimation")))
      return failure();

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

  // Derive iteration counts from the time budgets, like Triton's do_bench,
  // UNLESS explicit iteration overrides are given (params.repIters /
  // params.warmupIters > 0). The coarse pass of two-stage tuning uses the
  // iteration overrides so its ranking power is independent of GPU speed. When
  // adaptive stopping is on (params.relSemTarget > 0), `iterations` is
  // interpreted as the cap (maximum) rather than an exact count.
  //
  // An overridden budget is clamped to params.warmupMs so a coarse pass never
  // costs more than the precise pass it feeds.
  unsigned nWarmupFromMs = static_cast<unsigned>(params.warmupMs / estimateMs);
  unsigned nWarmup;
  if (params.warmupIters > 0) {
    unsigned floorIters =
        static_cast<unsigned>(params.warmupFloorMs / estimateMs);
    nWarmup = std::max<unsigned>(
        1, std::min(std::max(params.warmupIters, floorIters), nWarmupFromMs));
  } else {
    nWarmup = std::max<unsigned>(1, nWarmupFromMs);
  }
  // Same clamp for the measurement budget, against params.repMs. It applies to
  // both measurement modes below, since adaptive stopping can be off
  // (relSemTarget == 0) while the iteration override is on.
  unsigned nItersFromMs = static_cast<unsigned>(params.repMs / estimateMs);
  unsigned iterations = std::max<unsigned>(
      1, params.repIters > 0 ? std::min(params.repIters, nItersFromMs)
                             : nItersFromMs);

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
  if (failed(synchronizeStreamWithTimeout(stream, gpuRunDeadline,
                                          params.gpuRunTimeoutSec,
                                          result.perfConfig, "warmup")))
    return failure();

  // Measure runs. Two modes:
  //  * Adaptive (coarse pass, params.relSemTarget > 0): measure in chunks and
  //    stop as soon as the relative standard error of the mean is small enough
  //    to rank this config, floored at params.minRepIters (so the variance
  //    estimate is trustworthy) and capped at `iterations`. Noise, not a fixed
  //    count, decides when to stop, which makes the coarse budget hardware- and
  //    kernel-independent. This follows AdaTune's adaptive evaluator and the
  //    measure-until-precise rule of Georges et al. / Hoefler & Belli.
  //  * Fixed (precise pass, params.relSemTarget == 0): measure exactly
  //    `iterations` runs (the original do_bench-style behavior).
  std::vector<double> measurements;
  if (params.relSemTarget > 0.0) {
    // `iterations` is the cap here. It only binds for noisy, low-intensity
    // kernels whose configs never reach the SEM target.
    const unsigned chunk = std::max<unsigned>(1, params.chunkIters);
    const unsigned minIters = std::min(params.minRepIters, iterations);
    while (measurements.size() < iterations) {
      unsigned remaining =
          iterations - static_cast<unsigned>(measurements.size());
      unsigned thisChunk = std::min(chunk, remaining);
      if (failed(measureKernel(
              thisChunk, stream, functions, blockSizes, gridSizes, numCTAsList,
              argPointers, measurements, params.flushLastLevelCache,
              gpuRunDeadline, params.gpuRunTimeoutSec, result.perfConfig)))
        return failure();
      if (measurements.size() >= minIters &&
          computeRelSem(measurements) < params.relSemTarget)
        break;
    }
  } else {
    if (failed(measureKernel(iterations, stream, functions, blockSizes,
                             gridSizes, numCTAsList, argPointers, measurements,
                             params.flushLastLevelCache, gpuRunDeadline,
                             params.gpuRunTimeoutSec, result.perfConfig)))
      return failure();
  }

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

// Shared timing path used by both the in-process run and --benchmark-artifacts,
// guaranteeing identical measurement semantics. Allocates host/device buffers
// from `layout`, then sequentially benchmarks each config that is due a precise
// measurement, printing `perfConfig\t<ns|Discarded|N/A>`. Timed-out configs are
// reported as N/A, like not-applicable configs. When two-stage tuning is on
// (params.twoStageTopK > 0) a coarse pass parameterized by `coarseConfig` runs
// first and narrows the precise pass to the K fastest configs; the rest are
// reported as Discarded. For artifact results (empty hsacoBinary + nonempty
// blobPath) each HSACO is decompressed just before launch and freed right
// after. A decompression, launch, or timing failure aborts the run.
//
// Returns how many configs were measured. Having measured none is only fatal
// once the whole run is over, which the caller decides: a search strategy may
// well propose a batch that turns out to be entirely inapplicable and still
// have somewhere to go next. When `searchResults` is set, the outcome of every
// config is appended to it in the form the strategy that proposed them expects.
static FailureOr<int64_t>
runBenchmarkPhase(MutableArrayRef<CompilationResult> results,
                  const BufferLayout &layout, const BenchmarkParams &params,
                  const CoarseConfig &coarseConfig,
                  std::vector<rock::BenchmarkResult> *searchResults = nullptr) {
  std::vector<void *> hostBuffers;
  std::vector<void *> gpuBuffers;
  llvm::scope_exit bufferCleanup([&]() {
    for (void *buffer : hostBuffers)
      free(buffer);
    for (void *buffer : gpuBuffers) {
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
  for (auto [byteLength, dataType] :
       llvm::zip(layout.byteLengths, layout.dataTypes)) {
    void *hostBuffer = benchmark::allocAndFill(dataType, byteLength);
    void *gpuBuffer = nullptr;
    hipError_t hipStatus = hipMalloc(&gpuBuffer, byteLength);
    if (hipStatus != hipSuccess) {
      free(hostBuffer);
      llvm::errs() << "HIP error in hipMalloc(gpuBuffer): "
                   << hipGetErrorString(hipStatus) << "\n";
      return failure();
    }
    hostBuffers.push_back(hostBuffer);
    gpuBuffers.push_back(gpuBuffer);
  }

  // Time one already-compiled config. For artifact results the HSACO is
  // decompressed just before launch and freed right after.
  auto timeConfig =
      [&](CompilationResult &result,
          const BenchmarkParams &timingParams) -> FailureOr<double> {
    bool lazyLoaded = result.hsacoBinary.empty() && !result.blobPath.empty();
    if (lazyLoaded) {
      FailureOr<std::string> framed = readFileContents(result.blobPath);
      FailureOr<std::string> raw = failure();
      if (succeeded(framed))
        raw = decompressFramed(*framed);
      if (failed(raw))
        return failure();
      result.hsacoBinary = std::move(*raw);
    }

    FailureOr<double> timing = benchmarkKernels(
        result, hostBuffers, gpuBuffers, layout.byteLengths, timingParams);

    if (lazyLoaded) {
      result.hsacoBinary.clear();
      result.hsacoBinary.shrink_to_fit();
    }
    return timing;
  };

  // Which configs get a precise (reported) benchmark. Only Success carries a
  // code object, so in single-pass mode that is every Success config; in
  // two-stage mode it is narrowed to the coarse-pass shortlist below.
  // Everything else gets a status token instead of a timing: "Discarded" for a
  // config dropped by top-K, "N/A" for one that never became measurable
  // (NotApplicable, CompilationFailed or TimedOut).
  std::vector<bool> reportPrecise(results.size(), false);
  for (size_t i = 0; i < results.size(); ++i)
    reportPrecise[i] = results[i].status == CompilationStatus::Success;

  // Coarse timings of the configs the coarse pass times but top-K does not
  // shortlist for a precise re-run. They are too rough to report, but they are
  // still a measurement, so a search strategy is told about them rather than
  // being left to assume the config failed.
  std::vector<double> coarseTimeByIdx(results.size(),
                                      std::numeric_limits<double>::infinity());

  // Two-stage tuning is only meaningful when we are searching a space; in
  // single-config benchmark mode there is nothing to shortlist.
  if (params.twoStageTopK > 0 && params.benchmarkConfig.empty()) {
    // COARSE PASS: benchmark every applicable config once at the cheap budget.
    // Stats/measurement printing is suppressed so this pass emits no stdout;
    // only the shortlist is reported (with precise timings) below.
    BenchmarkParams coarseParams = params;
    // Adaptive, iteration-based coarse ranking (hardware-independent): stop
    // once the relative SEM is below target, floored at minRepIters and capped
    // at repIters. warmupFloorMs is the DVFS floor layered under warmupIters;
    // warmupMs is deliberately left at the precise budget, which
    // benchmarkKernels uses to cap the coarse warmup. A relSemTarget of 0
    // degrades to a fixed repIters-iteration measurement.
    coarseParams.repIters = coarseConfig.repIters;
    coarseParams.warmupIters = coarseConfig.warmupIters;
    coarseParams.warmupFloorMs = coarseConfig.warmupFloorMs;
    coarseParams.relSemTarget = coarseConfig.relSemTarget;
    coarseParams.chunkIters = coarseConfig.chunkIters;
    coarseParams.minRepIters = coarseConfig.minRepIters;
    coarseParams.showStats = false;
    coarseParams.showAllMeasurements = false;

    llvm::errs() << "Two-stage tuning budget: rep-iters="
                 << coarseParams.repIters
                 << " min-rep-iters=" << coarseParams.minRepIters
                 << " rel-sem-target="
                 << llvm::format("%g", coarseParams.relSemTarget)
                 << " chunk-iters=" << coarseParams.chunkIters
                 << " warmup-iters=" << coarseParams.warmupIters
                 << " warmup-floor-ms=" << coarseParams.warmupFloorMs << "\n";

    SmallVector<std::pair<size_t, double>> coarseTimes; // (index, ns)
    for (size_t i = 0; i < results.size(); ++i) {
      if (!reportPrecise[i])
        continue;
      FailureOr<double> timing = timeConfig(results[i], coarseParams);
      if (failed(timing)) {
        llvm::errs() << "Kernel execution failed (coarse pass) for config: "
                     << results[i].perfConfig << "\n";
        return failure();
      }
      coarseTimes.emplace_back(i, timing.value());
      coarseTimeByIdx[i] = timing.value();
    }

    // Shortlist the K fastest (smallest ns). Everything not shortlisted is
    // demoted to "Discarded" so the downstream winner selection (min time) can
    // only pick a config that was re-benchmarked at the precise budget.
    std::fill(reportPrecise.begin(), reportPrecise.end(), false);
    unsigned k = std::min<unsigned>(params.twoStageTopK, coarseTimes.size());
    std::partial_sort(
        coarseTimes.begin(), coarseTimes.begin() + k, coarseTimes.end(),
        [](const std::pair<size_t, double> &a,
           const std::pair<size_t, double> &b) { return a.second < b.second; });
    for (unsigned j = 0; j < k; ++j)
      reportPrecise[coarseTimes[j].first] = true;
  }

  // PRECISE (reporting) PASS. Sequential for accurate timing. A config that
  // compiled and ran but lost the coarse-pass shortlist is reported as
  // "Discarded", which keeps a deliberate two-stage demotion distinguishable
  // from a config that never produced a measurement at all ("N/A"); the sweep
  // keeps going either way, so one bad config cannot sink the whole run.
  // Re-benchmarking the shortlist reuses the binaries already in `results`, so
  // there is no recompilation.
  int64_t validResults = 0;
  for (size_t i = 0; i < results.size(); ++i) {
    CompilationResult &result = results[i];
    // Record the outcome as the search strategy sees it: a config that never
    // became measurable is a fact about the space, not a gap in it.
    auto recordForSearch = [&](double timeNs) {
      if (!searchResults)
        return;
      rock::BenchmarkResult searchResult;
      searchResult.perfConfig = result.perfConfig;
      searchResult.timeNs = timeNs;
      if (std::isfinite(timeNs))
        searchResult.status = rock::BenchmarkResult::Status::Success;
      else if (result.status == CompilationStatus::NotApplicable)
        searchResult.status = rock::BenchmarkResult::Status::NotApplicable;
      else
        searchResult.status = rock::BenchmarkResult::Status::Failed;
      searchResults->push_back(std::move(searchResult));
    };

    llvm::outs() << result.perfConfig << "\t";

    if (!reportPrecise[i]) {
      llvm::outs() << (result.status == CompilationStatus::Success ? "Discarded"
                                                                   : "N/A")
                   << "\n";
      recordForSearch(coarseTimeByIdx[i]);
      continue;
    }

    FailureOr<double> timing = timeConfig(result, params);
    if (failed(timing)) {
      llvm::errs() << "Kernel execution failed\n";
      return failure();
    }
    llvm::outs() << timing.value() << "\n";
    recordForSearch(timing.value());

    validResults++;
  }

  return validResults;
}

static LogicalResult runTuningLoop(ModuleOp source) {
  // Verify prerequisites
  SmallVector<func::FuncOp> funcs;
  if (failed(extractFuncOps(source, funcs)))
    return failure();

  // Capture per-arg buffer length AND benchmark::DataType so the benchmark
  // host (which has no MLIR) can allocate + fill identical buffers from the
  // manifest. byteLengths matches the existing buffer-length computation.
  BufferLayout layout;
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
    layout.byteLengths.push_back(llvm::divideCeil(sizeInBits, 8));
    layout.dataTypes.push_back(getDataType(getElementTypeOrSelf(argType)));
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

  // Target identity baked into the kernels during lowering (read off the
  // module, no HIP). Recorded in the manifest under --compile-only and checked
  // against the live GPU under --benchmark-artifacts. This is exactly the value
  // used to compute grid sizes.
  int64_t numCUs = rock::getNumCUValueOnFunc(funcs[0]);
  int64_t numChiplets = rock::getNumChipletsValueOnFunc(funcs[0]);

  // Host/device buffers are allocated inside runBenchmarkPhase (the shared
  // timing path), so --compile-only performs no HIP work at all.

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
                                           perfConfigTimeout,
                                           gpuRunTimeout,
                                           compileOnlyDir,
                                           lfboTrace,
                                           lfboEffort,
                                           twoStageTopK};

  const CoarseConfig coarseConfig = {coarseRepIters,     coarseMinRepIters,
                                     coarseRelSemTarget, coarseChunkIters,
                                     coarseWarmupIters,  coarseWarmupFloorMs};

  // The search decides which configs are worth trying. The brute-force spaces
  // hand out everything at once and ignore the timings; an adaptive search such
  // as LFBO uses each batch's timings to choose the next one, so the whole
  // compile-and-benchmark pass below runs once per batch.
  // --benchmark-config pins one config, which is just a search that proposes
  // that config and nothing else.
  std::unique_ptr<rock::TuningSearchStrategy> strategy =
      benchmarkParams.benchmarkConfig.empty()
          ? rock::createTuningSearchStrategy(
                source, benchmarkParams.tuningSpaceKind,
                benchmarkParams.lfboEffort, benchmarkParams.lfboTracePath)
          : rock::createFixedBatchSearchStrategy(
                {rock::PerfConfigString(benchmarkParams.benchmarkConfig)});
  // Only an iterative search reads the timings back, so only then is it worth
  // keeping a result record per config alive alongside the batch.
  const bool feedResultsBack = strategy->isIterative();

  // Asking for a trace of a search that has a single iteration would otherwise
  // leave no file and no explanation.
  if (!benchmarkParams.lfboTracePath.empty() && !feedResultsBack)
    llvm::errs() << "warning: --lfbo-trace: nothing to trace, since this run "
                    "hands out all of its configs in one batch\n";

  // Likewise for a budget this run has no way of spending.
  if (lfboEffort.getNumOccurrences() > 0 && !feedResultsBack)
    llvm::errs() << "warning: --lfbo-effort: nothing to budget, since this "
                    "run benchmarks every config it was going to anyway\n";

  // In --compile-only mode, stream each successful HSACO to disk as soon as
  // it is compiled (freeing the in-memory copy) so peak memory is bounded by
  // the in-flight compiles (~numThreads) rather than the entire config space,
  // which for an exhaustive space can reach many GiB. Prepare the staging
  // bundle up front (this also fails fast if zstd is unavailable, before any
  // compile time is spent). Use the snapshot: PassManager::run() resets the
  // cl::opt globals during compilation.
  const bool streamBlobs = !benchmarkParams.compileOnlyDir.empty();
  if (streamBlobs && feedResultsBack) {
    llvm::errs() << "error: --compile-only cannot be used with the '"
                 << rock::getSearchStrategyKindName(
                        benchmarkParams.tuningSpaceKind)
                 << "' tuning space, which needs the timings of one batch to "
                    "decide what to compile next\n";
    return failure();
  }
  if (streamBlobs &&
      failed(beginArtifactBundle(benchmarkParams.compileOnlyDir)))
    return failure();

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
    if (llvm::sys::fs::createTemporaryFile("rocmlir-tuning-in", "mlir", inputFd,
                                           sharedInputPath)) {
      llvm::errs() << "Failed to create temp input file for tuning\n";
      return failure();
    }
    sharedInputRemover.emplace(sharedInputPath);
    llvm::raw_fd_ostream inputOs(inputFd, /*shouldClose=*/true);
    inputOs << sourceModuleStr;
  }

  const bool moduleHasFusions = doesModuleHaveFusions(source);

  // Timings of the batch just benchmarked, handed back to the search so it can
  // propose the next one. Empty on the first round, as the protocol requires.
  std::vector<rock::BenchmarkResult> prevResults;
  int64_t totalValidResults = 0;
  bool isFirstBatch = true;

  // Main tuning pass: collect perf configs, compile, and benchmark.
  while (true) {
    // PHASE 1: Ask the search what to try next.
    std::vector<rock::PerfConfigString> configs =
        strategy->getPerfConfigBatch(prevResults);
    prevResults.clear();

    if (configs.empty()) {
      if (isFirstBatch) {
        llvm::errs() << "Tuning range is empty\n";
        return failure();
      }
      break;
    }
    isFirstBatch = false;

    // Determine number of parallel threads
    unsigned numThreads = (benchmarkParams.numCompileThreads > 0)
                              ? benchmarkParams.numCompileThreads
                              : std::thread::hardware_concurrency();
    if (numThreads == 0)
      numThreads = 4; // fallback

    // Don't create more threads than configs to compile
    numThreads = std::min(numThreads, static_cast<unsigned>(configs.size()));

    std::vector<bool> configIsFusible(configs.size(), true);
    if (moduleHasFusions) {
      for (size_t idx = 0; idx < configs.size(); ++idx)
        configIsFusible[idx] = rock::isModuleFusible(source, configs[idx]);
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

          CompilationResult result;
          if (!configIsFusible[idx]) {
            result.perfConfig = configs[idx];
            result.status = CompilationStatus::NotApplicable;
          } else {
            result = (benchmarkParams.perfConfigTimeoutSec > 0)
                         ? compileConfigViaSubprocess(
                               configs[idx], driverPath, sharedInputPath,
                               archName, benchmarkParams.perfConfigTimeoutSec,
                               outputMutex, compilationFailed)
                         : compileConfig(idx);
          }

          // Stream the compiled HSACO straight to the staging bundle and drop
          // it from memory. Without this, every Success blob would live in
          // `compilationResults` until serialization at the end, which is the
          // source of the OOM on large --compile-only runs.
          if (streamBlobs && result.status == CompilationStatus::Success) {
            FailureOr<std::string> blobRel = writeArtifactBlob(
                benchmarkParams.compileOnlyDir, idx, result.hsacoBinary);
            if (failed(blobRel)) {
              {
                std::lock_guard<std::mutex> lock(outputMutex);
                llvm::errs() << "Failed to write artifact blob for config: "
                             << result.perfConfig << "\n";
              }
              compilationFailed.store(true, std::memory_order_relaxed);
            } else {
              result.blobPath = std::move(*blobRel);
            }
            // The blob is on disk now (or we are aborting): release the large
            // binary so it does not accumulate across the whole config space.
            std::string().swap(result.hsacoBinary);
          }

          compilationResults[idx] = std::move(result);
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

    // --compile-only: serialize the kernel bundle and return before any HIP
    // call (no hipMalloc, no stream). Only non-iterative searches get here, so
    // this batch is the only one and the bundle is complete.
    if (!benchmarkParams.compileOnlyDir.empty()) {
      bool hasSuccessfulConfig = false;
      for (const CompilationResult &result : compilationResults) {
        if (result.status == CompilationStatus::Success) {
          hasSuccessfulConfig = true;
          break;
        }
      }
      if (!hasSuccessfulConfig) {
        llvm::errs() << "No configurations compiled successfully\n";
        return failure();
      }

      // Blobs were streamed into <dir>.tmp during compilation; finalize just
      // writes the manifest and atomically renames the bundle into place.
      if (failed(finalizeArtifactBundle(benchmarkParams.compileOnlyDir,
                                        compilationResults, layout, archName,
                                        numCUs, numChiplets)))
        return failure();
      return success();
    }

    // Sequential benchmarking phase via the shared timing path. The timings are
    // collected into `prevResults` so an iterative search can learn from this
    // batch before proposing the next one.
    FailureOr<int64_t> validResults = runBenchmarkPhase(
        compilationResults, layout, benchmarkParams, coarseConfig,
        feedResultsBack ? &prevResults : nullptr);
    if (failed(validResults))
      return failure();
    totalValidResults += *validResults;
  } // End tuning pass

  if (totalValidResults == 0) {
    llvm::errs() << "No valid configurations found\n";
    return failure();
  }
  return success();
}

// Benchmark a previously compiled kernel bundle (--benchmark-artifacts). Runs
// the commit / runtime-version / target-identity guardrails, then the shared
// timing path with just-in-time HSACO decompression and per-config crash
// resilience.
static LogicalResult runBenchmarkFromArtifacts(StringRef dir) {
  std::vector<CompilationResult> results;
  BufferLayout layout;
  ManifestInfo info;
  if (failed(loadArtifacts(dir, results, layout, info)))
    return failure();

  if (info.numConfigs != static_cast<int64_t>(results.size())) {
    llvm::errs() << "error: manifest numConfigs=" << info.numConfigs
                 << " but found " << results.size()
                 << " config entries (truncated space?)\n";
    return failure();
  }

  // Runtime-version guardrail (warn only; an incompatible code object will fail
  // hipModuleLoadData anyway, so this just makes that failure actionable).
  int runtimeVersion = 0;
  if (hipRuntimeGetVersion(&runtimeVersion) == hipSuccess) {
    if (static_cast<uint64_t>(runtimeVersion) != info.hipVersion) {
      llvm::errs()
          << "warning: artifact compiled against HIP " << info.hipVersion
          << " but running on HIP runtime " << runtimeVersion
          << " (code-object incompatibility may cause load failures)\n";
    }
  }

  // Target-identity guardrail (always a hard error): arch/numCUs/numChiplets
  // were baked into grid sizes at compile time, so any mismatch would produce
  // wrong launch dimensions and bogus timings.
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, 0));
  StringRef liveArch(props.gcnArchName);
  int64_t liveCU = props.multiProcessorCount;
  int64_t liveChiplets = rock::inferNumChiplets(liveArch, liveCU);

  RocmDeviceName manifestDev, liveDev;
  bool archMismatch = false;
  if (failed(manifestDev.parse(info.arch)) || failed(liveDev.parse(liveArch))) {
    archMismatch = true;
  } else {
    archMismatch =
        manifestDev.getChip() != liveDev.getChip() ||
        manifestDev.getFeaturesForBackend() != liveDev.getFeaturesForBackend();
  }
  bool cuMismatch = info.numCUs != liveCU;
  bool chipletMismatch = info.numChiplets != liveChiplets;
  if (archMismatch || cuMismatch || chipletMismatch) {
    std::string detail;
    llvm::raw_string_ostream d(detail);
    d << "artifact target (arch='" << info.arch << "', numCUs=" << info.numCUs
      << ", numChiplets=" << info.numChiplets << ") != live GPU (arch='"
      << liveArch << "', numCUs=" << liveCU << ", numChiplets=" << liveChiplets
      << ")";
    d.flush();
    llvm::errs() << "error: " << detail
                 << "; grid sizes were baked at compile time, so this would "
                    "produce wrong launches.\n";
    return failure();
  }

  BenchmarkParams benchmarkParams{};
  benchmarkParams.warmupMs = warmup;
  benchmarkParams.repMs = rep;
  benchmarkParams.useMedian = useMedian;
  benchmarkParams.trimPercent = trimPercent;
  benchmarkParams.sleepUs = sleepUs;
  benchmarkParams.showStats = showStats;
  benchmarkParams.showAllMeasurements = showAllMeasurements;
  benchmarkParams.tuningSpaceKind = rock::SearchStrategyKind::Full;
  benchmarkParams.numCompileThreads = numCompileThreads;
  benchmarkParams.benchmarkConfig = benchmarkConfig;
  benchmarkParams.flushLastLevelCache = flushLastLevelCache;
  benchmarkParams.perfConfigTimeoutSec = 0;
  benchmarkParams.gpuRunTimeoutSec = gpuRunTimeout;
  benchmarkParams.compileOnlyDir = "";
  benchmarkParams.twoStageTopK = twoStageTopK;
  const CoarseConfig coarseConfig = {coarseRepIters,     coarseMinRepIters,
                                     coarseRelSemTarget, coarseChunkIters,
                                     coarseWarmupIters,  coarseWarmupFloorMs};
  FailureOr<int64_t> validResults =
      runBenchmarkPhase(results, layout, benchmarkParams, coarseConfig);
  if (failed(validResults))
    return failure();
  if (*validResults == 0) {
    llvm::errs() << "No valid configurations found\n";
    return failure();
  }
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

  if (twoStageTopK > 0 && coarseWarmupFloorMs > warmup) {
    llvm::errs() << "coarse-warmup-floor-ms must not exceed warmup\n";
    return EXIT_FAILURE;
  }

  // --compile-only wins over --benchmark-artifacts: it is the CPU-only phase,
  // so honoring it stays valid even on the GPU-less hosts where the compile
  // phase normally runs, whereas the artifact path needs a device.
  bool runFromArtifacts = !benchmarkArtifactsDir.empty();
  if (!compileOnlyDir.empty() && runFromArtifacts) {
    llvm::errs() << "warning: ignoring --benchmark-artifacts="
                 << benchmarkArtifactsDir
                 << " because --compile-only takes precedence\n";
    runFromArtifacts = false;
  }

  // Only a live artifact run enumerates its configs from the bundle. Once
  // --compile-only has taken over, --benchmark-config is honored as the usual
  // single-config narrowing.
  if (runFromArtifacts && !benchmarkConfig.empty()) {
    llvm::errs() << "--benchmark-config is incompatible with "
                    "--benchmark-artifacts (the artifact already enumerates "
                    "configs)\n";
    return EXIT_FAILURE;
  }

  // --benchmark-artifacts: everything comes from the kernel bundle, so skip
  // parsing the MLIR input and the arch walk entirely.
  if (runFromArtifacts) {
    if (failed(runBenchmarkFromArtifacts(benchmarkArtifactsDir))) {
      llvm::errs() << "Benchmark from artifacts failed\n";
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
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
