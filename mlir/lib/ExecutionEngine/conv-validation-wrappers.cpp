//===- conv-validation-wrappers.cpp - conv validation wrapper library -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around convolution validations for easy linking in
// ORC jit.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <numeric>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cmath>
#include <unordered_map>

// Get program load time using function-local static (initialized on first call)
// This measures time from first access, which happens at library load via
// constructor
static std::chrono::steady_clock::time_point &getProgramLoadTime() {
  static std::chrono::steady_clock::time_point loadTime =
      std::chrono::steady_clock::now();
  return loadTime;
}

// Force initialization at library load via constructor attribute
// Disable global-constructors warning for this specific function
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
static struct ProgramLoadTimeInitializer {
  ProgramLoadTimeInitializer() { (void)getProgramLoadTime(); }
} programLoadTimeInit;
#pragma clang diagnostic pop

// Called at the start of main() to measure JIT compilation time
extern "C" void programStart() {
  auto now = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration<double, std::milli>(now - getProgramLoadTime())
          .count();
  printf("JIT compilation time: %.3f ms\n", elapsed);
}

// Timing utilities for CPU validation functions
static std::chrono::steady_clock::time_point cpuTimerStartPoint;

extern "C" void cpuTimerStart() {
  cpuTimerStartPoint = std::chrono::steady_clock::now();
}

extern "C" void cpuTimerStop() {
  auto endPoint = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration<double, std::milli>(endPoint - cpuTimerStartPoint)
          .count();
  printf("CPU validation time: %.3f ms\n", elapsed);
}

// Timing utilities for GPU kernel execution
static std::chrono::steady_clock::time_point gpuTimerStartPoint;

extern "C" void gpuTimerStart() {
  gpuTimerStartPoint = std::chrono::steady_clock::now();
}

extern "C" void gpuTimerStop() {
  auto endPoint = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration<double, std::milli>(endPoint - gpuTimerStartPoint)
          .count();
  printf("GPU kernel time: %.3f ms\n", elapsed);
}

// Timing utilities for memory initialization
static std::chrono::steady_clock::time_point initTimerStartPoint;

extern "C" void initTimerStart() {
  initTimerStartPoint = std::chrono::steady_clock::now();
}

extern "C" void initTimerStop() {
  auto endPoint = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration<double, std::milli>(endPoint - initTimerStartPoint)
          .count();
  printf("Memory init time: %.3f ms\n", elapsed);
}

extern "C" void seedRandomValues(uint32_t seed) {
  if (seed == 0)
    std::srand(time(0));
  else
    std::srand(seed);
}

extern "C" float randomIntegerValue(int16_t min, int16_t max) {
  if (min == max)
    return min;
  int16_t randVal = (std::rand() % (max - min)) + min;
  return static_cast<float>(randVal);
}

extern "C" float randomFloatValue(int16_t min, int16_t max) {
  auto minAsF = static_cast<float>(min);
  if (min == max)
    // Lower float values to prevent inf in big fp16 tests where not all sides
    // are randomized
    return minAsF * 0.1;
  return static_cast<float>((max - min) * static_cast<double>(std::rand()) /
                            static_cast<double>(RAND_MAX)) +
         minAsF;
}

size_t findIdxHistRelDiff(double relDiff, const double *BUCKET_BOUNDARIES,
                          size_t NUM_BOUNDARIES) {
  if (relDiff == 0.0)
    return 0;
  size_t i = 0;
  while (i < NUM_BOUNDARIES && relDiff > BUCKET_BOUNDARIES[i])
    i++;
  return i + 1;
}

void printDebugVerifyResults(long long dataSize, float maxAbsDiff,
                             float maxVAL_abs, float maxGPU_abs,
                             double aveAbsDiff, double maxRelDiff,
                             float maxVAL_rel, float maxGPU_rel,
                             double aveRelDiff, double err_RMS,
                             const double *BUCKET_BOUNDARIES,
                             size_t NUM_BUCKETS, int *hist_relDiff) {
  printf("Number of elements: %lld\n", dataSize);
  printf("maxAbsDiff info: maxAbsDiff = %f (valNum = %.5f, gpuNum = %.5f), "
         "average absDiff = %.1e\n",
         maxAbsDiff, maxVAL_abs, maxGPU_abs, aveAbsDiff);
  printf("maxRelDiff info: maxRelDiff = %.1e (valNum = %.10f, gpuNum = %.10f), "
         "average relDiff = %.1e\n",
         maxRelDiff, maxVAL_rel, maxGPU_rel, aveRelDiff);
  printf("RMS = %.1e\n", err_RMS);
  printf("Histogram of relDiff: \n");
  for (size_t i = 0; i < NUM_BUCKETS; ++i) {
    if (i == 0)
      printf("        relDiff = 0     ");
    else if (i == 1)
      printf("     0 < relDiff < %.0e", BUCKET_BOUNDARIES[i - 1]);
    else if (i == NUM_BUCKETS - 2) // second to the last bucket
      printf("%.0e < relDiff < inf   ", BUCKET_BOUNDARIES[i - 2]);
    else if (i == NUM_BUCKETS - 1) // last bucket
      printf("        relDiff = inf   ");
    else
      printf("%.0e < relDiff <= %.0e", BUCKET_BOUNDARIES[i - 2],
             BUCKET_BOUNDARIES[i - 1]);

    printf(": %d/%lld (%lf%%)\n", hist_relDiff[i], dataSize,
           100.0 * static_cast<double>(hist_relDiff[i]) /
               static_cast<double>(dataSize));
  }
}

enum class PrintOption : char {
  Always = 3,  // always print debug info
  Failure = 2, // print elem-wise diff + summary only if the test fails
  Summary = 1, // print summary info only if the test fails
  Off = 0      // do not print debug info
};

template <typename T>
void mcpuVerify(T *gpuResults, T *validationResults, long long dataSize,
                float thr_RMS, float thr_absDiff, float thr_relDiff,
                char printDebug, bool isFP32) {
  float valNum, gpuNum;
  // metric maxAbsDiff
  float maxAbsDiff = 0.0f;
  double sumAbsDiff = 0.0;
  float maxVAL_abs = 0.0f;
  float maxGPU_abs = 0.0f;
  // metric maxRelDiff
  double maxRelDiff = 0.0;
  double sumRelDiff = 0.0;
  float maxVAL_rel = 0.0f;
  float maxGPU_rel = 0.0f;
  // Metric RMS
  float maxMag = 0.0f;
  double sumDiffSq = 0.0;
  // histogram of relDiff metric
  // bucket index --> interval:
  //     0: 0
  //     1: 0 - 1e-6
  //     2: 1e-6 - 1e-5
  //     3: 1e-5 - 1e-4
  //     4: 1e-4 - 1e-3
  //     5: 1e-3 - 1e-2
  //     6: 1e-2 - 0.1
  //     7: 0.1 - 1
  //     8: >= 1
  //     9: Inf
  constexpr size_t NUM_BOUNDARIES = 7;
  static const double BUCKET_BOUNDARIES[NUM_BOUNDARIES] = {
      1.0e-06, 1.0e-05, 1.0e-04, 1.0e-03, 1.0e-02, 0.1, 1.0};
  // 3 more buckets compared to the bucket boundaries
  // 1. relDiff = 0
  // 2. largest boundary < relDiff <= inf
  // 3. relDiff = inf
  constexpr size_t NUM_BUCKETS = NUM_BOUNDARIES + 3;
  int hist_relDiff[NUM_BUCKETS] = {0};
  // Obtain print debug info option
  PrintOption print_option = static_cast<PrintOption>(printDebug);

  for (long long i = 0; i < dataSize; ++i) {
    valNum = static_cast<float>(validationResults[i]);
    gpuNum = static_cast<float>(gpuResults[i]);
    // Update the max magnitude value
    float maxNum = std::max(fabs(valNum), fabs(gpuNum));
    maxMag = std::max(maxMag, maxNum);

    if (valNum == gpuNum) {
      hist_relDiff[0]++;
    } else if ((std::fpclassify(valNum) == FP_SUBNORMAL) && isFP32) {
      // Since we are comparing the output of the kernel, and not the direct
      // output of operations there is a chance that fusion can modify f32
      // values such that the sign of the GPU and CPU result will differ even
      // though the results are correct. In this case we are going to treat
      // f32 subnormals as always being correct
      hist_relDiff[0]++;
    } else {
      // AIROCMLIR-911: Replace this hard-coded fp16 clamp with a
      // dtype-aware clamp (or remove it once the comparator handles inf
      // natively). Currently it masks real mismatches for non-fp16 types.
      constexpr float fp16MaxVal = 65504;
      if (std::isinf(valNum))
        valNum = (valNum > 0 ? fp16MaxVal : -fp16MaxVal);
      if (std::isinf(gpuNum))
        gpuNum = (gpuNum > 0 ? fp16MaxVal : -fp16MaxVal);
      float absDiff = fabs(valNum - gpuNum);
      // Update maxAbsDiff and its corresponding pair of values
      if (absDiff > maxAbsDiff) {
        maxVAL_abs = valNum;
        maxGPU_abs = gpuNum;
        maxAbsDiff = absDiff;
      }
      sumAbsDiff += static_cast<double>(absDiff);
      // Update maxRelDiff only if cpuVal != 0
      double relDiff = 0.0;
      if (valNum != 0.0f) {
        // Normalize relDiff by taking the max between valNum and epsilon to
        // avoid large/inf relDiff results
        constexpr float epsilon = 1e-8f;
        double denominator = std::max(static_cast<double>(fabs(valNum)),
                                      static_cast<double>(epsilon));
        relDiff = static_cast<double>(absDiff) / denominator;
        hist_relDiff[findIdxHistRelDiff(relDiff, BUCKET_BOUNDARIES,
                                        NUM_BOUNDARIES)]++;
        if (relDiff > maxRelDiff) {
          maxVAL_rel = valNum;
          maxGPU_rel = gpuNum;
          maxRelDiff = relDiff;
        }
        sumRelDiff += relDiff;
      } else {
        // relDiff = inf goes to the last bucket
        hist_relDiff[NUM_BUCKETS - 1]++;
      }
      // Accumulate square root
      sumDiffSq += static_cast<double>(absDiff) * static_cast<double>(absDiff);
      // Print out values if print mode is Always||Failure
      // and difference is larger than threshold
      if ((print_option == PrintOption::Always ||
           print_option == PrintOption::Failure) &&
          (absDiff > thr_absDiff || relDiff > thr_relDiff))
        printf("%lld: %f %f %f %lf\n", i, valNum, gpuNum, absDiff, relDiff);
    }
  }
  double aveAbsDiff = sumAbsDiff / static_cast<double>(dataSize);
  double aveRelDiff = sumRelDiff / static_cast<double>(dataSize);
  double err_RMS = sqrt(sumDiffSq) / (static_cast<double>(maxMag) *
                                      sqrt(static_cast<double>(dataSize)));
  // Check if pass based on all three metrics: RMS, maxAbsDiff, maxRelDiff
  int RMS_pass = (err_RMS <= thr_RMS) ? 1 : 0;
  int absDiff_pass = (maxAbsDiff <= thr_absDiff) ? 1 : 0;
  int relDiff_pass = (maxRelDiff <= thr_relDiff) ? 1 : 0;
  int all_pass = (RMS_pass && absDiff_pass && relDiff_pass) ? 1 : 0;
  // Verbose information about the difference
  if (print_option == PrintOption::Always ||
      ((print_option == PrintOption::Failure ||
        print_option == PrintOption::Summary) &&
       all_pass == 0))
    printDebugVerifyResults(dataSize, maxAbsDiff, maxVAL_abs, maxGPU_abs,
                            aveAbsDiff, maxRelDiff, maxVAL_rel, maxGPU_rel,
                            aveRelDiff, err_RMS, BUCKET_BOUNDARIES, NUM_BUCKETS,
                            hist_relDiff);
  printf("[%d %d %d]\n", RMS_pass, absDiff_pass, relDiff_pass);
}

// Compare the results in f32
extern "C" void
mcpuVerifyFloat(float *gpuAllocated, float *gpuAligned, int64_t gpuOffset,
                int64_t gpuSize, int64_t gpuStride, float *valAllocated,
                float *valAligned, int64_t valOffset, int64_t valSize,
                int64_t valStride, float thr_RMS, float thr_absDiff,
                float thr_relDiff, char printDebug, bool isFP32) {
  assert(gpuSize == valSize);
  mcpuVerify<float>(gpuAligned, valAligned, valSize, thr_RMS, thr_absDiff,
                    thr_relDiff, printDebug, isFP32);
}

// Allclose-style verification, matching numpy.allclose /
// torch.testing.assert_close:
//   |gpuNum - valNum| <= atol + rtol * |valNum|
//
// To preserve the "[%d %d %d]" output format that existing FileCheck tests
// check, the allclose result is repeated three times.
//
// On failure we print allclose-specific diagnostics:
// - The worst-offender element (the one with the largest absDiff/tolerance
// ratio),
// - The histogram of absDiff/tolerance ratios
// - The "smallest atol/rtol that would pass everything else, holding the other
// one fixed" hints. We intentionally do *not* print the legacy
// RMS/maxAbsDiff/maxRelDiff/relDiff stats here: they tell you nothing about why
// allclose failed or what tolerance would make it pass, and they confuse the
// diagnostic by suggesting thresholds that are no longer being checked. Use
// --comparator=legacy if you need those numbers.
//
// "ratio" buckets follow a 0/passing/failing structure:
//   [0]: ratio == 0                          (exact match or subnormal-equal)
//   [1..3]: 0 < ratio <= {0.1, 0.5, 1.0}     (PASSING -- with headroom)
//   [4..7]: 1.0 < ratio <= {2.0, 10.0, 100.0, +inf}   (FAILING)
//   [8]: ratio == +inf                       (tolerance == 0 and absDiff > 0)
//
// The 7 boundaries split the open interval (0, +inf) into the 7 buckets that
// sit between the "ratio == 0" prefix bucket and the "ratio == inf" suffix
// bucket, so NUM_RATIO_BUCKETS == NUM_RATIO_BOUNDARIES + 2.
static constexpr size_t NUM_RATIO_BOUNDARIES = 7;
static const double RATIO_BOUNDARIES[NUM_RATIO_BOUNDARIES] = {
    0.1, 0.5, 1.0, 2.0, 10.0, 100.0, INFINITY};
static constexpr size_t NUM_RATIO_BUCKETS = NUM_RATIO_BOUNDARIES + 2;

static void printAllcloseStats(long long dataSize, long long failingElements,
                               float atol, float rtol, float maxRatio,
                               float maxRatioValNum, float maxRatioGpuNum,
                               double maxRatioAbsDiff, double maxRatioTolerance,
                               bool maxRatioIsInf, long long ratioInfCount,
                               long long nanCount, float nanValNum,
                               float nanGpuNum, double minAtolForCurrentRtol,
                               double minRtolForCurrentAtol,
                               bool minRtolWellDefined, const int *hist_ratio) {
  printf("allclose statistics:\n");
  if (failingElements == 0) {
    printf("  all elements within tolerance (atol=%.3e, rtol=%.3e)\n", atol,
           rtol);
  } else if (nanCount > 0) {
    // NaN takes precedence: no finite tolerance can mask a NaN, so the user
    // has to see it first and any calibration hint would be misleading.
    printf("  worst element: valNum=%g gpuNum=%g (NaN-mismatch)\n", nanValNum,
           nanGpuNum);
    printf("  no tolerance can mask a NaN-mismatch; fix the kernel\n");
  } else {
    if (maxRatioIsInf) {
      printf("  worst element: valNum=%g gpuNum=%g absDiff=%.3e tolerance=0 "
             "(ratio=inf)\n",
             maxRatioValNum, maxRatioGpuNum, maxRatioAbsDiff);
    } else {
      printf("  worst element: valNum=%g gpuNum=%g absDiff=%.3e tolerance=%.3e "
             "(ratio=%.2fx)\n",
             maxRatioValNum, maxRatioGpuNum, maxRatioAbsDiff, maxRatioTolerance,
             maxRatio);
    }
    // Calibration hints: smallest atol/rtol that would make everything pass
    // (holding the other fixed).
    printf("  to pass with current rtol=%.3e: atol >= %.3e\n", rtol,
           minAtolForCurrentRtol);
    if (minRtolWellDefined) {
      printf("  to pass with current atol=%.3e: rtol >= %.3e\n", atol,
             minRtolForCurrentAtol);
    } else {
      printf("  to pass with current atol=%.3e: rtol >= n/a "
             "(failures only at valNum == 0; increase atol)\n",
             atol);
    }
  }
  printf("  histogram of absDiff/tolerance:\n");
  for (size_t i = 0; i < NUM_RATIO_BUCKETS; ++i) {
    if (i == 0)
      printf("           ratio == 0     ");
    else if (i == 1)
      printf("       0 < ratio <= %5.1f ", RATIO_BOUNDARIES[0]);
    else if (i == NUM_RATIO_BUCKETS - 2)
      printf("   %5.1f < ratio <  inf  ", RATIO_BOUNDARIES[i - 2]);
    else if (i == NUM_RATIO_BUCKETS - 1)
      printf("           ratio == inf   ");
    else
      printf("   %5.1f < ratio <= %5.1f ", RATIO_BOUNDARIES[i - 2],
             RATIO_BOUNDARIES[i - 1]);
    printf(": %d/%lld (%.4f%%)%s\n", hist_ratio[i], dataSize,
           100.0 * static_cast<double>(hist_ratio[i]) /
               static_cast<double>(dataSize),
           (i >= 4 && hist_ratio[i] > 0) ? "  <-- failing" : "");
  }
  // NaN gets its own dedicated histogram row; it is not bucketed by ratio
  // because NaN is unordered with respect to every finite boundary and would
  // otherwise be silently counted as a passing low ratio.
  if (nanCount > 0)
    printf("           ratio == nan   : %lld/%lld (%.4f%%)  <-- failing\n",
           nanCount, dataSize,
           100.0 * static_cast<double>(nanCount) /
               static_cast<double>(dataSize));
  if (ratioInfCount > 0)
    printf("  note: %lld element(s) had tolerance == 0 with absDiff > 0 "
           "(only possible when -atol=0 and valNum==0)\n",
           ratioInfCount);
  if (nanCount > 0)
    printf("  note: %lld element(s) had NaN on either side\n", nanCount);
}

template <typename T>
void mcpuVerifyAllclose(T *gpuResults, T *validationResults, long long dataSize,
                        float atol, float rtol, char printDebug, bool isFP32) {
  float valNum, gpuNum;
  PrintOption print_option = static_cast<PrintOption>(printDebug);

  // Ratio histogram. RATIO_BOUNDARIES, NUM_RATIO_BOUNDARIES and
  // NUM_RATIO_BUCKETS are file-scope (defined just above printAllcloseStats)
  // so both functions stay in sync by construction.
  int hist_ratio[NUM_RATIO_BUCKETS] = {0};
  // Worst offender by ratio = absDiff / tolerance. We track it separately
  // from maxAbsDiff because the legacy "worst by absDiff" element is often
  // *not* the element that drives the allclose verdict (e.g. an element with
  // large absolute value can have a large absDiff but still pass thanks to
  // the rtol*|valNum| component).
  float maxRatio = 0.0f;
  float maxRatioValNum = 0.0f;
  float maxRatioGpuNum = 0.0f;
  double maxRatioAbsDiff = 0.0;
  double maxRatioTolerance = 0.0;
  bool maxRatioIsInf = false;
  long long ratioInfCount = 0;
  // NaN bookkeeping. We can't fold NaN into the ratio histogram because
  // NaN-vs-finite comparisons are always false (NaN > x is false), so a NaN
  // element would silently end up in the (0, 0.1] passing bucket and never
  // update the worst-element ratio. Track them separately and surface them in
  // the diagnostic as a dedicated row.
  long long nanCount = 0;
  float nanValNum = 0.0f;
  float nanGpuNum = 0.0f;
  // Calibration hints: smallest atol/rtol that would make every failing
  // element pass, holding the other tolerance fixed.
  //   minAtolForCurrentRtol = max over elements of (absDiff - rtol*|valNum|)
  //   minRtolForCurrentAtol = max over elements of (absDiff - atol)/|valNum|
  // Both clamped at 0 (a "negative" tolerance is meaningless). The rtol hint
  // skips elements with valNum == 0, since rtol can't fix those.
  double minAtolForCurrentRtol = 0.0;
  double minRtolForCurrentAtol = 0.0;
  bool minRtolWellDefined = false;

  long long failingElements = 0;
  for (long long i = 0; i < dataSize; ++i) {
    valNum = static_cast<float>(validationResults[i]);
    gpuNum = static_cast<float>(gpuResults[i]);

    if (valNum == gpuNum) {
      hist_ratio[0]++;
      continue;
    }
    // NaN handling has to come *before* the subnormal/infinity normalization
    // and the absDiff/tolerance arithmetic, because every comparison and
    // arithmetic op with NaN produces NaN-or-false, which would otherwise
    // mis-bucket the element and skip the worst-element update.
    if (std::isnan(valNum) || std::isnan(gpuNum)) {
      failingElements++;
      if (nanCount == 0) {
        nanValNum = valNum;
        nanGpuNum = gpuNum;
      }
      nanCount++;
      if (print_option == PrintOption::Always ||
          print_option == PrintOption::Failure)
        printf("%lld: valNum=%f gpuNum=%f (NaN-mismatch)\n", i, valNum, gpuNum);
      continue;
    }
    if ((std::fpclassify(valNum) == FP_SUBNORMAL) && isFP32) {
      // Treat FP32 subnormals on the CPU side as exact matches; fusion can
      // legitimately reorder ops to produce a different signed subnormal on
      // the GPU side without the result being wrong. Same affordance as the
      // legacy mcpuVerify path.
      hist_ratio[0]++;
      continue;
    }
    // Clamp +/-inf on both sides to fp16 max so that the subsequent
    // arithmetic is well-defined (otherwise inf - finite = inf, |inf| = inf,
    // and the allclose tolerance becomes inf which would mask true failures).
    // Both CPU and GPU can produce inf when the true result is near the fp16
    // boundary; non-deterministic accumulation order decides which side
    // overflows first, so we treat both symmetrically.
    constexpr float fp16MaxVal = 65504;
    if (std::isinf(valNum))
      valNum = (valNum > 0 ? fp16MaxVal : -fp16MaxVal);
    if (std::isinf(gpuNum))
      gpuNum = (gpuNum > 0 ? fp16MaxVal : -fp16MaxVal);
    float absDiff = fabs(valNum - gpuNum);

    // The allclose predicate (asymmetric, matches PyTorch/NumPy convention).
    double tolerance =
        static_cast<double>(atol) +
        static_cast<double>(rtol) * static_cast<double>(fabs(valNum));
    bool elemPass = static_cast<double>(absDiff) <= tolerance;

    // allclose-specific bookkeeping: ratio, worst-offender, calibration.
    if (tolerance == 0.0) {
      // absDiff > 0 (we are past the valNum == gpuNum early-out) and
      // tolerance == 0 can only happen with atol == 0 && (rtol == 0 ||
      // valNum == 0). Record as ratio == inf.
      ratioInfCount++;
      hist_ratio[NUM_RATIO_BUCKETS - 1]++;
      if (!maxRatioIsInf) {
        maxRatioIsInf = true;
        maxRatio = std::numeric_limits<float>::infinity();
        maxRatioValNum = valNum;
        maxRatioGpuNum = gpuNum;
        maxRatioAbsDiff = absDiff;
        maxRatioTolerance = 0.0;
      }
    } else {
      double ratio = static_cast<double>(absDiff) / tolerance;
      // Find the ratio histogram bucket: 1..NUM_RATIO_BOUNDARIES cover the
      // boundaries, 0 is exact-match (already handled by early-out),
      // NUM_RATIO_BUCKETS - 1 is ratio == inf.
      // We start at 1 because bucket 0 is reserved for ratio == 0, which
      // cannot reach this loop: the (valNum == gpuNum) and f32 subnormal
      // cases above are the only path into bucket 0.
      size_t bucket = 1;
      while (bucket < NUM_RATIO_BOUNDARIES &&
             ratio > RATIO_BOUNDARIES[bucket - 1])
        bucket++;
      hist_ratio[bucket]++;
      if (!maxRatioIsInf && ratio > maxRatio) {
        maxRatio = static_cast<float>(ratio);
        maxRatioValNum = valNum;
        maxRatioGpuNum = gpuNum;
        maxRatioAbsDiff = absDiff;
        maxRatioTolerance = tolerance;
      }
    }
    if (!elemPass) {
      failingElements++;
      // Update calibration hints (clamped at 0 implicitly: we only consider
      // failing elements, and for those absDiff > tolerance >= atol).
      double atolNeeded =
          static_cast<double>(absDiff) -
          static_cast<double>(rtol) * static_cast<double>(fabs(valNum));
      if (atolNeeded > minAtolForCurrentRtol)
        minAtolForCurrentRtol = atolNeeded;
      if (valNum != 0.0f) {
        double rtolNeeded =
            (static_cast<double>(absDiff) - static_cast<double>(atol)) /
            static_cast<double>(fabs(valNum));
        if (rtolNeeded > minRtolForCurrentAtol) {
          minRtolForCurrentAtol = rtolNeeded;
          minRtolWellDefined = true;
        }
      }
      if (print_option == PrintOption::Always ||
          print_option == PrintOption::Failure) {
        double ratio = (tolerance == 0.0)
                           ? std::numeric_limits<double>::infinity()
                           : static_cast<double>(absDiff) / tolerance;
        printf("%lld: valNum=%f gpuNum=%f absDiff=%f tol=%.3e ratio=%.2fx\n", i,
               valNum, gpuNum, absDiff, tolerance, ratio);
      }
    }
  }
  int all_pass = (failingElements == 0) ? 1 : 0;
  if (print_option == PrintOption::Always ||
      ((print_option == PrintOption::Failure ||
        print_option == PrintOption::Summary) &&
       all_pass == 0)) {
    printf("allclose(atol=%.3e, rtol=%.3e): %lld/%lld failing element(s)\n",
           atol, rtol, failingElements, dataSize);
    printAllcloseStats(dataSize, failingElements, atol, rtol, maxRatio,
                       maxRatioValNum, maxRatioGpuNum, maxRatioAbsDiff,
                       maxRatioTolerance, maxRatioIsInf, ratioInfCount,
                       nanCount, nanValNum, nanGpuNum, minAtolForCurrentRtol,
                       minRtolForCurrentAtol, minRtolWellDefined, hist_ratio);
    printf("zero_diff: %lld/%lld\n", (long long)hist_ratio[0], dataSize);
  }
  printf("[%d %d %d]\n", all_pass, all_pass, all_pass);
}

extern "C" void mcpuVerifyFloatAllclose(float *gpuAllocated, float *gpuAligned,
                                        int64_t gpuOffset, int64_t gpuSize,
                                        int64_t gpuStride, float *valAllocated,
                                        float *valAligned, int64_t valOffset,
                                        int64_t valSize, int64_t valStride,
                                        float atol, float rtol, char printDebug,
                                        bool isFP32) {
  assert(gpuSize == valSize);
  mcpuVerifyAllclose<float>(gpuAligned, valAligned, valSize, atol, rtol,
                            printDebug, isFP32);
}

// Compare the results in int32
template <typename GPUTYPE, typename VALTYPE>
void mcpuVerifyInt(GPUTYPE *gpuAligned, VALTYPE *valAligned, long long dataSize,
                   char printDebug) {
  long long failure_count = 0;  // the number of incorrect elements
  long long overflow_count = 0; // the number of overflow elements
  long long maxAbsDiff = 0;
  int64_t max = std::numeric_limits<GPUTYPE>::max();
  int64_t min = std::numeric_limits<GPUTYPE>::min();
  PrintOption print_option = static_cast<PrintOption>(printDebug);
  for (long long i = 0; i < dataSize; ++i) {
    auto valNum = static_cast<long long>(valAligned[i]);
    int32_t gpuNum = gpuAligned[i];
    if (valNum > max || valNum < min) {
      overflow_count++;
      if (print_option == PrintOption::Always)
        printf("overflow at element : %lld, gpu=%d, val=%lld\n", i, gpuNum,
               valNum);
    }

    if (gpuNum != valNum) {
      failure_count++;
      long long absDiff = std::abs(valNum - gpuNum);
      if (absDiff > maxAbsDiff)
        maxAbsDiff = absDiff;

      // Print out individual failing elements if print mode is Always||Failure
      if (print_option == PrintOption::Always ||
          print_option == PrintOption::Failure) {
        printf("%lld: gpu=%d val=%lld absDiff=%lld\n", i, gpuNum, valNum,
               absDiff);
      }
    }
  }

  if (failure_count == 0) {
    if ((print_option == PrintOption::Always ||
         print_option == PrintOption::Summary) &&
        overflow_count > 0) {
      printf("Number of elements: %lld\n", dataSize);
      printf("Number of overflow elements: %lld\n", overflow_count);
    }
    printf("[1 1 1]\n");
  } else {
    if (print_option == PrintOption::Always ||
        print_option == PrintOption::Failure ||
        print_option == PrintOption::Summary) {
      printf("Number of elements: %lld\n", dataSize);
      printf("Number of incorrect elements: %lld\n", failure_count);
      printf("maxAbsDiff: %lld\n", maxAbsDiff);
      printf("Number of overflow elements: %lld\n", overflow_count);
    }
    printf("[0 0 0]");
  }
}

extern "C" void mcpuVerifyInt32Int32(int32_t *gpuAllocated, int32_t *gpuAligned,
                                     int64_t gpuOffset, int64_t gpuSize,
                                     int64_t gpuStride, int32_t *valAllocated,
                                     int32_t *valAligned, int32_t valOffset,
                                     int64_t valSize, int64_t valStride,
                                     char printDebug) {

  assert(gpuSize == valSize);
  mcpuVerifyInt<int32_t, int32_t>(gpuAligned, valAligned, valSize, printDebug);
}

extern "C" void mcpuVerifyInt32Int64(int32_t *gpuAllocated, int32_t *gpuAligned,
                                     int64_t gpuOffset, int64_t gpuSize,
                                     int64_t gpuStride, int64_t *valAllocated,
                                     int64_t *valAligned, int64_t valOffset,
                                     int64_t valSize, int64_t valStride,
                                     char printDebug) {

  assert(gpuSize == valSize);
  mcpuVerifyInt<int32_t, int64_t>(gpuAligned, valAligned, valSize, printDebug);
}

extern "C" void mcpuVerifyInt8Int64(int8_t *gpuAllocated, int8_t *gpuAligned,
                                    int64_t gpuOffset, int64_t gpuSize,
                                    int64_t gpuStride, int64_t *valAllocated,
                                    int64_t *valAligned, int64_t valOffset,
                                    int64_t valSize, int64_t valStride,
                                    char printDebug) {

  assert(gpuSize == valSize);
  mcpuVerifyInt<int8_t, int64_t>(gpuAligned, valAligned, valSize, printDebug);
}

template <typename T>
void mcpuVerifyNaive(T *gpuAligned, T *valAligned, long long dataSize,
                     char printDebug) {
  long long failure_count = 0; // the number of incorrect elements
  T maxAbsDiff = 0;

  PrintOption print_option = static_cast<PrintOption>(printDebug);
  for (int64_t i = 0; i < dataSize; ++i) {
    T valNum = valAligned[i];
    T gpuNum = gpuAligned[i];

    if (gpuNum != valNum) {
      failure_count++;
      T absDiff = std::abs(valNum - gpuNum);
      if (absDiff > maxAbsDiff)
        maxAbsDiff = absDiff;

      // Print out individual failing elements if print mode is Always||Failure
      if (print_option == PrintOption::Always ||
          print_option == PrintOption::Failure) {
        std::cout << i << ": gpu=" << gpuNum << " val=" << valNum
                  << " absDiff=" << absDiff << std::endl;
      }
    }
  }

  if (failure_count == 0) {
    printf("[1 1 1]\n");
  } else {
    if (print_option == PrintOption::Always ||
        print_option == PrintOption::Failure ||
        print_option == PrintOption::Summary) {
      printf("Number of elements: %lld\n", dataSize);
      printf("Number of incorrect elements: %lld\n", failure_count);
      std::cout << "maxAbsDiff: " << maxAbsDiff << std::endl;
    }
    printf("[0 0 0]");
  }
}

extern "C" void mcpuVerifyInt8Int8(int8_t *gpuAllocated, int8_t *gpuAligned,
                                   int64_t gpuOffset, int64_t gpuSize,
                                   int64_t gpuStride, int8_t *valAllocated,
                                   int8_t *valAligned, int32_t valOffset,
                                   int64_t valSize, int64_t valStride,
                                   char printDebug) {

  assert(gpuSize == valSize);
  mcpuVerifyNaive(gpuAligned, valAligned, valSize, printDebug);
}
