// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Exhaustive ULP measurement of the rock-legalize-math-for-triton f32 tanh
// approximation, over every one of the 2^32 f32 bit patterns.
//
// This is the tool behind the accuracy table in the emitApproxTanh comment in
// mlir/lib/Dialect/Rock/Transforms/LegalizeMathForTriton.cpp. Re-run it if that
// lowering changes.
//
// Fidelity matters more than anything else here, so approxTanh below is written
// to compile to exactly the instructions the pipeline emits. Confirm with:
//
//   rocmlir-gen --clone-harness -arch gfx1100 -fut mlir_convolution_tanh \
//       mlir/test/rocmlir-driver/tanh-isa.mlir \
//     | rocmlir-driver -arch=gfx1100 -kernel-pipeline=migraphx,highlevel \
//         -host-pipeline=migraphx,highlevel \
//     | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1100 \
//         -kernel-pipeline=gpu,triton,binary -o /dev/null
//
// which gives v_mul_f32 0xc038aa3b / v_exp_f32 / v_add_f32 1.0 / v_rcp_f32 /
// v_fma_f32 2.0, -1.0. __builtin_amdgcn_exp2f and __builtin_amdgcn_rcpf are
// those hardware instructions rather than a model of them, so the
// transcendental units' own error is included in the result.
//
// Three lowerings are swept side by side:
//
//   approx  the five-instruction approximation this file is named for.
//   expand  the math dialect's own tanh expansion, which is what an f32 tanh
//           is still left to and what every other type took before the
//           approximation existed.
//   ocml    tanhf, i.e. __ocml_tanh_f32, which is what --disable-fast-math
//           emits. Do not build this with -ffast-math, or that column stops
//           being OCML and the comparison loses its baseline.
//
// Reference for all three is __ocml_tanh_f64, which carries 29 more mantissa
// bits than f32 and so contributes about 2^-29 ULP of noise.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// One bucket per binary exponent of |x|: ilogb(|x|) + 150 covers subnormals
// through infinity.
#define NBUCKET 300

// log2(e), the scale a `math.exp` picks up on its way to the hardware's
// base-2 exponential. The approximation folds it into its own constant; the
// expansion cannot, because there the exp is a separate op.
constexpr float kLog2E = 1.44269504088896340736f;

__device__ __forceinline__ float approxTanh(float x) {
  float scaled = x * -2.885390082f; // 0xc038aa3b, the f32 value of -2*log2(e)
  float e = __builtin_amdgcn_exp2f(scaled);
  float denom = e + 1.0f;
  float recip = __builtin_amdgcn_rcpf(denom);
  return __builtin_fmaf(recip, 2.0f, -1.0f);
}

/// The math dialect's own tanh expansion:
///   sign = x < 0 ? -1 : 1;  sign * (1 - exp(-2*sign*x)) / (1 + exp(-2*sign*x))
///
/// Swept alongside the approximation because it is not a hypothetical: it is
/// the lowering an f32 tanh is still left to, and the one every other type took
/// before the approximation existed. Reading the two columns against each other
/// is the argument for declining f32, so both have to come off the same sweep.
///
/// The sign steps the expansion really emits are a uitofp of the compare scaled
/// by -2 and offset by 1. Those are exactly +/-1 with no rounding, so a select
/// stands in for them. The exp and the divide are spelled as the hardware
/// instructions they become under fast math: `math.exp` is a v_exp_f32 on a
/// log2(e)-scaled argument, and `arcp` turns the divide into a v_rcp_f32 and a
/// multiply. tanh-isa.mlir pins that this is the pair the f32 path emits.
__device__ __forceinline__ float expansionTanh(float x) {
  float sign = x < 0.0f ? -1.0f : 1.0f;
  float ax = sign * x;
  float e = __builtin_amdgcn_exp2f(-2.0f * ax * kLog2E);
  return sign * ((1.0f - e) * __builtin_amdgcn_rcpf(1.0f + e));
}

/// Spacing of f32 around `ref`, with subnormals flattened to the minimum.
__device__ __forceinline__ double ulpOf(double ref) {
  if (ref == 0.0 || !isfinite(ref))
    return ldexp(1.0, -149);
  int e = ilogb(ref);
  if (e < -126)
    return ldexp(1.0, -149);
  return ldexp(1.0, e - 23);
}

// Errors are non-negative, so their bit patterns order the same way they do and
// an integer atomicMax is a float max. Integer atomics rather than float ones
// on purpose: these are exact and order-independent, so the result does not
// depend on how the sweep happened to be scheduled. The buffers come from
// hipMalloc, i.e. coarse-grained device memory, which is what makes the atomics
// reliable; on fine-grained or host-allocated memory some parts can drop them.
__device__ __forceinline__ void bump(unsigned *slot, float v) {
  atomicMax(slot, __float_as_uint(v));
}

__global__ void sweep(uint32_t base, uint32_t count, unsigned *ulpApprox,
                      unsigned *ulpExpand, unsigned *ulpOcml,
                      unsigned *absApprox, unsigned *absExpand,
                      unsigned *absOcml, unsigned long long *nanCount,
                      unsigned long long *satBad) {
  uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    float x = __uint_as_float(base + i);
    if (isnan(x)) {
      atomicAdd(nanCount, 1ULL);
      continue;
    }

    float got = approxTanh(x);
    float expand = expansionTanh(x);

    // The approximation has no sign handling: -1 is reached only by the
    // exponential overflowing to +inf and rcp(+inf) being exactly +0. Check
    // that rather than assume it. The expansion reaches the ends the other way,
    // through its own sign restore, and is checked alongside so that a column
    // is never quietly compared against a saturation it did not reach.
    if (isinf(x)) {
      float want = x > 0.0f ? 1.0f : -1.0f;
      if (got != want || expand != want)
        atomicAdd(satBad, 1ULL);
      continue;
    }
    // Must not fall through: a NaN error would be fed to bump(), whose integer
    // atomicMax would then latch NaN's bit pattern and swallow every real
    // measurement in that bucket.
    if (isnan(got) || isnan(expand)) {
      atomicAdd(satBad, 1ULL);
      continue;
    }

    float ocml = tanhf(x);
    double ref = tanh((double)x);
    double u = ulpOf(ref);
    double errApprox = fabs((double)got - ref);
    double errExpand = fabs((double)expand - ref);
    double errOcml = fabs((double)ocml - ref);

    int e = (x == 0.0f) ? -149 : ilogbf(fabsf(x));
    int b = std::min(std::max(e + 150, 0), NBUCKET - 1);
    bump(&ulpApprox[b], (float)(errApprox / u));
    bump(&ulpExpand[b], (float)(errExpand / u));
    bump(&ulpOcml[b], (float)(errOcml / u));
    bump(&absApprox[b], (float)errApprox);
    bump(&absExpand[b], (float)errExpand);
    bump(&absOcml[b], (float)errOcml);
  }
}

static float asFloat(unsigned bits) {
  union {
    unsigned u;
    float f;
  } c;
  c.u = bits;
  return c.f;
}

int main() {
  unsigned *dUlpA, *dUlpE, *dUlpO, *dAbsA, *dAbsE, *dAbsO;
  unsigned long long *dNan, *dSat;
  size_t sz = NBUCKET * sizeof(unsigned);
  hipMalloc(&dUlpA, sz);
  hipMalloc(&dUlpE, sz);
  hipMalloc(&dUlpO, sz);
  hipMalloc(&dAbsA, sz);
  hipMalloc(&dAbsE, sz);
  hipMalloc(&dAbsO, sz);
  hipMalloc(&dNan, sizeof(unsigned long long));
  hipMalloc(&dSat, sizeof(unsigned long long));
  hipMemset(dUlpA, 0, sz);
  hipMemset(dUlpE, 0, sz);
  hipMemset(dUlpO, 0, sz);
  hipMemset(dAbsA, 0, sz);
  hipMemset(dAbsE, 0, sz);
  hipMemset(dAbsO, 0, sz);
  hipMemset(dNan, 0, sizeof(unsigned long long));
  hipMemset(dSat, 0, sizeof(unsigned long long));

  const uint32_t chunk = 1u << 26;
  const uint64_t total = 1ULL << 32;
  for (uint64_t off = 0; off < total; off += chunk) {
    uint32_t n = (uint32_t)std::min<uint64_t>(chunk, total - off);
    hipLaunchKernelGGL(sweep, dim3(4096), dim3(256), 0, 0, (uint32_t)off, n,
                       dUlpA, dUlpE, dUlpO, dAbsA, dAbsE, dAbsO, dNan, dSat);
    hipDeviceSynchronize();
    if ((off / chunk) % 8 == 0)
      fprintf(stderr, "\r  %.0f%%", 100.0 * off / total);
  }
  fprintf(stderr, "\r      \r");

  std::vector<unsigned> ulpA(NBUCKET), ulpE(NBUCKET), ulpO(NBUCKET),
      absA(NBUCKET), absE(NBUCKET), absO(NBUCKET);
  hipMemcpy(ulpA.data(), dUlpA, sz, hipMemcpyDeviceToHost);
  hipMemcpy(ulpE.data(), dUlpE, sz, hipMemcpyDeviceToHost);
  hipMemcpy(ulpO.data(), dUlpO, sz, hipMemcpyDeviceToHost);
  hipMemcpy(absA.data(), dAbsA, sz, hipMemcpyDeviceToHost);
  hipMemcpy(absE.data(), dAbsE, sz, hipMemcpyDeviceToHost);
  hipMemcpy(absO.data(), dAbsO, sz, hipMemcpyDeviceToHost);
  unsigned long long nans = 0, sat = 0;
  hipMemcpy(&nans, dNan, sizeof nans, hipMemcpyDeviceToHost);
  hipMemcpy(&sat, dSat, sizeof sat, hipMemcpyDeviceToHost);

  printf("f32 tanh, exhaustive over all 2^32 bit patterns\n\n");
  printf("%12s %13s %13s %13s %13s %13s\n", "|x| >= 2^", "approx ULP",
         "expand ULP", "ocml ULP", "approx abs", "expand abs");

  // Walk from the top down so each row is the worst case at or above that
  // magnitude, which is the shape the accuracy claim is stated in.
  float runUlpA = 0, runUlpE = 0, runUlpO = 0;
  float runAbsA = 0, runAbsE = 0, runAbsO = 0;
  for (int b = NBUCKET - 1; b >= 0; --b) {
    runUlpA = fmaxf(runUlpA, asFloat(ulpA[b]));
    runUlpE = fmaxf(runUlpE, asFloat(ulpE[b]));
    runUlpO = fmaxf(runUlpO, asFloat(ulpO[b]));
    runAbsA = fmaxf(runAbsA, asFloat(absA[b]));
    runAbsE = fmaxf(runAbsE, asFloat(absE[b]));
    runAbsO = fmaxf(runAbsO, asFloat(absO[b]));
    int e = b - 150;
    if (e >= -12 && e <= 3)
      printf("%12d %13.1f %13.2f %13.2f %13.3e %13.3e\n", e, runUlpA, runUlpE,
             runUlpO, runAbsA, runAbsE);
  }

  printf("\n  whole range: approx %.1f ULP / %.3e abs,"
         " expansion %.2f ULP / %.3e abs,"
         " ocml %.2f ULP / %.3e abs\n",
         runUlpA, runAbsA, runUlpE, runAbsE, runUlpO, runAbsO);
  printf("  NaN inputs skipped: %llu\n", nans);
  printf("  saturation or NaN failures: %llu\n", sat);
  return sat != 0;
}
