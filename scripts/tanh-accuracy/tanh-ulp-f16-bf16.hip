// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Exhaustive ULP measurement of the rock-legalize-math-for-triton tanh
// approximation for f16 and bf16, over all 65536 bit patterns of each.
//
// Companion to tanh-ulp-f32.hip; see that file for the methodology and for how
// to check the emitted ISA.
//
// Both narrow types are evaluated at f32 and rounded back, which is what the
// pass emits: bf16 arrives already promoted by math-extend-to-supported-types,
// and f16 is widened by the pass itself. Each is also swept under the math
// dialect expansion the approximation replaces, so the two can be read against
// each other.
//
// The baseline column in every case is what --disable-fast-math emits: widen to
// f32, call __ocml_tanh_f32, round back. Reference is __ocml_tanh_f64.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#define NBUCKET 300

// log2(e), the scale a `math.exp` picks up on its way to the hardware's
// base-2 exponential. The approximation folds it into its own constant; the
// expansion cannot, because there the exp is a separate op.
constexpr float kLog2E = 1.44269504088896340736f;

// There is no __builtin_amdgcn_rcph, hence the inline asm.
__device__ __forceinline__ _Float16 rcph(_Float16 x) {
  _Float16 r;
  asm("v_rcp_f16 %0, %1" : "=v"(r) : "v"(x));
  return r;
}

__device__ __forceinline__ float approxTanhF32(float x) {
  float scaled = x * -2.885390082f;
  float e = __builtin_amdgcn_exp2f(scaled);
  float denom = e + 1.0f;
  float recip = __builtin_amdgcn_rcpf(denom);
  return __builtin_fmaf(recip, 2.0f, -1.0f);
}

/// f16 under the math dialect expansion, the lowering the approximation
/// replaces. Only the exp is promoted: math-extend-to-supported-types widens
/// that one op to f32 and rounds its result back, while the surrounding
/// arithmetic stays f16 and the divide stays a v_rcp_f16. That split shows up
/// in the ISA as a v_exp_f32 paired with a v_rcp_f16.
__device__ __forceinline__ _Float16 expansionTanhF16(_Float16 x) {
  _Float16 sign = x < (_Float16)0.0f ? (_Float16)-1.0f : (_Float16)1.0f;
  _Float16 ax = sign * x;
  _Float16 y = (_Float16)-2.0f * ax;
  _Float16 e = (_Float16)__builtin_amdgcn_exp2f((float)y * kLog2E);
  _Float16 num = (_Float16)1.0f - e;
  _Float16 den = (_Float16)1.0f + e;
  return sign * (num * rcph(den));
}

__device__ __forceinline__ float bf16ToF32(uint16_t b) {
  union {
    uint32_t u;
    float f;
  } c;
  c.u = (uint32_t)b << 16;
  return c.f;
}

__device__ __forceinline__ uint16_t f32ToBf16(float f) {
  union {
    uint32_t u;
    float f;
  } c;
  c.f = f;
  if (isnan(f))
    return 0x7fc0;
  uint32_t r = c.u + 0x7fff + ((c.u >> 16) & 1); // round to nearest even
  return (uint16_t)(r >> 16);
}

__device__ __forceinline__ float roundBf16(float f) {
  return bf16ToF32(f32ToBf16(f));
}

/// bf16 under the math dialect expansion, the lowering the approximation
/// replaces. No target here has bf16 arithmetic, so every op in the expansion
/// is promoted to f32 individually and its result rounded back to bf16, which
/// is the extf/truncf pair the arith emulation wraps around each one. Rounding
/// after each step rather than once at the end is the whole difference from the
/// shipped path, and it is what the seven roundBf16 calls below are for.
__device__ __forceinline__ float expansionTanhBf16(float x) {
  float sign = x < 0.0f ? -1.0f : 1.0f;
  float ax = roundBf16(sign * x);
  float y = roundBf16(-2.0f * ax);
  float e = roundBf16(__builtin_amdgcn_exp2f(y * kLog2E));
  float num = roundBf16(1.0f - e);
  float den = roundBf16(1.0f + e);
  return roundBf16(sign * roundBf16(num * __builtin_amdgcn_rcpf(den)));
}

__device__ __forceinline__ void bump(unsigned *slot, float v) {
  atomicMax(slot, __float_as_uint(v));
}

enum Lowering { kApprox = 0, kExpansion = 1 };

__global__ void sweepF16(int mode, unsigned *ulpApprox, unsigned *ulpOcml,
                         unsigned *absApprox, unsigned long long *satBad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 65536)
    return;
  union {
    uint16_t u;
    _Float16 h;
  } in;
  in.u = (uint16_t)i;
  float x = (float)in.h;
  if (isnan(x))
    return;

  _Float16 got = mode == kExpansion ? expansionTanhF16(in.h)
                                    : (_Float16)approxTanhF32(x);
  if (isinf(x)) {
    if ((float)got != (x > 0.0f ? 1.0f : -1.0f))
      atomicAdd(satBad, 1ULL);
    return;
  }
  if (isnan((float)got)) {
    atomicAdd(satBad, 1ULL);
    return;
  }

  _Float16 ocml = (_Float16)tanhf(x);
  double ref = tanh((double)x);
  // f16 keeps 10 explicit mantissa bits; subnormal spacing is 2^-24.
  double u = (ref == 0.0) ? ldexp(1.0, -24)
                          : fmax(ldexp(1.0, ilogb(ref) - 10), ldexp(1.0, -24));

  int e = (x == 0.0f) ? -24 : ilogbf(fabsf(x));
  int b = std::min(std::max(e + 150, 0), NBUCKET - 1);
  bump(&ulpApprox[b], (float)(fabs((double)(float)got - ref) / u));
  bump(&ulpOcml[b], (float)(fabs((double)(float)ocml - ref) / u));
  bump(&absApprox[b], (float)fabs((double)(float)got - ref));
}

__global__ void sweepBf16(int mode, unsigned *ulpApprox, unsigned *ulpOcml,
                          unsigned *absApprox, unsigned long long *satBad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 65536)
    return;
  float x = bf16ToF32((uint16_t)i);
  if (isnan(x))
    return;

  float got = mode == kExpansion ? expansionTanhBf16(x)
                                 : roundBf16(approxTanhF32(x));
  if (isinf(x)) {
    if (got != (x > 0.0f ? 1.0f : -1.0f))
      atomicAdd(satBad, 1ULL);
    return;
  }
  if (isnan(got)) {
    atomicAdd(satBad, 1ULL);
    return;
  }

  float ocml = roundBf16(tanhf(x));
  double ref = tanh((double)x);
  // bf16 keeps 7 explicit mantissa bits.
  double u = (ref == 0.0) ? ldexp(1.0, -133)
                          : fmax(ldexp(1.0, ilogb(ref) - 7), ldexp(1.0, -133));

  int e = (x == 0.0f) ? -133 : ilogbf(fabsf(x));
  int b = std::min(std::max(e + 150, 0), NBUCKET - 1);
  bump(&ulpApprox[b], (float)(fabs((double)got - ref) / u));
  bump(&ulpOcml[b], (float)(fabs((double)ocml - ref) / u));
  bump(&absApprox[b], (float)fabs((double)got - ref));
}

static float asFloat(unsigned bits) {
  union {
    unsigned u;
    float f;
  } c;
  c.u = bits;
  return c.f;
}

static bool report(const char *name, unsigned *dUlpA, unsigned *dUlpO,
                   unsigned *dAbsA, unsigned long long *dSat) {
  unsigned ulpA[NBUCKET], ulpO[NBUCKET], absA[NBUCKET];
  unsigned long long sat = 0;
  hipMemcpy(ulpA, dUlpA, sizeof ulpA, hipMemcpyDeviceToHost);
  hipMemcpy(ulpO, dUlpO, sizeof ulpO, hipMemcpyDeviceToHost);
  hipMemcpy(absA, dAbsA, sizeof absA, hipMemcpyDeviceToHost);
  hipMemcpy(&sat, dSat, sizeof sat, hipMemcpyDeviceToHost);

  printf("\n%s, exhaustive over all 65536 bit patterns\n\n", name);
  printf("%12s %13s %13s %13s\n", "|x| >= 2^", "ULP", "ocml ULP", "abs");
  float runUlpA = 0, runUlpO = 0, runAbsA = 0;
  for (int b = NBUCKET - 1; b >= 0; --b) {
    runUlpA = fmaxf(runUlpA, asFloat(ulpA[b]));
    runUlpO = fmaxf(runUlpO, asFloat(ulpO[b]));
    runAbsA = fmaxf(runAbsA, asFloat(absA[b]));
    int e = b - 150;
    if (e >= -20 && e <= 3)
      printf("%12d %13.1f %13.2f %13.3e\n", e, runUlpA, runUlpO, runAbsA);
  }
  printf("\n  whole range: %.1f ULP / %.3e abs, ocml %.2f ULP\n", runUlpA,
         runAbsA, runUlpO);
  printf("  saturation or NaN failures: %llu\n", sat);
  return sat != 0;
}

int main() {
  unsigned *dUlpA, *dUlpO, *dAbsA;
  unsigned long long *dSat;
  size_t sz = NBUCKET * sizeof(unsigned);
  hipMalloc(&dUlpA, sz);
  hipMalloc(&dUlpO, sz);
  hipMalloc(&dAbsA, sz);
  hipMalloc(&dSat, sizeof(unsigned long long));

  auto reset = [&] {
    hipMemset(dUlpA, 0, sz);
    hipMemset(dUlpO, 0, sz);
    hipMemset(dAbsA, 0, sz);
    hipMemset(dSat, 0, sizeof(unsigned long long));
  };

  auto run = [&](auto kernel, int mode, const char *name) {
    reset();
    hipLaunchKernelGGL(kernel, dim3(256), dim3(256), 0, 0, mode, dUlpA, dUlpO,
                       dAbsA, dSat);
    hipDeviceSynchronize();
    return report(name, dUlpA, dUlpO, dAbsA, dSat);
  };

  bool bad = run(sweepF16, kApprox,
                 "f16 tanh, approximation at f32 (what the pass emits)");
  bad |= run(sweepF16, kExpansion,
             "f16 tanh, math dialect expansion (the lowering it replaces)");
  bad |= run(sweepBf16, kApprox,
             "bf16 tanh, approximation at f32 (what the pass emits)");
  bad |= run(sweepBf16, kExpansion,
             "bf16 tanh, math dialect expansion (the lowering it replaces)");
  return bad;
}
