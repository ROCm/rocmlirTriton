// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Exhaustive ULP measurement of the rock-legalize-math-for-triton tanh
// approximation for the narrow float types, over all 256 bit patterns of each
// fp8 format and all 16 of fp4.
//
// Companion to tanh-ulp-f32.hip; see that file for the methodology and for how
// to check the emitted ISA. Every one of these is evaluated the same way the
// pass emits it: widen to f32, run the five-instruction sequence, round the
// result back. Each is also swept under the math dialect expansion the
// approximation replaces, so the two can be read against each other. The
// baseline column is what --disable-fast-math emits instead, __ocml_tanh_f32
// around the same widening. Reference is __ocml_tanh_f64.
//
// The formats are handled by table rather than by bit twiddling. Each one is
// small enough to enumerate, so rounding an f32 to it is a search for the
// nearest representable value with ties to even, and its ULP at a point is the
// gap between the two representable values bracketing that point. Both are
// exact by construction, which matters more here than speed: the whole sweep is
// under 1300 points.
//
// f8E8M0FNU is left out on purpose. It is the MX scale format, unsigned, with
// no zero and no mantissa, so it cannot hold a tanh result below 2^-127 at all
// and is never an arithmetic operand.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

// log2(e), the scale a `math.exp` picks up on its way to the hardware's
// base-2 exponential. The approximation folds it into its own constant; the
// expansion cannot, because there the exp is a separate op.
constexpr float kLog2E = 1.44269504088896340736f;

struct Fmt {
  const char *name;
  int expBits, mantBits, bias, nBits;
  bool hasInf;   // exponent all ones with zero mantissa is an infinity
  bool topIsNaN; // exponent all ones with full mantissa is a NaN
  bool fnuz;     // the sign bit alone is the only NaN, and there is no -0
};

// Bias and the NaN/Inf rules are what separate these; the OCP formats follow
// IEEE, the FNUZ ones shift the bias by one and spend the whole top of the
// range on finite values.
static const Fmt kFormats[] = {
    {"f8E4M3FN", 4, 3, 7, 8, false, true, false},
    {"f8E4M3FNUZ", 4, 3, 8, 8, false, false, true},
    {"f8E5M2", 5, 2, 15, 8, true, true, false},
    {"f8E5M2FNUZ", 5, 2, 16, 8, false, false, true},
    {"f4E2M1FN", 2, 1, 1, 4, false, false, false},
};
constexpr int kNumFormats = sizeof(kFormats) / sizeof(kFormats[0]);

enum Kind { kFinite = 0, kPosInf = 1, kNegInf = 2, kNaN = 3 };

__device__ __host__ int decode(const Fmt &f, unsigned bits, float &out) {
  unsigned mantMask = (1u << f.mantBits) - 1;
  unsigned expMask = (1u << f.expBits) - 1;
  unsigned mant = bits & mantMask;
  unsigned exp = (bits >> f.mantBits) & expMask;
  unsigned sign = bits >> (f.mantBits + f.expBits);

  if (f.fnuz && sign && exp == 0 && mant == 0)
    return kNaN;
  if (exp == expMask) {
    if (f.hasInf && mant == 0)
      return sign ? kNegInf : kPosInf;
    if (f.hasInf || (f.topIsNaN && mant == mantMask))
      return kNaN;
  }

  float scale = 1.0f / (float)(1u << f.mantBits);
  float v = exp == 0 ? ldexpf((float)mant * scale, 1 - f.bias)
                     : ldexpf(1.0f + (float)mant * scale, (int)exp - f.bias);
  out = sign ? -v : v;
  return kFinite;
}

/// The bit pattern nearest `v`, ties to even. `v` has to be finite and inside
/// the format's range: the search ranks candidates by their distance to `v`,
/// which separates nothing once that distance swamps the whole range. roundTo
/// is the entry point that guarantees it.
__device__ unsigned encode(const Fmt &f, double v) {
  unsigned best = 0;
  double bestDiff = INFINITY;
  for (unsigned b = 0; b < (1u << f.nBits); ++b) {
    float c;
    if (decode(f, b, c) != kFinite)
      continue;
    double d = fabs((double)c - v);
    // Ascending order means the even pattern is already held on a tie.
    if (d < bestDiff) {
      bestDiff = d;
      best = b;
    }
  }
  return best;
}

/// The gap between the two representable magnitudes bracketing |ref|, which is
/// the format's ULP there. Subnormals need no special case: their spacing comes
/// out of the same table.
__device__ double spacing(const Fmt &f, double ref) {
  double r = fabs(ref);
  double below = -1.0, above = INFINITY, top = 0.0, nextTop = 0.0;
  for (unsigned b = 0; b < (1u << f.nBits); ++b) {
    float c;
    if (decode(f, b, c) != kFinite || c < 0.0f)
      continue;
    double cd = c;
    if (cd <= r && cd > below)
      below = cd;
    if (cd > r && cd < above)
      above = cd;
    if (cd > top) {
      nextTop = top;
      top = cd;
    } else if (cd > nextTop && cd < top) {
      nextTop = cd;
    }
  }
  if (below < 0.0)
    below = 0.0;
  if (!isfinite(above)) // |ref| is at or past the largest finite value
    return top - nextTop;
  return above - below;
}

__device__ __forceinline__ float approxTanhF32(float x) {
  float scaled = x * -2.885390082f;
  float e = __builtin_amdgcn_exp2f(scaled);
  float denom = e + 1.0f;
  float recip = __builtin_amdgcn_rcpf(denom);
  return __builtin_fmaf(recip, 2.0f, -1.0f);
}

/// The largest finite magnitude the format holds.
__device__ double maxFinite(const Fmt &f) {
  double top = 0.0;
  for (unsigned b = 0; b < (1u << f.nBits); ++b) {
    float c;
    if (decode(f, b, c) == kFinite && (double)c > top)
      top = c;
  }
  return top;
}

/// `v` rounded to the nearest value the format holds, saturating on overflow,
/// which is what a truncf down to one of these does.
__device__ __forceinline__ float roundTo(const Fmt &f, double v) {
  if (isnan(v))
    return (float)v;
  // Clamping before the search is not just the saturation: encode() ranks
  // candidates by their distance to v, and an out-of-range v makes that
  // distance so large that every candidate ties at it and the first one wins.
  double top = maxFinite(f);
  float r;
  decode(f, encode(f, fmin(fmax(v, -top), top)), r);
  return r;
}

/// The math dialect expansion, the lowering the approximation replaces:
///   sign = x < 0 ? -1 : 1;  sign * (1 - exp(-2*sign*x)) / (1 + exp(-2*sign*x))
///
/// No target has fp8 or fp4 arithmetic, so every op in it is promoted to f32
/// on its own and its result rounded back, the extf/truncf pair the arith
/// emulation wraps around each one. Rounding after each step rather than once
/// at the end is the whole difference from the shipped path, and at three or
/// fewer mantissa bits it is not a small one. The sign steps the expansion
/// really emits are exactly +/-1 with no rounding, so a select stands in.
__device__ float expansionTanh(const Fmt &f, float x) {
  float sign = x < 0.0f ? -1.0f : 1.0f;
  float ax = roundTo(f, sign * x);
  float y = roundTo(f, -2.0f * ax);
  float e = roundTo(f, __builtin_amdgcn_exp2f(y * kLog2E));
  float num = roundTo(f, 1.0f - e);
  float den = roundTo(f, 1.0f + e);
  return roundTo(f, sign * roundTo(f, num * __builtin_amdgcn_rcpf(den)));
}

__device__ __forceinline__ void bump(unsigned *slot, float v) {
  atomicMax(slot, __float_as_uint(v));
}

__global__ void sweep(Fmt f, unsigned *ulpA, unsigned *ulpE, unsigned *ulpO,
                      unsigned *absA, unsigned *absE, unsigned *missA,
                      unsigned *missE, unsigned *missO, unsigned *satBad,
                      unsigned *nPoints) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (1u << f.nBits))
    return;

  float x;
  int kind = decode(f, i, x);
  if (kind == kNaN)
    return;
  if (kind != kFinite) { // an infinity has to saturate to exactly +/-1
    float inf = kind == kPosInf ? INFINITY : -INFINITY;
    float want = kind == kPosInf ? 1.0f : -1.0f;
    if (roundTo(f, approxTanhF32(inf)) != want ||
        expansionTanh(f, inf) != want)
      atomicAdd(satBad, 1u);
    return;
  }
  atomicAdd(nPoints, 1u);

  double ref = tanh((double)x);
  float got = roundTo(f, approxTanhF32(x));
  float expand = expansionTanh(f, x);
  float ocml = roundTo(f, tanhf(x));
  float exact = roundTo(f, ref);

  if (isnan(got) || isnan(expand) || isnan(ocml)) {
    atomicAdd(satBad, 1u);
    return;
  }
  // A correctly rounded result is the same code point the true value rounds to.
  // At these widths that is the claim worth making; the ULP figure is dominated
  // by how coarse the format is rather than by the approximation.
  if (got != exact)
    atomicAdd(missA, 1u);
  if (expand != exact)
    atomicAdd(missE, 1u);
  if (ocml != exact)
    atomicAdd(missO, 1u);

  double u = spacing(f, ref);
  bump(ulpA, (float)(fabs((double)got - ref) / u));
  bump(ulpE, (float)(fabs((double)expand - ref) / u));
  bump(ulpO, (float)(fabs((double)ocml - ref) / u));
  bump(absA, (float)fabs((double)got - ref));
  bump(absE, (float)fabs((double)expand - ref));
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
  unsigned *d;
  // ulpA ulpE ulpO absA absE missA missE missO satBad nPoints
  constexpr int kSlots = 10;
  hipMalloc(&d, kSlots * sizeof(unsigned));

  printf("tanh rounded to narrow floats, exhaustive over every bit pattern.\n"
         "'approx' is what the pass emits, 'expand' the math dialect expansion "
         "it replaces.\n\n");
  printf("%-12s %7s %8s %8s %8s %10s %10s %7s %7s %7s\n", "format", "points",
         "approx", "expand", "ocml", "approx abs", "expand abs", "!RN", "e !RN",
         "o !RN");

  bool bad = false;
  for (int k = 0; k < kNumFormats; ++k) {
    Fmt f = kFormats[k];
    hipMemset(d, 0, kSlots * sizeof(unsigned));
    hipLaunchKernelGGL(sweep, dim3(1), dim3(256), 0, 0, f, d + 0, d + 1, d + 2,
                       d + 3, d + 4, d + 5, d + 6, d + 7, d + 8, d + 9);
    hipDeviceSynchronize();

    unsigned h[kSlots];
    hipMemcpy(h, d, sizeof h, hipMemcpyDeviceToHost);
    printf("%-12s %7u %8.2f %8.2f %8.2f %10.3e %10.3e %7u %7u %7u\n", f.name,
           h[9], asFloat(h[0]), asFloat(h[1]), asFloat(h[2]), asFloat(h[3]),
           asFloat(h[4]), h[5], h[6], h[7]);
    if (h[8] != 0) {
      printf("  saturation failures: %u\n", h[8]);
      bad = true;
    }
  }

  printf("\n  The ULP columns are worst case. '!RN' counts inputs whose result "
         "differs\n  from the correctly rounded one, per lowering.\n");
  hipFree(d);
  return bad;
}
