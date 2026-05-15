# Allclose comparator

This document describes the `allclose`-style verification comparator in
rocmlirTriton: how other projects in the same space approach the problem,
and what rocmlirTriton's current implementation does.

## 1. Comparison of different projects

### 1.1 Summary table

| Project | Formula | Per-dtype defaults? | Per-K scaling? | Per-op overrides? |
|---------|---------|---------------------|----------------|-------------------|
| **NumPy** | `\|a - b\| <= atol + rtol*\|b\|`, asymmetric | No (single global) | No | No (caller's responsibility) |
| **PyTorch** | `\|a - b\| <= atol + rtol*\|b\|`, asymmetric | Yes (4 floating dtypes) | No (built-in) | Yes (`toleranceOverride` decorator) |
| **rocBLAS** (the canonical AMD GEMM tester) | Two parallel paths: element-wise `near_check` with K-scaled tolerance, OR Frobenius `norm_check` | Yes (per-T `sum_error_tolerance`) | Yes (literally multiplies by `K`) | Yes (init method selects path) |
| **hipBLASLt** | Inherits rocBLAS's `near_check` / `norm_check` pattern; picks one based on data characteristics | Same as rocBLAS | Same as rocBLAS | Yes (`unit_check` vs `norm_check` per test) |

### 1.2 NumPy

The simplest case, and the original of the formula. The comparator is
[`numpy.allclose`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)
(and its assertion-throwing sibling
[`numpy.testing.assert_allclose`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html)):

```
absolute(a - b) <= atol + rtol * absolute(b)
```

with defaults `rtol=1e-5, atol=1e-8`. The formula is asymmetric in `a` and
`b`; the docs explicitly note this:

> "The above equation is not symmetric in `a` and `b`, so that
> `allclose(a, b)` might be different from `allclose(b, a)` in some rare
> cases."

NumPy has no per-dtype defaults and no built-in K-aware scaling. The user is
expected to pick `atol` and `rtol` for their workload. This works for
NumPy's intended audience (Python users testing their own scientific code)
but pushes the calibration problem to every downstream test author.

### 1.3 PyTorch -- `torch.testing.assert_close`

PyTorch uses the same asymmetric formula as NumPy. Defaults are per-dtype,
from [`torch/testing/_comparison.py`](https://github.com/pytorch/pytorch/blob/e9ebbd3b/torch/testing/_comparison.py#L49-L57):

```python
_DTYPE_PRECISIONS = {
    torch.float16:   (0.001, 1e-5),
    torch.bfloat16:  (0.016, 1e-5),
    torch.float32:   (1.3e-6, 1e-5),
    torch.float64:   (1e-7, 1e-7),
    torch.complex32: (0.001, 1e-5),
    torch.complex64: (1.3e-6, 1e-5),
    torch.complex128:(1e-7, 1e-7),
}
```

These tuples are `(rtol, atol)`. Calling `assert_close(a, b)` without
explicit tolerances picks them up from the dtype of `b`.

#### 1.3.1 The defaults are not used for matmul / reduction tests

The fp32 default `rtol=1.3e-6` is calibrated for element-wise ops and very
small reductions. For matmul, conv, attention, and other reduction-heavy
kernels, PyTorch's own tests **override these defaults** using the
`toleranceOverride` decorator from
`torch.testing._internal.common_device_type`. Real examples from the live
[`test/test_matmul_cuda.py`](https://github.com/pytorch/pytorch/blob/main/test/test_matmul_cuda.py):

```python
# Line 205-207 (FP8 matmul test):
@toleranceOverride({torch.float16:  xtol(atol=1e-4, rtol=1e-4),
                    torch.bfloat16: xtol(atol=1e-4, rtol=1e-4),
                    torch.float32:  xtol(atol=1e-4, rtol=1e-4)})

# Line 220-221 (mixed-dtype gemm):
@toleranceOverride({torch.float16:  xtol(atol=2e-3, rtol=2e-3),
                    torch.bfloat16: xtol(atol=2e-3, rtol=2e-3)})

# Line 299 (scaled mm fp16):
@toleranceOverride({torch.float16: xtol(atol=1e-3, rtol=3e-3)})
```

and from [`test/test_linalg.py`](https://github.com/pytorch/pytorch/blob/main/test/test_linalg.py):

```python
# Line 10352-10355 (large reduction test):
@toleranceOverride({
    torch.float32:  tol(atol=1e-05, rtol=1e-05),
    torch.float16:  tol(atol=0.6,   rtol=1e-03),    # <-- atol=0.6
    torch.bfloat16: tol(atol=5.0,   rtol=1e-03)})   # <-- atol=5.0
```

The fp16 and bf16 atol values here -- `0.6` and `5.0` -- are six and seven
orders of magnitude looser than the per-dtype default `atol=1e-5`. They are
not "loose because the test author was sloppy"; they are loose because that
is the actual precision floor for the underlying computation.

#### 1.3.2 Takeaway for us

PyTorch's `_DTYPE_PRECISIONS` table is the right baseline for element-wise
ops, but it is not the right default for GEMM/conv/attention. If we want to
pass realistic GEMM/attention tests without per-test `-atol`/`-rtol`
overrides, we need either (a) a calibration story for K, or (b) per-op
defaults that loosen for reduction-bearing ops.

### 1.4 rocBLAS -- the established AMD pattern

[rocBLAS](https://github.com/ROCm/rocBLAS) is the authoritative reference
for "how does AMD test GEMM." Source files of interest:

- [`clients/include/blas3/testing_gemm.hpp`](https://github.com/ROCm/rocBLAS/blob/develop/clients/include/blas3/testing_gemm.hpp)
- [`clients/include/near.hpp`](https://github.com/ROCm/rocBLAS/blob/develop/clients/include/near.hpp)
- [`clients/include/norm.hpp`](https://github.com/ROCm/rocBLAS/blob/develop/clients/include/norm.hpp)

rocBLAS runs **either or both** of two checks per GEMM, gated by
command-line flags `--unit_check` and `--norm_check`. The two checks answer
different questions:

- `near_check_general` -- element-wise: "is every element within tolerance?"
- `norm_check_general` -- aggregate: "is the relative Frobenius-norm error
  within tolerance?"

#### 1.4.1 Path A: `near_check_general` with K-scaled tolerance

The tolerance is computed from the reduction dimension `K`. From
[`testing_gemm.hpp` (rocm-6.2.2)](https://github.com/ROCm/rocBLAS/blob/rocm-6.2.2/clients/include/blas3/testing_gemm.hpp):

```cpp
if(arg.unit_check)
{
    if(arch_major == 11 && std::is_same_v<TiA, rocblas_half>) {
        const double tol = K * sum_error_tolerance_for_gfx11<T, T, T>;
        near_check_general<T>(M, N, ldc, hC_gold, hC_1, tol);
    } else {
        const double tol = K * sum_error_tolerance<T>;
        near_check_general<T>(M, N, ldc, hC_gold, hC_1, tol);
    }
}
```

`tol = K * sum_error_tolerance<T>` -- the tolerance is literally K times a
per-type constant. From [`near.hpp`](https://github.com/ROCm/rocBLAS/blob/develop/clients/include/near.hpp):

```cpp
template <typename T>
inline constexpr double sum_error_tolerance = get_epsilon<T>();

template <> inline constexpr double sum_error_tolerance<rocblas_half>     = 1 / 100.0;
template <> inline constexpr double sum_error_tolerance<rocblas_bfloat16> = 1 / 900.0;
template <> inline constexpr double sum_error_tolerance<float>            = 1 / 10000.0;
template <> inline constexpr double sum_error_tolerance<double>           = 1 / 1000000.0;
```

So for fp32 GEMM with `K=4096`, rocBLAS's effective absolute tolerance is
`4096/10000 = 0.4096` -- not `1e-5` like PyTorch's default.

There is also a separate `gfx11`-specific table with **looser** values
(e.g., fp16 -> `1/10` instead of `1/100`) because gfx11 GPUs use
lower-precision wmma intrinsics. This shows AMD already encodes
per-architecture tolerance differences.

The `arg.initialization` field controls when to use this path:

```cpp
template <typename T>
inline bool reduction_requires_near(const Arguments& arg, int64_t n) {
    return arg.initialization == rocblas_initialization::hpl
        || (std::is_same_v<T, rocblas_half> && n > 10000);
}
```

The tolerance method is selected based on input distribution AND `K`. HPL
(LINPACK-style) initialization produces inputs of size `O(1)` per element,
so the natural growth of accumulation error matters; for unit-valued
integer inputs, the GPU result should be bit-exact.

#### 1.4.2 Path B: `norm_check_general` with relative Frobenius error

From [`norm.hpp`](https://github.com/ROCm/rocBLAS/blob/develop/clients/include/norm.hpp):

```cpp
double cpu_norm = ref_lapack_xlange(norm_type, M, N, hCPU_double, M, work);
m_axpy_64(size, &alpha, hCPU_double, incx, hGPU_double, incx);
double error
    = ref_lapack_xlange(norm_type, M, N, hGPU_double, M, work) / cpu_norm;
return error;
```

In words: this computes `||CPU - GPU||_F / ||CPU||_F` -- the Frobenius norm
of the *difference matrix*, normalized by the Frobenius norm of the
reference. The result is a single scalar you compare against a threshold.
The threshold can be much tighter than the per-element threshold because
individual outliers get averaged into the global norm.

This is conceptually closer to our existing legacy `RMS` metric, but
**normalized by the reference norm** (which our RMS metric is not -- ours
normalizes by `maxMag`, which is sensitive to outliers).

### 1.5 hipBLASLt -- same pattern as rocBLAS, with explicit gating

[hipBLASLt](https://github.com/ROCm/hipBLASLt) inherits rocBLAS's
`unit_check` / `norm_check` infrastructure and uses
[PR #674](https://github.com/ROCm/hipBLASLt/pull/674) to switch between them
based on data characteristics. From the PR description:

> "**unit_check** is appropriate when ScaleC/D is false (scale factors are 0
> or false), test initialization uses integer value 1, all initializations
> are integers."
>
> "**norm_check** is appropriate when ScaleC/D is true with F8 or B8 output
> types, initialization uses random_small values [0.0~1.0] with fractions,
> the tolerance of F8 (0.125) or B8 (0.25) cannot be ignored."

So hipBLASLt explicitly acknowledges that fp8/bf8 per-element tolerance is
so loose (`0.125` / `0.25`) that element-wise checking is meaningless --
you have to fall back to the norm-based check. The bf8 `atol` of `0.25` is
**25,000x looser** than PyTorch's bf16 default of `1e-5`.

## 2. The rocmlirTriton approach

rocmlirTriton blends the PyTorch and rocBLAS patterns: same asymmetric
allclose formula as PyTorch, per-dtype baselines from PyTorch's
`_DTYPE_PRECISIONS`, plus a K-scaled atol term lifted from rocBLAS's
`near_check`.

### 2.1 Formula

The allclose comparator (`mcpuVerifyFloatAllclose` in
`mlir/lib/ExecutionEngine/conv-validation-wrappers.cpp`) uses the same
asymmetric NumPy/PyTorch formula:

```
|a - b| <= atol + rtol * |b|
```

where `b` is the CPU reference and `a` is the GPU output. The kernel
passes if *every* element satisfies this predicate; otherwise it fails.


### 2.2 Per-dtype baseline tolerances

`allcloseBaseline(Type)` in `rocmlir-gen.cpp` mirrors PyTorch's
`_DTYPE_PRECISIONS`, extended with placeholder values for the f8 / f4
dtypes PyTorch does not cover:

| dtype | atol | rtol |
|---|---|---|
| fp16 | `1e-5` | `1e-3` |
| bf16 | `1e-5` | `1.6e-2` |
| fp32 | `1e-5` | `1.3e-6` |
| fp64 | `1e-7` | `1e-7` |
| fp8 (all variants) | `1e-2` | `1.6e-2` |
| fp4 | `1e-2` | `1.25e-1` |

These are the *element-wise* defaults -- what you would use for an
add/mul/relu. They are deliberately tight for reduction-heavy kernels;
that gap is closed by the K-scaling below.

### 2.3 K-scaled atol (rocBLAS-style)

For reduction-bearing operations (GEMM, conv, attention) the effective
atol is computed as:

```
atol_eff = baseAtol + K_eff * sum_error_tolerance<T>
```

This mirrors rocBLAS's `tol = K * sum_error_tolerance<T>` in
`clients/include/blas3/testing_gemm.hpp`, with the PyTorch baseline added
back so element-wise ops (K_eff=1) match PyTorch's defaults.

`sumErrorTolerance(Type)` in `rocmlir-gen.cpp` uses rocBLAS's constants for
fp16/bf16/fp32/fp64, plus extrapolated values for the f8/f4 dtypes
rocBLAS does not test:

| dtype | sum_error_tolerance |
|---|---|
| fp16 | `1/100` |
| bf16 | `1/900` |
| fp32 | `1/10000` |
| fp64 | `1/1000000` |
| fp8 (all variants) | `1/100` |
| fp4 | `1/10` |

Worked example -- fp32 GEMM with K=4096:

```
atol_eff = 1e-5 + 4096 * 1e-4 = 0.4097
```

vs PyTorch's flat `1e-5` (would fail) or rocBLAS's flat `0.4096` (matches).

### 2.4 `K_eff` per operation

`computeReductionK(GenParams)` returns the dot-product length that drives
accumulation error growth:

| Operation | K_eff |
|---|---|
| GEMM | `K` |
| Attention | `head_dim_qk + seq_len_k` (additive: two cascaded reductions) |
| Conv (any direction) | `Cin * product(filter_spatial_dims)`, i.e. the im2col K |
| Element-wise / other | `1` |

The attention model is the conservative choice: errors from the QK^T
reduction (over `head_dim_qk`) and the softmax-weighted-V reduction (over
`seq_len_k`) are treated as independent and additive. A multiplicative
model would be tighter but would also overpredict error for short
sequences; the additive model is what mirrors the actual numerical
behavior we see in tests.

Conv excludes the K (output channel) and G (group) dimensions because they
do not participate in the per-output-element reduction.

### 2.5 Manual overrides

`-atol=<value>` and `-rtol=<value>` override the computed defaults
component-wise. Either flag implies `--comparator=allclose`. Use these for
per-test calibration when the dtype default is wrong for a specific
kernel shape -- analogous to PyTorch's `toleranceOverride` decorator.

### 2.6 IR scanning for clone-harness flows

Many tests do not run `rocmlir-gen` with `-operation gemm/conv/attention`;
they use `--clone-harness` and a pre-lowered MLIR kernel from
TOSA/MIGraphX. In those cases, we cannot infer the K dimension easily from the
command invocation.

In such case, we scan the IR (`rock.gemm`, `rock.conv`, etc) and take the
largest `K_eff` across all matmul-like ops. For chained reductions
(e.g. `tosa.reduce_sum(tosa.matmul(A, B))`), the scanner walks
`rock.reduce` ops backward through `rock.transform` and shape-preserving
elementwise ops (`arith.addf`, `arith.mulf`, `arith.extf` ...) to find
the matmul that feeds them. The matmul's K is then *multiplied* by the
reduce axis extent, since each output element of the reduce accumulates
`K_gemm * N_reduce` products into a single scalar. 

This is the classical forward-error bound for chained inner products: 

`err = O(K * N * eps)`

([Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
2002, ch. 3](https://epubs.siam.org/doi/book/10.1137/1.9780898718027)).

There is no library that scales tolerances this way --
rocBLAS / cuBLAS / hipBLASLt test primitive GEMM only and so do not
face cases like this one, but the rule is necessary once kernels
fuse a matmul with a reduction: a `(K=64) x (N=256)` fp16 chain needs
`atol ≈ 16 384 / 100 ≈ 164` ulps, not `64 / 100 ≈ 0.64`.

### 2.7 Precision floor from the narrowest float in the dataflow

Some kernels compute in a narrow dtype and only widen at the very end
(e.g. `rock.gemm <f16> -> arith.extf f16 to f32 -> return f32`). The
output dtype declares fp32, but the values being returned are at best
fp16-accurate. In those cases, we scan the IR for the narrowest float type appearing
on any matmul-like op (`getAType` / `getBType` / `getCType` / `getOutType`
via `RockGemmWrapperInterface` and `RockGemmGemmWrapperInterface`). If
that is narrower than the kernel's output dtype, that narrower dtype's
baseline is used instead. This mirrors how hipBLASLt implicitly switches
to `norm_check` for f8/b8 output types (see § 1.5): the test must not
demand precision the kernel cannot deliver.

## 3. References

- NumPy: <https://numpy.org/doc/stable/reference/generated/numpy.allclose.html>
- PyTorch `_DTYPE_PRECISIONS`: <https://github.com/pytorch/pytorch/blob/e9ebbd3b/torch/testing/_comparison.py#L49-L57>
- PyTorch `toleranceOverride` (matmul): <https://github.com/pytorch/pytorch/blob/main/test/test_matmul_cuda.py>
- PyTorch `toleranceOverride` (linalg): <https://github.com/pytorch/pytorch/blob/main/test/test_linalg.py>
- rocBLAS `testing_gemm.hpp`: <https://github.com/ROCm/rocBLAS/blob/develop/clients/include/blas3/testing_gemm.hpp>
- rocBLAS `near.hpp`: <https://github.com/ROCm/rocBLAS/blob/develop/clients/include/near.hpp>
- rocBLAS `norm.hpp`: <https://github.com/ROCm/rocBLAS/blob/develop/clients/include/norm.hpp>
- hipBLASLt PR #674 (unit_check vs norm_check): <https://github.com/ROCm/hipBLASLt/pull/674>
- Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed., SIAM, 2002), ch. 3 -- forward-error bounds for chained inner products: <https://epubs.siam.org/doi/book/10.1137/1.9780898718027>
- rocmlirTriton implementation: `mlir/tools/rocmlir-gen/rocmlir-gen.cpp` (`computeReductionK`, `sumErrorTolerance`, `allcloseBaseline`); `mlir/lib/ExecutionEngine/conv-validation-wrappers.cpp` (`mcpuVerifyFloatAllclose`)
