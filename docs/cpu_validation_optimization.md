## 1. Introduction
The code is under `mlir/lib/Conversion/CPU/Transforms`.
We use the [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/) to optimize the CPU code.
You can use the `--cpu-timers` option to measure different parts of CPU runtime.

## 1.1 Directory Structure

```
mlir/lib/Conversion/CPU/Transforms/
├── LowerCpuVerifier              # Main pass that orchestrates the lowering pipeline
├── ConvToGemm                    # im2col-style conversion of conv ops into fused matmul ops
├── Schedules                     # Creates and applies transform schedules
└── Schedules/                    # Individual transform schedule implementations
    ├── ScheduleUtils             # Common utilities (module creation, matmul matching)
    ├── FusedConvToMatmulSchedule # Schedule 1. Tiles fused conv ops so they become pure matmuls
    ├── TilingSchedule            # Schedule 2. Tiles linalg.generic ops to prevent huge vectors
    ├── VectorizationSchedule     # Schedule 3. Vectorizes tiled operations
    ├── UnrollSchedule            # Schedule 4. Unrolls small loops for better codegen
    ├── PrePostSchedules          # Pre (canonicalize+cse) and Post (LICM+hoisting) passes
    └── LowerToLLVMSchedule       # Schedule 5. Bufferization + LLVM dialect lowering
```

## 1.2 Key Components

**`Schedules/` subdirectory** - Each schedule is a C++ file that programmatically builds MLIR transform dialect IR:

| Schedule | Name      | Purpose |
|----------| ----------|---------|
| 1.       | `FusedConvToMatmulSchedule` | Tiles the fused conv-as-matmul op produced by `ConvToGemm` so its inner body becomes a pure matmul |
| 2.       | `TilingSchedule` | Tiles elementwise ops (1D/2D/3D) with tile size 8, and matmul with cache-friendly tiles |
| 3.       | `VectorizationSchedule` | Applies `transform.vectorize` to convert loops to vector operations |
| 4.       | `UnrollSchedule` | Unrolls inner loops for better instruction-level parallelism |
| 5.       | `LowerToLLVMSchedule` | Bufferization, vector lowering, and conversion to LLVM dialect |

We also have these supporting files:

| Name               | Purpose |
|--------------------|---------|
| `PrePostSchedules` | Pre: canonicalize+CSE. Post: LICM + redundant transfer hoisting |
| `ScheduleUtils`    | Shared helpers (module creation, matmul matching, etc.) |
| `ConvToGemm`       | Non-schedule pass that rewrites convolutions into a single fused matmul-like op (see §5) |

## 2. Transform dialect (this approach) vs. passes (rocMLIR approach)
Note that the best approach may vary depending on the use case. There is no single right answer to this question. 

However, for our specific goals, the transform dialect offers several advantages:

1. **IR-level optimization design**: Optimization and lowering logic is expressed at the IR level (via transform sequences) rather than in C++ code. This makes experimentation and debugging significantly easier and faster, since you can modify the IR directly (and then run `--transform-interpreter` on it) instead of recompiling every time you change a pass.
2. **Ecosystem alignment**: The transform dialect is widely used across CPU compilers in MLIR. Applying upstream MLIR optimizations without it would introduce additional complexity.
3. **Reduced boilerplate**: It requires fewer lines of code compared to implementing each pass in C++. A C++ pass would need much more boilerplate and error-prone implementation, whereas the transform dialect lets you compose existing transform ops that (in theory) are already correct and tested.

## 3. Isolating the CPU function before applying the transforms
Before applying each transform sequence, `LowerCpuVerifier` uses `detachFuncs` and `reattachFuncs` to temporarily remove all functions from the module except the CPU host function. 

This isolation is necessary because transform sequences operate on the entire module rather than individual functions, which could lead to unintended modifications of other functions.

Consider loop pipelining as a concrete example. Suppose our module contains both a CPU host function and a GPU kernel function, each with their own `scf.for` loops:

```mlir
func.func @cpu_host_naive_gemm(...) {
  scf.for %i = ... {  // We want to pipeline this loop
    ...
  }
}

func.func @main(...) {
  scf.for %j = ... {  // We do NOT want to touch this loop
    ...
  }
}
```

If we apply a loop pipelining transform that targets all `scf.for` operations without isolating the CPU function first, the transform would also modify the GPU kernel's loops—which is not what we want. By detaching all other functions before running the transform, we ensure that only the CPU host function is affected.

Previously (in rocMLIR, or before the CPU optimization landed on develop), this level of isolation wasn't necessary. The original CPU code path only performed bufferization and LLVM lowering—passes that are safe to apply to the entire module without side effects on unrelated functions. However, the transform dialect operations used for CPU optimization (such as tiling, vectorization, and loop pipelining) require more precise targeting, making function isolation neccesary.

## 3.1 Debugging: dumping per-step IR and transform schedules

`rocmlir-driver` accepts the optional flag `-dump-cpu-schedules=<dir>`.
When set, the pass writes one `.mlir` file per (function, schedule) pair, capturing the CPU verifier IR *before* the step runs together with the transform sequence that is about to be applied.

## 4. Optimization strategy
The high-level idea is following GOTO's paper [1] while trying to be reasonably simple.
In other words, optimize the code while keeping complexity low.
We want our CPU verifier to be fast enough, not cutting edge performance.

Using MLIR to optimize GEMM for CPU has already been studied in previous work [2] with high success.
In particular, the last trend is to use the transform dialect [3] and upstream MLIR to do so [4].

# 5. Convolution
To optimize convolutions, the approach is using an im2col approach to translate the convolution into a matrix multiplication. Then, the matrix multiplication will be optimized using the standard optimization path. This conversion is implemented in mlir/lib/Conversion/CPU/Transforms/ConvToGemm.cpp. 

Originally, the im2col was implemented like this:

```
%0 = tensor.empty
%1 = linalg.generic  %0 { yield }
%2 = matmul(%2)
```

Where the yield essentially did the im2col layout conversion. For big convs, that tensor.empty (which materialised im2col matrix) can potentially be huge. For example:

```
%0 = tensor.empty() : tensor<1x256x230x230x64x7x7xf32>
```

That case is 170GB!

The solution to avoid huge tensors is to merge the yield with the matmul-like op in a single op, thus avoiding to allocate the whole tensor at once. This has the big disadvantage that the new op is not a matmul anymore (because it's 8D, not 2D/3D and it has multiple reduction dimensions). The solution to that problem is simple: We tile this op later (in FusedConvToMatmulSchedule) and tile only the dimensions that come from the convolution itself. This way, the result is purely a matmul. 

This conversion supports all types of convolutions except for forward convolutions with stride > 1 or dilation > 1, and 3-D convolutions (regardless of direction). If such a convolution is given, it will not be matched and converted into a matmul; it stays as the original convolution-shaped `linalg.generic` and is executed by the standard tiling/vectorization path with no im2col fusion.

### 5.1 Support matrix

The matcher in `ConvToGemm.cpp` walks the `linalg.generic`'s affine maps via `linalg::inferConvolutionDims` and only accepts ops with **exactly 2 output-spatial dims, 2 filter-loop dims, unit strides, and unit dilations**. Everything else is layout- and direction-agnostic. The user-facing flags below describe what `rocmlir-gen` is called with; what matters for the matcher is whatever the verifier function's `linalg.generic` *ends up looking like* after `rocmlir-gen` has lowered it.

| Dimensions | Direction          | Stride | Dilation | Padding | Groups (G ≥ 1) | Converted? | Why                                                                                                                                                                                                                            |
| ---------- | ------------------ | ------ | -------- | ------- | -------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1D         | Forward            | 1      | 1        | any     | any            | ✅         | `rocmlir-gen` has no native 1-D path; we model 1-D as a 2-D conv with `in_h = fil_h = 1`.
| 1D         | Backward           | any    | any      | any     | any            | ✅         | Modeled as 2D (see above). |
| 2D         | Forward            | 1      | 1        | any     | any            | ✅         | Canonical supported case. The group axis is reported by `inferConvolutionDims` as a single `depth` dim, so any `G ≥ 1` matches. |
| 2D         | Forward            | > 1    | 1        | any     | any            | ❌         | No support for strided convs. |
| 2D         | Forward            | 1      | > 1      | any     | any            | ❌         | No support for dilation. |
| 2D         | Backward           | any    | any      | any     | any            | ✅         | `rocmlir-gen` generates bwd-data as a stride-1 / dilation-1 conv (zero-stuffed gradient + rotated filter), so conversion works even with stride > 1 and/or dilation > 1. |
| 3D         | Forward / Backward | any    | any      | any     | any            | ❌         | 3D convs are not supported |

Tests for each row live in `mlir/test/Conversion/CPU/cpu_conv_to_gemm.mlir`.

## References
[1] https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf
[2] https://arxiv.org/pdf/2003.00532
[3] https://arxiv.org/pdf/2409.03864
[4] https://arxiv.org/pdf/2404.15204
