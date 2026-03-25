## 1. Introduction
The code is under `mlir/lib/Dialect/CPU/Transforms`.
We use the [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/) to optimize the CPU code.
You can use the `--cpu-timers` option to measure different parts of CPU runtime.

## 1.1 Directory Structure

```
mlir/lib/Dialect/CPU/Transforms/
├── LowerCpuVerifier.cpp          # Main pass that orchestrates the lowering pipeline
├── Schedules.cpp                 # Creates and applies transform schedules
└── Schedules/                    # Individual transform schedule implementations
    ├── ScheduleUtils.cpp         # Common utilities (module creation, matmul matching)
    ├── TilingSchedule.cpp        # Schedule 1. Tiles linalg.generic ops to prevent huge vectors
    ├── VectorizationSchedule.cpp # Schedule 2. Vectorizes tiled operations
    ├── UnrollSchedule.cpp        # Schedule 3. Unrolls small loops for better codegen
    ├── PrePostSchedules.cpp      # Pre (canonicalize+cse) and Post (LICM+hoisting) passes 
    └── LowerToLLVMSchedule.cpp   # Schedule 4. Bufferization + LLVM dialect lowering
```

## 1.2 Key Components

**`Schedules/` subdirectory** - Each schedule is a C++ file that programmatically builds MLIR transform dialect IR:

| Schedule | Name      | Purpose |
|----------| ----------|---------|
| 1.       | `TilingSchedule` | Tiles elementwise ops (1D/2D/3D) with tile size 8, and matmul with cache-friendly tiles |
| 2.       | `VectorizationSchedule` | Applies `transform.vectorize` to convert loops to vector operations |
| 3.       | `UnrollSchedule` | Unrolls inner loops for better instruction-level parallelism |
| 4.       | `LowerToLLVMSchedule` | Bufferization, vector lowering, and conversion to LLVM dialect |

We also have other 2 files:

| Name               | Purpose |
|--------------------|---------|
| `PrePostSchedules` | Pre: canonicalize+CSE. Post: LICM + redundant transfer hoisting |
| `ScheduleUtils`    | Shared helpers |

## 2. Optimization strategy
The high-level idea is following GOTO's paper [1] while trying to be reasonably simple.
In other words, optimize the code while keeping complexity low.
We want our CPU verifier to be fast enough, not cutting edge performance.

Using MLIR to optimize GEMM for CPU has already been studied in previous work [2] with high success.
In particular, the last trend is to use the transform dialect [3] and upstream MLIR to do so [4].

## 3. Large JIT time issue
## 3.1 Description
With certain IR inputs, we can get huge JIT times if we are not careful (in order of several seconds).
This is because we unconditionally vectorize all IR in the `VectorizationSchedule`. 
So, if the `linalg.generic` was not tiled, it can be vectorized with huge vector sizes, so the JIT will have a very hard time with this IR.

To overcome this issue, in `TilingSchedule`, we make sure that we tile all the `linalg.generic` in the IR to a reasonable vector size (8).

## 3.2 Can we vectorize only linalg.generics that are tiled?
Yes, but we need to use `transform.structured.vectorize` which, in my experience, is less mature compared to `transform.structured.vectorize_children_and_apply_patterns`. The former will attempt to vectorize a specific op, whereas the latter will attempt to vectorize all ops and apply different paterns. The problem with the former is that it's very brittle in practice. If the target op is not ready for vectorization, it will fail. If the target op is ready, but needs some cleanup before vectorization, it will fail as well. The latter, however, will apply cleanup patterns before/after vectorization, so it is more likely that it will work. 

We may want to use `vectorize` in the long term if we find corner cases with `vectorize_children_and_apply_patterns` since it will attempt vectorizing all ops, but in general `vectorize_children_and_apply_patterns` should be more reliable.

# 4. Convolution

```
$ ./build/bin/rocmlir-gen --cpu-timers -batchsize=64 -in_channels=512 -in_h=16 -in_w=16 -out_channels=512 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 --operation conv_bwd_data -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t bf16  --arch gfx950:sramecc+:xnack- -pv   | ./build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner -O0 --shared-libs=/home/pamartin/rocmlirTriton/external/triton/llvm-project/build/./lib/libmlir_rocm_runtime.so,/home/pamartin/rocmlirTriton/build/lib/libconv-validation-wrappers.so,/home/pamartin/rocmlirTriton/external/triton/llvm-project/build/./lib/libmlir_runner_utils.so,/home/pamartin/rocmlirTriton/external/triton/llvm-project/build/./lib/libmlir_float16_utils.so,/home/pamartin/rocmlirTriton/external/triton/llvm-project/build/./lib/libmlir_c_runner_utils.so --entry-point-result=void
JIT compilation time: 186.891 ms
Memory init time: 18.601 ms
GPU kernel time: 95.314 ms


--- CPU Verifier Per-Op Statistics ---
Operation                        Total (ms)    Count     Avg (ms)
------------------------------ ------------ -------- ------------
extf (filter bf16->f32)               2.097        1        2.097
extf (input bf16->f32)                0.000        1        0.000
extf (output bf16->f32)               1.474        1        1.474
fill (zero init)                      7.582        1        7.582
generate (dilate)                     5.484        1        5.484
pad/crop (bwd_data)                  21.274        1       21.274
generate (flip filter)                3.348        1        3.348
generic (convolution)            110650.496        1   110650.496
truncf (f32->bf16)                    5.427        1        5.427
------------------------------ ------------ -------- ------------
Total                            110697.181
--------------------------------------

CPU validation time: 110697.810 ms
```

Seems like we just need to focus on optimizing the convolution kernel itself

## References
[1] https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf
[2] https://arxiv.org/pdf/2003.00532
[3] https://arxiv.org/pdf/2409.03864
[4] https://arxiv.org/pdf/2404.15204
