## 1. Introduction
The code is under `mlir/lib/Conversion/CPU/Transforms`.
We use the [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/) to optimize the CPU code.
You can use the `--cpu-timers` option to measure different parts of CPU runtime.

## 1.1 Directory Structure

```
mlir/lib/Conversion/CPU/Transforms/
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

## 4. Optimization strategy
The high-level idea is following GOTO's paper [1] while trying to be reasonably simple.
In other words, optimize the code while keeping complexity low.
We want our CPU verifier to be fast enough, not cutting edge performance.

Using MLIR to optimize GEMM for CPU has already been studied in previous work [2] with high success.
In particular, the last trend is to use the transform dialect [3] and upstream MLIR to do so [4].

## 5. Large JIT time issue
## 5.1 Description
With certain IR inputs, we can get huge JIT times if we are not careful (in order of several seconds).
This is because we unconditionally vectorize all IR in the `VectorizationSchedule`. 
So, if the `linalg.generic` was not tiled, it can be vectorized with huge vector sizes, so the JIT will have a very hard time with this IR.

To overcome this issue, in `TilingSchedule`, we make sure that we tile all the `linalg.generic` in the IR to a reasonable vector size (8).

## 5.2 Can we vectorize only linalg.generics that are tiled?
Yes, but we need to use `transform.structured.vectorize` which, in my experience, is less mature compared to `transform.structured.vectorize_children_and_apply_patterns`. The former will attempt to vectorize a specific op, whereas the latter will attempt to vectorize all ops and apply different paterns. The problem with the former is that it's very brittle in practice. If the target op is not ready for vectorization, it will fail. If the target op is ready, but needs some cleanup before vectorization, it will fail as well. The latter, however, will apply cleanup patterns before/after vectorization, so it is more likely that it will work. 

We may want to use `vectorize` in the long term if we find corner cases with `vectorize_children_and_apply_patterns` since it will attempt vectorizing all ops, but in general `vectorize_children_and_apply_patterns` should be more reliable.

## References
[1] https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf
[2] https://arxiv.org/pdf/2003.00532
[3] https://arxiv.org/pdf/2409.03864
[4] https://arxiv.org/pdf/2404.15204
