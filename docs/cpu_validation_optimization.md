## 1. Introduction
The code is under `mlir/lib/Dialect/CPU/Transforms`.
We use the [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/) to optimize the CPU code.

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

## 3. Issues 
## 3.1 Large JIT time
With certain IR inputs, we can get huge JIT times if we are not careful (in order of several seconds).
This is because we unconditionally vectorize all IR in the `VectorizationSchedule`. 
So, if the `linalg.generic` was not tiled, it can be vectorized with huge vector sizes, so the JIT will have a very hard time with this IR.

To overcome this issue, in `TilingSchedule`, we make sure that we tile all the `linalg.generic` in the IR to a reasonable vector size (8).

## 3.2 Loop Pipelining 

I tried pipelining but it does not work

### The Problem

MLIR's `transform.loop.pipeline` fails on the innermost K-reduction loop because of a 
loop-carried dependency through the accumulator tensor.

Current IR structure:
```mlir
%result = scf.for %k = ... iter_args(%acc_tensor = %init) -> (tensor) {
  %acc_tile = vector.transfer_read %acc_tensor[...]   // Read accumulator every iteration
  %a = vector.transfer_read %A[...]
  %b = vector.transfer_read %B[...]
  %new_acc = vector.contract %a, %b, %acc_tile
  %out = vector.transfer_write %new_acc, %acc_tensor[...]
  scf.yield %out
}
```

The pipeliner's scheduling algorithm (`loopScheduling` in SCFTransformOps.cpp) ignores 
block arguments when computing dependencies (`if (!def) continue`), so it doesn't account 
for the recurrence through `%acc_tensor`. The schedule verification then fails with 
"operation scheduled before its operands".

### The Solution: Pad/Pack with Register-Resident Accumulator

Following the BLIS/Goto approach (see papers above), the solution is to:

1. **Pack A and B** into contiguous, cache-friendly layouts
2. **Hoist the accumulator** out of the innermost loop, keeping it in vector registers

Target IR structure:
```mlir
%acc_vec = vector.transfer_read %C_tile[...]          // Load accumulator ONCE

%result = scf.for %k = ... iter_args(%acc = %acc_vec) -> (vector) {  // Vector, not tensor!
  %a = vector.transfer_read %packed_A[%k, ...]        // Only input reads in loop
  %b = vector.transfer_read %packed_B[%k, ...]
  %new_acc = vector.contract %a, %b, %acc             // Accumulator in registers
  scf.yield %new_acc                                   // No memory write
}

vector.transfer_write %result, %C_tile[...]           // Store accumulator ONCE
```

I think this should work, because it eliminates the tensor recurrence—the iter_arg becomes a `vector` type that stays in registers. The only memory operations in the inner loop are reads from the (packed) A and B matrices, which CAN be pipelined with `read_latency > 0`.

## References
[1] https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf
[2] https://arxiv.org/pdf/2003.00532
[3] https://arxiv.org/pdf/2409.03864
[4] https://arxiv.org/pdf/2404.15204
