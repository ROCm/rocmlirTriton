https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf
https://arxiv.org/pdf/2003.00532
https://arxiv.org/pdf/2404.15204
https://arxiv.org/pdf/2409.03864

## Large JIT time
With certain IR inputs, we can get huge JIT times if we are not careful (in order of several seconds).
This is because we unconditionally vectorize all IR in the `VectorizationSchedule`. 
So, if the `linalg.generic` was not tiled, it can be vectorized with huge vector sizes, so the JIT will have a very hard time with this IR.

To overcome this issue, in `TilingSchedule`, we make sure that we tile all the `linalg.generic` in the IR to a reasonable vector size (8).

## Loop Pipelining 

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