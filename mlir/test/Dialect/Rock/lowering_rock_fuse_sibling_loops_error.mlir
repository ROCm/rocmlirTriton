// Error tests for rock-fuse-sibling-loops pass.

// RUN: rocmlir-opt -rock-fuse-sibling-loops -verify-diagnostics --split-input-file %s

// ============================================================
// Error: a side-effecting op (memref.store) means the IR is not value-semantic,
// so the sibling loops' independence cannot be proven and fusion is unsafe. The
// pass only runs this early in the pipeline, where every op is memory-effect-
// free; encountering one that is not is a hard error.
// ============================================================

func.func @error_memory_effects(%buf: memref<8xf32>, %v: f32) attributes {rock.kernel} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    // expected-error @below {{rock-fuse-sibling-loops requires value-semantic IR but found an op with memory effects}}
    memref.store %v, %buf[%i] : memref<8xf32>
  }
  scf.for %i = %lb to %ub step %step {
    memref.store %v, %buf[%i] : memref<8xf32>
  }
  return
}
