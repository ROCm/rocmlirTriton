// Verify that `rocmlir-driver --host-pipeline=...,backend` rejects modules
// that still contain `rock.kernel` functions.
//
// The host backend pipeline ends with `convert-func-to-llvm`, which lowers
// every `func.func` (including kernels and their detach stubs) to
// `llvm.func` and every `func.call` to `llvm.call`. The kernel pipeline's
// `rock-restore-host-code` pass only matches `func::CallOp`, so once host
// backend lowering has run, kernel calls can no longer be rewritten into
// `gpu.launch_func`. The supported way to drive both sides is the two-stage
// flow `--host-pipeline=highlevel | rocmlir-driver -c`, where the host
// backend never runs in the same invocation as the kernel pipeline.
//
// The driver detects this combination up front and emits a diagnostic
// instead of producing invalid IR (or, before the bail-out was added,
// segfaulting in `mlir::reattachFuncs` on the freed kernel stub once
// `convert-func-to-llvm` had erased it -- see `mlir/utils/DetachReattach.cpp`).
//
// The two cases below are split into separate inputs via `split-file`
// because each must be fed to its own `rocmlir-driver` invocation (the
// driver doesn't honour MLIR's `// -----` separator).  Each chunk's RUN
// and CHECK lines live inside the chunk; lit reads them from the original
// file (`%s`) so they are picked up regardless, and MLIR's parser ignores
// them as comments.

// RUN: split-file %s %t

//===----------------------------------------------------------------------===//
// Case 1: kernel without host-side caller. The driver rejects the module
// up front, before any pass runs.
//===----------------------------------------------------------------------===//

//--- no_call.mlir

// RUN: not rocmlir-driver --host-pipeline=backend %t/no_call.mlir 2>&1 | FileCheck %s --check-prefix=NOCALL

// NOCALL: error: --host-pipeline=...,backend cannot be run on a module with kernel functions

module {
  func.func @kernel(%arg0: memref<4xf32>) attributes {rock.kernel} {
    return
  }
  func.func @host_no_call(%arg0: memref<4xf32>) {
    return
  }
}

//===----------------------------------------------------------------------===//
// Case 2: kernel WITH host-side caller. The driver rejects the same way --
// the bail-out is purely on the presence of any `rock.kernel` function, not
// on whether it has callers.
//===----------------------------------------------------------------------===//

//--- with_call.mlir

// RUN: not rocmlir-driver --host-pipeline=backend %t/with_call.mlir 2>&1 | FileCheck %s --check-prefix=WITHCALL

// WITHCALL: error: --host-pipeline=...,backend cannot be run on a module with kernel functions

module {
  func.func @kernel(%arg0: memref<4xf32>) attributes {rock.kernel} {
    return
  }
  func.func @host_with_call(%arg0: memref<4xf32>) {
    call @kernel(%arg0) : (memref<4xf32>) -> ()
    return
  }
}
