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

// RUN: not rocmlir-driver --host-pipeline=backend %s 2>&1 | FileCheck %s

// CHECK: error: --host-pipeline=...,backend cannot be run on a module with kernel functions

module {
  func.func @kernel(%arg0: memref<4xf32>) attributes {rock.kernel} {
    return
  }
  func.func @host(%arg0: memref<4xf32>) {
    call @kernel(%arg0) : (memref<4xf32>) -> ()
    return
  }
}
