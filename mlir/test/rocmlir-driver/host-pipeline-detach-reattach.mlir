// The two cases below are split into separate inputs via `split-file`
// because each must be fed to its own `rocmlir-driver` invocation.

// RUN: split-file %s %t

// Case 1: kernel without host-side caller.
// The kernel is detached, the host-backend pipeline runs `convert-func-to-llvm`
// (erasing the stub), and reattach must re-find the symbol and splice the real
// kernel back in.

//--- no_call.mlir

// RUN: rocmlir-driver --host-pipeline=backend %t/no_call.mlir -o - | FileCheck %s --check-prefix=NOCALL

// NOCALL-LABEL: module
// NOCALL:       func.func @kernel
// NOCALL-SAME:    attributes {rock.kernel}
// NOCALL:       llvm.func @host_no_call

module {
  func.func @kernel(%arg0: memref<4xf32>) attributes {rock.kernel} {
    return
  }
  func.func @host_no_call(%arg0: memref<4xf32>) {
    return
  }
}

// Case 2: kernel with host-side caller -- documents a known limitation of
// the current pipeline.
//
// When the host code calls into the kernel (`func.call @kernel`),
// `convert-func-to-llvm` lowers the call to `llvm.call @kernel` while the
// kernel itself is still detached out of the module.  After reattach, the
// real `func.func @kernel` is spliced back in, but the LLVM dialect verifier
// rejects the module because `llvm.call` requires its callee to be an
// `llvm.func` / IFunc / alias -- not a `func.func`.

//--- with_call.mlir

// RUN: rocmlir-driver --host-pipeline=backend %t/with_call.mlir -o - | not rocmlir-opt 2>&1 | FileCheck %s --check-prefix=CALL-INVALID

// CALL-INVALID: 'llvm.call' op 'kernel' does not reference a valid LLVM function

module {
  func.func @kernel(%arg0: memref<4xf32>) attributes {rock.kernel} {
    return
  }
  func.func @host_with_call(%arg0: memref<4xf32>) {
    call @kernel(%arg0) : (memref<4xf32>) -> ()
    return
  }
}
