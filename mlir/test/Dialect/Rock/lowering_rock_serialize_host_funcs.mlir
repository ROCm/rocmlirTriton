// Unit tests for rock-serialize-host-funcs pass.
// The pass serializes non-kernel functions into a "rock.host_functions"
// module attribute (as an array of strings) and erases them.
// Kernel functions (with rock.kernel attr) are left unchanged.

// RUN: rocmlir-opt -rock-serialize-host-funcs -mlir-print-local-scope %s | FileCheck %s

// The module attribute contains serialized strings for both host funcs.
// CHECK: rock.host_functions = [
// CHECK-SAME: "func.func @host_add
// CHECK-SAME: "func.func @host_identity

// Kernel function is preserved as-is.
// CHECK-LABEL: func.func @kernel
// CHECK-SAME: attributes {rock.kernel}

// Host functions are erased from the module body.
// CHECK-NOT: func.func @host_add
// CHECK-NOT: func.func @host_identity

module {
  func.func @kernel(%a: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.kernel} {
    return %a : tensor<64x64xf32>
  }

  func.func @host_add(%x: f32) -> f32 {
    %r = arith.addf %x, %x : f32
    return %r : f32
  }

  func.func @host_identity(%x: i32) -> i32 {
    return %x : i32
  }
}
