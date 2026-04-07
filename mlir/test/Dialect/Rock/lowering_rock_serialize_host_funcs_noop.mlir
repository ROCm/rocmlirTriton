// Test that rock-serialize-host-funcs is a no-op when there are no host
// functions: no rock.host_functions attribute is added and all kernels remain.

// RUN: rocmlir-opt -rock-serialize-host-funcs -mlir-print-local-scope %s | FileCheck %s

// CHECK-NOT: rock.host_functions
// CHECK-LABEL: func.func @kernel1
// CHECK-LABEL: func.func @kernel2

module {
  func.func @kernel1(%a: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.kernel} {
    return %a : tensor<64x64xf32>
  }

  func.func @kernel2(%a: tensor<8x8xf16>) -> tensor<8x8xf16> attributes {rock.kernel} {
    return %a : tensor<8x8xf16>
  }
}
