// Exercise rock-rewrite-div-by-reciprocal: each arith.divf is replaced with the
// same operands and type, with fastmath extended by `arcp`.

// RUN: rocmlir-opt -rock-rewrite-div-by-reciprocal -mlir-print-local-scope %s | FileCheck %s

// Further lowering (scalar `arith.divf` → `llvm.fdiv`; tensor case still needs
// a tensor/vector pipeline before `arith` fully disappears). Example:
//   rocmlir-opt -rock-rewrite-div-by-reciprocal \
//     -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts %s

module {

  // CHECK-LABEL: func.func @divf_scalar_adds_arcp
  // CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<arcp> : f32
  func.func @divf_scalar_adds_arcp(%a: f32, %b: f32) -> f32 {
    %0 = arith.divf %a, %b : f32
    return %0 : f32
  }

  // CHECK-LABEL: func.func @divf_tensor_adds_arcp
  // CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<arcp> : tensor<2x3xf32>
  func.func @divf_tensor_adds_arcp(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = arith.divf %x, %y : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // Prior fast-math bits are kept; `arcp` is merged in.
  // CHECK-LABEL: func.func @divf_preserves_other_fastmath
  // CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<arcp> : f32
  func.func @divf_preserves_other_fastmath(%a: f32, %b: f32) -> f32 {
    %0 = arith.divf %a, %b fastmath<nnan> : f32
    return %0 : f32
  }
}
