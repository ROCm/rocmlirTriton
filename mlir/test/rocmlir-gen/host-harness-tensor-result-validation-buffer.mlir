// A host-only function that returns a tensor gets a fresh result buffer in the
// harness. With clone validation the harness also keeps validation buffers
// indexed like the regular ones, so the result needs its own validation buffer
// for the CPU reference call and the verifier to use.

// RUN: rocmlir-gen -ph -pr -fut host_collapse --verifier clone %s | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK: %[[IN:.+]] = memref.alloc() : memref<2x3xf32>
// CHECK: %[[VAL_IN:.+]] = memref.alloc() : memref<2x3xf32>
// CHECK: %[[RES:.+]] = memref.alloc() : memref<6xf32>
// CHECK-NEXT: %[[VAL_RES:.+]] = memref.alloc() : memref<6xf32>
// CHECK: %[[OUT:.+]] = call @host_collapse(%{{.+}}) : (tensor<2x3xf32>) -> tensor<6xf32>
// CHECK-NEXT: %[[OUT_BUF:.+]] = bufferization.to_buffer %[[OUT]]
// CHECK-NEXT: memref.copy %[[OUT_BUF]], %[[VAL_RES]]
// CHECK: %[[REF:.+]] = call @host_collapse_cpu_host(%{{.+}}) : (tensor<2x3xf32>) -> tensor<6xf32>
// CHECK-NEXT: %[[REF_BUF:.+]] = bufferization.to_buffer %[[REF]]
// CHECK-NEXT: memref.copy %[[REF_BUF]], %[[VAL_RES]]
// CHECK-NEXT: call @host_collapse_verify1(%[[RES]], %[[VAL_RES]])
// CHECK-DAG: memref.dealloc %[[RES]]
// CHECK-DAG: memref.dealloc %[[VAL_RES]]

func.func @host_collapse_cpu_host(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  return %0 : tensor<6xf32>
}

func.func @host_collapse(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  return %0 : tensor<6xf32>
}
