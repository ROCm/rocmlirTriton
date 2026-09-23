// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Under clone validation, a host-only root is the function under test: it
// must read and write the regular buffers, while the _cpu_host reference reads
// and writes the validation buffers. The verifier then compares the two.
// Small-float parameters used to route the root through the validation
// buffers too, which left the regular result buffer unwritten.

// RUN: rocmlir-gen -ph -pr -fut host_add --verifier clone %s | rocmlir-opt | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK: %[[A:.+]] = memref.alloc() : memref<6xf16>
// CHECK: %[[VAL_A:.+]] = memref.alloc() : memref<6xf16>
// CHECK: %[[B:.+]] = memref.alloc() : memref<6xf16>
// CHECK: %[[VAL_B:.+]] = memref.alloc() : memref<6xf16>
// CHECK: %[[RES:.+]] = memref.alloc() : memref<6xf16>
// CHECK-NEXT: %[[VAL_RES:.+]] = memref.alloc() : memref<6xf16>

// CHECK-NEXT: %[[A_T:.+]] = bufferization.to_tensor %[[A]]
// CHECK-NEXT: %[[B_T:.+]] = bufferization.to_tensor %[[B]]
// CHECK-NEXT: %[[OUT:.+]] = call @host_add(%[[A_T]], %[[B_T]])
// CHECK-NEXT: %[[OUT_M:.+]] = bufferization.to_buffer %[[OUT]]
// CHECK-NEXT: memref.copy %[[OUT_M]], %[[RES]]

// CHECK-NEXT: %[[VAL_A_T:.+]] = bufferization.to_tensor %[[VAL_A]]
// CHECK-NEXT: %[[VAL_B_T:.+]] = bufferization.to_tensor %[[VAL_B]]
// CHECK-NEXT: %[[REF:.+]] = call @host_add_cpu_host(%[[VAL_A_T]], %[[VAL_B_T]])
// CHECK-NEXT: %[[REF_M:.+]] = bufferization.to_buffer %[[REF]]
// CHECK-NEXT: memref.copy %[[REF_M]], %[[VAL_RES]]

// CHECK-NEXT: call @host_add_verify2(%[[RES]], %[[VAL_RES]])
// CHECK: call @_memcpy_f16_f32_6(%[[RES]], %{{.+}})
// CHECK: call @printMemrefF32

func.func @host_add_cpu_host(%arg0: tensor<6xf16>, %arg1: tensor<6xf16>) -> tensor<6xf16> {
  %0 = arith.addf %arg0, %arg1 : tensor<6xf16>
  return %0 : tensor<6xf16>
}

func.func @host_add(%arg0: tensor<6xf16>, %arg1: tensor<6xf16>) -> tensor<6xf16> {
  %0 = arith.addf %arg0, %arg1 : tensor<6xf16>
  return %0 : tensor<6xf16>
}
