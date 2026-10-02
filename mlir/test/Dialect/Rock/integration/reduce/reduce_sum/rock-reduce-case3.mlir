// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This test is checking for a reduction in highest dimension

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s \
// RUN:   | rocmlir-gen -fut test_reduce --arch %arch --clone-harness - \
// RUN:   | rocmlir-driver -kernel-pipeline highlevel -host-pipeline highlevel \
// RUN:   | rocmlir-gen -ph -print-results -fut test_reduce --verifier clone -rand none - \
// RUN:   | rocmlir-driver -c -arch %arch \
// RUN:   | rocm-run | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: Unranked Memref base

func.func @test_reduce(%arg0: tensor<20x30x1xf32>, %arg1: tensor<20x1x10xf32>) -> tensor<1x30x10xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %gemm = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) {acc_type = f32} : (tensor<20x30x1xf32>, tensor<20x1x10xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<20x30x10xf32>
  %reduced = "tosa.reduce_sum"(%gemm) {axis = 0 : i32} : (tensor<20x30x10xf32>) -> tensor<1x30x10xf32>
  return %reduced : tensor<1x30x10xf32>
}
