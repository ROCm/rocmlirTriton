// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen -fut test_fusion --arch %arch --clone-harness - | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel -arch %arch | rocmlir-gen -ph -print-results -fut test_fusion -rand 1 --verifier clone - | rocmlir-driver -c --nongemm -arch %arch | rocm-run | FileCheck %s --check-prefix=CLONE
//
// The shape a nongemm kernel is meant to take: a view, two elementwise ops
// and a reduction at the end. The reshape is pushed down to the leaf, the
// elementwise ops are retyped onto the tile, and the reduction is the only
// thing left needing a schedule.
//
// `-rand 1` draws integers in [-5, 5), so every element differs but each sum
// stays exact in f32. The reduction order on the GPU is not the CPU
// verifier's, and only exact arithmetic makes that difference invisible.

module {
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base
  func.func @test_fusion(%arg0: tensor<8x4x128xf32>, %arg1: tensor<32x128xf32>) -> tensor<32x1xf32> attributes {rock.kernel, rock.nongemm, rock.arch = "##TOKEN_ARCH##"} {
    %s = "tosa.const_shape"() {values = dense<[32, 128]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %r = tosa.reshape %arg0, %s : (tensor<8x4x128xf32>, !tosa.shape<2>) -> tensor<32x128xf32>
    %0 = tosa.add %r, %arg1 : (tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    %1 = tosa.tanh %0 : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %2 = "tosa.reduce_sum"(%1) {axis = 1 : i32} : (tensor<32x128xf32>) -> tensor<32x1xf32>
    return %2 : tensor<32x1xf32>
  }
}
