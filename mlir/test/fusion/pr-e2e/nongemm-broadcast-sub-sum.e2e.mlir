// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen -fut test_fusion --arch %arch --clone-harness - | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel -arch %arch | rocmlir-gen -ph -print-results -fut test_fusion -rand 1 --verifier clone - | rocmlir-driver -c --nongemm -arch %arch | rocm-run | FileCheck %s --check-prefix=CLONE
//
// The two shapes a broadcast leaf can take ahead of a reduction. `%arg1`
// spreads along the reduced axis, which is the `x - mean` an earlier
// reduction would have produced: every lane of a row reads the same element
// of it, uniform rather than strided. `%arg2` spreads along the rows, so its
// row is read by every workgroup.
//
// `-rand 1` draws integers in [-5, 5), so every element differs but each sum
// stays exact in f32. The reduction order on the GPU is not the CPU
// verifier's, and only exact arithmetic makes that difference invisible.

module {
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base
  func.func @test_fusion(%arg0: tensor<8x512xf32>, %arg1: tensor<8x1xf32>, %arg2: tensor<1x512xf32>) -> tensor<8x1xf32> attributes {rock.kernel, rock.nongemm, rock.arch = "##TOKEN_ARCH##"} {
    %0 = tosa.sub %arg0, %arg1 : (tensor<8x512xf32>, tensor<8x1xf32>) -> tensor<8x512xf32>
    %shift = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %1 = "tosa.mul"(%0, %arg2, %shift) : (tensor<8x512xf32>, tensor<1x512xf32>, tensor<1xi8>) -> tensor<8x512xf32>
    %2 = "tosa.reduce_sum"(%1) {axis = 1 : i32} : (tensor<8x512xf32>) -> tensor<8x1xf32>
    return %2 : tensor<8x1xf32>
  }
}
