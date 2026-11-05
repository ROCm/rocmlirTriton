// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen -fut test_fusion --arch %arch --clone-harness - | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel -arch %arch | rocmlir-gen -ph -print-results -fut test_fusion -rand 1 --verifier clone - | rocmlir-driver -c --reduce -arch %arch | rocm-run | FileCheck %s --check-prefix=CLONE
//
// `-rand 1` draws integers in [-5, 5), so every element differs but each sum
// stays exact in f32. The reduction order on the GPU is not the CPU verifier's,
// and only exact arithmetic makes that difference invisible.

module {
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base
  func.func @test_fusion(%arg0: tensor<4x512xf32>, %arg1: tensor<4x512xf32>) -> tensor<4x1xf32> attributes {rock.kernel, rock.reduce, rock.arch = "##TOKEN_ARCH##"} {
    %0 = tosa.add %arg0, %arg1 : (tensor<4x512xf32>, tensor<4x512xf32>) -> tensor<4x512xf32>
    %1 = "tosa.reduce_sum"(%0) {axis = 1 : i32} : (tensor<4x512xf32>) -> tensor<4x1xf32>
    return %1 : tensor<4x1xf32>
  }
}
