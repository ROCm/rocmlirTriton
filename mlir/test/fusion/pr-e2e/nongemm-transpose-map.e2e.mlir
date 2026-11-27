// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen -fut test_fusion --arch %arch --clone-harness - | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel -arch %arch | rocmlir-gen -ph -print-results -fut test_fusion -rand 1 --verifier clone - | rocmlir-driver -c --nongemm -arch %arch | rocm-run | FileCheck %s --check-prefix=CLONE
//
// A view in the middle of the graph, which `rock-regularize-nongemm` pushes
// down to the leaf it came from. A map places no demand on the layout of its
// inputs, so a transposed one is as legal as any other: the tile covers the
// flattened result either way.

module {
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base
  func.func @test_fusion(%arg0: tensor<64x8xf32>, %arg1: tensor<8x64xf32>) -> tensor<8x64xf32> attributes {rock.kernel, rock.nongemm, rock.arch = "##TOKEN_ARCH##"} {
    %t = tosa.transpose %arg0 perms([1, 0]) : (tensor<64x8xf32>) -> tensor<8x64xf32>
    %0 = tosa.add %t, %arg1 : (tensor<8x64xf32>, tensor<8x64xf32>) -> tensor<8x64xf32>
    %1 = tosa.tanh %0 : (tensor<8x64xf32>) -> tensor<8x64xf32>
    return %1 : tensor<8x64xf32>
  }
}
