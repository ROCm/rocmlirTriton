// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen -fut test_fusion --arch %arch --clone-harness - | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel -arch %arch | rocmlir-gen -ph -print-results -fut test_fusion -rand 1 -rand_type float --verifier clone - | rocmlir-driver -c --elementwise -arch %arch | rocm-run | FileCheck %s --check-prefix=CLONE

module {
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base
  func.func @test_fusion(%arg0: tensor<1x64xf32>, %arg1: tensor<1x64xf32>) -> tensor<1x64xf32> attributes {rock.kernel, rock.elementwise, rock.arch = "##TOKEN_ARCH##", rock.block_size = 64 : i32} {
    %0 = tosa.add %arg0, %arg1 : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
    return %0 : tensor<1x64xf32>
  }
}
