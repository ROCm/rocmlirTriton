// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The epilogue after conv+gemm's second GEMM has to survive being applied once
// per split and then summed by the atomic_add. A round trip through an integer
// type is neither linear nor zero-preserving, so it is only fusible when the
// split factor is 1.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,4,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:0
// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @rock_conv_gemm(%filter: tensor<1x4x1x1x2xf32>, %input: tensor<2x2x2x1x2xf32>, %c: tensor<1x4x3xf32>, %out: tensor<1x8x3xf32>) -> tensor<1x8x3xf32> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %r = rock.conv_elementwise_gemm{
     ab = conv(%filter, %input) : tensor<1x4x1x1x2xf32>, tensor<2x2x2x1x2xf32>
     ab = elementwise {
     ^bb0(%ab_in: tensor<1x4x8xf32>):
       rock.yield %ab_in : tensor<1x4x8xf32>
     }
     out = ab * %c : tensor<1x4x3xf32>
    } {dilations = [1 : index, 1 : index],
       filter_layout = ["g", "k", "0", "1", "c"],
       input_layout = ["ni", "0i", "1i", "gi", "ci"],
       padding = [0 : index, 0 : index, 0 : index, 0 : index],
       strides = [1 : index, 1 : index]} -> tensor<1x8x3xf32>
    %quant = arith.fptoui %r : tensor<1x8x3xf32> to tensor<1x8x3xi8>
    %dequant = arith.sitofp %quant : tensor<1x8x3xi8> to tensor<1x8x3xf32>
    %stored = rock.store %dequant to %out by set : tensor<1x8x3xf32> -> tensor<1x8x3xf32> to tensor<1x8x3xf32>
    return %stored : tensor<1x8x3xf32>
  }
}
