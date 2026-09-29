// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The carve-out only covers an extf on its own. Chaining a second op onto it
// puts arithmetic back in front of the LSE combine, which is what makes
// output fusions illegal under splitKV > 1 in the first place.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:0
func.func @attn_splitkv_output_extf_chain_not_fusible(
    %queries: tensor<1x64x1024xf16>, %keys: tensor<1x64x1024xf16>,
    %values: tensor<1x1024x64xf16>, %scale: tensor<4x1024x64xf32>,
    %lse: tensor<4x1024xf32>, %out: tensor<4x1024x64xf32>)
    -> (tensor<4x1024x64xf32>, tensor<4x1024xf32>)
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  %result, %lseOut = rock.attention{
    qk = tr %queries * %keys : tensor<1x64x1024xf16>, tensor<1x64x1024xf16>
    softmax(qk) * %values : tensor<1x1024x64xf16>
  } {
    numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 4 : i32
  } -> tensor<4x1024x64xf16>, tensor<4x1024xf32>
  %cvt = arith.extf %result : tensor<4x1024x64xf16> to tensor<4x1024x64xf32>
  %scaled = arith.mulf %cvt, %scale : tensor<4x1024x64xf32>
  %outStore = rock.store %scaled to %out by set : tensor<4x1024x64xf32> -> tensor<4x1024x64xf32> to tensor<4x1024x64xf32>
  %lseStore = rock.store %lseOut to %lse by set : tensor<4x1024xf32> -> tensor<4x1024xf32> to tensor<4x1024xf32>
  return %outStore, %lseStore : tensor<4x1024x64xf32>, tensor<4x1024xf32>
}
