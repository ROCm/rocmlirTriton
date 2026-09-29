// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Output fusions are not allowed for attention ops with splitKV > 1. Each
// split only produces a partial result, which an LSE-based combine has yet to
// rescale in a later stage, so an epilogue applied per-split does not survive
// the combine. The split factor in the perf config is 1, so the plain split-k
// legality check is not what rejects this.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:0
func.func @attn_splitkv_output_fusion_not_fusible(
    %queries: tensor<1x64x1024xf32>, %keys: tensor<1x64x1024xf32>,
    %values: tensor<1x1024x64xf32>, %bias: tensor<4x1024x64xf32>,
    %lse: tensor<4x1024xf32>, %out: tensor<4x1024x64xf32>)
    -> (tensor<4x1024x64xf32>, tensor<4x1024xf32>)
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  %result, %lseOut = rock.attention{
    qk = tr %queries * %keys : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    softmax(qk) * %values : tensor<1x1024x64xf32>
  } {
    numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 4 : i32
  } -> tensor<4x1024x64xf32>, tensor<4x1024xf32>
  %add = arith.addf %result, %bias : tensor<4x1024x64xf32>
  %outStore = rock.store %add to %out by set : tensor<4x1024x64xf32> -> tensor<4x1024x64xf32> to tensor<4x1024x64xf32>
  %lseStore = rock.store %lseOut to %lse by set : tensor<4x1024xf32> -> tensor<4x1024xf32> to tensor<4x1024xf32>
  return %outStore, %lseStore : tensor<4x1024x64xf32>, tensor<4x1024xf32>
}
