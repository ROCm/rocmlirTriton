// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A `rock.reduce` on the attention output is an output fusion like any other,
// and reducing over partial results before the LSE combine has rescaled them
// gives the wrong answer. Unlike split-k, where a sum reduction commutes with
// the per-split sums, splitKV needs the LSE rescale first, so even reduce sum
// is rejected here.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:0
func.func @attn_splitkv_output_reduce_not_fusible(
    %queries: tensor<1x64x1024xf32>, %keys: tensor<1x64x1024xf32>,
    %values: tensor<1x1024x64xf32>,
    %lse: tensor<4x1024xf32>, %out: tensor<4x1024x1xf32>)
    -> (tensor<4x1024x1xf32>, tensor<4x1024xf32>)
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  %result, %lseOut = rock.attention{
    qk = tr %queries * %keys : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    softmax(qk) * %values : tensor<1x1024x64xf32>
  } {
    numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 4 : i32
  } -> tensor<4x1024x64xf32>, tensor<4x1024xf32>
  %red = rock.reduce sum %result {axis = 2 : index} : tensor<4x1024x64xf32> -> tensor<4x1024x1xf32>
  %outStore = rock.store %red to %out by set : tensor<4x1024x1xf32> -> tensor<4x1024x1xf32> to tensor<4x1024x1xf32>
  %lseStore = rock.store %lseOut to %lse by set : tensor<4x1024xf32> -> tensor<4x1024xf32> to tensor<4x1024xf32>
  return %outStore, %lseStore : tensor<4x1024x1xf32>, tensor<4x1024xf32>
}
