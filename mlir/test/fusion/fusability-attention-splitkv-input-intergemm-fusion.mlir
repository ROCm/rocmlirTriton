// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Only output fusions are rejected under splitKV > 1. Input fusions and
// fusions between the two gemms both run before the softmax normalizes over
// the split dimension, so each split still computes its own partial result
// correctly and the LSE combine stays valid.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:1
func.func @attn_splitkv_input_and_intergemm_fusible(
    %queries: tensor<1x64x1024xf32>, %keys: tensor<1x64x1024xf32>,
    %values: tensor<1x1024x64xf32>, %scale: tensor<1x64x1024xf32>,
    %lse: tensor<4x1024xf32>, %out: tensor<4x1024x64xf32>)
    -> (tensor<4x1024x64xf32>, tensor<4x1024xf32>)
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // Input fusion: scale the queries before feeding them into attention.
  %scaledQueries = arith.mulf %queries, %scale : tensor<1x64x1024xf32>
  %result, %lseOut = rock.attention{
    qk = tr %scaledQueries * %keys : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    // Inter-gemm fusion: element-wise negation between the two gemms.
    qk = elementwise {
    ^bb0(%qkIn: tensor<4x1024x256xf32>):
      %neg = arith.negf %qkIn : tensor<4x1024x256xf32>
      rock.yield %neg : tensor<4x1024x256xf32>
    }
    softmax(qk) * %values : tensor<1x1024x64xf32>
  } {
    numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 4 : i32
  } -> tensor<4x1024x64xf32>, tensor<4x1024xf32>
  %outStore = rock.store %result to %out by set : tensor<4x1024x64xf32> -> tensor<4x1024x64xf32> to tensor<4x1024x64xf32>
  %lseStore = rock.store %lseOut to %lse by set : tensor<4x1024xf32> -> tensor<4x1024xf32> to tensor<4x1024xf32>
  return %outStore, %lseStore : tensor<4x1024x64xf32>, tensor<4x1024xf32>
}
