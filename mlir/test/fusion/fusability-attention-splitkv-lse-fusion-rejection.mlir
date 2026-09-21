// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The LSE is a per-split partial log-sum-exp under splitKV > 1, just like the
// main result, so an epilogue on it has to be rejected too. Here the result
// itself is stored untouched and only the LSE carries a fusion.
//
// Note this is not covered by attention being non-fusible under split-k:
// testFusionLegalitySplitK only runs when the perf config requests a split,
// and flash decoding uses splitKFactor = 1 (as below).

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:0
func.func @attn_splitkv_lse_fusion_not_fusible(
    %queries: tensor<1x64x1024xf32>, %keys: tensor<1x64x1024xf32>,
    %values: tensor<1x1024x64xf32>, %scale: tensor<4x1024xf32>,
    %lse: tensor<4x1024xf32>, %out: tensor<4x1024x64xf32>)
    -> (tensor<4x1024x64xf32>, tensor<4x1024xf32>)
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  %result, %lseOut = rock.attention{
    qk = tr %queries * %keys : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    softmax(qk) * %values : tensor<1x1024x64xf32>
  } {
    numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 4 : i32
  } -> tensor<4x1024x64xf32>, tensor<4x1024xf32>
  %scaledLse = arith.mulf %lseOut, %scale : tensor<4x1024xf32>
  %outStore = rock.store %result to %out by set : tensor<4x1024x64xf32> -> tensor<4x1024x64xf32> to tensor<4x1024x64xf32>
  %lseStore = rock.store %scaledLse to %lse by set : tensor<4x1024xf32> -> tensor<4x1024xf32> to tensor<4x1024xf32>
  return %outStore, %lseStore : tensor<4x1024x64xf32>, tensor<4x1024xf32>
}
