// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A lone element-wise extf on the attention output is safe to fuse with
// splitKV > 1: widening is lossless, so applying it to each partial result
// and then rescaling in the LSE combine gives the same answer as combining
// first.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:1
func.func @attn_splitkv_output_extf_fusible(
    %queries: tensor<1x64x1024xf16>, %keys: tensor<1x64x1024xf16>,
    %values: tensor<1x1024x64xf16>,
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
  %outStore = rock.store %cvt to %out by set : tensor<4x1024x64xf32> -> tensor<4x1024x64xf32> to tensor<4x1024x64xf32>
  %lseStore = rock.store %lseOut to %lse by set : tensor<4x1024xf32> -> tensor<4x1024xf32> to tensor<4x1024xf32>
  return %outStore, %lseStore : tensor<4x1024x64xf32>, tensor<4x1024xf32>
}
