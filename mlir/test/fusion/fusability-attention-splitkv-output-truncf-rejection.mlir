// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The carve-out for element-wise conversions on splitKV > 1 attention covers
// widening only. truncf is lossy, so rounding each partial result before the
// LSE combine is not the same as rounding the combined result.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:0
func.func @attn_splitkv_output_truncf_not_fusible(
    %queries: tensor<1x64x1024xf32>, %keys: tensor<1x64x1024xf32>,
    %values: tensor<1x1024x64xf32>,
    %lse: tensor<4x1024xf32>, %out: tensor<4x1024x64xf16>)
    -> (tensor<4x1024x64xf16>, tensor<4x1024xf32>)
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  %result, %lseOut = rock.attention{
    qk = tr %queries * %keys : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    softmax(qk) * %values : tensor<1x1024x64xf32>
  } {
    numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 4 : i32
  } -> tensor<4x1024x64xf32>, tensor<4x1024xf32>
  %cvt = arith.truncf %result : tensor<4x1024x64xf32> to tensor<4x1024x64xf16>
  %outStore = rock.store %cvt to %out by set : tensor<4x1024x64xf16> -> tensor<4x1024x64xf16> to tensor<4x1024x64xf16>
  %lseStore = rock.store %lseOut to %lse by set : tensor<4x1024xf32> -> tensor<4x1024xf32> to tensor<4x1024xf32>
  return %outStore, %lseStore : tensor<4x1024x64xf16>, tensor<4x1024xf32>
}
