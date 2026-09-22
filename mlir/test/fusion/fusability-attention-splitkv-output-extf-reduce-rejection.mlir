// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The lone-extf carve-out must not let a reduction in behind it. The widening
// itself is lossless, but the reduce that consumes it still runs on partial
// results, ahead of the LSE combine.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:0
func.func @attn_splitkv_output_extf_reduce_not_fusible(
    %queries: tensor<1x64x1024xf16>, %keys: tensor<1x64x1024xf16>,
    %values: tensor<1x1024x64xf16>,
    %lse: tensor<4x1024xf32>, %out: tensor<4x1024x1xf32>)
    -> (tensor<4x1024x1xf32>, tensor<4x1024xf32>)
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  %result, %lseOut = rock.attention{
    qk = tr %queries * %keys : tensor<1x64x1024xf16>, tensor<1x64x1024xf16>
    softmax(qk) * %values : tensor<1x1024x64xf16>
  } {
    numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 4 : i32
  } -> tensor<4x1024x64xf16>, tensor<4x1024xf32>
  %cvt = arith.extf %result : tensor<4x1024x64xf16> to tensor<4x1024x64xf32>
  %red = rock.reduce sum %cvt {axis = 2 : index} : tensor<4x1024x64xf32> -> tensor<4x1024x1xf32>
  %outStore = rock.store %red to %out by set : tensor<4x1024x1xf32> -> tensor<4x1024x1xf32> to tensor<4x1024x1xf32>
  %lseStore = rock.store %lseOut to %lse by set : tensor<4x1024xf32> -> tensor<4x1024xf32> to tensor<4x1024xf32>
  return %outStore, %lseStore : tensor<4x1024x1xf32>, tensor<4x1024xf32>
}
