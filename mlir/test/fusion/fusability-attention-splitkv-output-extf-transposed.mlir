// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A transposing view between the attention output and a lone element-wise
// extf stays fusible with splitKV > 1. Upstream rocMLIR rejects this case
// because there the transpose is folded into a linalg.generic indexing map,
// which would silently reorder the buffer the LSE combine reads back. Here
// the transpose is an explicit rock.transform view and the layout is carried
// by the rock.store, so widening each partial result is still lossless.
//
// Counterpart of rocMLIR's
// fusability-attention-splitkv-output-extf-transposed-rejection.mlir, with
// the expectation inverted for the view-based IR.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:1

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [4, 64, 1024] -> [4, 1024, 64]>

func.func @attn_splitkv_output_extf_transposed_fusible(
    %queries: tensor<1x64x1024xf16>, %keys: tensor<1x64x1024xf16>,
    %values: tensor<1x1024x64xf16>,
    %lse: tensor<4x1024xf32>, %out: tensor<4x64x1024xf32>)
    -> (tensor<4x64x1024xf32>, tensor<4x1024xf32>)
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  %result, %lseOut = rock.attention{
    qk = tr %queries * %keys : tensor<1x64x1024xf16>, tensor<1x64x1024xf16>
    softmax(qk) * %values : tensor<1x1024x64xf16>
  } {
    numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 4 : i32
  } -> tensor<4x1024x64xf16>, tensor<4x1024xf32>
  %tr = rock.transform %result by #transform_map : tensor<4x1024x64xf16> to tensor<4x64x1024xf16>
  %cvt = arith.extf %tr : tensor<4x64x1024xf16> to tensor<4x64x1024xf32>
  %outStore = rock.store %cvt to %out by set : tensor<4x64x1024xf32> -> tensor<4x64x1024xf32> to tensor<4x64x1024xf32>
  %lseStore = rock.store %lseOut to %lse by set : tensor<4x1024xf32> -> tensor<4x1024xf32> to tensor<4x1024xf32>
  return %outStore, %lseStore : tensor<4x64x1024xf32>, tensor<4x1024xf32>
}
