// RUN: rocmlir-opt --rock-detect-flash-decoding %s | FileCheck %s

// Ported from rocMLIR's detect-flash-decoding-multibroadcast.mlir and adapted
// to this repository's rock.attention assembly (LSE is a result rather than an
// operand, and the pre-softmax region operates on tensors).
//
// Several Broadcast transforms are live in the function at once: one carries
// the split-KV dimension on Q, another only widens the pre-softmax fusion
// operand. Detection must pick the split-KV one and ignore the rest. Q is also
// sliced from 96 heads down to 32, and V arrives through a plain flat Unmerge
// with no split-KV structure, so the split can only be corroborated by K's
// Merge. Batch 64 = 32 heads * 2 split-KV, so the batch becomes 32 and the key
// sequence is stitched back from 1024 to 2048.

#map = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 128 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 2 + d2) * 1024 + d3) * 128 + d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>
#map5 = affine_map<(d0, d1, d2) -> (0, d0 floordiv 2, d0 mod 2, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> ((d0 * 1024 + d1) * 128 + d2)>
#map7 = affine_map<(d0, d1, d2) -> (d0)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{96, 128} ["exp1", "exp4"] at [1, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>, <AddDim{1} ["unit3"] at [3] -> [] at []>] bounds = [1, 96, 1, 1, 128] -> [12288]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [1, 96, 2, 1, 128] -> [1, 96, 1, 1, 128]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{32, 2, 1024, 128} ["exp1", "exp2", "exp3", "exp4"] at [1, 2, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 32, 2, 1024, 128] -> [8388608]>
#transform_map3 = #rock.transform_map<#map3 by [<Slice{0, 1, 0, 32, 0, 2, 0, 1, 0, 128} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced", "dim4_sliced"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim3", "dim4"] at [0, 1, 2, 3, 4]>] bounds = [1, 32, 2, 1, 128] -> [1, 96, 2, 1, 128]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 4, 3]>] bounds = [1, 32, 2, 128, 1024] -> [1, 32, 2, 1024, 128]>
#transform_map5 = #rock.transform_map<#map5 by [<Merge{1, 32, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [64, 1, 128] -> [1, 32, 2, 1, 128]>
#transform_map6 = #rock.transform_map<#map5 by [<Merge{1, 32, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [64, 128, 1024] -> [1, 32, 2, 128, 1024]>
#transform_map7 = #rock.transform_map<#map6 by [<Unmerge{64, 1024, 128} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [64, 1024, 128] -> [8388608]>
#transform_map8 = #rock.transform_map<#map7 by [<Unmerge{64} ["exp0"] at [0] -> ["dim0"] at [0]>, <AddDim{1} ["unit1"] at [1] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [64, 1, 1] -> [64]>
#transform_map9 = #rock.transform_map<#map8 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [64, 1, 1024] -> [64, 1, 1]>
module {
  // CHECK-LABEL: func.func @mlir_attention
  // CHECK: rock.attention{
  // CHECK-NEXT: qk = %{{.*}} * %{{.*}} : tensor<32x1x128xf16>, tensor<32x128x2048xf16>
  // CHECK-NEXT: qk = elementwise
  // CHECK: softmax(qk) * %{{.*}} : tensor<32x2048x128xf16>
  // CHECK: preSoftmaxHasSplitKVTransforms = true
  // CHECK-SAME: splitKV = 2 : i32
  func.func @mlir_attention(%arg0: tensor<12288xf16>, %arg1: tensor<8388608xf16>, %arg2: tensor<8388608xf16>, %arg3: tensor<64xf16>) -> (tensor<64x1x128xf16>, tensor<64x1xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel = "mixr", rock.num_cu = 48 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<12288xf16> to tensor<1x96x1x1x128xf16>
    %1 = rock.transform %0 by #transform_map1 : tensor<1x96x1x1x128xf16> to tensor<1x96x2x1x128xf16>
    %2 = rock.transform %arg1 by #transform_map2 : tensor<8388608xf16> to tensor<1x32x2x1024x128xf16>
    %3 = rock.transform %1 by #transform_map3 : tensor<1x96x2x1x128xf16> to tensor<1x32x2x1x128xf16>
    %4 = rock.transform %2 by #transform_map4 : tensor<1x32x2x1024x128xf16> to tensor<1x32x2x128x1024xf16>
    %q = rock.transform %3 by #transform_map5 : tensor<1x32x2x1x128xf16> to tensor<64x1x128xf16>
    %k = rock.transform %4 by #transform_map6 : tensor<1x32x2x128x1024xf16> to tensor<64x128x1024xf16>
    %v = rock.transform %arg2 by #transform_map7 : tensor<8388608xf16> to tensor<64x1024x128xf16>
    %b0 = rock.transform %arg3 by #transform_map8 : tensor<64xf16> to tensor<64x1x1xf16>
    %scale = rock.transform %b0 by #transform_map9 : tensor<64x1x1xf16> to tensor<64x1x1024xf16>
    %result, %lseOut = rock.attention{
     qk = %q * %k : tensor<64x1x128xf16>, tensor<64x128x1024xf16>
     qk = elementwise otherIns(%scale : tensor<64x1x1024xf16>) {
    ^bb0(%qkIn: tensor<64x1x1024xf16>, %scaleIn: tensor<64x1x1024xf16>):
      %scaled = arith.mulf %qkIn, %scaleIn : tensor<64x1x1024xf16>
      rock.yield %scaled : tensor<64x1x1024xf16>
    }
     softmax(qk) * %v : tensor<64x1024x128xf16>
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32} -> tensor<64x1x128xf16>, tensor<64x1xf32>
    return %result, %lseOut : tensor<64x1x128xf16>, tensor<64x1xf32>
  }
}
