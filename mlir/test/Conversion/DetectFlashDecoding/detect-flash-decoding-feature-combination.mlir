// RUN: rocmlir-opt --rock-detect-flash-decoding %s | FileCheck %s

// Ported from rocMLIR's detect-flash-decoding-feature-combination.mlir and
// adapted to this repository's rock.attention assembly (LSE is a result rather
// than an operand, and the pre-softmax region operates on tensors).
//
// Combines every optional attention feature with split-KV: a KV-cache
// currentSeqLen, a prefix-causal prefixOffset, causal masking and a pre-softmax
// elementwise fusion. Note that V reaches the op through a plain flat Unmerge
// with no split-KV structure left in it, so split-KV can only be detected from
// Q's Broadcast plus K's Merge. Batch 8 = 2 batch * 2 heads * 2 split-KV, so
// after promotion the batch becomes 4 and both i32 tensors are sliced to
// tensor<4xi32>.

#map = affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 6 + d1) * 2 + d3) * 2 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> ((((d0 * 2 + d1) * 2 + d2) * 2 + d3) * 2 + d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0 floordiv 4, (d0 mod 4) floordiv 2, d0 mod 2, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> ((d0 * 2 + d1) * 2 + d2)>
#map7 = affine_map<(d0, d1, d2) -> (d0)>
#map8 = affine_map<(d0, d1, d2) -> (d0, 0, 0)>
#map9 = affine_map<(d0) -> (d0 floordiv 4, (d0 mod 4) floordiv 2, d0 mod 2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{2, 6, 2, 2} ["exp0", "exp1", "exp3", "exp4"] at [0, 1, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [2, 6, 1, 2, 2] -> [48]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [2, 6, 2, 2, 2] -> [2, 6, 1, 2, 2]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{2, 2, 2, 2, 2} ["exp0", "exp1", "exp2", "exp3", "exp4"] at [0, 1, 2, 3, 4] -> ["dim0"] at [0]>] bounds = [2, 2, 2, 2, 2] -> [32]>
#transform_map3 = #rock.transform_map<#map3 by [<Slice{0, 2, 0, 2, 0, 2, 0, 2, 0, 2} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced", "dim4_sliced"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim3", "dim4"] at [0, 1, 2, 3, 4]>] bounds = [2, 2, 2, 2, 2] -> [2, 6, 2, 2, 2]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 4, 3]>] bounds = [2, 2, 2, 2, 2] -> [2, 2, 2, 2, 2]>
#transform_map5 = #rock.transform_map<#map5 by [<Merge{2, 2, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [8, 2, 2] -> [2, 2, 2, 2, 2]>
#transform_map6 = #rock.transform_map<#map6 by [<Unmerge{8, 2, 2} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [8, 2, 2] -> [32]>
#transform_map7 = #rock.transform_map<#map7 by [<Unmerge{2} ["exp0"] at [0] -> ["dim0"] at [0]>, <AddDim{1} ["unit1"] at [1] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [2, 1, 1] -> [2]>
#transform_map8 = #rock.transform_map<#map8 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [2, 2, 2] -> [2, 1, 1]>
#transform_map9 = #rock.transform_map<#map9 by [<Merge{2, 2, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [8] -> [2, 2, 2]>
module {
  // CHECK-LABEL: func.func @mlir_attention
  // CHECK: rock.attention{
  // CHECK-NEXT: qk = %{{.*}} * %{{.*}} : tensor<4x2x2xf16>, tensor<4x2x4xf16>
  // CHECK-NEXT: currentSeqLen = (%{{.*}} : tensor<4xi32>)
  // CHECK-NEXT: prefixOffset = (%{{.*}} : tensor<4xi32>)
  // CHECK-NEXT: causal
  // CHECK-NEXT: qk = elementwise
  // CHECK: softmax(qk) * %{{.*}} : tensor<4x4x2xf16>
  // CHECK: preSoftmaxHasSplitKVTransforms = true
  // CHECK-SAME: splitKV = 2 : i32
  func.func @mlir_attention(%arg0: tensor<48xf16>, %arg1: tensor<32xf16>, %arg2: tensor<2xi32>, %arg3: tensor<32xf16>, %arg4: tensor<2xi32>, %arg5: tensor<32xf16>) -> (tensor<8x2x2xf16>, tensor<8x2xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel = "mixr", rock.num_cu = 48 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<48xf16> to tensor<2x6x1x2x2xf16>
    %1 = rock.transform %0 by #transform_map1 : tensor<2x6x1x2x2xf16> to tensor<2x6x2x2x2xf16>
    %2 = rock.transform %arg1 by #transform_map2 : tensor<32xf16> to tensor<2x2x2x2x2xf16>
    %3 = rock.transform %1 by #transform_map3 : tensor<2x6x2x2x2xf16> to tensor<2x2x2x2x2xf16>
    %4 = rock.transform %2 by #transform_map4 : tensor<2x2x2x2x2xf16> to tensor<2x2x2x2x2xf16>
    %q = rock.transform %3 by #transform_map5 : tensor<2x2x2x2x2xf16> to tensor<8x2x2xf16>
    %k = rock.transform %4 by #transform_map5 : tensor<2x2x2x2x2xf16> to tensor<8x2x2xf16>
    %v = rock.transform %arg3 by #transform_map6 : tensor<32xf16> to tensor<8x2x2xf16>
    %scale = rock.transform %arg5 by #transform_map6 : tensor<32xf16> to tensor<8x2x2xf16>
    %c0 = rock.transform %arg2 by #transform_map7 : tensor<2xi32> to tensor<2x1x1xi32>
    %c1 = rock.transform %c0 by #transform_map8 : tensor<2x1x1xi32> to tensor<2x2x2xi32>
    %csl = rock.transform %c1 by #transform_map9 : tensor<2x2x2xi32> to tensor<8xi32>
    %p0 = rock.transform %arg4 by #transform_map7 : tensor<2xi32> to tensor<2x1x1xi32>
    %p1 = rock.transform %p0 by #transform_map8 : tensor<2x1x1xi32> to tensor<2x2x2xi32>
    %poff = rock.transform %p1 by #transform_map9 : tensor<2x2x2xi32> to tensor<8xi32>
    %result, %lseOut = rock.attention{
     qk = %q * %k : tensor<8x2x2xf16>, tensor<8x2x2xf16>
     currentSeqLen = (%csl : tensor<8xi32>)
     prefixOffset = (%poff : tensor<8xi32>)
     causal
     qk = elementwise otherIns(%scale : tensor<8x2x2xf16>) {
    ^bb0(%qkIn: tensor<8x2x2xf16>, %scaleIn: tensor<8x2x2xf16>):
      %scaled = arith.mulf %qkIn, %scaleIn : tensor<8x2x2xf16>
      rock.yield %scaled : tensor<8x2x2xf16>
    }
     softmax(qk) * %v : tensor<8x2x2xf16>
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32} -> tensor<8x2x2xf16>, tensor<8x2xf32>
    return %result, %lseOut : tensor<8x2x2xf16>, tensor<8x2xf32>
  }
}
