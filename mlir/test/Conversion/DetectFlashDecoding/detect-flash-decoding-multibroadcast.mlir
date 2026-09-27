// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Flash decoding where Q is sliced out of a packed QKV tensor and then
// multi-broadcast along the splitKV dimension. V's splitKV transform has been
// folded away, so only Q and K still carry it.

// RUN: rocmlir-opt --rock-detect-flash-decoding %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 128 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 2 + d2) * 1024 + d3) * 128 + d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>
#map5 = affine_map<(d0, d1, d2) -> (0, d0 floordiv 2, d0 mod 2, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> ((d0 * 1024 + d1) * 128 + d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map9 = affine_map<(d0, d1, d2) -> (d0, 0, 0, d2)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 * 1024 + d4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, 0, d2, d3, d4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d2, d3, d4)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d2, d4)>
#map14 = affine_map<(d0) -> (0, d0 floordiv 2, d0 mod 2, 0, 0)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>
#map16 = affine_map<(d0) -> (0, d0 floordiv 4096, 0, (d0 floordiv 128) mod 32, d0 mod 128)>
#transform_map = #rock.transform_map<#map by [<Unmerge{96, 128} ["exp1", "exp4"] at [1, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>, <AddDim{1} ["unit3"] at [3] -> [] at []>] bounds = [1, 96, 1, 1, 128] -> [12288]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [1, 96, 2, 1, 128] -> [1, 96, 1, 1, 128]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{32, 2, 1024, 128} ["exp1", "exp2", "exp3", "exp4"] at [1, 2, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 32, 2, 1024, 128] -> [8388608]>
#transform_map3 = #rock.transform_map<#map3 by [<Slice{0, 1, 0, 32, 0, 2, 0, 1, 0, 128} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced", "dim4_sliced"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim3", "dim4"] at [0, 1, 2, 3, 4]>] bounds = [1, 32, 2, 1, 128] -> [1, 96, 2, 1, 128]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 4, 3]>] bounds = [1, 32, 2, 128, 1024] -> [1, 32, 2, 1024, 128]>
#transform_map5 = #rock.transform_map<#map5 by [<Merge{1, 32, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [64, 1, 128] -> [1, 32, 2, 1, 128]>
#transform_map6 = #rock.transform_map<#map5 by [<Merge{1, 32, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [64, 128, 1024] -> [1, 32, 2, 128, 1024]>
#transform_map7 = #rock.transform_map<#map6 by [<Unmerge{64, 1024, 128} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [64, 1024, 128] -> [8388608]>
#transform_map8 = #rock.transform_map<#map7 by [<Unmerge{1} ["exp3"] at [3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit1"] at [1] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [1, 1, 1, 1] -> [1]>
#transform_map9 = #rock.transform_map<#map8 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 1, 1, 2048] -> [1, 1, 1, 1]>
#transform_map10 = #rock.transform_map<#map9 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Merge{1, 1} ["dim1"] at [1] -> ["col1", "col2"] at [1, 2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [1, 1, 2048] -> [1, 1, 1, 2048]>
#transform_map11 = #rock.transform_map<#map10 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Unmerge{2, 1024} ["exp2", "exp4"] at [2, 4] -> ["dim2"] at [2]>, <AddDim{1} ["unit3"] at [3] -> [] at []>] bounds = [1, 1, 2, 1, 1024] -> [1, 1, 2048]>
#transform_map12 = #rock.transform_map<#map11 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [1, 32, 2, 1, 1024] -> [1, 1, 2, 1, 1024]>
#transform_map13 = #rock.transform_map<#map12 by [<Unmerge{32, 2} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [3] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [4] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 32, 2, 1, 1024] -> [64, 1, 1024]>
#transform_map14 = #rock.transform_map<#map13 by [<Unmerge{32, 2} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <Unmerge{1} ["exp4"] at [4] -> ["dim1"] at [1]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit3"] at [3] -> [] at []>] bounds = [1, 32, 2, 1, 1] -> [64, 1]>
#transform_map15 = #rock.transform_map<#map14 by [<Merge{1, 32, 2, 1, 1} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [64] -> [1, 32, 2, 1, 1]>
#transform_map16 = #rock.transform_map<#map12 by [<Unmerge{32, 2} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [3] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [4] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 32, 2, 1, 128] -> [64, 1, 128]>
#transform_map17 = #rock.transform_map<#map15 by [<PassThrough ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 1, 2, 3, 4] -> ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 2, 3, 1, 4]>] bounds = [1, 2, 1, 32, 128] -> [1, 32, 2, 1, 128]>
#transform_map18 = #rock.transform_map<#map16 by [<Merge{1, 2, 1, 32, 128} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [8192] -> [1, 2, 1, 32, 128]>
module {
  func.func @mlir_attention(%arg0: tensor<12288xf16>, %arg1: tensor<8388608xf16>, %arg2: tensor<1xi32>, %arg3: tensor<8388608xf16>) -> (tensor<8192xf16>, tensor<64xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel = "mixr"} {
    %0 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x32x2x1x1024xf16>}> : () -> tensor<1x32x2x1x1024xf16>
    %1 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x32x2x1x1024xf16>}> : () -> tensor<1x32x2x1x1024xf16>
    %2 = "tosa.const"() <{values = dense<0> : tensor<1x1x2x1x1024xi32>}> : () -> tensor<1x1x2x1x1024xi32>
    %3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = rock.transform %arg0 by #transform_map : tensor<12288xf16> to tensor<1x96x1x1x128xf16>
    %5 = rock.transform %4 by #transform_map1 : tensor<1x96x1x1x128xf16> to tensor<1x96x2x1x128xf16>
    %6 = rock.transform %arg1 by #transform_map2 : tensor<8388608xf16> to tensor<1x32x2x1024x128xf16>
    %7 = rock.transform %5 by #transform_map3 : tensor<1x96x2x1x128xf16> to tensor<1x32x2x1x128xf16>
    %8 = rock.transform %6 by #transform_map4 : tensor<1x32x2x1024x128xf16> to tensor<1x32x2x128x1024xf16>
    %9 = rock.transform %7 by #transform_map5 : tensor<1x32x2x1x128xf16> to tensor<64x1x128xf16>
    %10 = rock.transform %8 by #transform_map6 : tensor<1x32x2x128x1024xf16> to tensor<64x128x1024xf16>
    %11 = rock.transform %arg3 by #transform_map7 : tensor<8388608xf16> to tensor<64x1024x128xf16>

    // CHECK: rock.attention{
    // CHECK-NEXT: qk = %{{.*}} * %{{.*}} : tensor<32x1x128xf16>, tensor<32x128x2048xf16>
    // CHECK-NEXT: qk = elementwise
    // CHECK: softmax(qk) * %{{.*}} : tensor<32x2048x128xf16>
    // CHECK: preSoftmaxHasSplitKVTransforms = true
    // CHECK-SAME: splitKV = 2

    %result, %lseOut = rock.attention{
     qk = %9 * %10 : tensor<64x1x128xf16>, tensor<64x128x1024xf16>
     qk = elementwise otherIns(%arg2 : tensor<1xi32>) {
    ^bb0(%arg4: tensor<64x1x1024xf16>, %arg5: tensor<1xi32>):
      %21 = rock.transform %arg5 by #transform_map8 : tensor<1xi32> to tensor<1x1x1x1xi32>
      %22 = rock.transform %21 by #transform_map9 : tensor<1x1x1x1xi32> to tensor<1x1x1x2048xi32>
      %23 = rock.transform %22 by #transform_map10 : tensor<1x1x1x2048xi32> to tensor<1x1x2048xi32>
      %24 = rock.transform %23 by #transform_map11 : tensor<1x1x2048xi32> to tensor<1x1x2x1x1024xi32>
      %25 = tosa.greater %2, %24 : (tensor<1x1x2x1x1024xi32>, tensor<1x1x2x1x1024xi32>) -> tensor<1x1x2x1x1024xi1>
      %26 = tosa.cast %25 : (tensor<1x1x2x1x1024xi1>) -> tensor<1x1x2x1x1024xi32>
      %27 = tosa.cast %26 : (tensor<1x1x2x1x1024xi32>) -> tensor<1x1x2x1x1024xi8>
      %28 = rock.transform %27 by #transform_map12 : tensor<1x1x2x1x1024xi8> to tensor<1x32x2x1x1024xi8>
      %29 = tosa.cast %28 : (tensor<1x32x2x1x1024xi8>) -> tensor<1x32x2x1x1024xi1>
      %30 = rock.transform %arg4 by #transform_map13 : tensor<64x1x1024xf16> to tensor<1x32x2x1x1024xf16>
      %31 = tosa.mul %30, %0, %3 : (tensor<1x32x2x1x1024xf16>, tensor<1x32x2x1x1024xf16>, tensor<1xi8>) -> tensor<1x32x2x1x1024xf16>
      %32 = tosa.select %29, %1, %31 : (tensor<1x32x2x1x1024xi1>, tensor<1x32x2x1x1024xf16>, tensor<1x32x2x1x1024xf16>) -> tensor<1x32x2x1x1024xf16>
      %33 = tosa.cast %32 : (tensor<1x32x2x1x1024xf16>) -> tensor<1x32x2x1x1024xf32>
      rock.yield %33 : tensor<1x32x2x1x1024xf32>
    }
     softmax(qk) * %11 : tensor<64x1024x128xf16>
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32} -> tensor<64x1x128xf16>, tensor<64x1xf32>
    %14 = rock.transform %lseOut by #transform_map14 : tensor<64x1xf32> to tensor<1x32x2x1x1xf32>
    %15 = rock.transform %14 by #transform_map15 : tensor<1x32x2x1x1xf32> to tensor<64xf32>
    %16 = rock.transform %result by #transform_map16 : tensor<64x1x128xf16> to tensor<1x32x2x1x128xf16>
    %17 = rock.transform %16 by #transform_map17 : tensor<1x32x2x1x128xf16> to tensor<1x2x1x32x128xf16>
    %18 = rock.transform %17 by #transform_map18 : tensor<1x2x1x32x128xf16> to tensor<8192xf16>
    return %18, %15 : tensor<8192xf16>, tensor<64xf32>
  }
}
