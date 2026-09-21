// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// KV-cache flash decoding: V's splitKV transform has been folded away by
// upstream canonicalization, so only Q and K still carry it. Detection has to
// go through with the two tensors that agree rather than insisting all three
// do.

// REQUIRES: asserts
// RUN: rocmlir-opt --rock-detect-flash-decoding --debug-only=rock-detect-flash-decoding %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEBUG
// RUN: rocmlir-opt --rock-detect-flash-decoding %s | FileCheck %s --check-prefix=CHECK-IR

// CHECK-DEBUG: Analyzing Q tensor for splitKV:
// CHECK-DEBUG: Q: Found 4D Broadcast at dim 1, splitKV = 2
// CHECK-DEBUG: Analyzing K tensor for splitKV:
// CHECK-DEBUG: K: Found Merge{1, 32, 2}, splitKV = 2, dimensionality = 5D
// CHECK-DEBUG: Analyzing V tensor for splitKV:
// CHECK-DEBUG: V: No Merge pattern found
// CHECK-DEBUG: Flash decoding detected: splitKV = 2, dimensionality = 4D

// CHECK-IR-LABEL: @mlir_attention
// CHECK-IR: rock.attention{
// CHECK-IR: splitKV = 2
func.func @mlir_attention(%arg0: tensor<61440xf16>, %arg1: tensor<4194304xf16>, %arg2: tensor<1xi32>, %arg3: tensor<4194304xf16>) -> (tensor<40960xf16>, tensor<320xf32>) attributes {rock.arch = "gfx950", rock.kernel = "mixr", rock.num_cu = 256 : i64} {
  %0 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x32x2x5x512xf16>}> : () -> tensor<1x32x2x5x512xf16>
  %1 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x32x2x5x512xf16>}> : () -> tensor<1x32x2x5x512xf16>
  %2 = "tosa.const"() <{values = dense<0> : tensor<1x1x2x1x512xi32>}> : () -> tensor<1x1x2x1x512xi32>
  %3 = "tosa.const"() <{values = dense<0> : tensor<1x1x5x1024xi8>}> : () -> tensor<1x1x5x1024xi8>
  %4 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %5 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (d2 * 12288 + d3)> by [<Unmerge{5, 12288} ["exp2", "exp3"] at [2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit1"] at [1] -> [] at []>] bounds = [1, 1, 5, 12288] -> [61440]> : tensor<61440xf16> to tensor<1x1x5x12288xf16>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 2, 5, 12288] -> [1, 1, 5, 12288]> : tensor<1x1x5x12288xf16> to tensor<1x2x5x12288xf16>
  %7 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 2 + d2) * 512 + d3) * 128 + d4)> by [<Unmerge{32, 2, 512, 128} ["exp1", "exp2", "exp3", "exp4"] at [1, 2, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 32, 2, 512, 128] -> [4194304]> : tensor<4194304xf16> to tensor<1x32x2x512x128xf16>
  %8 = rock.transform %6 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 * 128 + d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <Unmerge{96, 128} ["exp3", "exp4"] at [3, 4] -> ["dim3"] at [3]>] bounds = [1, 2, 5, 96, 128] -> [1, 2, 5, 12288]> : tensor<1x2x5x12288xf16> to tensor<1x2x5x96x128xf16>
  %9 = rock.transform %8 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d1, d4)> by [<PassThrough ["dim0", "dim3", "dim1", "dim2", "dim4"] at [0, 1, 2, 3, 4] -> ["dim0", "dim3", "dim1", "dim2", "dim4"] at [0, 3, 1, 2, 4]>] bounds = [1, 96, 2, 5, 128] -> [1, 2, 5, 96, 128]> : tensor<1x2x5x96x128xf16> to tensor<1x96x2x5x128xf16>
  %10 = rock.transform %9 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)> by [<Slice{0, 1, 0, 32, 0, 2, 0, 5, 0, 128} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced", "dim4_sliced"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim3", "dim4"] at [0, 1, 2, 3, 4]>] bounds = [1, 32, 2, 5, 128] -> [1, 96, 2, 5, 128]> : tensor<1x96x2x5x128xf16> to tensor<1x32x2x5x128xf16>
  %11 = rock.transform %7 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)> by [<PassThrough ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 4, 3]>] bounds = [1, 32, 2, 128, 512] -> [1, 32, 2, 512, 128]> : tensor<1x32x2x512x128xf16> to tensor<1x32x2x128x512xf16>
  %12 = rock.transform %10 by <affine_map<(d0, d1, d2) -> (0, d0 floordiv 2, d0 mod 2, d1, d2)> by [<Merge{1, 32, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [64, 5, 128] -> [1, 32, 2, 5, 128]> : tensor<1x32x2x5x128xf16> to tensor<64x5x128xf16>
  %13 = rock.transform %11 by <affine_map<(d0, d1, d2) -> (0, d0 floordiv 2, d0 mod 2, d1, d2)> by [<Merge{1, 32, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [64, 128, 512] -> [1, 32, 2, 128, 512]> : tensor<1x32x2x128x512xf16> to tensor<64x128x512xf16>
  %14 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> ((d0 * 512 + d1) * 128 + d2)> by [<Unmerge{64, 512, 128} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [64, 512, 128] -> [4194304]> : tensor<4194304xf16> to tensor<64x512x128xf16>
  %result, %lse = rock.attention{
   qk = %12 * %13 : tensor<64x5x128xf16>, tensor<64x128x512xf16>
   qk = elementwise otherIns(%2, %arg2, %3 : tensor<1x1x2x1x512xi32>, tensor<1xi32>, tensor<1x1x5x1024xi8>) {
  ^bb0(%arg4: tensor<64x5x512xf16>, %arg5: tensor<1x1x2x1x512xi32>, %arg6: tensor<1xi32>, %arg7: tensor<1x1x5x1024xi8>):
    %20 = rock.transform %arg6 by <affine_map<(d0, d1, d2, d3) -> (d3)> by [<Unmerge{1} ["exp3"] at [3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit1"] at [1] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [1, 1, 1, 1] -> [1]> : tensor<1xi32> to tensor<1x1x1x1xi32>
    %21 = rock.transform %20 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 1, 1, 1024] -> [1, 1, 1, 1]> : tensor<1x1x1x1xi32> to tensor<1x1x1x1024xi32>
    %22 = rock.transform %21 by <affine_map<(d0, d1, d2) -> (d0, 0, 0, d2)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Merge{1, 1} ["dim1"] at [1] -> ["col1", "col2"] at [1, 2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [1, 1, 1024] -> [1, 1, 1, 1024]> : tensor<1x1x1x1024xi32> to tensor<1x1x1024xi32>
    %23 = rock.transform %22 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 * 512 + d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Unmerge{2, 512} ["exp2", "exp4"] at [2, 4] -> ["dim2"] at [2]>, <AddDim{1} ["unit3"] at [3] -> [] at []>] bounds = [1, 1, 2, 1, 512] -> [1, 1, 1024]> : tensor<1x1x1024xi32> to tensor<1x1x2x1x512xi32>
    %24 = tosa.greater %arg5, %23 : (tensor<1x1x2x1x512xi32>, tensor<1x1x2x1x512xi32>) -> tensor<1x1x2x1x512xi1>
    %25 = tosa.cast %24 : (tensor<1x1x2x1x512xi1>) -> tensor<1x1x2x1x512xi32>
    %26 = tosa.cast %25 : (tensor<1x1x2x1x512xi32>) -> tensor<1x1x2x1x512xi8>
    %27 = rock.transform %26 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, 0, d2, 0, d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [1, 32, 2, 5, 512] -> [1, 1, 2, 1, 512]> : tensor<1x1x2x1x512xi8> to tensor<1x32x2x5x512xi8>
    %28 = tosa.cast %27 : (tensor<1x32x2x5x512xi8>) -> tensor<1x32x2x5x512xi1>
    %29 = rock.transform %arg7 by <affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 32, 5, 1024] -> [1, 1, 5, 1024]> : tensor<1x1x5x1024xi8> to tensor<1x32x5x1024xi8>
    %30 = rock.transform %29 by <affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 1024, d2 mod 1024)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Merge{5, 1024} ["dim2"] at [2] -> ["col2", "col3"] at [2, 3]>] bounds = [1, 32, 5120] -> [1, 32, 5, 1024]> : tensor<1x32x5x1024xi8> to tensor<1x32x5120xi8>
    %31 = rock.transform %30 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, (d2 * 5 + d3) * 512 + d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Unmerge{2, 5, 512} ["exp2", "exp3", "exp4"] at [2, 3, 4] -> ["dim2"] at [2]>] bounds = [1, 32, 2, 5, 512] -> [1, 32, 5120]> : tensor<1x32x5120xi8> to tensor<1x32x2x5x512xi8>
    %32 = tosa.cast %31 : (tensor<1x32x2x5x512xi8>) -> tensor<1x32x2x5x512xi1>
    %33 = rock.transform %arg4 by <affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d2, d3, d4)> by [<Unmerge{32, 2} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [3] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [4] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 32, 2, 5, 512] -> [64, 5, 512]> : tensor<64x5x512xf16> to tensor<1x32x2x5x512xf16>
    %34 = tosa.mul %33, %0, %4 : (tensor<1x32x2x5x512xf16>, tensor<1x32x2x5x512xf16>, tensor<1xi8>) -> tensor<1x32x2x5x512xf16>
    %35 = tosa.select %32, %1, %34 : (tensor<1x32x2x5x512xi1>, tensor<1x32x2x5x512xf16>, tensor<1x32x2x5x512xf16>) -> tensor<1x32x2x5x512xf16>
    %36 = tosa.select %28, %1, %35 : (tensor<1x32x2x5x512xi1>, tensor<1x32x2x5x512xf16>, tensor<1x32x2x5x512xf16>) -> tensor<1x32x2x5x512xf16>
    %37 = tosa.cast %36 : (tensor<1x32x2x5x512xf16>) -> tensor<1x32x2x5x512xf32>
    rock.yield %37 : tensor<1x32x2x5x512xf32>
  }
   softmax(qk) * %14 : tensor<64x512x128xf16>
  } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32} -> tensor<64x5x128xf16>, tensor<64x5xf32>
  %15 = rock.transform %lse by <affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d2, d3)> by [<Unmerge{32, 2} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <Unmerge{5} ["exp3"] at [3] -> ["dim1"] at [1]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit4"] at [4] -> [] at []>] bounds = [1, 32, 2, 5, 1] -> [64, 5]> : tensor<64x5xf32> to tensor<1x32x2x5x1xf32>
  %16 = rock.transform %15 by <affine_map<(d0) -> (0, d0 floordiv 10, (d0 mod 10) floordiv 5, d0 mod 5, 0)> by [<Merge{1, 32, 2, 5, 1} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [320] -> [1, 32, 2, 5, 1]> : tensor<1x32x2x5x1xf32> to tensor<320xf32>
  %17 = rock.transform %result by <affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d2, d3, d4)> by [<Unmerge{32, 2} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [3] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [4] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 32, 2, 5, 128] -> [64, 5, 128]> : tensor<64x5x128xf16> to tensor<1x32x2x5x128xf16>
  %18 = rock.transform %17 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)> by [<PassThrough ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 1, 2, 3, 4] -> ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 2, 3, 1, 4]>] bounds = [1, 2, 5, 32, 128] -> [1, 32, 2, 5, 128]> : tensor<1x32x2x5x128xf16> to tensor<1x2x5x32x128xf16>
  %19 = rock.transform %18 by <affine_map<(d0) -> (0, d0 floordiv 20480, (d0 floordiv 4096) mod 5, (d0 floordiv 128) mod 32, d0 mod 128)> by [<Merge{1, 2, 5, 32, 128} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [40960] -> [1, 2, 5, 32, 128]> : tensor<1x2x5x32x128xf16> to tensor<40960xf16>
  return %19, %16 : tensor<40960xf16>, tensor<320xf32>
}

