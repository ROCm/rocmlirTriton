// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,5,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:1
// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1

module {
  func.func @mlir_convolution_multi_reduce(%arg0: tensor<320xf32>, %arg1: tensor<32768xf32>, %arg2: tensor<11520xf32>, %arg3: tensor<64xf32> {rock.prefill = 0.000000e+00 : f32}, %arg4: tensor<2621440xf32> {rock.prefill = 0.000000e+00 : f32}) -> (tensor<64xf32>, tensor<2621440xf32>) attributes {rock.arch = "gfx942:sramecc+:xnack-", rock.enable_splitk_for_tuning, rock.kernel = "mixr"} {
    %cst = arith.constant dense<2.44140629E-5> : tensor<2x32x10x64x64xf32>
    %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 64 + d2) * 64 + d3)> by [<Unmerge{2, 4, 64, 64} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [2, 4, 64, 64] -> [32768]> : tensor<32768xf32> to tensor<2x4x64x64xf32>
    %1 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 3 + d2) * 3 + d3)> by [<Unmerge{320, 4, 3, 3} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [320, 4, 3, 3] -> [11520]> : tensor<11520xf32> to tensor<320x4x3x3xf32>
    %2 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 4 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 4} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [2, 1, 4, 64, 64] -> [2, 4, 64, 64]> : tensor<2x4x64x64xf32> to tensor<2x1x4x64x64xf32>
    %3 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 320 + d1, d2, d3, d4)> by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 320, 4, 3, 3] -> [320, 4, 3, 3]> : tensor<320x4x3x3xf32> to tensor<1x320x4x3x3xf32>
    %empty = tensor.empty() : tensor<2x1x320x64x64xf32>
    %conv = rock.conv(%3, %2, %empty) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : tensor<1x320x4x3x3xf32>, tensor<2x1x4x64x64xf32>, tensor<2x1x320x64x64xf32> -> tensor<2x1x320x64x64xf32>
    %conv_4d = rock.transform %conv by <affine_map<(d0, d1, d2, d3) -> (d0, 0, d1, d2, d3)> by [<PassThrough ["n", "h", "w"] at [0, 2, 3] -> ["n", "h", "w"] at [0, 3, 4]>, <Merge{1, 320} ["k"] at [1] -> ["g", "k"] at [1, 2]>] bounds = [2, 320, 64, 64] -> [2, 1, 320, 64, 64]> : tensor<2x1x320x64x64xf32> to tensor<2x320x64x64xf32>
    %conv_reshaped = rock.transform %conv_4d by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 10 + d2, d3, d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Unmerge{32, 10} ["exp1", "exp2"] at [1, 2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [4] -> ["dim3"] at [3]>] bounds = [2, 32, 10, 64, 64] -> [2, 320, 64, 64]> : tensor<2x320x64x64xf32> to tensor<2x32x10x64x64xf32>
    %bias_reshaped = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (d1 * 10 + d2)> by [<Unmerge{32, 10} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit3"] at [3] -> [] at []>, <AddDim{1} ["unit4"] at [4] -> [] at []>] bounds = [1, 32, 10, 1, 1] -> [320]> : tensor<320xf32> to tensor<1x32x10x1x1xf32>
    %bias_broadcast = rock.transform %bias_reshaped by <affine_map<(d0, d1, d2, d3, d4) -> (0, d1, d2, 0, 0)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>, <Broadcast{1} ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [2, 32, 10, 64, 64] -> [1, 32, 10, 1, 1]> : tensor<1x32x10x1x1xf32> to tensor<2x32x10x64x64xf32>
    %fused_add = arith.addf %conv_reshaped, %bias_broadcast : tensor<2x32x10x64x64xf32>
    %fused_flat = rock.transform %fused_add by <affine_map<(d0) -> (d0 floordiv 1310720, (d0 mod 1310720) floordiv 40960, (d0 mod 40960) floordiv 4096, (d0 mod 4096) floordiv 64, d0 mod 64)> by [<Merge{2, 32, 10, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [2621440] -> [2, 32, 10, 64, 64]> : tensor<2x32x10x64x64xf32> to tensor<2621440xf32>
    %scaled = arith.mulf %fused_add, %cst : tensor<2x32x10x64x64xf32>
    %scaled_flat = rock.transform %scaled by <affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 4096, (d2 mod 4096) floordiv 64, d2 mod 64)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Merge{10, 64, 64} ["dim2"] at [2] -> ["col2", "col3", "col4"] at [2, 3, 4]>] bounds = [2, 32, 40960] -> [2, 32, 10, 64, 64]> : tensor<2x32x10x64x64xf32> to tensor<2x32x40960xf32>
    %reduced = rock.reduce sum %scaled_flat {axis = 2 : index} : tensor<2x32x40960xf32> -> tensor<2x32x1xf32>
    %reduced_flat = rock.transform %reduced by <affine_map<(d0) -> (d0 floordiv 32, d0 mod 32, 0)> by [<Merge{2, 32, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [64] -> [2, 32, 1]> : tensor<2x32x1xf32> to tensor<64xf32>
    %out3 = rock.store %reduced_flat to %arg3 by set : tensor<64xf32> -> tensor<64xf32> to tensor<64xf32>
    %out4 = rock.store %fused_flat to %arg4 by set : tensor<2621440xf32> -> tensor<2621440xf32> to tensor<2621440xf32>
    return %out3, %out4 : tensor<64xf32>, tensor<2621440xf32>
  }
}
