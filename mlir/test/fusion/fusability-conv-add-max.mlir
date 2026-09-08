// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,5,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:0
// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
// RUN: rocmlir-gen --emit-tuning-key - < %s | FileCheck %s --check-prefix=CHECK-TUNING-KEY
// CHECK-TUNING-KEY: -supportsSplitK false
module {
  func.func @mlir_convolution_add_relu(%arg0: tensor<64x1x1x1xf32>, %arg1: tensor<1x256x56x56xf32>, %arg2: tensor<64x256x1x1xf32>, %arg3: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x64x56x56xf32>
    %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)> by [<PassThrough ["dim3", "dim0", "dim1", "dim2"] at [0, 1, 2, 3] -> ["dim3", "dim0", "dim1", "dim2"] at [3, 0, 1, 2]>] bounds = [1, 64, 1, 1] -> [64, 1, 1, 1]> : tensor<64x1x1x1xf32> to tensor<1x64x1x1xf32>
    %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 64, 56, 56] -> [1, 64, 1, 1]> : tensor<1x64x1x1xf32> to tensor<1x64x56x56xf32>
    %2 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 256 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 256} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [1, 1, 256, 56, 56] -> [1, 256, 56, 56]> : tensor<1x256x56x56xf32> to tensor<1x1x256x56x56xf32>
    %3 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 64 + d1, d2, d3, d4)> by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 64} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 64, 256, 1, 1] -> [64, 256, 1, 1]> : tensor<64x256x1x1xf32> to tensor<1x64x256x1x1xf32>
    %5 = rock.conv(%3, %2) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : tensor<1x64x256x1x1xf32>, tensor<1x1x256x56x56xf32> -> tensor<1x1x64x56x56xf32>
    %6 = rock.transform %5 by <affine_map<(d0, d1, d2) -> (0, 0, d0, d1, d2)> by [<Merge{1, 1, 64} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1", "dim2"] at [1, 2] -> ["dim1", "dim2"] at [3, 4]>] bounds = [64, 56, 56] -> [1, 1, 64, 56, 56]> : tensor<1x1x64x56x56xf32> to tensor<64x56x56xf32>
    %7 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 64} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [64, 56, 56] -> [1, 64, 56, 56]> : tensor<1x64x56x56xf32> to tensor<64x56x56xf32>
    %8 = arith.addf %6, %7 : tensor<64x56x56xf32>
    %9 = arith.constant dense<0.000000e+00> : tensor<64x56x56xf32>
    %10 = arith.maximumf %8, %9 : tensor<64x56x56xf32>
    %11 = rock.transform %10 by <affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1, d2, d3)> by [<Unmerge{1, 64} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>] bounds = [1, 64, 56, 56] -> [64, 56, 56]> : tensor<64x56x56xf32> to tensor<1x64x56x56xf32>
    %12 = rock.store %11 to %arg3 by  set : tensor<1x64x56x56xf32> -> tensor<1x64x56x56xf32> to tensor<1x64x56x56xf32>
    return %12 : tensor<1x64x56x56xf32>
  }
}
