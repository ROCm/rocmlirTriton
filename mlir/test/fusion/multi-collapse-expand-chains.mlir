// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline gpu -arch %arch | rocmlir-opt | FileCheck %s

// CHECK: tt.dot
// CHECK: arith.addf
// CHECK: tt.store
#map = affine_map<(d0, d1, d2, d3) -> (((d0 * 3 + d1) * 3 + d2) * 3 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 3 + d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 4 + d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d0, d1, d2, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d1, d2, d3)>
#map8 = affine_map<(d0) -> (d0 floordiv 4, d0 mod 4, 0, 0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{4, 3, 3, 3} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [4, 3, 3, 3] -> [108]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{4} ["exp1"] at [1] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>, <AddDim{1} ["unit3"] at [3] -> [] at []>] bounds = [1, 4, 1, 1] -> [4]>
#transform_map2 = #rock.transform_map<#map2 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [4, 4, 1, 1] -> [1, 4, 1, 1]>
#transform_map3 = #rock.transform_map<#map3 by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 3} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [4, 1, 3, 3, 3] -> [4, 3, 3, 3]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 4} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 4, 3, 3, 3] -> [4, 3, 3, 3]>
#transform_map5 = #rock.transform_map<#map5 by [<PassThrough ["dim1", "dim2", "dim3", "dim0", "dim4"] at [0, 1, 2, 3, 4] -> ["dim1", "dim2", "dim3", "dim0", "dim4"] at [1, 2, 3, 0, 4]>] bounds = [4, 3, 3, 1, 3] -> [1, 4, 3, 3, 3]>
#transform_map6 = #rock.transform_map<#map6 by [<PassThrough ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 1, 2, 3, 4] -> ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 2, 3, 1, 4]>] bounds = [4, 3, 3, 1, 3] -> [4, 1, 3, 3, 3]>
#transform_map7 = #rock.transform_map<#map7 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Merge{1, 4} ["dim1"] at [1] -> ["col1", "col2"] at [1, 2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [4]>] bounds = [4, 4, 1, 1] -> [4, 1, 4, 1, 1]>
#transform_map8 = #rock.transform_map<#map8 by [<Merge{4, 4, 1, 1} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3]>] bounds = [16] -> [4, 4, 1, 1]>
module {
  func.func @test(%arg0: tensor<4xf32>, %arg1: tensor<108xf32>, %arg2: tensor<108xf32>, %arg3: tensor<16xf32>) -> tensor<16xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %cst = arith.constant dense<3.40282347E+38> : tensor<4x4x1x1xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x4x1x1xf32>
    %0 = rock.transform %arg2 by #transform_map : tensor<108xf32> to tensor<4x3x3x3xf32>
    %1 = rock.transform %arg1 by #transform_map : tensor<108xf32> to tensor<4x3x3x3xf32>
    %2 = rock.transform %arg0 by #transform_map1 : tensor<4xf32> to tensor<1x4x1x1xf32>
    %3 = rock.transform %2 by #transform_map2 : tensor<1x4x1x1xf32> to tensor<4x4x1x1xf32>
    %4 = rock.transform %1 by #transform_map3 : tensor<4x3x3x3xf32> to tensor<4x1x3x3x3xf32>
    %5 = rock.transform %0 by #transform_map4 : tensor<4x3x3x3xf32> to tensor<1x4x3x3x3xf32>
    %6 = rock.transform %5 by #transform_map5 : tensor<1x4x3x3x3xf32> to tensor<4x3x3x1x3xf32>
    %7 = rock.transform %4 by #transform_map6 : tensor<4x1x3x3x3xf32> to tensor<4x3x3x1x3xf32>
    %8 = rock.conv(%5, %4) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : tensor<1x4x3x3x3xf32>, tensor<4x1x3x3x3xf32> -> tensor<4x1x4x1x1xf32>
    %9 = rock.transform %8 by #transform_map7 : tensor<4x1x4x1x1xf32> to tensor<4x4x1x1xf32>
    %10 = arith.addf %9, %3 : tensor<4x4x1x1xf32>
    %11 = arith.maximumf %10, %cst_0 : tensor<4x4x1x1xf32>
    %12 = arith.minimumf %11, %cst : tensor<4x4x1x1xf32>
    %13 = rock.transform %12 by #transform_map8 : tensor<4x4x1x1xf32> to tensor<16xf32>
    %14 = rock.store %13 to %arg3 by  set : tensor<16xf32> -> tensor<16xf32> to tensor<16xf32>
    return %14 : tensor<16xf32>
  }
}
