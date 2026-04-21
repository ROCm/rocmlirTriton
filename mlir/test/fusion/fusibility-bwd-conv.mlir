// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefix=NOSPLITK
// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,4,2,0,0 - < %s | FileCheck %s --check-prefix=SPLITK
// NOSPLITK: fusible:0
// SPLITK: fusible:0
module {
  func.func @mlir_conv_bwd_data_add_relu(
      %arg0: tensor<1x64x3x7x7xf32>,
      %arg1: tensor<256x1x3x230x230xf32>,
      %arg3: tensor<256x1x64x112x112xf32>
  ) -> tensor<256x1x64x112x112xf32>
  attributes {enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
    %conv = rock.conv_bwd_data(%arg0, %arg1) {
      dilations = [1 : index, 1 : index],
      filter_layout = ["g", "k", "c", "y", "x"],
      input_layout = ["ni", "gi", "ci", "hi", "wi"],
      output_layout = ["no", "go", "ko", "ho", "wo"],
      padding = [0 : index, 0 : index, 0 : index, 0 : index],
      strides = [2 : index, 2 : index]
    } : tensor<1x64x3x7x7xf32>, tensor<256x1x3x230x230xf32> -> tensor<256x1x64x112x112xf32>
    %add = "tosa.add"(%conv, %arg3) : (tensor<256x1x64x112x112xf32>, tensor<256x1x64x112x112xf32>) -> tensor<256x1x64x112x112xf32>
    %relu = tosa.clamp %add {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<256x1x64x112x112xf32>) -> tensor<256x1x64x112x112xf32>
    return %relu : tensor<256x1x64x112x112xf32>
  }
}
