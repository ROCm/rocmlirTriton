// RUN: rocmlir-driver -kernel-pipeline highlevel %s | FileCheck %s

// Verify that transpose is converted as transform and add is fused.
// CHECK-DAG: #{{.*}} = #rock.transform_map<{{.*}} by [<PassThrough [{{.*}}] at [0, 1, 2, 3] -> [{{.*}}] at [0, 2, 3, 1]>]
// CHECK: rock.conv
// CHECK: arith.addf
// CHECK-COUNT-1: rock.store
// CHECK-NOT: rock.store

// NOTE: using gfx906 arch to make sure we get non-accel path

func.func @test_fusion(%arg0: tensor<256x28x28x128xf32>, %arg1: tensor<64x3x3x128xf32>, %arg2: tensor<256x64x28x28xf32>) -> tensor<256x28x28x64xf32> attributes {rock.kernel, rock.arch = "gfx906"} {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %cst_0, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, expected_filter_layout = "kyxc", expected_input_layout = "nhwc", expected_output_layout = "nhwk", pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<256x28x28x128xf32>, tensor<64x3x3x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<256x28x28x64xf32>
    %1 = "tosa.transpose"(%arg2) {perms = array<i32: 0, 2, 3, 1>} : (tensor<256x64x28x28xf32>) -> tensor<256x28x28x64xf32>
    %2 = "tosa.add"(%0, %1) : (tensor<256x28x28x64xf32>, tensor<256x28x28x64xf32>) -> tensor<256x28x28x64xf32>
    return %2 : tensor<256x28x28x64xf32>
}
