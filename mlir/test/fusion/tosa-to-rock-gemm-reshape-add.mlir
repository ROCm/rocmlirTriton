// RUN: rocmlir-driver -kernel-pipeline highlevel %s | FileCheck %s

// Verify that reshape is converted as transform and add is fused.
// CHECK: rock.gemm
// CHECK: arith.addf
// CHECK-COUNT-1: rock.store
// CHECK-NOT: rock.store

// NOTE: using gfx906 arch to make sure we get non-accel path

func.func @test_fusion(%arg0: tensor<1x1x512xf32>, %arg1: tensor<1x512x1000xf32>, %arg2: tensor<1x1000xf32>) -> (tensor<1x1000xf32>) attributes {rock.kernel, rock.arch = "gfx906"} {
    %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) {acc_type = f32} : (tensor<1x1x512xf32>, tensor<1x512x1000xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1000xf32>
    %const_shape = "tosa.const_shape"() { values = dense<[1, 1000]> : tensor<2xindex> } : () -> !tosa.shape<2>
    %3 = "tosa.reshape"(%2, %const_shape) : (tensor<1x1x1000xf32>, !tosa.shape<2>) -> tensor<1x1000xf32>
    %4 = "tosa.add"(%3, %arg2) : (tensor<1x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %4 : tensor<1x1000xf32>
}
