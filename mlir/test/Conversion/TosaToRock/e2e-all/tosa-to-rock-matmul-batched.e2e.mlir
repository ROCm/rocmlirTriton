// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline highlevel \
// RUN: | rocmlir-gen -ph -rand=none -print-results - \
// RUN: | rocmlir-driver -kernel-pipeline full -arch %arch \
// RUN: | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [327680] strides = [1] data =
// CHECK-NEXT: [64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,

func.func @test_fusion(%a: tensor<10x128x64xf32>, %b: tensor<10x64x256xf32>) -> tensor<10x128x256xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = "tosa.matmul"(%a, %b, %a_zp, %b_zp) {acc_type = f32} : (tensor<10x128x64xf32>, tensor<10x64x256xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<10x128x256xf32>

  return %0 : tensor<10x128x256xf32>
}

// -----
