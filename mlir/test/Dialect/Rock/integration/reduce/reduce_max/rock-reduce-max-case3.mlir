// This test is checking for a reduction in highest dimension

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s \
// RUN:   | rocmlir-gen -fut test_reduce --arch %arch --clone-harness - \
// RUN:   | rocmlir-driver -kernel-pipeline highlevel -host-pipeline highlevel \
// RUN:   | rocmlir-gen -ph -print-results -fut test_reduce --verifier clone -rand none - \
// RUN:   | rocmlir-driver -c -arch %arch \
// RUN:   | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: Unranked Memref base

func.func @test_reduce(%arg0: tensor<20x30x1xf32>, %arg1: tensor<20x1x10xf32>) -> tensor<1x30x10xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %gemm = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) : (tensor<20x30x1xf32>, tensor<20x1x10xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<20x30x10xf32>
  %reduced = "tosa.reduce_max"(%gemm) {axis = 0 : i32} : (tensor<20x30x10xf32>) -> tensor<1x30x10xf32>
  return %reduced : tensor<1x30x10xf32>
}
