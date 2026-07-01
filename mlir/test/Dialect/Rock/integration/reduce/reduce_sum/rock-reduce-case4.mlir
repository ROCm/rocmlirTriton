// This test is checking for larger reductions with larger block and grid sizes

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s \
// RUN:   | rocmlir-gen -fut test_reduce --arch %arch --clone-harness - \
// RUN:   | rocmlir-driver -kernel-pipeline highlevel -host-pipeline highlevel \
// RUN:   | rocmlir-gen -ph -print-results -fut test_reduce --verifier clone -rand none - \
// RUN:   | rocmlir-driver -c -arch %arch \
// RUN:   | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: Unranked Memref base

func.func @test_reduce(%arg0: tensor<1000x250x1xf32>, %arg1: tensor<1000x1x100xf32>) -> tensor<1x250x100xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %gemm = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) {acc_type = f32} : (tensor<1000x250x1xf32>, tensor<1000x1x100xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1000x250x100xf32>
  %reduced = "tosa.reduce_sum"(%gemm) {axis = 0 : i32} : (tensor<1000x250x100xf32>) -> tensor<1x250x100xf32>
  return %reduced : tensor<1x250x100xf32>
}
