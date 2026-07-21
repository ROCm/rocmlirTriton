// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen -ph -verifier clone -fut test - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]

// Non-power-of-two kPerBlock (48) exercised via perf_config on a simple GEMM.
module {
  func.func @test(%arg0: !migraphx.shaped<1x128x96xf32, 12288x96x1>, %arg1: !migraphx.shaped<1x96x128xf32, 12288x128x1>) -> !migraphx.shaped<1x128x128xf32, 16384x128x1> attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 {perf_config = "gemm:v1:64,64,48,1,1,4,16,1,1,1,1"} : <1x128x96xf32, 12288x96x1>, <1x96x128xf32, 12288x128x1> -> <1x128x128xf32, 16384x128x1>
    return %0 : !migraphx.shaped<1x128x128xf32, 16384x128x1>
  }
}
