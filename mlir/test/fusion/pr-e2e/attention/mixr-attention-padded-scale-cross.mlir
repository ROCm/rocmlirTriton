// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention  --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x128x64xf32, 8192x64x1>,
                                    %arg1: !migraphx.shaped<1x64x27xf32, 1728x27x1>,
                                    %arg2: !migraphx.shaped<1x27x64xf32, 1728x64x1>,
                                    %arg3: !migraphx.shaped<1x128x27xf32, 3456x27x1>) 
                                    -> (!migraphx.shaped<1x128x64xf32, 8192x64x1>)  attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1: <1x128x64xf32, 8192x64x1>, <1x64x27xf32, 1728x27x1> -> <1x128x27xf32, 3456x27x1>
    %biased = migraphx.mul %0, %arg3 : <1x128x27xf32, 3456x27x1>, <1x128x27xf32, 3456x27x1> -> <1x128x27xf32, 3456x27x1>
    %1 = migraphx.softmax %biased{axis = 2 : i64} : <1x128x27xf32, 3456x27x1> -> <1x128x27xf32, 3456x27x1>
    %2 = migraphx.dot %1, %arg2 {perf_config = "attn:v1:32,32,32,1,1,2,0,1,1,0,0"} : <1x128x27xf32, 3456x27x1>, <1x27x64xf32, 1728x64x1> -> <1x128x64xf32, 8192x64x1>
    return %2 : !migraphx.shaped<1x128x64xf32, 8192x64x1>
  }
}
