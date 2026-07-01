// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention  --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @mlir_attention(%v: !migraphx.shaped<2x2x32x32xf32, 2048x1024x32x1>, 
                            %q: !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>, 
                            %k: !migraphx.shaped<2x2x32x32xf32, 2048x1024x32x1>) 
                            -> (!migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>)  attributes {rock.kernel} {
    %vbroadcast = migraphx.multibroadcast %v {out_dyn_dims = [], out_lens = [2, 2, 2, 32, 32]} : <2x2x32x32xf32, 2048x1024x32x1> -> <2x2x2x32x32xf32, 2048x1024x0x32x1>
    %vreshaped = migraphx.reshape %vbroadcast {dims = [2, 4, 32, 32]} : <2x2x2x32x32xf32, 2048x1024x0x32x1> -> <2x4x32x32xf32, 2048x1024x32x1>
    %kbroadcast = migraphx.multibroadcast %k {out_dyn_dims = [], out_lens = [2, 2, 2, 32, 32]} : <2x2x32x32xf32, 2048x1024x32x1> -> <2x2x2x32x32xf32, 2048x1024x0x32x1>
    %kreshaped = migraphx.reshape %kbroadcast {dims = [2, 4, 32, 32]} : <2x2x2x32x32xf32, 2048x1024x0x32x1> -> <2x4x32x32xf32, 2048x1024x32x1>
    %kt = migraphx.transpose %kreshaped {permutation = [0, 1, 3, 2]} : <2x4x32x32xf32, 2048x1024x32x1> -> <2x4x32x32xf32, 2048x1024x32x1>
    %qk = migraphx.dot %q, %kt : <2x4x32x32xf32, 4096x1024x32x1>, <2x4x32x32xf32, 2048x1024x32x1> -> <2x4x32x32xf32, 4096x1024x32x1>
    %att = migraphx.softmax %qk {axis = 3 : i64} : <2x4x32x32xf32, 4096x1024x32x1> -> <2x4x32x32xf32, 4096x1024x32x1>
    %res = migraphx.dot %att, %vreshaped : <2x4x32x32xf32, 4096x1024x32x1>, <2x4x32x32xf32, 2048x1024x32x1> -> <2x4x32x32xf32, 4096x1024x32x1>
    return %res : !migraphx.shaped<2x4x32x32xf32, 4096x1024x32x1>
  }
}
