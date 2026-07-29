// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @mlir_attention(%q: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>,
                            %k: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>,
                            %v: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>)
      -> !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1> attributes {rock.kernel} {
    %scale = migraphx.literal(dense<1.250000e-01> : tensor<1xf16>) : <1xf16, 0>
    %scale_b = migraphx.multibroadcast %scale {out_dyn_dims = [], out_lens = [1, 12, 256, 256]} : <1xf16, 0> -> <1x12x256x256xf16, 0x0x0x0>
    %k_t = migraphx.transpose %k {permutation = [0, 1, 3, 2]} : <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x256xf16, 786432x65536x1x256>
    %qk = migraphx.dot %q, %k_t : <1x12x256x256xf16, 786432x65536x256x1>, <1x12x256x256xf16, 786432x65536x1x256> -> <1x12x256x256xf16, 786432x65536x256x1>
    %qk_scaled = migraphx.mul %qk, %scale_b : <1x12x256x256xf16, 786432x65536x256x1>, <1x12x256x256xf16, 0x0x0x0> -> <1x12x256x256xf16, 786432x65536x256x1>
    %att = migraphx.softmax %qk_scaled {axis = 3 : i64} : <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x256xf16, 786432x65536x256x1>
    %out = migraphx.dot %att, %v : <1x12x256x256xf16, 786432x65536x256x1>, <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x256xf16, 786432x65536x256x1>
    return %out : !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>
  }
}
