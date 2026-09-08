// RUN: rocmlir-gen -fut dot_dynamic_m --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut dot_dynamic_m -m 256 --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// The kernel is compiled without knowing M: it takes M as a runtime argument
// and the launch derives its grid from that. `-m` picks the M this run
// allocates and verifies against. It has to be a multiple of the tile height,
// because the kernel does not yet mask the tail of the M axis.

// CHECK: [1 1 1]
module {
  func.func @dot_dynamic_m(%arg0: !migraphx.shaped<?x64xf32, 64x1>, %arg1: !migraphx.shaped<64x64xf32, 64x1>) -> !migraphx.shaped<?x64xf32, 64x1> attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 : <?x64xf32, 64x1>, <64x64xf32, 64x1> -> <?x64xf32, 64x1>
    return %0 : !migraphx.shaped<?x64xf32, 64x1>
  }
}
