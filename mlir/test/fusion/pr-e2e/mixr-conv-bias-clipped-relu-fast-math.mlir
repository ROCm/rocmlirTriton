// The numerical half of the clip story that clip-nnan-ttir.mlir and
// clip-nnan-isa.mlir tell structurally. The same fused conv/bias/clip is
// compiled twice, once with the kernel path's default no-NaN assumption (the
// clip becomes an nnan minnumf/maxnumf pair and folds into one tt.clampf) and
// once with -disable-fast-math (it stays on the NaN-propagating pair). On
// finite data the two must agree with the reference down to the last result:
// the assumption buys cheaper instructions, not different numbers.
//
// --clone-harness builds the reference from the same source, and the reference
// is lowered by upstream tosa-to-linalg, which spells IGNORE as the propagating
// op plus a compare and select and never folds. So even the default run is
// comparing two genuinely different lowerings rather than a kernel against
// itself.

// RUN: rocmlir-gen -fut conv_add_clip --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut conv_add_clip --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// RUN: rocmlir-gen -fut conv_add_clip --arch %arch --clone-harness %s | rocmlir-driver -disable-fast-math -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut conv_add_clip --verifier clone - | rocmlir-driver -disable-fast-math -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @conv_add_clip(%bias: !migraphx.shaped<1x4x1x1xf32, 4x1x1x1>, %filter: !migraphx.shaped<4x3x3x3xf32, 27x9x3x1>, %input: !migraphx.shaped<4x3x3x3xf32, 27x9x3x1>) -> !migraphx.shaped<4x4x1x1xf32, 4x1x1x1> attributes {rock.kernel, rock.arch = ""} {
    %0 = migraphx.multibroadcast %bias {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : <1x4x1x1xf32, 4x1x1x1> -> <4x4x1x1xf32, 0x1x1x1>
    %1 = migraphx.literal (dense<6.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %2 = migraphx.literal (dense<0.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %3 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : <1xf32, 0> -> <4x4x1x1xf32, 0x0x0x0>
    %4 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [4, 4, 1, 1]} : <1xf32, 0> -> <4x4x1x1xf32, 0x0x0x0>
    %5 = migraphx.convolution %filter, %input {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <4x3x3x3xf32, 27x9x3x1>, <4x3x3x3xf32, 27x9x3x1> -> <4x4x1x1xf32, 4x1x1x1>
    %6 = migraphx.add %5, %0 : <4x4x1x1xf32, 4x1x1x1>, <4x4x1x1xf32, 0x1x1x1> -> <4x4x1x1xf32, 4x1x1x1>
    %7 = migraphx.clip %6, %3, %4 : <4x4x1x1xf32, 4x1x1x1>, <4x4x1x1xf32, 0x0x0x0>, <4x4x1x1xf32, 0x0x0x0> -> <4x4x1x1xf32, 4x1x1x1>
    return %7 : !migraphx.shaped<4x4x1x1xf32, 4x1x1x1>
  }
}
