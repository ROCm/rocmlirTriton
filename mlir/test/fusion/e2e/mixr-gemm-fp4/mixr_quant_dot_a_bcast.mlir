// RUN: rocmlir-gen -fut mlir_quant_dot_a_bcast --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_quant_dot_a_bcast --verifier clone - | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// COM: Exercises the A-batch-broadcast path in `rock-fold-broadcast`.
// COM: A is replicated across the G dimension (stride 0), while B and the
// COM: scales have normal strides. With G=2 this triggers FoldBroadcast's
// COM: `isABatchBroadcast` branch and folds the batch dim of B and scaleB
// COM: (regression test for the `op.getBScaleTransposed()` sign bug).

module {
  func.func @mlir_quant_dot_a_bcast(
      %A2d: !migraphx.shaped<256x768xf4E2M1FN, 768x1>,
      %B: !migraphx.shaped<2x768x256xf4E2M1FN, 196608x256x1>,
      %sA: !migraphx.shaped<2x256x768xf32, 196608x768x1>,
      %sB: !migraphx.shaped<2x768x256xf32, 196608x256x1>)
      -> !migraphx.shaped<2x256x256xf32, 65536x256x1>
      attributes {rock.kernel} {
    %A = migraphx.multibroadcast %A2d {out_dyn_dims = [], out_lens = [2, 256, 768]} : <256x768xf4E2M1FN, 768x1> -> <2x256x768xf4E2M1FN, 0x768x1>
    %sE8A = migraphx.convert %sA : !migraphx.shaped<2x256x768xf32, 196608x768x1> to !migraphx.shaped<2x256x768xf8E8M0FNU, 196608x768x1>
    %sE8B = migraphx.convert %sB : !migraphx.shaped<2x768x256xf32, 196608x256x1> to !migraphx.shaped<2x768x256xf8E8M0FNU, 196608x256x1>
    %r = migraphx.quant_dot %A scaled by %sE8A, %B scaled by %sE8B : <2x256x768xf4E2M1FN, 0x768x1> scaled by !migraphx.shaped<2x256x768xf8E8M0FNU, 196608x768x1>, <2x768x256xf4E2M1FN, 196608x256x1> scaled by !migraphx.shaped<2x768x256xf8E8M0FNU, 196608x256x1> -> <2x256x256xf32, 65536x256x1>
    return %r : !migraphx.shaped<2x256x256xf32, 65536x256x1>
  }
}
