// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut mlir_convolution_add_leaky_relu --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline migraphx | FileCheck %s --check-prefix=TOSA
// RUN: rocmlir-gen -fut mlir_convolution_add_leaky_relu --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -fut mlir_convolution_add_leaky_relu --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

// MIGraphX decomposes LeakyReLU into greater/mul/convert/where, and the convert
// of the f16 comparison result is what MIGraphXToTosa turns into a
// `fp_to_int_cast` custom op. The comparison only ever yields 0.0 or 1.0, so
// migraphx-tosa-simplify must fold the whole round trip away and drive the
// select from the comparison directly.
// TOSA-LABEL: func.func @mlir_convolution_add_leaky_relu(
// TOSA-NOT: fp_to_int_cast
// TOSA: tosa.select

// Random integer-valued inputs (the -rand_type default) keep the convolution
// exact in f16, so the GPU and CPU paths agree on the sign of every element and
// take the same LeakyReLU branch.
// CLONE: [1 1 1]

module {
  func.func @mlir_convolution_add_leaky_relu(%arg0: !migraphx.shaped<1x8x4x4xf16, 128x16x4x1>, %arg1: !migraphx.shaped<8x8x3x3xf16, 72x9x3x1>, %arg2: !migraphx.shaped<8xf16, 1>) -> !migraphx.shaped<1x8x4x4xf16, 128x16x4x1> attributes {rock.kernel} {
    %0 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %1 = migraphx.literal(dense<2.998050e-01> : tensor<1xf16>) : <1xf16, 1>
    %2 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf16, 128x16x4x1>, <8x8x3x3xf16, 72x9x3x1> -> <1x8x4x4xf16, 128x16x4x1>
    %3 = migraphx.broadcast %arg2 {axis = 1 : i64, out_dyn_dims = [], out_lens = [1, 8, 4, 4]} : <8xf16, 1> -> <1x8x4x4xf16, 0x1x0x0>
    %4 = migraphx.add %2, %3 : <1x8x4x4xf16, 128x16x4x1>, <1x8x4x4xf16, 0x1x0x0> -> <1x8x4x4xf16, 128x16x4x1>
    %5 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 8, 4, 4]} : <1xf16, 1> -> <1x8x4x4xf16, 0x0x0x0>
    %6 = migraphx.greater %4, %5 : <1x8x4x4xf16, 128x16x4x1>, <1x8x4x4xf16, 0x0x0x0> -> <1x8x4x4xf16, 128x16x4x1>
    %7 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 8, 4, 4]} : <1xf16, 1> -> <1x8x4x4xf16, 0x0x0x0>
    %8 = migraphx.mul %4, %7 : <1x8x4x4xf16, 128x16x4x1>, <1x8x4x4xf16, 0x0x0x0> -> <1x8x4x4xf16, 128x16x4x1>
    %9 = migraphx.convert %6 {target_type = 0 : i64} : <1x8x4x4xf16, 128x16x4x1> to <1x8x4x4xsi8, 128x16x4x1>
    %10 = migraphx.where %9, %4, %8 : <1x8x4x4xsi8, 128x16x4x1>, <1x8x4x4xf16, 128x16x4x1>, <1x8x4x4xf16, 128x16x4x1> -> <1x8x4x4xf16, 128x16x4x1>
    return %10 : !migraphx.shaped<1x8x4x4xf16, 128x16x4x1>
  }
}
