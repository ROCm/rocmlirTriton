// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-opt --migraphx-to-tosa --split-input-file -verify-diagnostics %s | FileCheck %s

// Scaled quant_dot with f8E4M3FN scales (NVFP4-style) lowers to
// tosa.matmul_t_block_scaled with block size 16 and keeps the scale type.
// CHECK-LABEL: func.func @quant_dot_e4m3_scales
// CHECK: tosa.slice {{.*}} -> tensor<1x64x8x1xf8E4M3FN>
// CHECK: tosa.slice {{.*}} -> tensor<1x8x1x64xf8E4M3FN>
// CHECK: tosa.matmul_t_block_scaled {{.*}} {acc_type = f32, block_size = BLOCK_SIZE_16} : (tensor<1x64x128xf4E2M1FN>, tensor<1x64x8xf8E4M3FN>, tensor<1x64x128xf4E2M1FN>, tensor<1x64x8xf8E4M3FN>) -> tensor<1x64x64xf32>
func.func @quant_dot_e4m3_scales(
    %arg0: !migraphx.shaped<1x64x128xf4E2M1FN, 8192x128x1>,
    %arg1: !migraphx.shaped<1x128x64xf4E2M1FN, 8192x64x1>,
    %arg2: !migraphx.shaped<1x64x128xf8E4M3FN, 8192x128x1>,
    %arg3: !migraphx.shaped<1x128x64xf8E4M3FN, 8192x64x1>
) -> !migraphx.shaped<1x64x64xf32, 4096x64x1> attributes {rock.kernel} {
  %0 = migraphx.quant_dot %arg0 scaled by %arg2, %arg1 scaled by %arg3
    : <1x64x128xf4E2M1FN, 8192x128x1> scaled by !migraphx.shaped<1x64x128xf8E4M3FN, 8192x128x1>,
      <1x128x64xf4E2M1FN, 8192x64x1> scaled by !migraphx.shaped<1x128x64xf8E4M3FN, 8192x64x1>
    -> <1x64x64xf32, 4096x64x1>
  return %0 : !migraphx.shaped<1x64x64xf32, 4096x64x1>
}

// -----

// K=40 is not a multiple of the f8E4M3FN scale block size (16).
func.func @quant_dot_e4m3_scales_k_not_multiple_of_block_size(
    %arg0: !migraphx.shaped<1x64x40xf4E2M1FN, 2560x40x1>,
    %arg1: !migraphx.shaped<1x40x64xf4E2M1FN, 2560x64x1>,
    %arg2: !migraphx.shaped<1x64x40xf8E4M3FN, 2560x40x1>,
    %arg3: !migraphx.shaped<1x40x64xf8E4M3FN, 2560x64x1>
) -> !migraphx.shaped<1x64x64xf32, 4096x64x1> attributes {rock.kernel} {
  // expected-error @+2 {{K dimension (40) must be a multiple of blockSize (16)}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.quant_dot' that was explicitly marked illegal}}
  %0 = migraphx.quant_dot %arg0 scaled by %arg2, %arg1 scaled by %arg3
    : <1x64x40xf4E2M1FN, 2560x40x1> scaled by !migraphx.shaped<1x64x40xf8E4M3FN, 2560x40x1>,
      <1x40x64xf4E2M1FN, 2560x64x1> scaled by !migraphx.shaped<1x40x64xf8E4M3FN, 2560x64x1>
    -> <1x64x64xf32, 4096x64x1>
  return %0 : !migraphx.shaped<1x64x64xf32, 4096x64x1>
}
