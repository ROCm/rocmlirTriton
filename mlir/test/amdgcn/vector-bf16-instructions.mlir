// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: sed -e 's/###TOKEN_ARCH###/gfx1250/' %s \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --arch gfx1250 -kernel-pipeline=migraphx,highlevel,gpu,triton,binary --elementwise -o /dev/null 2>&1 \
// RUN: | FileCheck --check-prefix=GFX1250 %s

func.func @mlir_add_mul_max_tanh(%arg0: !migraphx.shaped<1x8x4x4xbf16, 128x16x4x1>, %arg1: !migraphx.shaped<1x8x4x4xbf16, 128x16x4x1>) -> !migraphx.shaped<1x8x4x4xbf16, 128x16x4x1> attributes {rock.kernel, rock.elementwise, rock.arch = "###TOKEN_ARCH###"} {
  %1 = migraphx.add %arg0, %arg1 : <1x8x4x4xbf16, 128x16x4x1>, <1x8x4x4xbf16, 128x16x4x1> -> <1x8x4x4xbf16, 128x16x4x1>
  %2 = migraphx.mul %1, %arg1 : <1x8x4x4xbf16, 128x16x4x1>, <1x8x4x4xbf16, 128x16x4x1> -> <1x8x4x4xbf16, 128x16x4x1>
  %3 = migraphx.max %2, %1 : <1x8x4x4xbf16, 128x16x4x1>, <1x8x4x4xbf16, 128x16x4x1> -> <1x8x4x4xbf16, 128x16x4x1>
  %4 = migraphx.tanh %3 : <1x8x4x4xbf16, 128x16x4x1> -> <1x8x4x4xbf16, 128x16x4x1>
  return %4 : !migraphx.shaped<1x8x4x4xbf16, 128x16x4x1>
}

// GFX1250: v_pk_add_bf16
// GFX1250: v_pk_mul_bf16
// GFX1250: v_pk_max_num_bf16
// GFX1250: v_tanh_f32
// TODO: Investigate why v_tanh_bf16 is not produced
