// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A depthwise convolution has one output channel per group, so its GEMM M
// dimension is 1. When mPerBlock is tuned to 1 the result tensor still gets the
// matrix instruction's layout, and gfx11's WMMA duplicates its result across
// the register dimension rather than offsetting it (see the version 1 vs 2/3
// note in wmmaDotOperandToLinearLayout). Every register in a lane then holds
// the same output element.
//
// The store conversions in Triton's AMD backend drop those redundant registers
// using the "register" free-variable mask of getFreeVariableMasks(), and the
// epilogue feeding the dropped stores dies with them, which is why the exp2
// count falls along with the store count. Without that deduplication gfx1100
// emits 32 stores and 32 exp2 calls instead of 4 and 4.
//
// gfx12 uses WMMA version 2, which offsets its result across registers instead
// of duplicating it, so there is nothing to deduplicate there; it is checked
// here to pin that contrast.
//
// perf_config is pinned so the counts do not move when the tuning heuristics
// change. mPerBlock is the first field, and it is what makes the layout
// degenerate.

// RUN: sed s/##TOKEN_ARCH##/gfx1100/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx1100 | env LLVM_IR_ENABLE_DUMP=1 rocmlir-driver -kernel-pipeline=gpu,triton,binary -arch gfx1100 -o /dev/null 2>&1 | FileCheck %s --check-prefix=GFX1100
// RUN: sed s/##TOKEN_ARCH##/gfx1200/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx1200 | env LLVM_IR_ENABLE_DUMP=1 rocmlir-driver -kernel-pipeline=gpu,triton,binary -arch gfx1200 -o /dev/null 2>&1 | FileCheck %s --check-prefix=GFX1200

// GFX1100-COUNT-4: call{{.*}}@llvm.amdgcn.wmma
// GFX1100-NOT: call{{.*}}@llvm.amdgcn.wmma
// GFX1100-COUNT-4: call{{.*}}@llvm.exp2
// GFX1100-NOT: call{{.*}}@llvm.exp2
// GFX1100-COUNT-4: call void @llvm.amdgcn.raw.ptr.buffer.store
// GFX1100-NOT: call void @llvm.amdgcn.raw.ptr.buffer.store

// GFX1200-COUNT-4: call{{.*}}@llvm.amdgcn.wmma
// GFX1200-NOT: call{{.*}}@llvm.amdgcn.wmma
// GFX1200-COUNT-8: call{{.*}}@llvm.exp2
// GFX1200-NOT: call{{.*}}@llvm.exp2
// GFX1200-COUNT-4: call void @llvm.amdgcn.raw.ptr.buffer.store
// GFX1200-NOT: call void @llvm.amdgcn.raw.ptr.buffer.store

module {
  func.func @mlir_quant_convolution_dequantizelinear_sigmoid_mul_quantizelinear(%arg0: !migraphx.shaped<1x80x80x80xsi8, 512000x6400x80x1>, %arg1: !migraphx.shaped<80x1x3x3xsi8, 9x9x3x1>) -> !migraphx.shaped<1x80x80x80xsi8, 512000x6400x80x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.enable_splitk_for_tuning, rock.kernel} {
    %0 = migraphx.literal(dense<3.29618277E-10> : tensor<1xf32>) : <1xf32, 0>
    %1 = migraphx.literal(dense<1.1920929E-7> : tensor<1xf32>) : <1xf32, 0>
    %2 = migraphx.quant_convolution %arg0, %arg1 {dilation = [1, 1], group = 80 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, perf_config = "gemm:v5:1,256,16,1,1,4,0,1,1,0,0,-1,-1,-1,-1,-1,-1,-1", stride = [1, 1]} : <1x80x80x80xsi8, 512000x6400x80x1>, <80x1x3x3xsi8, 9x9x3x1> -> <1x80x80x80xsi32, 512000x6400x80x1>
    %3 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 80, 80, 80]} : <1xf32, 0> -> <1x80x80x80xf32, 0x0x0x0>
    %4 = migraphx.dequantizelinear %2, %3 : <1x80x80x80xsi32, 512000x6400x80x1>, <1x80x80x80xf32, 0x0x0x0> -> <1x80x80x80xf32, 512000x6400x80x1>
    %5 = migraphx.sigmoid %4 : <1x80x80x80xf32, 512000x6400x80x1> -> <1x80x80x80xf32, 512000x6400x80x1>
    %6 = migraphx.mul %4, %5 : <1x80x80x80xf32, 512000x6400x80x1>, <1x80x80x80xf32, 512000x6400x80x1> -> <1x80x80x80xf32, 512000x6400x80x1>
    %7 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 80, 80, 80]} : <1xf32, 0> -> <1x80x80x80xf32, 0x0x0x0>
    %8 = migraphx.quantizelinear %6, %7 {out_type = 5 : i64} : <1x80x80x80xf32, 512000x6400x80x1>, <1x80x80x80xf32, 0x0x0x0> -> <1x80x80x80xsi8, 512000x6400x80x1>
    return %8 : !migraphx.shaped<1x80x80x80xsi8, 512000x6400x80x1>
  }
}
