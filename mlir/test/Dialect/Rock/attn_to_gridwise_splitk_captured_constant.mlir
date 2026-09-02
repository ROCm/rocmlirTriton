// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Split-k applied to a gemm+gemm whose inter-gemm body scales by a splat
// literal. Driven through the real pipelines rather than a hand-written pass
// list, because the point of this test is an interaction between passes:
// rock-regularize-inter-gemm-fusion inlines the literal into the body, and
// then rock-conv-to-gemm drives the greedy rewriter, whose folder CSEs
// constants across parent-ancestor regions and merges that copy back into the
// identical one at function scope. So the constant attn-to-gridwise has to
// reshape is one the body reads from outside the region.

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-driver -kernel-pipeline=gpu -mlir-print-ir-after=rock-attn-to-gridwise -mlir-print-local-scope -o /dev/null 2>&1 | FileCheck %s

// The second GEMM reduces over gemm0's N, so a 4-way slice of it folds into G
// and the body is retyped to that per-split shape.
// CHECK-LABEL: @gemm_gemm_splitk_const_scale
// CHECK: rock.gridwise_attention
// CHECK-NEXT: ^bb0(%[[AB:.*]]: tensor<4x64x16xf32>):

// The captured constant has to move with the body. Were it left at the
// unsplit shape, this multiply would have mismatched operands and the IR
// would not verify.
// CHECK-NEXT: %[[CST:.*]] = arith.constant dense<2.500000e-01> : tensor<4x64x16xf32>
// CHECK-NEXT: arith.mulf %[[AB]], %[[CST]] : tensor<4x64x16xf32>

// CHECK: params1 = #rock.gemm_params<{{.*}}splitKFactor = 4
// CHECK: rock.store %{{.*}} by atomic_add
module {
  func.func @gemm_gemm_splitk_const_scale(%arg0: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg1: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg2: !migraphx.shaped<1x64x64xf32, 4096x64x1>) -> !migraphx.shaped<1x64x64xf32, 4096x64x1> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
    %0 = migraphx.dot %arg0, %arg1 : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %cst = migraphx.literal(dense<2.500000e-01> : tensor<1x64x64xf32>) : <1x64x64xf32, 0x0x0>
    %scaled = migraphx.mul %0, %cst : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf32, 4096x64x1>
    %1 = migraphx.dot %scaled, %arg2 {perf_config="attn:mPerBlockG0=128,nPerBlockG0=64,kPerBlock=32,numWaves=4,matrixInstrNonkdim=0,splitKFactor=4"} : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    return %1 : !migraphx.shaped<1x64x64xf32, 4096x64x1>
  }
}
