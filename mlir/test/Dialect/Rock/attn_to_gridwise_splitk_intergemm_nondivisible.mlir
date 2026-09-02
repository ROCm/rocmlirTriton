// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Split-k on a gemm+gemm inter-gemm fusion where neither extent is convenient:
// gemm0's N is 7 and the split factor is 4, so N has to be padded to 8 before
// it can be split, and gemm0's M is 7 against an mPerBlock of 32, so M picks up
// tile padding afterwards. The two paddings land in different places, and the
// elementwise input has to follow only the first of them - it is padded along
// gemmN with the keys, but keeps the pre-tile-pad M extent. The e2e test of the
// same shape only checks the numbers come out right; this pins down where each
// pad is applied.

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-driver -kernel-pipeline=gpu -mlir-print-ir-after=rock-attn-to-gridwise -mlir-print-local-scope -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: @gemm_gemm_splitk_intergemm_nondivisible

// The elementwise input is padded along gemmN only: gemmM passes through, so
// the pad matches the one given to the keys and not the one given to gemm0's M.
// CHECK: %[[EW_PAD:.*]] = rock.transform %{{.*}} by <{{.*}}<PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 1} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>{{.*}}> : tensor<1x7x7xf32> to tensor<1x7x8xf32>

// Only once padded is it splittable four ways, and the split index folds into
// gemmG exactly as it does for the keys.
// CHECK: %[[EW_SPLIT:.*]] = rock.transform %[[EW_PAD]] by <{{.*}}<Unmerge{4, 2} ["gemmNSplit", "gemmN"] at [1, 2] -> ["gemmN"] at [2]>{{.*}}> : tensor<1x7x8xf32> to tensor<1x4x2x7xf32>
// CHECK-NEXT: %[[EW:.*]] = rock.transform %[[EW_SPLIT]] by <{{.*}}<Merge{1, 4} ["gemmG"] at [0] -> ["gemmG", "gemmNSplit"] at [0, 1]>{{.*}}> : tensor<1x4x2x7xf32> to tensor<4x7x2xf32>

// Matrix A is tile-padded to 32x32 after the split. The elementwise input does
// not get this padding, which is why the op records the pre-pad extents below.
// CHECK: %[[A:.*]] = rock.transform %{{.*}} by <{{.*}}<Pad{0, 25} ["gemm0MPad"] at [1] -> ["gemm0M"] at [1]>{{.*}}> : tensor<4x7x3xf32> to tensor<4x32x32xf32>

// The body is retyped to the pre-tile-pad M and the per-split N.
// CHECK: rock.gridwise_attention(%[[A]], %{{.*}}, %{{.*}}, %[[EW]])
// CHECK-NEXT: ^bb0(%[[AB:.*]]: tensor<4x7x2xf32>, %[[BIAS:.*]]: tensor<4x7x2xf32>):
// CHECK-NEXT: arith.addf %[[AB]], %[[BIAS]] : tensor<4x7x2xf32>

// Both pre-pad extents are recorded, and they are what the verifier checks the
// elementwise input against: M is the pre-tile-pad 7, N the per-split 2.
// CHECK: splitKFactor = 4
// CHECK-SAME: prePadG0M = 7 : index, prePadG0N = 2 : index
// CHECK: rock.store %{{.*}} by atomic_add
module {
  func.func @gemm_gemm_splitk_intergemm_nondivisible(%arg0: !migraphx.shaped<1x7x3xf32, 21x3x1>, %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1>, %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1>, %arg3: !migraphx.shaped<1x7x7xf32, 49x7x1>) -> (!migraphx.shaped<1x7x3xf32, 21x3x1>) attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
    %0 = migraphx.dot %arg0, %arg1 : <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1> -> <1x7x7xf32, 49x7x1>
    %biased = migraphx.add %0, %arg3 : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
    %1 = migraphx.dot %biased, %arg2 {perf_config="attn:mPerBlockG0=32,nPerBlockG0=32,kPerBlock=32,numWaves=4,matrixInstrNonkdim=0,splitKFactor=4"} : <1x7x7xf32, 49x7x1>, <1x7x3xf32, 21x3x1> -> <1x7x3xf32, 21x3x1>
    return %1 : !migraphx.shaped<1x7x3xf32, 21x3x1>
  }
}
