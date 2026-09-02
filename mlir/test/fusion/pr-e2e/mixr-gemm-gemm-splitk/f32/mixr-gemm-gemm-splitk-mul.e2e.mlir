// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut gemm_gemm_splitk_mul --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut gemm_gemm_splitk_mul --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// CHECK: [1 1 1]

// A multiplicative epilogue distributes over the split, so it must NOT pick up
// a division by the split factor.
// RUN: rocmlir-gen -fut gemm_gemm_splitk_mul --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-opt --rock-affix-params --rock-fusion-splitk-regularization | FileCheck %s --check-prefix=SPLITK
// SPLITK: splitKFactor = 4
// SPLITK-NOT: arith.divf
module {
  func.func @gemm_gemm_splitk_mul(%arg0: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg1: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg2: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg3: !migraphx.shaped<1x64x64xf32, 4096x64x1>) -> (!migraphx.shaped<1x64x64xf32, 4096x64x1>) attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %1 = migraphx.dot %0, %arg2 {perf_config="attn:mPerBlockG0=128,nPerBlockG0=64,kPerBlock=32,numWaves=4,matrixInstrNonkdim=0,splitKFactor=4"} : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %2 = migraphx.mul %1, %arg3 : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    return %2 : !migraphx.shaped<1x64x64xf32, 4096x64x1>
  }
}
