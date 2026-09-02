// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut gemm_gemm_splitk_linear --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut gemm_gemm_splitk_linear --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// CHECK: [1 1 1]

// A scale-and-bias tree: the two adds that mix the gemm result with an
// external tensor each need their own division, while the add whose operands
// both come from the gemm chain must be left alone.
// RUN: rocmlir-gen -fut gemm_gemm_splitk_linear --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-opt --rock-affix-params --rock-fusion-splitk-regularization | FileCheck %s --check-prefix=SPLITK
// SPLITK: splitKFactor = 4
// SPLITK: arith.divf
module {
  func.func @gemm_gemm_splitk_linear(%arg0: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg1: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg2: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg3: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg4: !migraphx.shaped<1x64x64xf32, 4096x64x1>) -> (!migraphx.shaped<1x64x64xf32, 4096x64x1>) attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %1 = migraphx.dot %0, %arg2 {perf_config="attn:mPerBlockG0=128,nPerBlockG0=64,kPerBlock=32,numWaves=4,matrixInstrNonkdim=0,splitKFactor=4"} : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %cst = migraphx.literal(dense<3.000000e+00> : tensor<1x64x64xf32>) : <1x64x64xf32, 0x0x0>
    %scale = migraphx.add %arg3, %cst : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf32, 4096x64x1>
    %scaled = migraphx.mul %1, %scale : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %biased = migraphx.add %1, %arg4 : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %2 = migraphx.add %scaled, %biased : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    return %2 : !migraphx.shaped<1x64x64xf32, 4096x64x1>
  }
}
