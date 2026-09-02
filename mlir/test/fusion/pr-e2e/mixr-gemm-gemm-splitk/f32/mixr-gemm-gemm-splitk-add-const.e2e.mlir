// RUN: rocmlir-gen -fut gemm_gemm_splitk_add_const --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut gemm_gemm_splitk_add_const --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// CHECK: [1 1 1]

// A splat bias is divided at compile time, so 10.0 folds to 2.5 rather than
// producing a runtime arith.divf.
// RUN: rocmlir-gen -fut gemm_gemm_splitk_add_const --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-opt --rock-affix-params --rock-fusion-splitk-regularization | FileCheck %s --check-prefix=SPLITK
// SPLITK: splitKFactor = 4
// SPLITK: dense<2.500000e+00>
module {
  func.func @gemm_gemm_splitk_add_const(%arg0: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg1: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg2: !migraphx.shaped<1x64x64xf32, 4096x64x1>) -> (!migraphx.shaped<1x64x64xf32, 4096x64x1>) attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %1 = migraphx.dot %0, %arg2 {perf_config="attn:mPerBlockG0=128,nPerBlockG0=64,kPerBlock=32,numWaves=4,matrixInstrNonkdim=0,splitKFactor=4"} : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %cst = migraphx.literal(dense<1.000000e+01> : tensor<1x64x64xf32>) : <1x64x64xf32, 0x0x0>
    %2 = migraphx.add %1, %cst : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf32, 4096x64x1>
    return %2 : !migraphx.shaped<1x64x64xf32, 4096x64x1>
  }
}
