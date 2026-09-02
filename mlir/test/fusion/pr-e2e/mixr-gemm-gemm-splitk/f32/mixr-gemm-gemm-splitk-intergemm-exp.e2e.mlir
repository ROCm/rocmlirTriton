// RUN: rocmlir-gen -fut gemm_gemm_splitk_intergemm_exp --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut gemm_gemm_splitk_intergemm_exp --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// CHECK: [1 1 1]

// The only inter-gemm body in this directory that does not preserve zero.
// gemm0's N is 7 and the split factor is 4, so the split pads N up to 8, and
// in that padded column both gemm0's accumulator and the scale read as zero --
// but exp turns them into 1 rather than leaving 0. Those lanes then feed
// gemm1, whose matching padded K row of %arg2 is zero, so the 1s are
// annihilated by the accumulation; on top of that, the scale's load carries a
// Pad mask, so rock-preserve-masked-load-semantics detects the chain as
// non-zero-preserving and selects the lanes back to zero. Both guards have to
// hold for this to match the unfused reference.
// RUN: rocmlir-gen -fut gemm_gemm_splitk_intergemm_exp --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=SPLITK
// SPLITK: splitKFactor = 4
module {
  func.func @gemm_gemm_splitk_intergemm_exp(%arg0: !migraphx.shaped<1x7x3xf32, 21x3x1>, %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1>, %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1>, %arg3: !migraphx.shaped<1x7x7xf32, 49x7x1>) -> (!migraphx.shaped<1x7x3xf32, 21x3x1>) attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 : <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1> -> <1x7x7xf32, 49x7x1>
    %scaled = migraphx.mul %0, %arg3 : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
    %exp = migraphx.exp %scaled : <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
    %1 = migraphx.dot %exp, %arg2 {perf_config="attn:mPerBlockG0=32,nPerBlockG0=32,kPerBlock=32,numWaves=4,matrixInstrNonkdim=0,splitKFactor=4"} : <1x7x7xf32, 49x7x1>, <1x7x3xf32, 21x3x1> -> <1x7x3xf32, 21x3x1>
    return %1 : !migraphx.shaped<1x7x3xf32, 21x3x1>
  }
}
