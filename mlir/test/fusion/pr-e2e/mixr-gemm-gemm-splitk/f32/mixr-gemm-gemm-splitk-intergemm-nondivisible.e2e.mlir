// RUN: rocmlir-gen -fut gemm_gemm_splitk_intergemm_nondivisible --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut gemm_gemm_splitk_intergemm_nondivisible --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// CHECK: [1 1 1]

// gemm0's N is 7 and the split factor is 4, so the split pads N up to 8 before
// splitting and the body is evaluated on the padded column too. The bias is
// padded along with gemm0's output space, so that column reads 0 + 0 and the
// body stays zero-preserving; what this pins down is that the pad is applied
// to the elementwise input and to the keys in the same place, since a bias
// that kept its unpadded extent would slide by one column per split.
// RUN: rocmlir-gen -fut gemm_gemm_splitk_intergemm_nondivisible --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=SPLITK
// SPLITK: splitKFactor = 4
module {
  func.func @gemm_gemm_splitk_intergemm_nondivisible(%arg0: !migraphx.shaped<1x7x3xf32, 21x3x1>, %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1>, %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1>, %arg3: !migraphx.shaped<1x7x7xf32, 49x7x1>) -> (!migraphx.shaped<1x7x3xf32, 21x3x1>) attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 : <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1> -> <1x7x7xf32, 49x7x1>
    %biased = migraphx.add %0, %arg3 : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
    %1 = migraphx.dot %biased, %arg2 {perf_config="attn:mPerBlockG0=32,nPerBlockG0=32,kPerBlock=32,numWaves=4,matrixInstrNonkdim=0,splitKFactor=4"} : <1x7x7xf32, 49x7x1>, <1x7x3xf32, 21x3x1> -> <1x7x3xf32, 21x3x1>
    return %1 : !migraphx.shaped<1x7x3xf32, 21x3x1>
  }
}
