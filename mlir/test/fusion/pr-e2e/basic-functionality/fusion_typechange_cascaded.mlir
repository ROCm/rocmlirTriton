// Fused GEMM test (migraphx IR): C_f32 = extf(gemm(A_f16, B_f16) + bias_f16)
//
// Tests a cascaded output fusion: add a f16 bias to the f16 GEMM
// result, then extend the whole thing to f32 before the store.

// RUN: rocmlir-gen -fut rock_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rock_gemm --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @rock_gemm(%arg0: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %arg1: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %bias: !migraphx.shaped<1x100x100xf16, 10000x100x1>)
      -> !migraphx.shaped<1x100x100xf32, 10000x100x1>
      attributes {rock.kernel} {
    %gemm = migraphx.dot %arg0, %arg1 : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %fused = migraphx.add %gemm, %bias : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %result = migraphx.convert %fused : <1x100x100xf16, 10000x100x1> to <1x100x100xf32, 10000x100x1>
    return %result : !migraphx.shaped<1x100x100xf32, 10000x100x1>
  }
}
