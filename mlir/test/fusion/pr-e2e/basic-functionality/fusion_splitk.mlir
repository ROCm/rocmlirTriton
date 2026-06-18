// Fused GEMM test (migraphx IR): C_f32 = extf((A_f16 + input_f16) * (B_f16 + input_f16) + ofusion_f16)
//
// Same fused GEMM as fusion.mlir but with an f32 output (so an extf is added in
// the output epilogue). The "splitk" name in the original refers to the
// `rock.enable_splitk_for_tuning` attribute / perf_config used by the rock
// tuner; the migraphx pipeline auto-tunes the lowered rock kernel, so we do
// not need to (and cannot) pin a perf_config from migraphx IR.

// RUN: rocmlir-gen -fut rock_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rock_gemm --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @rock_gemm(%arg0: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %arg1: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %inputfusion: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %ofusion: !migraphx.shaped<1x100x100xf16, 10000x100x1>)
      -> !migraphx.shaped<1x100x100xf32, 10000x100x1>
      attributes {rock.kernel} {
    %a = migraphx.add %arg0, %inputfusion : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %b = migraphx.add %arg1, %inputfusion : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %gemm = migraphx.dot %a, %b : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %fused = migraphx.add %gemm, %ofusion : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %result = migraphx.convert %fused : <1x100x100xf16, 10000x100x1> to <1x100x100xf32, 10000x100x1>
    return %result : !migraphx.shaped<1x100x100xf32, 10000x100x1>
  }
}
