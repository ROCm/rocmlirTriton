// Fused GEMM test (migraphx IR): C = (A + input) * (B + input) + ofusion
//
// The migraphx pipeline lowers this to the same fused GEMM kernel
// (A + input) * (B + input) with the ofusion added in the output epilogue.

// RUN: rocmlir-gen -fut rock_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rock_gemm --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @rock_gemm(%arg0: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %arg1: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %inputfusion: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %ofusion: !migraphx.shaped<1x100x100xf16, 10000x100x1>)
      -> !migraphx.shaped<1x100x100xf16, 10000x100x1>
      attributes {rock.kernel} {
    %a = migraphx.add %arg0, %inputfusion : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %b = migraphx.add %arg1, %inputfusion : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %gemm = migraphx.dot %a, %b : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %result = migraphx.add %gemm, %ofusion : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    return %result : !migraphx.shaped<1x100x100xf16, 10000x100x1>
  }
}
