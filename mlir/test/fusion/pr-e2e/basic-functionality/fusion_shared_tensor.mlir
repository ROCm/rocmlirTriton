// Fused GEMM test (migraphx IR): C = (A + shared) * B + shared
//
// Tests reusing the same operand for both the input fusion on A and the
// output fusion on the GEMM result.

// RUN: rocmlir-gen -fut rock_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rock_gemm --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @rock_gemm(%arg0: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %arg1: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %shared: !migraphx.shaped<1x100x100xf16, 10000x100x1>)
      -> !migraphx.shaped<1x100x100xf16, 10000x100x1>
      attributes {rock.kernel} {
    %a = migraphx.add %arg0, %shared : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %gemm = migraphx.dot %a, %arg1 : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %result = migraphx.add %gemm, %shared : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    return %result : !migraphx.shaped<1x100x100xf16, 10000x100x1>
  }
}
