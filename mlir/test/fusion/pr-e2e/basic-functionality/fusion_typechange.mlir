// Fused GEMM test (migraphx IR): C_f32 = extf(gemm(truncf(A_f32), truncf(B_f32)))
//
// Tests type-changing fusions: a f32->f16 convert on each GEMM input and a
// f16->f32 convert on the output.

// RUN: rocmlir-gen -fut rock_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rock_gemm --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @rock_gemm(%arg0: !migraphx.shaped<1x100x100xf32, 10000x100x1>,
                       %arg1: !migraphx.shaped<1x100x100xf32, 10000x100x1>)
      -> !migraphx.shaped<1x100x100xf32, 10000x100x1>
      attributes {rock.kernel} {
    %a = migraphx.convert %arg0 : <1x100x100xf32, 10000x100x1> to <1x100x100xf16, 10000x100x1>
    %b = migraphx.convert %arg1 : <1x100x100xf32, 10000x100x1> to <1x100x100xf16, 10000x100x1>
    %gemm = migraphx.dot %a, %b : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %result = migraphx.convert %gemm : <1x100x100xf16, 10000x100x1> to <1x100x100xf32, 10000x100x1>
    return %result : !migraphx.shaped<1x100x100xf32, 10000x100x1>
  }
}
