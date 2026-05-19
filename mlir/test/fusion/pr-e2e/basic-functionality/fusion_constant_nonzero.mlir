// Fused GEMM test (migraphx IR): C = (A + 1) * (B + 1) + 1
//
// Tests constant fusion handling for the NON-zero-preserving case: the
// literal-1 broadcasts survive in the rock kernel and exercise the
// elementwise input/output fusion paths in the gridwise lowering.

// RUN: rocmlir-gen -fut rock_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rock_gemm --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @rock_gemm(%arg0: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %arg1: !migraphx.shaped<1x100x100xf16, 10000x100x1>)
      -> !migraphx.shaped<1x100x100xf16, 10000x100x1>
      attributes {rock.kernel} {
    %one = migraphx.literal(dense<1.000000e+00> : tensor<1xf16>) : <1xf16, 0>
    %one_bcast = migraphx.multibroadcast %one {out_dyn_dims = [], out_lens = [1, 100, 100]} : <1xf16, 0> -> <1x100x100xf16, 0x0x0>
    %a = migraphx.add %arg0, %one_bcast : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 0x0x0> -> <1x100x100xf16, 10000x100x1>
    %b = migraphx.add %arg1, %one_bcast : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 0x0x0> -> <1x100x100xf16, 10000x100x1>
    %gemm = migraphx.dot %a, %b : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %result = migraphx.add %gemm, %one_bcast : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 0x0x0> -> <1x100x100xf16, 10000x100x1>
    return %result : !migraphx.shaped<1x100x100xf16, 10000x100x1>
  }
}
