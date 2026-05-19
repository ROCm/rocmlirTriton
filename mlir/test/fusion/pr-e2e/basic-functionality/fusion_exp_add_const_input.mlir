// Fused GEMM test (migraphx IR): C = exp(A + 1.0) * B
// Tests a chain of two NON-zero-preserving input fusions
// (broadcast-add then exp) on the A operand before the GEMM.

// RUN: rocmlir-gen -fut rock_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rock_gemm --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @rock_gemm(%arg0: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %arg1: !migraphx.shaped<1x100x100xf16, 10000x100x1>)
      -> !migraphx.shaped<1x100x100xf16, 10000x100x1>
      attributes {rock.kernel} {
    %one = migraphx.literal(dense<1.000000e+00> : tensor<1xf16>) : <1xf16, 0>
    %one_bcast = migraphx.multibroadcast %one {out_dyn_dims = [], out_lens = [1, 100, 100]} : <1xf16, 0> -> <1x100x100xf16, 0x0x0>
    %a_add = migraphx.add %arg0, %one_bcast : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 0x0x0> -> <1x100x100xf16, 10000x100x1>
    %a = migraphx.exp %a_add : <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %gemm = migraphx.dot %a, %arg1 : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    return %gemm : !migraphx.shaped<1x100x100xf16, 10000x100x1>
  }
}
