// Fused GEMM test (migraphx IR): C = (A + 0) * (B + 0) + 0
//
// NOTE: When lowered through the migraphx pipeline the `+ 0` adds are folded
// away, so the rock kernel collapses to a plain GEMM. This mirrors what the
// rock fusion passes do on the original kernel (the add-zero is the
// zero-preserving case the original test was probing).

// RUN: rocmlir-gen -fut rock_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rock_gemm --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @rock_gemm(%arg0: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %arg1: !migraphx.shaped<1x100x100xf16, 10000x100x1>)
      -> !migraphx.shaped<1x100x100xf16, 10000x100x1>
      attributes {rock.kernel} {
    %zero = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 0>
    %zero_bcast = migraphx.multibroadcast %zero {out_dyn_dims = [], out_lens = [1, 100, 100]} : <1xf16, 0> -> <1x100x100xf16, 0x0x0>
    %a = migraphx.add %arg0, %zero_bcast : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 0x0x0> -> <1x100x100xf16, 10000x100x1>
    %b = migraphx.add %arg1, %zero_bcast : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 0x0x0> -> <1x100x100xf16, 10000x100x1>
    %gemm = migraphx.dot %a, %b : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %result = migraphx.add %gemm, %zero_bcast : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 0x0x0> -> <1x100x100xf16, 10000x100x1>
    return %result : !migraphx.shaped<1x100x100xf16, 10000x100x1>
  }
}
