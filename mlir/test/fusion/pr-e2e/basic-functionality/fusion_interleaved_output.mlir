// Fused GEMM test (migraphx IR): C = gemm(A, B) + extra1 + extra2
//
// The original kernel interleaves reshape and add fusions in different
// rank spaces (Unmerge {10,10} on N to 4D, then Merge back to 3D) before
// storing. Since Unmerge/Merge are row-major reshapes, the flat semantics are
// just element-wise C = gemm + extra1 + extra2. The migraphx pipeline
// produces an equivalent kernel using transforms generated from the shaped
// layouts.

// RUN: rocmlir-gen -fut rock_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rock_gemm --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @rock_gemm(%arg0: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %arg1: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %extra1: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %extra2: !migraphx.shaped<1x100x100xf16, 10000x100x1>)
      -> !migraphx.shaped<1x100x100xf16, 10000x100x1>
      attributes {rock.kernel} {
    %gemm = migraphx.dot %arg0, %arg1 : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %add1 = migraphx.add %gemm, %extra1 : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %add2 = migraphx.add %add1, %extra2 : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    return %add2 : !migraphx.shaped<1x100x100xf16, 10000x100x1>
  }
}
