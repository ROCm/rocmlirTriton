// Fused GEMM+reduce test (migraphx IR):
//   gemm   = A_f16 * B_f16                       (f16, [1, 100, 100])
//   fused  = extf( gemm + extra1 + extra2 )      (f32, [1, 100, 100])
//   result[1, 1, n] = sum_m fused[1, m, n]        (f32, [1, 1, 100])
//
// The original kernel interleaves an Unmerge{10,10} on N reshape and a
// Merge back to 3D between the two element-wise adds. Those reshapes are
// row-major no-ops on the flat memory, so the migraphx representation is
// just the two adds in 3D space.

// RUN: rocmlir-gen -fut rock_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rock_gemm --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @rock_gemm(%arg_a: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %arg_b: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %extra1: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %extra2: !migraphx.shaped<1x100x100xf16, 10000x100x1>)
      -> !migraphx.shaped<1x1x100xf32, 100x100x1>
      attributes {rock.kernel} {
    %gemm = migraphx.dot %arg_a, %arg_b : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %add1 = migraphx.add %gemm, %extra1 : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %add2 = migraphx.add %add1, %extra2 : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %add2_f32 = migraphx.convert %add2 : <1x100x100xf16, 10000x100x1> to <1x100x100xf32, 10000x100x1>
    %reduced = migraphx.reduce_sum %add2_f32 {axes = [1]} : <1x100x100xf32, 10000x100x1> -> <1x1x100xf32, 100x100x1>
    return %reduced : !migraphx.shaped<1x1x100xf32, 100x100x1>
  }
}
