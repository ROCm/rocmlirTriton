// End-to-end cover for migraphx.pow, which rock-legalize-math-for-triton always
// turns into a `__ocml_pow_f32` call. lowering_rock_legalize_math_for_triton.mlir
// pins the rewrite on hand-written funcs; what this adds is the real migraphx
// pipeline and a run on hardware, which is what checks that the hand-spelled
// symbol resolves against the `ocml.bc` TritonToHsaco links.
//
// -rand_min 0 keeps the base non-negative: pow(x, y) is NaN for negative x and
// fractional y, and the verifier cannot mask a NaN with any tolerance.
// Default tolerances suffice because the exponent lands in [0, 1), where pow
// attenuates rather than magnifies the convolution's error.

// RUN: rocmlir-gen -fut mlir_convolution_pow --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-driver -arch %arch -kernel-pipeline=gpu -mlir-print-ir-after=rock-legalize-math-for-triton -o /dev/null 2>&1 | FileCheck %s --check-prefix=OCML
// RUN: rocmlir-gen -fut mlir_convolution_pow --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -rand_min 0 -rand_max 1 -fut mlir_convolution_pow --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// Both operands go into the call, unlike the exp/log expansion powf used to get.
// OCML: tt.extern_elementwise %{{.*}}, %{{.*}} {{.*}}symbol = "__ocml_pow_f32"
// OCML-NOT: math.powf

module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_convolution_pow(%arg0: !migraphx.shaped<8x8x16x16xf32, 2048x256x16x1>, %arg1: !migraphx.shaped<16x8x3x3xf32, 72x9x3x1>, %arg2: !migraphx.shaped<8x16x14x14xf32, 3136x196x14x1>) -> !migraphx.shaped<8x16x14x14xf32, 3136x196x14x1> attributes {rock.kernel} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <8x8x16x16xf32, 2048x256x16x1>, <16x8x3x3xf32, 72x9x3x1> -> <8x16x14x14xf32, 3136x196x14x1>
    %1 = migraphx.pow %0, %arg2 : <8x16x14x14xf32, 3136x196x14x1>, <8x16x14x14xf32, 3136x196x14x1> -> <8x16x14x14xf32, 3136x196x14x1>
    return %1 : !migraphx.shaped<8x16x14x14xf32, 3136x196x14x1>
  }
}
