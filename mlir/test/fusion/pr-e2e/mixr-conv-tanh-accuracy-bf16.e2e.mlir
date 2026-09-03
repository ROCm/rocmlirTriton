// Accuracy of the inline tanh approximation in rock-legalize-math-for-triton at
// bf16, against the upstream CPU lowering of the same graph. This is
// mixr-conv-tanh-accuracy.e2e.mlir with the element type changed; see it for
// why the input range is bounded and why the tolerance is set by hand.
//
// bf16 is worth its own run because it reaches the pass by a different route:
// f16 is widened by the approximation itself, whereas bf16 would already have
// been promoted by math-extend-to-supported-types if the pass were not ordered
// ahead of it, and would then be indistinguishable from an f32 the user asked
// for.
//
// Both tolerances have to be given here. atol is one bf16 ULP at 1.0, and the
// 1.6e-2 rtol bf16 defaults to would on its own be twice the error the test is
// looking for. Together they come to 1e-2 at the top of the range, against a
// worst case of 1.9e-3 for the approximation and 2.3e-2 for a 5% perturbation
// of its -2*log2(e) scale.

// RUN: rocmlir-gen -fut mlir_convolution_tanh --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -rand_min -1 -rand_max 1 -atol 8e-3 -rtol 2e-3 -fut mlir_convolution_tanh --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// RUN: rocmlir-gen -fut mlir_convolution_tanh --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -rand_min -4 -rand_max 4 -atol 8e-3 -rtol 2e-3 -fut mlir_convolution_tanh --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_convolution_tanh(%arg0: !migraphx.shaped<8x8x16x16xbf16, 2048x256x16x1>, %arg1: !migraphx.shaped<16x8x3x3xbf16, 72x9x3x1>) -> !migraphx.shaped<8x16x14x14xbf16, 3136x196x14x1> attributes {rock.kernel} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <8x8x16x16xbf16, 2048x256x16x1>, <16x8x3x3xbf16, 72x9x3x1> -> <8x16x14x14xbf16, 3136x196x14x1>
    %1 = migraphx.tanh %0 : <8x16x14x14xbf16, 3136x196x14x1> -> <8x16x14x14xbf16, 3136x196x14x1>
    return %1 : !migraphx.shaped<8x16x14x14xbf16, 3136x196x14x1>
  }
}
