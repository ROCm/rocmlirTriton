// Accuracy of the inline tanh approximation in rock-legalize-math-for-triton,
// against the upstream CPU lowering of the same graph.
//
// Two things have to be set for this to measure anything, and it is worth
// nothing without either of them. Both are checked the same way, by perturbing
// the -2*log2(e) scale in the pass by 5% and seeing whether the test notices.
//
// The tolerance has to be given explicitly. The default atol is scaled by the
// convolution's reduction length, which is right for a bare convolution and far
// too loose once tanh has squashed the result into [-1, 1]: it comes out at
// 8e-2, eighty times the f16 spacing at 1.0, and the perturbation passes. At
// the 1e-3 below, one f16 ULP at 1.0, it misses by 14x on 16385 of the 25088
// elements. The approximation's own worst case is 2.4e-4 (see emitApproxTanh),
// a quarter of that atol.
//
// The input range has to be bounded. Leaving -rand at its default, as
// tosa-to-rock-tanh.e2e.mlir does, makes the convolution output large enough
// that all 25088 elements saturate to +1 and not one of them is negative, which
// passes whatever the approximation computes.
//
// The two RUNs differ only in that range:
//   -1..1 leaves 4842 distinct outputs with 2666 saturated, so it measures the
//         approximation where its error is largest.
//   -4..4 saturates 23058 of them, roughly half at each end. The approximation
//         has no sign handling, so -1 is reached by the exponential overflowing
//         to +inf and the reciprocal of that being exactly +0. If that stopped
//         holding, e.g. if `ninf` were added to the flags
//         rock-allow-fast-math-flags attaches, this RUN would see NaN.

// RUN: rocmlir-gen -fut mlir_convolution_tanh --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -rand_min -1 -rand_max 1 -atol 1e-3 -fut mlir_convolution_tanh --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// RUN: rocmlir-gen -fut mlir_convolution_tanh --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -rand_min -4 -rand_max 4 -atol 1e-3 -fut mlir_convolution_tanh --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_convolution_tanh(%arg0: !migraphx.shaped<8x8x16x16xf16, 2048x256x16x1>, %arg1: !migraphx.shaped<16x8x3x3xf16, 72x9x3x1>) -> !migraphx.shaped<8x16x14x14xf16, 3136x196x14x1> attributes {rock.kernel} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <8x8x16x16xf16, 2048x256x16x1>, <16x8x3x3xf16, 72x9x3x1> -> <8x16x14x14xf16, 3136x196x14x1>
    %1 = migraphx.tanh %0 : <8x16x14x14xf16, 3136x196x14x1> -> <8x16x14x14xf16, 3136x196x14x1>
    return %1 : !migraphx.shaped<8x16x14x14xf16, 3136x196x14x1>
  }
}
