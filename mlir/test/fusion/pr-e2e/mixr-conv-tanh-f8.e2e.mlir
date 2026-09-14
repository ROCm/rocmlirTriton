// Accuracy of the inline tanh approximation in rock-legalize-math-for-triton at
// fp8, against the upstream CPU lowering of the same graph. This is
// mixr-conv-tanh-f16.e2e.mlir with the element type changed.
//
// fp8 is the type that most needs the pass to run early, since
// arith-emulate-unsupported-floats would otherwise round every intermediate of
// the expansion back to three mantissa bits. tanh-f8-isa.mlir pins that
// statically; this runs it. Default tolerances suffice because fp8 is coarse
// enough that the result matches the CPU reference on all but a few elements.

// RUN: rocmlir-gen -fut mlir_convolution_tanh --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -rand_min -1 -rand_max 1 -fut mlir_convolution_tanh --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_convolution_tanh(%arg0: !migraphx.shaped<8x8x16x16xf8E4M3FN, 2048x256x16x1>, %arg1: !migraphx.shaped<16x8x3x3xf8E4M3FN, 72x9x3x1>) -> !migraphx.shaped<8x16x14x14xf8E4M3FN, 3136x196x14x1> attributes {rock.kernel} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <8x8x16x16xf8E4M3FN, 2048x256x16x1>, <16x8x3x3xf8E4M3FN, 72x9x3x1> -> <8x16x14x14xf8E4M3FN, 3136x196x14x1>
    %1 = migraphx.tanh %0 : <8x16x14x14xf8E4M3FN, 3136x196x14x1> -> <8x16x14x14xf8E4M3FN, 3136x196x14x1>
    return %1 : !migraphx.shaped<8x16x14x14xf8E4M3FN, 3136x196x14x1>
  }
}
