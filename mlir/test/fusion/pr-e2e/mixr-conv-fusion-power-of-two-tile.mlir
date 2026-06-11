// RUN: rocmlir-gen -fut mlir_conv --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_conv --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// A non-power-of-two block tile (mPerBlock/nPerBlock = 80) is requested via the
// convolution's perf_config, so rock-decompose-nonpow2-tiles splits the implicit
// GEMM (gemmM = out_channels = 160, gemmN = batch*out_h*out_w = 160) into a 2x2
// grid of power-of-two sub-tiles. The conv also has input fusion (the activation
// is the sum of two inputs) and output fusion (the result is scaled then
// offset), exercising recursive splitting through both fusion DAGs.

module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_conv(%in0: !migraphx.shaped<1x64x16x10xf32, 10240x160x10x1>,
                       %in1: !migraphx.shaped<1x64x16x10xf32, 10240x160x10x1>,
                       %fil: !migraphx.shaped<160x64x1x1xf32, 64x1x1x1>,
                       %scale: !migraphx.shaped<1x160x16x10xf32, 25600x160x10x1>,
                       %bias: !migraphx.shaped<1x160x16x10xf32, 25600x160x10x1>)
      -> !migraphx.shaped<1x160x16x10xf32, 25600x160x10x1> attributes {rock.kernel} {
    // Input fusion: activation = in0 + in1.
    %in = migraphx.add %in0, %in1 : <1x64x16x10xf32, 10240x160x10x1>, <1x64x16x10xf32, 10240x160x10x1> -> <1x64x16x10xf32, 10240x160x10x1>
    // The convolution drives the non-power-of-two (80x80) block tiling.
    %conv = migraphx.convolution %in, %fil {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1], perf_config = "gemm:v1:80,80,16,1,1,4,16,1,1,1,1"} : <1x64x16x10xf32, 10240x160x10x1>, <160x64x1x1xf32, 64x1x1x1> -> <1x160x16x10xf32, 25600x160x10x1>
    // Output fusion: (conv * scale) + bias.
    %mul = migraphx.mul %conv, %scale : <1x160x16x10xf32, 25600x160x10x1>, <1x160x16x10xf32, 25600x160x10x1> -> <1x160x16x10xf32, 25600x160x10x1>
    %add = migraphx.add %mul, %bias : <1x160x16x10xf32, 25600x160x10x1>, <1x160x16x10xf32, 25600x160x10x1> -> <1x160x16x10xf32, 25600x160x10x1>
    return %add : !migraphx.shaped<1x160x16x10xf32, 25600x160x10x1>
  }
}
