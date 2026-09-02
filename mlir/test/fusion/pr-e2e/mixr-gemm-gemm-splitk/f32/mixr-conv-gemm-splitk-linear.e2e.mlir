// RUN: rocmlir-gen -fut conv_gemm_splitk_linear --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut conv_gemm_splitk_linear --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// CHECK: [1 1 1]

// RUN: rocmlir-gen -fut conv_gemm_splitk_linear --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=SPLITK
// SPLITK: splitKFactor = 4
module {
  func.func @conv_gemm_splitk_linear(%arg0: !migraphx.shaped<2x8x8x16xf32, 1024x8x1x64>, %arg1: !migraphx.shaped<16x16x3x3xf32, 144x1x48x16>, %arg2: !migraphx.shaped<1x16x32xf32, 0x1x0>, %arg3: !migraphx.shaped<1x128x32xf32, 4096x32x1>, %arg4: !migraphx.shaped<1x128x32xf32, 4096x32x1>) -> !migraphx.shaped<1x128x32xf32, 4096x32x1> attributes {rock.kernel} {
    %transposed = migraphx.transpose %arg0 {permutation = [0, 3, 1, 2]} : <2x8x8x16xf32, 1024x8x1x64> -> <2x16x8x8xf32, 1024x64x8x1>
    %1 = migraphx.convolution %transposed, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <2x16x8x8xf32, 1024x64x8x1>, <16x16x3x3xf32, 144x1x48x16> -> <2x16x8x8xf32, 2048x1x128x16>
    %2 = migraphx.transpose %1 {permutation = [0, 2, 3, 1]} : <2x16x8x8xf32, 2048x1x128x16> -> <2x8x8x16xf32, 2048x128x16x1>
    %3 = migraphx.reshape %2 {dims = [1, 128, 16]} : <2x8x8x16xf32, 2048x128x16x1> -> <1x128x16xf32, 2048x16x1>
    %4 = migraphx.dot %3, %arg2 {perf_config="attn:mPerBlockG0=128,nPerBlockG0=64,kPerBlock=32,numWaves=4,matrixInstrNonkdim=0,splitKFactor=4"} : <1x128x16xf32, 2048x16x1>, <1x16x32xf32, 0x1x0> -> <1x128x32xf32, 4096x32x1>
    %scaled = migraphx.mul %4, %arg3 : <1x128x32xf32, 4096x32x1>, <1x128x32xf32, 4096x32x1> -> <1x128x32xf32, 4096x32x1>
    %5 = migraphx.add %scaled, %arg4 : <1x128x32xf32, 4096x32x1>, <1x128x32xf32, 4096x32x1> -> <1x128x32xf32, 4096x32x1>
    return %5 : !migraphx.shaped<1x128x32xf32, 4096x32x1>
  }
}
