// Fused conv test (migraphx IR): output = conv(filter, input + inputfusion) + ofusion
//
//   N=2, C=8, K=4, H=W=8, filter 3x3, pad=1, stride=1, group=1.
// The rock kernel is a 5D group-conv (G=1) flattened from <2x8x8x8> and
// <4x3x3x8>. The migraphx equivalent uses the standard NCHW/KCRS layouts:
//   input  : <2x8x8x8xf16>  strides 512x64x8x1
//   filter : <4x8x3x3xf16>  strides  72x 9x3x1
//   output : <2x4x8x8xf16>  strides 256x64x8x1

// RUN: rocmlir-gen -fut rock_conv --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rock_conv --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @rock_conv(%filter: !migraphx.shaped<4x8x3x3xf16, 72x9x3x1>,
                       %input: !migraphx.shaped<2x8x8x8xf16, 512x64x8x1>,
                       %inputfusion: !migraphx.shaped<2x8x8x8xf16, 512x64x8x1>,
                       %ofusion: !migraphx.shaped<2x4x8x8xf16, 256x64x8x1>)
      -> !migraphx.shaped<2x4x8x8xf16, 256x64x8x1>
      attributes {rock.kernel} {
    %a = migraphx.add %input, %inputfusion : <2x8x8x8xf16, 512x64x8x1>, <2x8x8x8xf16, 512x64x8x1> -> <2x8x8x8xf16, 512x64x8x1>
    %conv = migraphx.convolution %a, %filter {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <2x8x8x8xf16, 512x64x8x1>, <4x8x3x3xf16, 72x9x3x1> -> <2x4x8x8xf16, 256x64x8x1>
    %result = migraphx.add %conv, %ofusion : <2x4x8x8xf16, 256x64x8x1>, <2x4x8x8xf16, 256x64x8x1> -> <2x4x8x8xf16, 256x64x8x1>
    return %result : !migraphx.shaped<2x4x8x8xf16, 256x64x8x1>
  }
}
