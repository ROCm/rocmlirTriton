// RUN: rocmlir-gen -fut mlir_test --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -rand_min -1 -rand_max 1 -fut mlir_test --verifier clone -relDiff_threshold 0.01 -RMS_threshold 0.0001 - | rocmlir-driver -c | rocm-run | FileCheck %s
// RUN: rocmlir-gen -fut mlir_test --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-driver -arch %arch -c -mlir-print-ir-after=rock-remove-redundant-casts 2>&1 | FileCheck %s --check-prefix=NO-EXTF

// CHECK: [1 1 1]

// NO-EXTF-LABEL: func.func @mlir_test(
// NO-EXTF-NOT: arith.extf
// NO-EXTF: return

module {
  func.func @mlir_test(%arg0: !migraphx.shaped<1x16x7x7xf32, 784x49x7x1>,
                       %arg1: !migraphx.shaped<32x16x3x3xf16, 144x9x3x1>,
                       %arg2: !migraphx.shaped<32xf32, 1>)
      -> !migraphx.shaped<1x32x7x7xf32, 1568x49x7x1> attributes {rock.kernel} {
    %in_f16 = migraphx.convert %arg0 {target_type = 1 : i64} : <1x16x7x7xf32, 784x49x7x1> to <1x16x7x7xf16, 784x49x7x1>
    %conv_f16 = migraphx.convolution %in_f16, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x16x7x7xf16, 784x49x7x1>, <32x16x3x3xf16, 144x9x3x1> -> <1x32x7x7xf16, 1568x49x7x1>
    %bias = migraphx.broadcast %arg2 {axis = 1 : i64, out_lens = [1, 32, 7, 7]} : <32xf32, 1> -> <1x32x7x7xf32, 0x1x0x0>
    %conv_f32 = migraphx.convert %conv_f16 {target_type = 2 : i64} : <1x32x7x7xf16, 1568x49x7x1> to <1x32x7x7xf32, 1568x49x7x1>
    %biased = migraphx.add %conv_f32, %bias : <1x32x7x7xf32, 1568x49x7x1>, <1x32x7x7xf32, 0x1x0x0> -> <1x32x7x7xf32, 1568x49x7x1>
    %out = migraphx.relu %biased : <1x32x7x7xf32, 1568x49x7x1> -> <1x32x7x7xf32, 1568x49x7x1>
    return %out : !migraphx.shaped<1x32x7x7xf32, 1568x49x7x1>
  }
}
