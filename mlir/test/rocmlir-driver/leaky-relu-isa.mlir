// Like mlir/test/fusion/pr-e2e/mixr-conv-bias-leaky-relu.mlir, but this one
// does not need a GPU, it is just static checks.

// RUN: rocmlir-gen --clone-harness -arch gfx1101 -fut mlir_convolution_add_leaky_relu %s \
// RUN: | rocmlir-driver -arch=gfx1101 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1101 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx1101 2>&1
// RUN: FileCheck %s --check-prefix=GFX1101 < %t.gfx1101
// RUN: FileCheck /dev/null --implicit-check-not=v_cvt_i16_f16 \
// RUN:   --implicit-check-not=v_cvt_i32_f16 --implicit-check-not=v_cvt_u16_f16 \
// RUN:   --implicit-check-not=v_pk_max_f16 --implicit-check-not=v_pk_min_f16 \
// RUN:   --implicit-check-not=v_max_f16 --implicit-check-not=v_min_f16 \
// RUN:   < %t.gfx1101

// The comparison keeps its f16 operands (with the operands swapped, hence lt)
// and feeds the select directly.
// GFX1101-DAG: v_cmp_lt_f16
// GFX1101-DAG: v_mul_f16
// GFX1101-DAG: v_cndmask_b32

module {
  func.func @mlir_convolution_add_leaky_relu(%arg0: !migraphx.shaped<1x8x4x4xf16, 128x16x4x1>, %arg1: !migraphx.shaped<8x8x3x3xf16, 72x9x3x1>, %arg2: !migraphx.shaped<8xf16, 1>) -> !migraphx.shaped<1x8x4x4xf16, 128x16x4x1> attributes {rock.kernel} {
    %0 = migraphx.literal(dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %1 = migraphx.literal(dense<2.998050e-01> : tensor<1xf16>) : <1xf16, 1>
    %2 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf16, 128x16x4x1>, <8x8x3x3xf16, 72x9x3x1> -> <1x8x4x4xf16, 128x16x4x1>
    %3 = migraphx.broadcast %arg2 {axis = 1 : i64, out_dyn_dims = [], out_lens = [1, 8, 4, 4]} : <8xf16, 1> -> <1x8x4x4xf16, 0x1x0x0>
    %4 = migraphx.add %2, %3 : <1x8x4x4xf16, 128x16x4x1>, <1x8x4x4xf16, 0x1x0x0> -> <1x8x4x4xf16, 128x16x4x1>
    %5 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 8, 4, 4]} : <1xf16, 1> -> <1x8x4x4xf16, 0x0x0x0>
    %6 = migraphx.greater %4, %5 : <1x8x4x4xf16, 128x16x4x1>, <1x8x4x4xf16, 0x0x0x0> -> <1x8x4x4xf16, 128x16x4x1>
    %7 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 8, 4, 4]} : <1xf16, 1> -> <1x8x4x4xf16, 0x0x0x0>
    %8 = migraphx.mul %4, %7 : <1x8x4x4xf16, 128x16x4x1>, <1x8x4x4xf16, 0x0x0x0> -> <1x8x4x4xf16, 128x16x4x1>
    %9 = migraphx.convert %6 {target_type = 0 : i64} : <1x8x4x4xf16, 128x16x4x1> to <1x8x4x4xsi8, 128x16x4x1>
    %10 = migraphx.where %9, %4, %8 : <1x8x4x4xsi8, 128x16x4x1>, <1x8x4x4xf16, 128x16x4x1>, <1x8x4x4xf16, 128x16x4x1> -> <1x8x4x4xf16, 128x16x4x1>
    return %10 : !migraphx.shaped<1x8x4x4xf16, 128x16x4x1>
  }
}
