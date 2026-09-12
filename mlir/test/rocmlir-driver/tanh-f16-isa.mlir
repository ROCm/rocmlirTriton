// Static ISA checks for the f16 tanh lowering in rock-legalize-math-for-triton.
// The f32 case is in tanh-isa.mlir; this file pins the opposite property, that
// an f16 tanh is evaluated at f32 and rounded back rather than kept at f16 on
// the half-rate transcendentals. The v_exp_f16/v_rcp_f16 implicit-check-not
// lines are what enforce that, and they are the point of this test.
//
// Staying at f16 is the obvious implementation and it is worse on both counts:
// 2558 ULP against 1.0 (see emitApproxTanh), and more instructions, because
// nothing has to be widened where the input is an f32 accumulator and the
// narrowing folds into the closing fma. That fold is why no separate conversion
// appears below: gfx942, gfx1100 and gfx1200 emit v_fma_mixlo_f16 and gfx950
// pairs the conversions into v_cvt_pk_f16_f32. Neither form is checked, since
// which one appears is a backend decision.
//
// gfx1250 is the exception. It takes the OCML path for the native instruction,
// and libdevice only declares f32 and f64 entry points, so the operand is
// widened and the result is v_tanh_f32 rather than the native v_tanh_f16.

// RUN: rocmlir-gen --clone-harness -arch gfx942 -fut mlir_convolution_tanh %s \
// RUN: | rocmlir-driver -arch=gfx942 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx942 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx942 2>&1
// RUN: FileCheck %s --check-prefix=APPROX < %t.gfx942
// RUN: FileCheck /dev/null --implicit-check-not=v_exp_f16 --implicit-check-not=v_rcp_f16 \
// RUN:   --implicit-check-not=v_div_scale --implicit-check-not=v_div_fmas \
// RUN:   --implicit-check-not=v_div_fixup --implicit-check-not=v_tanh < %t.gfx942

// RUN: rocmlir-gen --clone-harness -arch gfx950 -fut mlir_convolution_tanh %s \
// RUN: | rocmlir-driver -arch=gfx950 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx950 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx950 2>&1
// RUN: FileCheck %s --check-prefix=APPROX < %t.gfx950
// RUN: FileCheck /dev/null --implicit-check-not=v_exp_f16 --implicit-check-not=v_rcp_f16 \
// RUN:   --implicit-check-not=v_div_scale --implicit-check-not=v_div_fmas \
// RUN:   --implicit-check-not=v_div_fixup --implicit-check-not=v_tanh < %t.gfx950

// RUN: rocmlir-gen --clone-harness -arch gfx1100 -fut mlir_convolution_tanh %s \
// RUN: | rocmlir-driver -arch=gfx1100 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1100 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx1100 2>&1
// RUN: FileCheck %s --check-prefix=APPROX < %t.gfx1100
// RUN: FileCheck /dev/null --implicit-check-not=v_exp_f16 --implicit-check-not=v_rcp_f16 \
// RUN:   --implicit-check-not=v_div_scale --implicit-check-not=v_div_fmas \
// RUN:   --implicit-check-not=v_div_fixup --implicit-check-not=v_tanh < %t.gfx1100

// RUN: rocmlir-gen --clone-harness -arch gfx1200 -fut mlir_convolution_tanh %s \
// RUN: | rocmlir-driver -arch=gfx1200 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1200 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx1200 2>&1
// RUN: FileCheck %s --check-prefix=APPROX < %t.gfx1200
// RUN: FileCheck /dev/null --implicit-check-not=v_exp_f16 --implicit-check-not=v_rcp_f16 \
// RUN:   --implicit-check-not=v_div_scale --implicit-check-not=v_div_fmas \
// RUN:   --implicit-check-not=v_div_fixup --implicit-check-not=v_tanh < %t.gfx1200

// RUN: rocmlir-gen --clone-harness -arch gfx1250 -fut mlir_convolution_tanh %s \
// RUN: | rocmlir-driver -arch=gfx1250 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1250 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx1250 2>&1
// RUN: FileCheck %s --check-prefix=NATIVE < %t.gfx1250
// RUN: FileCheck /dev/null --implicit-check-not=v_exp_f16 --implicit-check-not=v_rcp_f16 \
// RUN:   --implicit-check-not=v_exp_f32 --implicit-check-not=v_rcp_f32 \
// RUN:   --implicit-check-not=v_div_scale --implicit-check-not=v_div_fixup < %t.gfx1250

// Only the exp and the reciprocal identify the precision the sequence runs at,
// so those are what is matched; see tanh-isa.mlir for why the rest is not
// portable to check.
// APPROX-DAG: v_exp_f32
// APPROX-DAG: v_rcp_f32

// NATIVE-DAG: v_tanh_f32

module {
  func.func @mlir_convolution_tanh(%arg0: !migraphx.shaped<1x8x4x4xf16, 128x16x4x1>, %arg1: !migraphx.shaped<8x8x3x3xf16, 72x9x3x1>) -> !migraphx.shaped<1x8x4x4xf16, 128x16x4x1> attributes {rock.kernel} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf16, 128x16x4x1>, <8x8x3x3xf16, 72x9x3x1> -> <1x8x4x4xf16, 128x16x4x1>
    %1 = migraphx.tanh %0 : <1x8x4x4xf16, 128x16x4x1> -> <1x8x4x4xf16, 128x16x4x1>
    return %1 : !migraphx.shaped<1x8x4x4xf16, 128x16x4x1>
  }
}
