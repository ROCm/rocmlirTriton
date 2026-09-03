// Static ISA checks for the fp8 tanh lowering in rock-legalize-math-for-triton.
// The f32 case is in tanh-isa.mlir, f16 in tanh-f16-isa.mlir and bf16 in
// tanh-bf16-isa.mlir.
//
// fp8 is the type that most needs the pass to run early. Nothing here has fp8
// arithmetic, so arith-emulate-unsupported-floats wraps an extf/truncf pair
// around every individual op it is left with. Applied to the expansion that
// means rounding an intermediate to two or three mantissa bits seven times
// over, which is where its 8-16 ULP in the f8 table comes from. Because the
// pass is ordered ahead of both math-extend-to-supported-types and the arith
// emulation, it sees the fp8 tanh whole and emits one widening, the
// approximation at f32, and one rounding back.
//
// The encoding is immaterial to that lowering, which keys off the width, so
// only f8E4M3FNUZ is compiled here; the other three fp8 encodings and f4E2M1FN
// are covered in lowering_rock_legalize_math_for_triton.mlir.

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
// RUN: FileCheck /dev/null --implicit-check-not=v_exp_f32 --implicit-check-not=v_rcp_f32 \
// RUN:   --implicit-check-not=v_exp_f16 --implicit-check-not=v_rcp_f16 \
// RUN:   --implicit-check-not=v_div_scale --implicit-check-not=v_div_fixup < %t.gfx1250

// The approximation, evaluated at f32. The conversions around it belong to the
// fp8 storage format, not to the tanh, and which ones the backend picks is not
// portable to check; see tanh-isa.mlir.
// APPROX-DAG: v_exp_f32
// APPROX-DAG: v_rcp_f32

// NATIVE-DAG: v_tanh_f32

module {
  func.func @mlir_convolution_tanh(%arg0: !migraphx.shaped<1x8x4x4xf8E4M3FNUZ, 128x16x4x1>, %arg1: !migraphx.shaped<8x8x3x3xf8E4M3FNUZ, 72x9x3x1>) -> !migraphx.shaped<1x8x4x4xf8E4M3FNUZ, 128x16x4x1> attributes {rock.kernel} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf8E4M3FNUZ, 128x16x4x1>, <8x8x3x3xf8E4M3FNUZ, 72x9x3x1> -> <1x8x4x4xf8E4M3FNUZ, 128x16x4x1>
    %1 = migraphx.tanh %0 : <1x8x4x4xf8E4M3FNUZ, 128x16x4x1> -> <1x8x4x4xf8E4M3FNUZ, 128x16x4x1>
    return %1 : !migraphx.shaped<1x8x4x4xf8E4M3FNUZ, 128x16x4x1>
  }
}
