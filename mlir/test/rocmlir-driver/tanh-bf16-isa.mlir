// Static ISA checks for the bf16 tanh lowering in rock-legalize-math-for-triton.
// The f32 case is in tanh-isa.mlir and the f16 case in tanh-f16-isa.mlir.
//
// bf16 reaches the pass as bf16, which is the whole reason the pass is ordered
// ahead of math-extend-to-supported-types: that pass would otherwise promote it
// to f32 first, leaving it indistinguishable from an f32 the user asked for and
// so falling onto the expansion tanh-isa.mlir pins. It is widened to f32 here
// too, but by the approximation rather than around a library call, and the
// rounding back to bf16 is what makes the approximation admissible.

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

// The approximation, evaluated at f32. The surrounding v_cmp_o_f32 and
// v_and_b32_sdwa belong to the bf16 rounding, not to the tanh.
// APPROX-DAG: v_exp_f32
// APPROX-DAG: v_rcp_f32

// NATIVE-DAG: v_tanh_f32

module {
  func.func @mlir_convolution_tanh(%arg0: !migraphx.shaped<1x8x4x4xbf16, 128x16x4x1>, %arg1: !migraphx.shaped<8x8x3x3xbf16, 72x9x3x1>) -> !migraphx.shaped<1x8x4x4xbf16, 128x16x4x1> attributes {rock.kernel} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xbf16, 128x16x4x1>, <8x8x3x3xbf16, 72x9x3x1> -> <1x8x4x4xbf16, 128x16x4x1>
    %1 = migraphx.tanh %0 : <1x8x4x4xbf16, 128x16x4x1> -> <1x8x4x4xbf16, 128x16x4x1>
    return %1 : !migraphx.shaped<1x8x4x4xbf16, 128x16x4x1>
  }
}
