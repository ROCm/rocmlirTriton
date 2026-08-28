// Static ISA checks for the f32 tanh lowering in rock-legalize-math-for-triton.
// Like mlir/test/fusion/pr-e2e/tosa-to-rock-tanh.e2e.mlir, but this one does
// not need a GPU. The narrow types, which are the ones that get an inline
// approximation, are in tanh-f16-isa.mlir and tanh-bf16-isa.mlir.
//
// Two things are pinned here. On gfx1250 tanh must reach the native v_tanh_f32
// instruction, which it does by going out as __ocml_tanh_f32 for the AMD
// backend to rewrite. Everywhere else an f32 tanh keeps the math dialect's
// expansion, and that expansion has to come out as one v_exp_f32 and one
// v_rcp_f32. That reciprocal is the fragile part: it depends on the `arcp` that
// rock-allow-fast-math-flags attaches reaching the divide, and without it the
// backend emits the full IEEE expansion instead
// (v_div_scale/v_div_fmas/v_div_fixup). That is a large silent slowdown which
// shows up nowhere but the ISA, hence this test. Under --disable-fast-math
// there is no expansion at all and every type takes the OCML call, which is the
// point of the flag rather than a regression, so it is not covered here.
//
// Which of the expansion and the approximation produced the sequence is not
// something the ISA distinguishes, since both are an exp and a reciprocal.
// lowering_rock_legalize_math_for_triton.mlir is what pins that.

// RUN: rocmlir-gen --clone-harness -arch gfx942 -fut mlir_convolution_tanh %s \
// RUN: | rocmlir-driver -arch=gfx942 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx942 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx942 2>&1
// RUN: FileCheck %s --check-prefix=EXPAND < %t.gfx942
// RUN: FileCheck /dev/null --implicit-check-not=v_div_scale --implicit-check-not=v_div_fmas \
// RUN:   --implicit-check-not=v_div_fixup --implicit-check-not=v_tanh < %t.gfx942

// RUN: rocmlir-gen --clone-harness -arch gfx950 -fut mlir_convolution_tanh %s \
// RUN: | rocmlir-driver -arch=gfx950 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx950 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx950 2>&1
// RUN: FileCheck %s --check-prefix=EXPAND < %t.gfx950
// RUN: FileCheck /dev/null --implicit-check-not=v_div_scale --implicit-check-not=v_div_fmas \
// RUN:   --implicit-check-not=v_div_fixup --implicit-check-not=v_tanh < %t.gfx950

// RUN: rocmlir-gen --clone-harness -arch gfx1100 -fut mlir_convolution_tanh %s \
// RUN: | rocmlir-driver -arch=gfx1100 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1100 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx1100 2>&1
// RUN: FileCheck %s --check-prefix=EXPAND < %t.gfx1100
// RUN: FileCheck /dev/null --implicit-check-not=v_div_scale --implicit-check-not=v_div_fmas \
// RUN:   --implicit-check-not=v_div_fixup --implicit-check-not=v_tanh < %t.gfx1100

// RUN: rocmlir-gen --clone-harness -arch gfx1200 -fut mlir_convolution_tanh %s \
// RUN: | rocmlir-driver -arch=gfx1200 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1200 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx1200 2>&1
// RUN: FileCheck %s --check-prefix=EXPAND < %t.gfx1200
// RUN: FileCheck /dev/null --implicit-check-not=v_div_scale --implicit-check-not=v_div_fmas \
// RUN:   --implicit-check-not=v_div_fixup --implicit-check-not=v_tanh < %t.gfx1200

// RUN: rocmlir-gen --clone-harness -arch gfx1250 -fut mlir_convolution_tanh %s \
// RUN: | rocmlir-driver -arch=gfx1250 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1250 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx1250 2>&1
// RUN: FileCheck %s --check-prefix=NATIVE < %t.gfx1250
// RUN: FileCheck /dev/null --implicit-check-not=v_exp_f32 --implicit-check-not=v_rcp_f32 \
// RUN:   --implicit-check-not=v_div_scale --implicit-check-not=v_div_fixup < %t.gfx1250

// Only the exp and the reciprocal are matched, because the rest is not portable
// across the targets above: CDNA packs arithmetic two-wide (v_pk_mul_f32,
// v_pk_fma_f32) while RDNA does not, and the elements per thread follow the
// tiling. The implicit-check-not lines on each RUN are what pin the
// instructions that must be absent.
// EXPAND-DAG: v_exp_f32
// EXPAND-DAG: v_rcp_f32

// NATIVE-DAG: v_tanh_f32

module {
  func.func @mlir_convolution_tanh(%arg0: !migraphx.shaped<1x8x4x4xf32, 128x16x4x1>, %arg1: !migraphx.shaped<8x8x3x3xf32, 72x9x3x1>) -> !migraphx.shaped<1x8x4x4xf32, 128x16x4x1> attributes {rock.kernel} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf32, 128x16x4x1>, <8x8x3x3xf32, 72x9x3x1> -> <1x8x4x4xf32, 128x16x4x1>
    %1 = migraphx.tanh %0 : <1x8x4x4xf32, 128x16x4x1> -> <1x8x4x4xf32, 128x16x4x1>
    return %1 : !migraphx.shaped<1x8x4x4xf32, 128x16x4x1>
  }
}
