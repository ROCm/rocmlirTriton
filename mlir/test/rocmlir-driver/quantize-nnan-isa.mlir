// A `migraphx.quantizelinear` reduces its saturating float-to-int cast to a
// single v_med3.
//
// rock::createClampedFPToInt emits a NaN -> 0 sanitization (cmpf uno + select,
// so that arith.fptosi never sees a NaN and returns poison) ahead of the clamp
// into the integer range. The kernel path's default no-NaN assumption makes that
// prologue dead, leaving a bare `nnan` minnumf/maxnumf pair that RockToTTIR
// folds into `tt.clampf`.
//
// Only one kernel lives in this file, since every kernel in the module lands in
// the same ISA dump; the IEEE contrast below is the same kernel compiled with
// `-disable-fast-math`.
//
// The target is pinned rather than taken from %arch, as in minmax-isa.mlir: the
// mnemonic is target-specific (gfx12 renames the whole family, spelling this one
// v_med3_num_f32), so letting the host GPU pick would make the checks depend on
// the machine the suite runs on. See clip-nnan-isa.mlir for how much the fold
// is worth per target; it is a large win here on gfx1100 but a wash on gfx1250,
// so the med3 is the shape of the lowering rather than a fixed saving.

// RUN: rocmlir-gen --clone-harness -arch gfx1100 -fut quantize_nnan %s \
// RUN: | rocmlir-driver -arch=gfx1100 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1100 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t 2>&1
// RUN: FileCheck %s < %t
// Neither the sanitization nor a NaN-propagating clamp is left behind. Keep this
// as its own invocation: implicit negative checks do not apply to the run above.
// RUN: FileCheck /dev/null --implicit-check-not=v_cmp_o_f32 \
// RUN:   --implicit-check-not=v_cndmask_b32 --implicit-check-not=v_max_f32 \
// RUN:   --implicit-check-not=v_min_f32 < %t

// CHECK: v_med3_f32

// Under -disable-fast-math the sanitization stays, so the compare and select
// come back, and the clamp does not fold: only RockAllowFastMathFlags marking
// minnumf/maxnumf `nnan` earns the med3, and that pass does not run at all
// here, so none may appear. The clamp itself still takes the cheap
// non-propagating form, here the fused v_minmax, which is what makes the med3 a
// fold of the whole idiom rather than a repair of a NaN-propagating one.
// RUN: rocmlir-gen --clone-harness -arch gfx1100 -fut quantize_nnan %s \
// RUN: | rocmlir-driver -disable-fast-math -arch=gfx1100 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -disable-fast-math -arch=gfx1100 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.ieee 2>&1
// RUN: FileCheck %s --check-prefix=IEEE < %t.ieee
// RUN: FileCheck /dev/null --implicit-check-not=v_med3_f32 < %t.ieee

// IEEE-DAG: v_cmp_o_f32
// IEEE-DAG: v_cndmask_b32
// IEEE-DAG: v_minmax_f32

module {
  func.func @quantize_nnan(%a: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                           %b: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                           %scale: !migraphx.shaped<1x1x256xf32, 256x256x1>)
      -> !migraphx.shaped<1x256x256xi8, 65536x256x1>
      attributes {rock.kernel} {
    %d = migraphx.dot %a, %b : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %q = migraphx.quantizelinear %d, %scale : <1x256x256xf32, 65536x256x1>, <1x1x256xf32, 256x256x1> -> <1x256x256xi8, 65536x256x1>
    return %q : !migraphx.shaped<1x256x256xi8, 65536x256x1>
  }
}
