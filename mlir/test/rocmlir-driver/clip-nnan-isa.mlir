// A `migraphx.clip` collapses to a single v_med3. MIGraphXToTosa gives the tosa
// maximum/minimum pair `nan_mode = IGNORE`, RockTosaToElementwise turns that
// into `nnan`-flagged minnumf/maxnumf, and RockToTTIR folds the pair into
// `tt.clampf`.
//
// The contrast is minmax-isa.mlir, where the same clip under
// `-disable-fast-math` keeps IEEE-2019 NaN propagation and costs a v_maximum
// plus a fused maximumminimum. Only one kernel lives in this file, since every
// kernel in the module lands in the same ISA dump.
//
// The target is pinned rather than taken from %arch, as in minmax-isa.mlir: the
// mnemonic is target-specific (gfx12 renames the whole family, spelling this one
// v_med3_num_f32), so letting the host GPU pick would make the checks depend on
// the machine the suite runs on.
//
// How much the fold actually saves is target-specific too, so read the med3 as
// the shape of the lowering rather than as a fixed win. It is a large one on
// targets whose propagating min/max needs compare/select repair (gfx942 spends
// roughly six times the instructions on this clamp without it) and a smaller
// one where a propagating three-operand form exists (2x on gfx950). It is a
// wash on gfx1250, which fuses the propagating pair into a single
// v_maximumminimum_f32, and on packed f16 for gfx950/gfx1250, where the
// propagating v_pk_*imum3_f16 covers two elements per instruction while med3
// has no packed form on any target.

// RUN: rocmlir-gen --clone-harness -arch gfx1100 -fut clip_nnan %s \
// RUN: | rocmlir-driver -arch=gfx1100 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1100 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t 2>&1
// RUN: FileCheck %s < %t
// The NaN bookkeeping is gone entirely, not merely reduced. Keep this as its
// own invocation: implicit negative checks do not apply to the run above.
// RUN: FileCheck /dev/null --implicit-check-not=v_cmp_o_f32 \
// RUN:   --implicit-check-not=v_cndmask_b32 --implicit-check-not=v_max_f32 \
// RUN:   --implicit-check-not=v_min_f32 < %t

// CHECK: v_med3_f32

module {
  func.func @clip_nnan(%a: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                       %b: !migraphx.shaped<1x256x256xf32, 65536x256x1>)
      -> !migraphx.shaped<1x256x256xf32, 65536x256x1>
      attributes {rock.kernel} {
    %lo = migraphx.literal (dense<0.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %hi = migraphx.literal (dense<6.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %blo = migraphx.multibroadcast %lo {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1xf32, 0> -> <1x256x256xf32, 0x0x0>
    %bhi = migraphx.multibroadcast %hi {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1xf32, 0> -> <1x256x256xf32, 0x0x0>
    %d = migraphx.dot %a, %b : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %clipped = migraphx.clip %d, %blo, %bhi : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 0x0x0>, <1x256x256xf32, 0x0x0> -> <1x256x256xf32, 65536x256x1>
    return %clipped : !migraphx.shaped<1x256x256xf32, 65536x256x1>
  }
}
