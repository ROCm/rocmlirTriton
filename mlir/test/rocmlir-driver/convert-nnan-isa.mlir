// The float-to-int side of `migraphx.convert` reaches the same saturating cast
// as `migraphx.quantizelinear` (both go through createCastOp in MIGraphXToTosa
// and then rock::createClampedFPToInt), so it collapses to a single v_med3 in
// exactly the same way. quantize-nnan-isa.mlir carries the full explanation;
// this file pins the second entry point, and does it on an unsigned destination
// to cover the `unsigned_cast` spelling and its clamped fptoui alongside the
// signed `fp_to_int_cast`.
//
// Only one kernel lives in this file, since every kernel in the module lands in
// the same ISA dump; the IEEE contrast below is the same kernel compiled with
// `-disable-fast-math`. The target is pinned for the reason given there.

// RUN: rocmlir-gen --clone-harness -arch gfx1100 -fut convert_nnan %s \
// RUN: | rocmlir-driver -arch=gfx1100 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1100 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t 2>&1
// RUN: FileCheck %s < %t
// Neither the sanitization nor a NaN-propagating clamp is left behind. Keep this
// as its own invocation: implicit negative checks do not apply to the run above.
// RUN: FileCheck /dev/null --implicit-check-not=v_cmp_o_f32 \
// RUN:   --implicit-check-not=v_cndmask_b32 --implicit-check-not=v_max_f32 \
// RUN:   --implicit-check-not=v_min_f32 < %t

// CHECK: v_med3_f32

// Under -disable-fast-math the NaN -> 0 sanitization stays, so the compare and
// select come back, and the clamp does not fold: nothing hands out the `nnan`
// that earns the med3, so none may appear here.
// RUN: rocmlir-gen --clone-harness -arch gfx1100 -fut convert_nnan %s \
// RUN: | rocmlir-driver -disable-fast-math -arch=gfx1100 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -disable-fast-math -arch=gfx1100 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.ieee 2>&1
// RUN: FileCheck %s --check-prefix=IEEE < %t.ieee
// RUN: FileCheck /dev/null --implicit-check-not=v_med3_f32 < %t.ieee

// IEEE-DAG: v_cmp_o_f32
// IEEE-DAG: v_cndmask_b32
// IEEE-DAG: v_minmax_f32

module {
  func.func @convert_nnan(%a: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                          %b: !migraphx.shaped<1x256x256xf32, 65536x256x1>)
      -> !migraphx.shaped<1x256x256xui8, 65536x256x1>
      attributes {rock.kernel} {
    %d = migraphx.dot %a, %b : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %c = migraphx.convert %d : <1x256x256xf32, 65536x256x1> to <1x256x256xui8, 65536x256x1>
    return %c : !migraphx.shaped<1x256x256xui8, 65536x256x1>
  }
}
