// A `migraphx.max` next to a `migraphx.clip` is enough to see every shape the
// IEEE-754-2019 NaN-propagating min/max instructions take. MIGraphXToTosa turns
// the clip into `tosa.clamp` and RockTosaToElementwise expands that into an
// `arith.maximumf`/`arith.minimumf` pair, so the clip exercises the fused
// three-operand form while the standalone max exercises the two-operand one.
// The max has to come after the clip: fed into it instead, the two adjacent
// maxima fuse into one three-operand instruction and the fused max/min pair
// never appears.
//
// The same kernel appears at f32 and at f16 because the two select different
// instructions: the f16 elementwise tail vectorizes, so it reaches the packed
// v_pk_* forms, and there is no packed fused form for a mixed max/min pair to
// collapse into. Both kernels stay in the module no matter which one `-fut`
// names -- that only picks the harness entry point -- so one compile per arch
// covers both.
//
// buildKernelPipeline leaves the min/max ops unexpanded, so Triton's
// MinMaxFOpConversion is what chooses between the hardware instruction and its
// NaN-check emulation. The assembly comes from the `binary` stage's own
// AMDGCN_ENABLE_DUMP, so these are the instructions the shipped kernel holds
// rather than a reconstruction of the backend's flags.
//
// Which way round the fused pair comes out (a minimum of the maximum or the
// other way about) follows from the tuning config rather than the target, so
// the checks below accept either spelling.

// CDNA3 has no IEEE-2019 min/max in any form, packed or not: Triton emits the
// NaN-check emulation and the clamp collapses into the legacy non-propagating
// instructions.
// RUN: rocmlir-gen --clone-harness -arch gfx942 -fut mlir_minmax_f32 %s \
// RUN: | rocmlir-driver -arch=gfx942 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx942 -kernel-pipeline=gpu,triton,binary -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefix=GFX942 \
// RUN:   --implicit-check-not=v_maximum --implicit-check-not=v_pk_maximum \
// RUN:   --implicit-check-not=v_minimum --implicit-check-not=v_pk_minimum

// GFX942-DAG: v_max_f32
// GFX942-DAG: v_med3_f32
// GFX942-DAG: v_pk_max_f16
// GFX942-DAG: v_pk_min_f16

// On CDNA4 the family exists only in its three-operand minimum3/maximum3 form,
// packed included, so there is no two-operand pair for the backend to fuse and
// each op selects its own instruction with a duplicated operand.
// RUN: rocmlir-gen --clone-harness -arch gfx950 -fut mlir_minmax_f32 %s \
// RUN: | rocmlir-driver -arch=gfx950 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx950 -kernel-pipeline=gpu,triton,binary -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefix=GFX950 \
// RUN:   --implicit-check-not=v_maximumminimum --implicit-check-not=v_minimummaximum

// GFX950-DAG: v_maximum3_f32
// GFX950-DAG: v_minimum3_f32
// GFX950-DAG: v_pk_maximum3_f16
// GFX950-DAG: v_pk_minimum3_f16

// gfx1170 has the two-operand form, so the f32 max is a single instruction and
// the clip's pair fuses into one three-operand instruction, while the f16 tail
// takes the packed two-operand form. Nothing falls back to a compare/select.
// RUN: rocmlir-gen --clone-harness -arch gfx1170 -fut mlir_minmax_f32 %s \
// RUN: | rocmlir-driver -arch=gfx1170 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1170 -kernel-pipeline=gpu,triton,binary -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefix=GFX1170 --implicit-check-not=v_cndmask

// GFX1170-DAG: v_maximum_f32
// GFX1170-DAG: v_{{maximumminimum|minimummaximum}}_f32
// GFX1170-DAG: v_pk_maximum_f16
// GFX1170-DAG: v_pk_minimum_f16

// gfx1250 matches gfx1170 at f32, but its packed f16 ops come out in the
// three-operand form.
// RUN: rocmlir-gen --clone-harness -arch gfx1250 -fut mlir_minmax_f32 %s \
// RUN: | rocmlir-driver -arch=gfx1250 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1250 -kernel-pipeline=gpu,triton,binary -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefix=GFX1250 --implicit-check-not=v_cndmask

// GFX1250-DAG: v_maximum_f32
// GFX1250-DAG: v_{{maximumminimum|minimummaximum}}_f32
// GFX1250-DAG: v_pk_maximum3_f16
// GFX1250-DAG: v_pk_minimum3_f16

module {
  func.func @mlir_minmax_f32(%a: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                             %b: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                             %c: !migraphx.shaped<1x256x256xf32, 65536x256x1>)
      -> (!migraphx.shaped<1x256x256xf32, 65536x256x1>) attributes {rock.kernel} {
    %lo = migraphx.literal (dense<0.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %hi = migraphx.literal (dense<6.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %blo = migraphx.multibroadcast %lo {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1xf32, 0> -> <1x256x256xf32, 0x0x0>
    %bhi = migraphx.multibroadcast %hi {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1xf32, 0> -> <1x256x256xf32, 0x0x0>
    %d = migraphx.dot %a, %b : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %clipped = migraphx.clip %d, %blo, %bhi : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 0x0x0>, <1x256x256xf32, 0x0x0> -> <1x256x256xf32, 65536x256x1>
    %m = migraphx.max %clipped, %c : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    return %m : !migraphx.shaped<1x256x256xf32, 65536x256x1>
  }

  func.func @mlir_minmax_f16(%a: !migraphx.shaped<1x256x256xf16, 65536x256x1>,
                             %b: !migraphx.shaped<1x256x256xf16, 65536x256x1>,
                             %c: !migraphx.shaped<1x256x256xf16, 65536x256x1>)
      -> (!migraphx.shaped<1x256x256xf16, 65536x256x1>) attributes {rock.kernel} {
    %lo = migraphx.literal (dense<0.000000e+00> : tensor<1xf16>) : <1xf16, 0>
    %hi = migraphx.literal (dense<6.000000e+00> : tensor<1xf16>) : <1xf16, 0>
    %blo = migraphx.multibroadcast %lo {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1xf16, 0> -> <1x256x256xf16, 0x0x0>
    %bhi = migraphx.multibroadcast %hi {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1xf16, 0> -> <1x256x256xf16, 0x0x0>
    %d = migraphx.dot %a, %b : <1x256x256xf16, 65536x256x1>, <1x256x256xf16, 65536x256x1> -> <1x256x256xf16, 65536x256x1>
    %clipped = migraphx.clip %d, %blo, %bhi : <1x256x256xf16, 65536x256x1>, <1x256x256xf16, 0x0x0>, <1x256x256xf16, 0x0x0> -> <1x256x256xf16, 65536x256x1>
    %m = migraphx.max %clipped, %c : <1x256x256xf16, 65536x256x1>, <1x256x256xf16, 65536x256x1> -> <1x256x256xf16, 65536x256x1>
    return %m : !migraphx.shaped<1x256x256xf16, 65536x256x1>
  }
}
