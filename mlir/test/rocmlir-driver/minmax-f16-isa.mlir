// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The f16 half of minmax-isa.mlir: same clip-then-max kernel shape and the same
// `-disable-fast-math` requirement, see that file for why those two things are
// what put the ops on the IEEE-754-2019 NaN-propagating min/max forms. What f16
// adds is the packed forms, which is the whole reason for a second dtype.
//
// This is a separate file rather than a second kernel in minmax-isa.mlir because
// the Triton half of the kernel pipeline is parameterized per module
// (`ttg.num-warps` sets every kernel's block size, `ttg.shared` and
// `ttg.global_scratch_memory_size` are single module-wide values), so a module
// carrying more than one kernel is out of contract and aborts Triton's
// global-scratch allocation.

// CDNA3 has no IEEE-2019 min/max in any form, packed or not. LLVM lowers each
// operation to a legacy non-propagating min/max followed by compare/select
// fixups that preserve the required NaN behavior.
// Keep absence checks in a separate FileCheck invocation: implicit negative
// checks do not cover the interior of a single CHECK-DAG group.
// RUN: rocmlir-gen --clone-harness -arch gfx942 -fut mlir_minmax_f16 %s \
// RUN: | rocmlir-driver -disable-fast-math -arch=gfx942 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -disable-fast-math -arch=gfx942 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx942 2>&1
// RUN: FileCheck %s --check-prefix=GFX942 < %t.gfx942
// RUN: FileCheck /dev/null \
// RUN:   --implicit-check-not=v_maximum --implicit-check-not=v_pk_maximum \
// RUN:   --implicit-check-not=v_minimum --implicit-check-not=v_pk_minimum \
// RUN:   < %t.gfx942

// GFX942-DAG: v_max_f16
// GFX942-DAG: v_min_f16
// GFX942-DAG: v_cmp_o_f16
// GFX942-DAG: v_cndmask_b32

// On CDNA4 the family exists only in its three-operand minimum3/maximum3 form,
// packed included, so there is no two-operand pair for the backend to fuse and
// each op selects its own instruction with a duplicated operand.
// RUN: rocmlir-gen --clone-harness -arch gfx950 -fut mlir_minmax_f16 %s \
// RUN: | rocmlir-driver -disable-fast-math -arch=gfx950 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -disable-fast-math -arch=gfx950 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx950 2>&1
// RUN: FileCheck %s --check-prefix=GFX950 < %t.gfx950
// RUN: FileCheck /dev/null --implicit-check-not=v_maximumminimum \
// RUN:   --implicit-check-not=v_minimummaximum < %t.gfx950

// GFX950-DAG: v_pk_maximum3_f16
// GFX950-DAG: v_pk_minimum3_f16

// gfx1170 takes the packed two-operand form, and nothing falls back to a
// compare/select.
// Pin the GEMM shape so the whole-kernel absence check does not depend on
// unrelated address-selection instructions from quick-tuning list changes.
// RUN: rocmlir-gen --clone-harness -arch gfx1170 -fut mlir_minmax_f16 %s \
// RUN: | rocmlir-driver -disable-fast-math -arch=gfx1170 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -disable-fast-math -arch=gfx1170 -kernel-pipeline=gpu,triton,binary \
// RUN:   --perf-config="gemm:mPerBlock=128,nPerBlock=256,kPerBlock=64,kpack=1,numCTAs=1,numWaves=8,matrixInstrNonkdim=0,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0" \
// RUN:   -o /dev/null > %t.gfx1170 2>&1
// RUN: FileCheck %s --check-prefix=GFX1170 < %t.gfx1170
// RUN: FileCheck /dev/null --implicit-check-not=v_cndmask < %t.gfx1170

// GFX1170-DAG: v_pk_maximum_f16
// GFX1170-DAG: v_pk_minimum_f16

// gfx1250's packed f16 ops come out in the three-operand form instead.
// RUN: rocmlir-gen --clone-harness -arch gfx1250 -fut mlir_minmax_f16 %s \
// RUN: | rocmlir-driver -disable-fast-math -arch=gfx1250 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -disable-fast-math -arch=gfx1250 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx1250 2>&1
// RUN: FileCheck %s --check-prefix=GFX1250 < %t.gfx1250
// RUN: FileCheck /dev/null --implicit-check-not=v_cndmask < %t.gfx1250

// GFX1250-DAG: v_pk_maximum3_f16
// GFX1250-DAG: v_pk_minimum3_f16

module {
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
