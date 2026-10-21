// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The MI450 program separates the MI450 A0 and B0 code paths with a new
// compiler target, gfx1250-strict: A0 takes gfx1250-strict, B0 takes gfx1250,
// and a gfx1250-strict code object is not loadable on B0. The two therefore
// have to stay distinguishable all the way down to the emitted code object.
//
// Case 1: the "-strict" suffix survives rocMLIR's arch-string handling, Triton's
// TargetFeatures and LLVM's target selection, and lands in the code object's
// target string. A plain gfx1250 object must not claim to be the strict one.
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx1250-strict' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STRICT
// Case 2: the baseline it has to stay distinguishable from.
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx1250' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONSTRICT

// The vendor/environment spelling in the middle of the triple is an LLVM
// detail that has changed across bumps, so only the arch tail is pinned here.
// STRICT: .amdgcn_target "amdgcn-amd-amdhsa{{.*}}-gfx1250-strict"
// NONSTRICT: .amdgcn_target "amdgcn-amd-amdhsa{{.*}}-gfx1250"
// NONSTRICT-NOT: gfx1250-strict

// Strict is modelled as a modifier on gfx1250 rather than as its own GPU
// family, so it keeps reporting ISAFamily::GFX1250 and every arch gate that
// already fires for gfx1250 has to keep firing. Expert scheduling is such a
// gate: isExpertSchedulingEnabled() tests the arch prefix, so the mode-2 write
// to WAVE_SCHED_MODE on function entry must appear for both spellings. If this
// ever regresses, strict silently loses a gfx1250 codegen feature.
// STRICT: s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE{{.*}}), 2
// NONSTRICT: s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE{{.*}}), 2
module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> f32
    %1 = llvm.load %arg1 : !llvm.ptr -> f32
    %2 = llvm.fadd %0, %1 : f32
    llvm.store %2, %arg0 : f32, !llvm.ptr
    llvm.return
  }
}
