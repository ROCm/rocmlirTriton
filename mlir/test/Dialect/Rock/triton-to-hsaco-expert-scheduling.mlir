// Case 1: gfx1250 -> isExpertSchedulingEnabled() -> expert scheduling mode 2.
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx1250' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EXPERT
// Case 2: gfx1201 is also GFX12 (the hardware supports expert scheduling mode),
// but isExpertSchedulingEnabled() only returns true for gfx1250, so the mode
// must NOT be enabled here. This pins the arch gate, not just the hardware
// capability.
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx1201' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOEXPERT
// Case 3: gfx1250 with the use_expert_scheduling override forced off. Mirrors
// upstream `knobs.amd.use_expert_scheduling = False`: even on the supported
// arch the mode must NOT be enabled. Pins the tri-state override (the default
// -1 keeps it on for gfx1250, as Case 1 checks).
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt \
// RUN:   -triton-to-hsaco='arch=gfx1250 use-expert-scheduling=0' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOEXPERT

// Verify the gfx1250 expert-scheduling enablement in translateTritonToHsaco():
// when isExpertSchedulingEnabled(arch) is true we set the
// `amdgpu-expert-scheduling-mode` LLVM function attributes, which drive
// SIInsertWaitcnts to emit expert scheduling mode 2. That mode is not reflected
// in the source MLIR, so we inspect the AMDGCN assembly dump
// (AMDGCN_ENABLE_DUMP). In expert mode the backend writes the WAVE_SCHED_MODE
// hardware register to 2 on function entry; otherwise no such write is emitted.

// EXPERT: s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE{{.*}}), 2
// NOEXPERT-NOT: HW_REG_WAVE_SCHED_MODE
module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> f32
    %1 = llvm.load %arg1 : !llvm.ptr -> f32
    %2 = llvm.fadd %0, %1 : f32
    llvm.store %2, %arg0 : f32, !llvm.ptr
    llvm.return
  }
}
