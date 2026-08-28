// Case 1: gfx1170 (RDNA3.5 + OCP fp8 HW) -> disableTrue16 -> fake16 (plain VGPRs).
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx1170' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FAKE16
// Case 2: gfx1100 (other gfx11) -> disable does NOT apply -> true16 (.l/.h subregs).
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx1100' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TRUE16
// Case 3: gfx12 (RDNA4) -> disable does NOT apply -> true16 (.l/.h subregs).
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx1200' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TRUE16

// Verify the gfx117x true16 workaround in translateTritonToHsaco(): only
// gfx117* kernels get `-real-true16` (LCOMPILER-2609: packed OCP fp8 upcast
// ISel is fake16-only). Other gfx11 and gfx12 compile in real-true16 mode,
// matching upstream after triton-lang/triton#11188. That feature is not
// reflected in the LLVM IR, so we inspect the AMDGCN assembly dump
// (AMDGCN_ENABLE_DUMP). On gfx1170 the f16 ops use full 32-bit VGPRs (fake16);
// on gfx1100/gfx1200 the backend emits true16 `.l`/`.h` subregisters.

// FAKE16: v_add_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
// TRUE16: v_add_f16_e32 v{{[0-9]+}}.{{[lh]}}, v{{[0-9]+}}.{{[lh]}}, v{{[0-9]+}}.{{[lh]}}
module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> f16
    %1 = llvm.load %arg2 : !llvm.ptr -> f16
    %2 = llvm.fadd %0, %1 : f16
    %3 = llvm.fmul %2, %0 : f16
    llvm.store %3, %arg1 : f16, !llvm.ptr
    llvm.return
  }
}
