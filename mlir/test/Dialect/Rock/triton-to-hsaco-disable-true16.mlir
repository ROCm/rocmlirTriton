// Case 1: fp8 pointer arg present -> `disableTrue16` -> fake16 (plain VGPRs).
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx1100' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FAKE16
// Case 2: same kernel without the fp8 arg -> true16 left enabled (.l/.h subregs).
// RUN: sed 's/ {tt.pointee_type = f8E4M3FN}//' %s \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx1100' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TRUE16

// Verify the gfx11 (RDNA3) true16 workaround in translateTritonToHsaco(): when a
// kernel argument points to an fp8 element type, `disableTrue16` adds the
// `-real-true16` target feature to the AMDGCN/HSACO codegen TargetMachine. That
// feature is not reflected in the LLVM IR, so we inspect the AMDGCN assembly
// dump (AMDGCN_ENABLE_DUMP). With true16 disabled the f16 ops use full 32-bit
// VGPRs (fake16); otherwise the backend emits true16 `.l`/`.h` subregisters.

// FAKE16: v_add_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
// TRUE16: v_add_f16_e32 v{{[0-9]+}}.{{[lh]}}, v{{[0-9]+}}.{{[lh]}}, v{{[0-9]+}}.{{[lh]}}
module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr {tt.pointee_type = f8E4M3FN}, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> f16
    %1 = llvm.load %arg2 : !llvm.ptr -> f16
    %2 = llvm.fadd %0, %1 : f16
    %3 = llvm.fmul %2, %0 : f16
    llvm.store %3, %arg1 : f16, !llvm.ptr
    llvm.return
  }
}
