// RUN: env LLVM_IR_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx942' %s 2>&1 | FileCheck %s
// RUN: env LLVM_IR_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx90a' %s 2>&1 | FileCheck %s
// RUN: env LLVM_IR_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx950' %s 2>&1 | FileCheck %s

// Pin the `amdgpu-prealloc-sgpr-spill-vgprs` attribute stamped by
// setKernelAttributes() (TritonToHsaco.cpp). It contains an AMDGPU
// register-allocator miscompile: once SGPRs are spilled into VGPR lanes and the
// lane holders themselves spill to scratch, the allocator emits a reload of a
// WWM slot that was never stored, so the restored scalars are garbage. The
// attribute makes SIPreAllocateWWMRegs reserve those holders before allocation,
// which keeps the allocator out of that path.
//
// Stamped unconditionally rather than per-arch: the pass only acts on
// SI_SPILL_S32_TO_VGPR, so kernels that never spill SGPRs into lanes are
// unaffected. Read back from the post-optimization LLVM IR dump that
// LLVM_IR_ENABLE_DUMP emits.

// CHECK: attributes #{{[0-9]+}} = {
// CHECK-SAME: "amdgpu-prealloc-sgpr-spill-vgprs"

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.mlir.global internal @lds(#llvm.undef) {addr_space = 3 : i32} : !llvm.array<32768 x i8>

  llvm.func amdgpu_kernelcc @kernel() {
    %0 = llvm.mlir.addressof @lds : !llvm.ptr<3>
    %1 = llvm.mlir.constant(1 : i8) : i8
    llvm.store volatile %1, %0 : i8, !llvm.ptr<3>
    llvm.return
  }
}
