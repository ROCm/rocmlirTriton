// RUN: not rocmlir-opt -triton-to-hsaco %s > %t.log 2>&1 && FileCheck %s < %t.log

// Verify that SuppressWarningHandler in TritonToHsaco.cpp does not silence
// LLVM errors. The oversized LDS allocation makes the AMDGPU backend emit a
// resource-limit error through LLVMContext::diagnose.

// CHECK-NOT: failed to meet occupancy target given by 'amdgpu-waves-per-eu'
// CHECK: local memory (131072) exceeds limit
// CHECK-SAME: in function 'kernel'
// CHECK-NOT: failed to meet occupancy target given by 'amdgpu-waves-per-eu'

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.mlir.global internal @lds(#llvm.undef) {addr_space = 3 : i32} : !llvm.array<131072 x i8>

  llvm.func amdgpu_kernelcc @kernel() attributes {passthrough = [["amdgpu-waves-per-eu", "8,8"]]} {
    %0 = llvm.mlir.addressof @lds : !llvm.ptr<3>
    %1 = llvm.mlir.constant(1 : i8) : i8
    llvm.store volatile %1, %0 : i8, !llvm.ptr<3>
    llvm.return
  }
}
