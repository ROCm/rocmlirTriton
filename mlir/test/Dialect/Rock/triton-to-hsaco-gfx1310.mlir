// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx1310' %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa{{.*}}-gfx1310"
// CHECK-NOT: gfx1250
// CHECK-NOT: HW_REG_WAVE_SCHED_MODE
// CHECK-NOT: v_wmma
module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> f32
    %1 = llvm.load %arg1 : !llvm.ptr -> f32
    %2 = llvm.fadd %0, %1 : f32
    llvm.store %2, %arg0 : f32, !llvm.ptr
    llvm.return
  }
}
