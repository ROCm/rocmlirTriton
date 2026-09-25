// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// The compile-time guard is off for normal production compilation.
// RUN: rocmlir-opt -triton-to-hsaco='arch=gfx942' %s -o /dev/null

// Tuning callers can opt into rejection. Verify both the diagnostic and the
// marker that makes the failure a non-applicable result rather than a bug.
// RUN: not rocmlir-opt \
// RUN:   -triton-to-hsaco='arch=gfx942 max-live-values-per-block=3' %s \
// RUN:   --mlir-print-ir-after-failure -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: configuration is not applicable: optimized LLVM IR function @kernel has an estimated peak of 4 simultaneously-live SSA values in one basic block (limit 3)
// CHECK: rock.not_applicable

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr<1>,
                                    %arg1: !llvm.ptr<1>,
                                    %out: !llvm.ptr<1>) {
    %0 = llvm.load volatile %arg0 : !llvm.ptr<1> -> f32
    %1 = llvm.load volatile %arg1 : !llvm.ptr<1> -> f32
    %2 = llvm.fadd %0, %1 : f32
    llvm.store volatile %2, %out : f32, !llvm.ptr<1>
    llvm.return
  }
}
