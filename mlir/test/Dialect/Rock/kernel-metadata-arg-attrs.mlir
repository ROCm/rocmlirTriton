// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-opt -triton-to-hsaco='arch=gfx1100' %s > %t.log 2>&1 && \
// RUN:   FileCheck %s --implicit-check-not="Unhandled parameter attribute" < %t.log

// Triton annotates kernel parameters with metadata that has no LLVM IR
// counterpart, so translation to LLVM IR drops it. The interface registered by
// registerKernelMetadataDialectTranslation() accepts those attributes; without
// it, LLVMTranslationInterface warns once per attribute and each warning
// prints the whole kernel, which dominates the translation.
//
// The warning check is --implicit-check-not rather than a CHECK-NOT so that it
// spans the whole log. %t.log holds both streams, and a CHECK-NOT placed before
// the first positive CHECK would only cover the text ahead of it - here just
// the module line, since stdout is flushed after the warnings on stderr.
//
// `rock.prefill` is not covered here on purpose: it never reaches this point.
// RockTensorToTritonPtrPass keeps it off the tt.func once it has recorded it as
// a module attribute (see rock-tensor-to-triton-ptr.mlir), and `rock` is not
// registered for translation, so a leak would warn instead of being accepted.

// The attributes are accepted, not deleted: the MLIR kernel still carries them
// after translation.
// CHECK: llvm.func amdgpu_kernelcc @kernel
// CHECK-SAME: tt.divisibility = 16 : i32
// CHECK-SAME: tt.pointee_type = f32
// CHECK-SAME: tt.pointer_range = 32 : i32

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr<1> {llvm.noalias,
                                    tt.divisibility = 16 : i32,
                                    tt.pointee_type = f32,
                                    tt.pointer_range = 32 : i32}) {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    llvm.store %0, %arg0 : f32, !llvm.ptr<1>
    llvm.return
  }
}
