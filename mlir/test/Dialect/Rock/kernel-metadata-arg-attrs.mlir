// RUN: rocmlir-opt -triton-to-hsaco='arch=gfx1100' %s > %t.log 2>&1 && FileCheck %s < %t.log

// Rock and Triton annotate kernel parameters with metadata that has no LLVM IR
// counterpart, so translation to LLVM IR drops it. The interface registered by
// registerKernelMetadataDialectTranslation() accepts those attributes; without
// it, LLVMTranslationInterface warns once per attribute and each warning
// prints the whole kernel, which dominates the translation.

// CHECK-NOT: Unhandled parameter attribute

// The attributes are accepted, not deleted: the MLIR kernel still carries them
// after translation.
// CHECK: llvm.func amdgpu_kernelcc @kernel
// CHECK-SAME: rock.prefill = 0.000000e+00 : f32
// CHECK-SAME: tt.divisibility = 16 : i32
// CHECK-SAME: tt.pointee_type = f32
// CHECK-SAME: tt.pointer_range = 32 : i32

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr<1> {llvm.noalias,
                                    rock.prefill = 0.000000e+00 : f32,
                                    tt.divisibility = 16 : i32,
                                    tt.pointee_type = f32,
                                    tt.pointer_range = 32 : i32}) {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    llvm.store %0, %arg0 : f32, !llvm.ptr<1>
    llvm.return
  }
}
