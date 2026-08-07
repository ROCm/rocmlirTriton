// RUN: not rocmlir-opt -triton-to-hsaco='arch=gfx1200' %s -o %t 2>&1 \
// RUN:   | FileCheck %s

// CHECK: Unresolved AMD device library symbol after linking: __ocml_missing_f16

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func @__ocml_missing_f16(f16) -> f16

  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr) {
    %value = llvm.load %arg0 : !llvm.ptr -> f16
    %result = llvm.call @__ocml_missing_f16(%value) : (f16) -> f16
    llvm.store %result, %arg0 : f16, !llvm.ptr
    llvm.return
  }
}
