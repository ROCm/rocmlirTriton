// RUN: not rocmlir-opt -triton-to-hsaco='arch=gfx1200' %s -o %t 2>&1 \
// RUN:   | FileCheck %s

// CHECK-DAG: Unresolved AMD device library symbol after linking: __ocml_missing_f16
// CHECK-DAG: Unresolved AMD device library symbol after linking: __oclc_missing

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.mlir.global external @__oclc_missing() {addr_space = 4 : i32} : i8
  llvm.func @__ocml_missing_f16(f16) -> f16

  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr) {
    %value = llvm.load %arg0 : !llvm.ptr -> f16
    %result = llvm.call @__ocml_missing_f16(%value) : (f16) -> f16
    %control_ptr = llvm.mlir.addressof @__oclc_missing : !llvm.ptr<4>
    %control = llvm.load %control_ptr : !llvm.ptr<4> -> i8
    llvm.store %control, %arg0 : i8, !llvm.ptr
    llvm.store %result, %arg0 : f16, !llvm.ptr
    llvm.return
  }
}
