// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt \
// RUN:   -triton-to-hsaco='arch=gfx1200' %s -o %t 2>&1 \
// RUN:   | FileCheck %s

// The OCML implementation is packaged into rockCompiler and linked before
// code generation, so the generated assembly contains the lowered operation
// rather than a call to an unresolved device function.
// CHECK-NOT: __ocml_exp_f16
// CHECK-NOT: __ockl_clz_u32
// CHECK: v_exp_f32
// CHECK-NOT: __ocml_exp_f16
// CHECK-NOT: __ockl_clz_u32

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func @__ocml_exp_f16(f16) -> f16
  llvm.func @__ockl_clz_u32(i32) -> i32

  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr) {
    %value = llvm.load %arg0 : !llvm.ptr -> f16
    %result = llvm.call @__ocml_exp_f16(%value) : (f16) -> f16
    llvm.store %result, %arg0 : f16, !llvm.ptr
    llvm.return
  }

  llvm.func amdgpu_kernelcc @ockl_kernel(%arg0: !llvm.ptr) {
    %value = llvm.load %arg0 : !llvm.ptr -> i32
    %result = llvm.call @__ockl_clz_u32(%value) : (i32) -> i32
    llvm.store %result, %arg0 : i32, !llvm.ptr
    llvm.return
  }
}
