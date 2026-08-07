// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt \
// RUN:   -triton-to-hsaco='arch=gfx1200' %s -o %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=__ocml_ \
// RUN:       --implicit-check-not=__ockl_

// The OCML/OCKL implementations are embedded in rockCompiler and linked before
// code generation, so assembly contains lowered operations rather than calls
// to unresolved device functions.
// CHECK: v_exp_f32
// CHECK: s_clz_i32_u32

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
