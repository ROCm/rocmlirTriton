// RUN: env LLVM_IR_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx942 llvm-fn-attrs=rocmlir-test-flag,rocmlir-test-attr=enabled,rocmlir-test-empty=,denormal-fp-math-f32=preserve-sign' %s 2>&1 | FileCheck %s

// Verify the debug-only `llvm_fn_attrs` override loop in
// setKernelAttributes() (TritonToHsaco.cpp). The comma-separated list is
// parsed into kernel function attributes that are applied last, so they win
// over the attributes Triton stamps earlier. We inspect the post-optimization
// LLVM IR dump enabled by LLVM_IR_ENABLE_DUMP to read the resulting attribute
// group. 

// CHECK: attributes #{{[0-9]+}} = {
// CHECK-SAME: "denormal-fp-math-f32"="preserve-sign"
// CHECK-SAME: "rocmlir-test-attr"="enabled"
// CHECK-SAME: "rocmlir-test-empty" "rocmlir-test-flag"

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.mlir.global internal @lds(#llvm.undef) {addr_space = 3 : i32} : !llvm.array<32768 x i8>

  llvm.func amdgpu_kernelcc @kernel() {
    %0 = llvm.mlir.addressof @lds : !llvm.ptr<3>
    %1 = llvm.mlir.constant(1 : i8) : i8
    llvm.store volatile %1, %0 : i8, !llvm.ptr<3>
    llvm.return
  }
}
