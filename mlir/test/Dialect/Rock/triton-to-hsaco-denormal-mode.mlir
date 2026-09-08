// RUN: env LLVM_IR_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx942' %s 2>&1 | FileCheck %s --check-prefix=FLUSH-IR
// RUN: env LLVM_IR_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx942 allow-flush-denorm=false' %s 2>&1 | FileCheck %s --check-prefix=IEEE-IR

// setKernelAttributes() must express the denormal mode with the
// `denormal_fpenv` enum attribute. The legacy "denormal-fp-math-f32" string
// spelling is only auto-upgraded when a module is parsed, so a module built in
// memory would silently keep the IEEE default and leave f32 denormals unflushed.

// FLUSH-IR: attributes #{{[0-9]+}} = {
// FLUSH-IR-SAME: denormal_fpenv(float: preservesign)

// IEEE-IR-NOT: denormal_fpenv(float: preservesign)

// Carrying the attribute is only half of it: the point is that the backend
// reads it and programs the mode register accordingly. `.amdhsa_float_denorm_mode_32`
// is what the hardware actually runs with -- 0 flushes both f32 denormal inputs
// and outputs to signed zero, and 3 keeps both. The f16/f64 field beside it is a
// separate knob this option does not touch, which is why the checks name the
// _32 one.
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx942' %s 2>&1 | FileCheck %s --check-prefix=FLUSH-ASM
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx942 allow-flush-denorm=false' %s 2>&1 | FileCheck %s --check-prefix=IEEE-ASM

// FLUSH-ASM: .amdhsa_float_denorm_mode_32 0
// IEEE-ASM: .amdhsa_float_denorm_mode_32 3

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.mlir.global internal @lds(#llvm.undef) {addr_space = 3 : i32} : !llvm.array<32768 x i8>

  llvm.func amdgpu_kernelcc @kernel() {
    %0 = llvm.mlir.addressof @lds : !llvm.ptr<3>
    %1 = llvm.mlir.constant(1 : i8) : i8
    llvm.store volatile %1, %0 : i8, !llvm.ptr<3>
    llvm.return
  }
}
