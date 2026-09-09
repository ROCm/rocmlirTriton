// Case 1: no asan. AMDGCN_ENABLE_DUMP=0 emits an object directly, =1 takes the
// AMDGCN assembly round trip.
// RUN: rocmlir-opt -triton-to-hsaco='arch=gfx942' %s -o %t.plain.0.mlir
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt -triton-to-hsaco='arch=gfx942' %s \
// RUN:   -o %t.plain.1.mlir 2>/dev/null
// RUN: hsacoDisasm.py -i %t.plain.0.mlir -o %t.plain.0.s
// RUN: hsacoDisasm.py -i %t.plain.1.mlir -o %t.plain.1.s
// RUN: diff -u %t.plain.0.s %t.plain.1.s
// RUN: FileCheck --check-prefix=KERNEL %s < %t.plain.0.s

// Case 2: asan (+xnack), which stays on the assembly round trip either way.
// RUN: rocmlir-opt -triton-to-hsaco='arch=gfx942 features=+xnack' %s \
// RUN:   -o %t.asan.0.mlir
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt \
// RUN:   -triton-to-hsaco='arch=gfx942 features=+xnack' %s -o %t.asan.1.mlir 2>/dev/null
// RUN: hsacoDisasm.py -i %t.asan.0.mlir -o %t.asan.0.s
// RUN: hsacoDisasm.py -i %t.asan.1.mlir -o %t.asan.1.s
// RUN: diff -u %t.asan.0.s %t.asan.1.s
// RUN: FileCheck --check-prefix=KERNEL %s < %t.asan.0.s

// Binary emission has two paths: translateTritonToHsaco() normally asks the
// TargetMachine for an object directly, but falls back to printing AMDGCN
// assembly and re-parsing it with AMDGPUAsmParser when the text is actually
// wanted (AMDGCN_ENABLE_DUMP=1) or under asan, where the assembler carries a
// `+xnack` that `asmFeatures` (and hence `tmAsm`) does not. Whichever path runs,
// the kernel that comes out has to be the same.
//
// The comparison is on disassembly rather than on the ELF, which is not
// byte-comparable: the assembler path leaves extra local symbols behind and
// emits an empty .AMDGPU.csdata section, so .symtab, .strtab and .shstrtab all
// differ while .text does not.

// The disassembly must be real, so that a failure to extract or disassemble
// cannot make the diffs above pass by comparing two empty files.
// KERNEL: s_endpgm

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> f32
    %1 = llvm.fadd %0, %0 : f32
    llvm.store %1, %arg1 : f32, !llvm.ptr
    llvm.return
  }
}
