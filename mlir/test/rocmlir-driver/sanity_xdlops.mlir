// Sanity test to ensure every step of the XDLOPS lowering process gets valid MLIR
// and LLVM IR.

// fp32 tests.
// RUN: rocmlir-gen --arch gfx908 -p | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=gfx908 | rocmlir-opt

// fp16 tests.
// RUN: rocmlir-gen --arch gfx908 -p -t f16 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p -t f16 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=gfx908 | rocmlir-opt

// bf16 tests.
// RUN: rocmlir-gen --arch gfx908 -p -t bf16 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=gfx908 | rocmlir-opt

// i8 tests
// RUN: rocmlir-gen --arch gfx908 -p -t i8 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p -t i8 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=gfx908 | rocmlir-opt

