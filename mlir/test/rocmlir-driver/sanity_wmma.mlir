// Sanity test to ensure every step of the WMMA lowering process gets valid MLIR
// and LLVM IR.

// fp16 tests.
// RUN: rocmlir-gen --arch gfx1100 -p -t f16 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -p -t f16 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=gfx1100 | rocmlir-opt

// i8 tests
// RUN: rocmlir-gen --arch gfx1100 -p -t i8 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -p -t i8 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=gfx1100 | rocmlir-opt
