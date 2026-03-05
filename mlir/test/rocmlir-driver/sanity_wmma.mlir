// Sanity test to ensure every step of the WMMA lowering process gets valid MLIR
// and LLVM IR.

// fp16 tests.
// RUN: rocmlir-gen --arch gfx1100 -p -t f16 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -p -t f16 | rocmlir-driver -pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -p -t f16 | rocmlir-driver -pipeline=gpu,rocdl --verify-passes --arch=gfx1100 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -p -t f16 | rocmlir-driver -pipeline=gpu,triton,binary --verify-passes --arch=gfx1100 | rocmlir-opt

// i8 tests
// RUN: rocmlir-gen --arch gfx1100 -p -t i8 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -p -t i8 | rocmlir-driver -pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -p -t i8 | rocmlir-driver -pipeline=gpu,rocdl --verify-passes --arch=gfx1100 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -p -t i8 | rocmlir-driver -pipeline=gpu,triton,binary --verify-passes --arch=gfx1100 | rocmlir-opt
