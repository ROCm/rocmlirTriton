// Sanity test to ensure every step of the lowering process gets valid MLIR,
// LLVM IR, and AMD GCN ISA.

// fp32 tests.
// RUN: rocmlir-gen --arch %arch -p | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=%arch | rocmlir-opt

// fp16 tests.
// RUN: rocmlir-gen --arch %arch -p -t f16 | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=%arch | rocmlir-opt

// bf16 tests.
// RUN: rocmlir-gen --arch %arch -p -t bf16 | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=%arch | rocmlir-opt

// i8 tests
// RUN: rocmlir-gen --arch %arch -p -t i8 | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=%arch | rocmlir-opt
