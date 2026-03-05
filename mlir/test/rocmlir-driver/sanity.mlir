// Sanity test to ensure every step of the lowering process gets valid MLIR,
// LLVM IR, and AMD GCN ISA.

// fp32 tests.
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p | rocmlir-opt
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p | rocmlir-driver -kernel-pipeline=gpu,rocdl --arch=%arch | rocmlir-opt
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=%arch | rocmlir-opt

// fp16 tests.
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p -t f16 | rocmlir-opt
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p -t f16 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,rocdl --arch=%arch | rocmlir-opt
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=%arch | rocmlir-opt

// bf16 tests.
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p -t bf16 | rocmlir-opt
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu,rocdl --arch=%arch | rocmlir-opt
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=%arch | rocmlir-opt

// i8 tests
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p -t i8 | rocmlir-opt
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p -t i8 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,rocdl --arch=%arch | rocmlir-opt
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen --arch %arch -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=%arch | rocmlir-opt
