// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Sanity test to ensure every step of the XDLOPS lowering process gets valid MLIR
// and LLVM IR.

// fp32 tests.
// RUN: rocmlir-gen --arch gfx90a -p | rocmlir-opt
// RUN: rocmlir-gen --arch gfx90a -p | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch gfx90a -p | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=gfx90a | rocmlir-opt

// fp16 tests.
// RUN: rocmlir-gen --arch gfx90a -p -t f16 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx90a -p -t f16 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch gfx90a -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=gfx90a | rocmlir-opt

// bf16 tests.
// RUN: rocmlir-gen --arch gfx90a -p -t bf16 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx90a -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch gfx90a -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=gfx90a | rocmlir-opt

// i8 tests
// RUN: rocmlir-gen --arch gfx90a -p -t i8 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx90a -p -t i8 | rocmlir-driver -kernel-pipeline=gpu | rocmlir-opt
// RUN: rocmlir-gen --arch gfx90a -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --arch=gfx90a | rocmlir-opt

