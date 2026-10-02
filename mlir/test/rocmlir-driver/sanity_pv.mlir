// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Sanity test to -pv produces valid MLIR for all possible data types and
// lowering paths.

// RUN: rocmlir-gen --arch gfx908 -p -pv | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p -pv -t f16 | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p -pv -t bf16 | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p -pv -t i8 | rocmlir-driver -c | rocmlir-opt
