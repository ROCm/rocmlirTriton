// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen --arch %arch --operation gemm -t f32_f16 -to f32 -g 1 -m 1 -k 1 -n 1 -pv | rocmlir-driver -c | rocm-run | FileCheck %s
// RUN: rocmlir-gen --arch %arch --operation gemm -ta f16 -tb f32 -to f32 -g 1 -m 1 -k 1 -n 1 -pv | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
