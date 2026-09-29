// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %python %mlir_src_root/utils/performance/hipblaslt-benchmark-driver/verify_hipblaslt.py -m 128 -n 128 -k 64 -g 1 -t f32 --transA True --transB True \
// RUN:   --hipblaslt-path hipblaslt-benchmark-driver \
// RUN:   --rocmlir-gen-path rocmlir-gen \
// RUN:   --rocmlir-driver-path rocmlir-driver \
// RUN:   -arch %arch | FileCheck %s

// Verify hipblaslt GEMM f32 128x128x64 with transA and transB produces correct results
// CHECK: PASSED

