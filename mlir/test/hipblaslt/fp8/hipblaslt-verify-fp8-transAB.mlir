// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %python %mlir_src_root/utils/performance/hipblaslt-benchmark-driver/verify_hipblaslt.py -m 128 -n 128 -k 64 -g 1 -t fp8 --transA True --transB True \
// RUN:   --hipblaslt-path hipblaslt-benchmark-driver \
// RUN:   --rocmlir-gen-path rocmlir-gen \
// RUN:   --rocmlir-driver-path rocmlir-driver \
// RUN:   -arch %arch --tolerance 0.1 | FileCheck %s

// Verify hipblaslt GEMM fp8 128x128x64 with transA and transB produces correct results
// CHECK: PASSED

