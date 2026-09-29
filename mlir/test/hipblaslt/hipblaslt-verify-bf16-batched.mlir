// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %python %mlir_src_root/utils/performance/hipblaslt-benchmark-driver/verify_hipblaslt.py -m 128 -n 128 -k 64 -g 3 -t bf16 \
// RUN:   --hipblaslt-path hipblaslt-benchmark-driver \
// RUN:   --rocmlir-gen-path rocmlir-gen \
// RUN:   --rocmlir-driver-path rocmlir-driver \
// RUN:   -arch %arch --tolerance 0.05 | FileCheck %s

// Verify hipblaslt batched GEMM bf16 128x128x64 g=3 produces correct results
// CHECK: PASSED

