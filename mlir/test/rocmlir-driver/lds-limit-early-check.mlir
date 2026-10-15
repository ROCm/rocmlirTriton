// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Verifies that a perf config whose tile overflows LDS is rejected by
// rock-check-lds-limit as soon as shared memory is allocated, while the kernel
// is still TritonGPU IR, rather than after convert-triton-amdgpu-to-llvm. The
// tile (mPerBlock=nPerBlock=256, kPerBlock=512, f32) needs more shared memory
// than any supported arch has.
//
// RUN: rocmlir-gen --operation gemm --arch gfx942 -t f32 -out_datatype f32 \
// RUN:   -g 1 -m 256 -n 256 -k 512 --perf_config="gemm:v1:256,256,512,1,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-driver --kernel-pipeline=gpu,triton --arch gfx942 --mlir-print-ir-after-failure 2>&1 \
// RUN: | FileCheck %s

// CHECK: error: ttg.shared ({{[0-9]+}}) exceeds LDS limit (65536) for gfx942
// CHECK: IR Dump After RockCheckLdsLimitPass Failed: rock-check-lds-limit{arch=gfx942}
// CHECK-NOT: llvm.func
