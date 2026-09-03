// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -pv --operation conv_bwd_data -t f16 --arch %arch --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 8 --in_channels 512 --in_h 8 --in_w 112 --out_channels 64 --fil_h 2 --fil_w 5 --dilation_h 2 --dilation_w 2 --conv_stride_h 1 --conv_stride_w 6 --padding_h 3 --padding_w_l 3 --padding_w_r 2 --groupsize 4 --perf_config=gemm:v1:128,128,48,1,1,1,16,1,1,2,0 | rocmlir-driver -c | rocm-run | FileCheck %s

// kPerBlock=48 decomposes the K loop into power-of-two segments (32 + 16) that use
// different MFMA opcodes accumulating into the same registers. On gfx950 the
// switch between them needs wait states that the AMDGPU hazard recognizer used
// to omit, so the kernel silently produced wrong results (llvm-patches/
// patch218363.patch). The miscompile is only observable at runtime, so this
// compares the GPU output against the CPU verifier.

// CHECK: [1 1 1]
