//===- attention-transposed-bias-direct-load-gfx942.mlir -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A physically transposed 128x256 bias previously required a 128 KiB
// cross-wave layout-conversion buffer and exceeded gfx942's 64 KiB LDS limit.

// RUN: rocmlir-gen -operation attention -t f16 --arch gfx942 \
// RUN:   -g 1 -seq_len_q 128 -seq_len_k 256 \
// RUN:   -num_heads_q 1 -num_heads_kv 1 \
// RUN:   -head_dim_qk 32 -head_dim_v 32 \
// RUN:   -with-attn-scale=False -with-attn-bias=True -transBias=True \
// RUN:   -transQ=False -transK=False -transV=False -transO=False \
// RUN:   -causal=False -return_lse=False -split_kv=1 \
// RUN:   --perf_config=attn:v2:128,256,32,1,1,2,32,1,1,0,0,-1,-1,-1,-1,-1,1 \
// RUN:   -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | FileCheck %s

// CHECK: gpu.binary @rock_kernels
