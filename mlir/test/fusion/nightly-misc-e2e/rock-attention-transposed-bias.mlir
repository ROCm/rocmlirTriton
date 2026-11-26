// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -operation attention -t f16 --arch %arch -g 1 -seq_len_q 32 -seq_len_k 16 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 32 -head_dim_v 32 -with-attn-scale=false -with-attn-bias=true -transBias=true -transQ=false -transK=false -transV=false -transO=false -causal=false -return_lse=false -split_kv=1 -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
