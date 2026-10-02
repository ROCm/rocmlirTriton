// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 --causal --prefix_offset=3 --last_valid_kv_index=4 -seq_len_q 8 -seq_len_k 16 -head_dim_qk 32 -head_dim_v 32 -rand 1 -rand_type float -pv | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]

