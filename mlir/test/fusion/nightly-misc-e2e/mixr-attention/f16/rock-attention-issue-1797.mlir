// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 8 -seq_len_k 8 -head_dim_qk 8 -head_dim_v 8 --transQ=true --transK=true --transV=false --transO=false -rand 1 -rand_type int -pv | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
