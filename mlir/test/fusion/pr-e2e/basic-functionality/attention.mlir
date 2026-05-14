// basic attention f16
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -rand 1 -g 1 -seq_len_q 64 -seq_len_k 64 -head_dim_qk 64 -head_dim_v 64 -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// grouped-query attention (GQA) f16, 4 Q-heads sharing 2 KV-heads
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -rand 1 -g 1 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 32 -seq_len_k 32 -head_dim_qk 32 -head_dim_v 32 -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// causal attention f16
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -rand 1 --causal -g 1 -seq_len_q 64 -seq_len_k 64 -head_dim_qk 64 -head_dim_v 64 -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// causal + GQA attention f16
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -rand 1 --causal -g 1 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 32 -seq_len_k 32 -head_dim_qk 32 -head_dim_v 32 -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
