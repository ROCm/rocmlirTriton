// RUN: rocmlir-gen -operation attention -t f16 --arch %arch -g 1 -seq_len_q 32 -seq_len_k 16 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 32 -head_dim_v 32 -with-attn-scale=False -with-attn-bias=True -transBias=True -transQ=False -transK=False -transV=False -transO=False -causal=False -return_lse=False -split_kv=1 -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
