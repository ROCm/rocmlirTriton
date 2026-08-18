// Some old behavior in Triton was implemented such that `shuffleXor` on RDNA
// fell through to the CDNA DPP chain (ROW_HALF_MIRROR / ROW_ROR / QUAD_PERM),
// which assumes 64-lane wavefronts. Executed on RDNA's 32-lane wavefronts it
// combined the wrong lanes, corrupting any softmax-style reduction inside
// attention and producing NaN/garbage outputs. This was fixed with
// https://github.com/triton-lang/triton/pull/10065. The rocmlir-gen commands
// below reproduce the shapes from AIROCMLIR-84.

// RUN: rocmlir-gen --arch %arch --operation attention -t i8 -g 1 -seq_len_q 20 -seq_len_k 3900 -num_heads_q 128 -num_heads_kv 32 -head_dim_qk 191 -head_dim_v 122 -with-attn-scale=True -with-attn-bias=True -transQ=False -transK=False -transV=True -transO=True -causal=True -return_lse=True -split_kv=32 --perf_config=attn:v1:16,32,128,2,1,8,16,1,3,2,1 -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]

// RUN: rocmlir-gen --arch %arch --operation attention -t f16 --num_cu 70 --num_chiplets 1 -g 5 -seq_len_q 1 -seq_len_k 331 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 69 -head_dim_v 208 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=True -transV=True -transO=False -causal=False -return_lse=True -split_kv=8 --perf_config=attn:v1:128,64,32,2,1,16,32,1,1,4,4 --current_seq_len=255,18,268,69,317 -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s --check-prefix=KVCACHE

// KVCACHE: [1 1 1]

// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -g 1 -seq_len_q 1 -seq_len_k 117 -num_heads_q 64 -num_heads_kv 4 -head_dim_qk 91 -head_dim_v 61 -with-attn-scale=False -with-attn-bias=True -transQ=False -transK=False -transV=True -transO=True -causal=False -return_lse=True -split_kv=1 --perf_config=attn:v1:64,32,16,2,1,8,32,1,1,0,1 -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s --check-prefix=GQA

// GQA: [1 1 1]
