// Regression coverage for PreserveMaskedLoadSemantics: this i8 causal
// attention shape previously kept invalid masked lanes alive in the softmax max
// reduction. Those lanes are loaded as zero, which is not the neutral value for
// max and can dominate real negative scores unless the pass remasks them to the
// type minimum before reduction.
// RUN: rocmlir-gen -operation attention -t i8 --arch %arch -g 3 -seq_len_q 1216 -seq_len_k 162 -num_heads_q 32 -num_heads_kv 2 -head_dim_qk 217 -head_dim_v 128 -with-attn-scale=True -with-attn-bias=False -transQ=True -transK=True -transV=True -transO=False -causal=True -return_lse=False -split_kv=1 -rand 1 -rand_type int -rand_min_int -1 -rand_max_int 1 --perf_config=attn:v1:16,256,64,2,1,8,16,1,1,2,2 -pv | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
