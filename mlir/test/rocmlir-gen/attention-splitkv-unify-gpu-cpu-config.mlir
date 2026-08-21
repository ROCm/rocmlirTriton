// Regression test: with no --perf_config, the split-KV block size used by the
// CPU reference (rocmlir-gen) must equal the one the compiler picks for the GPU
// kernel (AffixTuningParameters). rocmlir-gen resolves it from the *same*
// rock.attention op the compiler tunes; before the fix it used the raw first
// quick-tuning entry, which the compiler discards, so CPU and GPU disagreed and
// GPU-vs-CPU verification failed.
//
// With head_dim_v=208 on gfx942 the first quick-tuning entry overflows LDS, so
// the compiler falls back to the conservative default nPerBlockG0=32.

// CPU side: the reference split-KV partition is derived from gemm0's
// nPerBlockG0. nPerBlockG0=32 -> valid split mask [8, 1, 5, 3, 5]. The buggy
// nPerBlockG0=64 would instead give [4, 1, 5, 2, 5].
// RUN: rocmlir-gen --arch gfx942 --operation attention -t f16 -g 5 -seq_len_q 1 -seq_len_k 331 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 69 -head_dim_v 208 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=True -transV=True -transO=False -causal=False -return_lse=True -split_kv=8 --last_valid_kv_index=255,18,268,69,317 -pv | rocmlir-opt | FileCheck %s --check-prefix=CPU-CONFIG

// GPU side: the config the compiler actually assigns to the kernel's gemm0
// (params0) must have the same nPerBlock=32 that the CPU mask above assumes.
// RUN: rocmlir-gen --arch gfx942 --operation attention -t f16 -g 5 -seq_len_q 1 -seq_len_k 331 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 69 -head_dim_v 208 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=True -transV=True -transO=False -causal=False -return_lse=True -split_kv=8 --last_valid_kv_index=255,18,268,69,317 -pv | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=GPU-CONFIG

// CPU-CONFIG-LABEL: func.func @rock_attention_gpu
// CPU-CONFIG: "tosa.const"() <{values = dense<{{\[+}}8{{\]+}}, {{\[+}}1{{\]+}}, {{\[+}}5{{\]+}}, {{\[+}}3{{\]+}}, {{\[+}}5{{\]+}}> : tensor<5x1x1x1xi32>}>

// GPU-CONFIG-DAG: [[GEMM0:#gemm_params[0-9]*]] = #rock.gemm_params<mPerBlock = {{[0-9]+}}, nPerBlock = 32,
// GPU-CONFIG: rock.attention
// GPU-CONFIG: params0 = [[GEMM0]],
