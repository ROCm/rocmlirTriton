// Verifies the BF16 precision difference between GPU and CPU for attention
// with scale and bias (the pre-softmax elementwise fusion: scale*QK + bias).
//
// On the GPU (RDNA3/RDNA4), the kernel uses v_dot2_bf16_bf16 to compute
// scale*QK+bias, performing the multiply-add in bf16 precision.
//
// On the CPU (host reference), the first GEMM accumulates in f32
// (llvm.fmul/fadd : f32), and although scale*QK+bias is expressed as bf16
// ops in MLIR, the LLVM backend lowers bf16 arithmetic by extending to f32,
// so the CPU effectively computes scale*QK+bias in f32.  This causes
// numerical divergence from the GPU — especially visible with
// current_seq_len masking in GQA + KV Cache configurations.  This is why
// PrAttentionBF16.toml uses relaxed thresholds for those tests.

// --- GPU assembly: gfx1100 (RDNA3) ---
// RUN: rocmlir-gen --arch gfx1100 --operation attention -rand 1 -current_seq_len=17 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --with-attn-bias -t bf16 -p \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GFX1100

// --- GPU assembly: gfx1200 (RDNA4) ---
// RUN: rocmlir-gen --arch gfx1200 --operation attention -rand 1 -current_seq_len=17 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --with-attn-bias -t bf16 -p \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GFX1200

// --- CPU LLVM IR (host reference) ---
// RUN: rocmlir-gen --arch gfx1100 --operation attention -rand 1 -current_seq_len=17 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --with-attn-bias -t bf16 -pv \
// RUN:   | rocmlir-driver --host-pipeline=highlevel \
// RUN:   | rocmlir-driver -c \
// RUN:   | FileCheck %s --check-prefix=CPU

// GPU uses bf16 dot for the pre-softmax scale*QK+bias fusion
// GFX1100: v_dot2_bf16_bf16
// GFX1200: v_dot2_bf16_bf16

// CPU first GEMM accumulates in f32
// CPU: llvm.func @host_naive_attention
// CPU: llvm.fmul {{.*}} : f32
// CPU: llvm.fadd {{.*}} : f32
