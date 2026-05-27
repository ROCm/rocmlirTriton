// Documents the BF16 precision contract between the GPU kernel and the CPU
// reference for attention with scale and bias (the pre-softmax elementwise
// fusion: scale*QK + bias).
//
// On the GPU (RDNA3/RDNA4), the kernel stores the first-GEMM result as bf16,
// loads + extends it back to f32 for the fused chain, and computes scale*QK
// + bias in f32. The pattern is selected as `v_dot2_bf16_bf16` (or
// `v_pk_fma_bf16`), which performs the multiply-add at higher precision and
// rounds to bf16 only once at the end.
//
// On the CPU (host reference, strict mode) we mirror this: the first GEMM
// stays at bf16 (matching the GPU's narrow store), QK is cast back to f32
// before the fused chain, and scale*QK + bias runs at f32. The LLVM backend
// emits `llvm.fmul`/`llvm.fadd` at f32 for these operations. Without this
// matching, the CPU would otherwise do `bf16 mul -> truncate -> bf16 add
// -> truncate`, picking up two extra roundings versus the GPU's one, which
// can flip the softmax argmax with KV-cache masking and produce a wildly
// different output (~8.6x abs-diff). See PrAttentionBF16.toml.

// --- GPU assembly: gfx1100 (RDNA3) ---
// RUN: rocmlir-gen --arch gfx1100 --operation attention -rand 1 -current_seq_len=17 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --with-attn-bias -t bf16 -p \
// RUN:   | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GFX1100

// --- GPU assembly: gfx1200 (RDNA4) ---
// RUN: rocmlir-gen --arch gfx1200 --operation attention -rand 1 -current_seq_len=17 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --with-attn-bias -t bf16 -p \
// RUN:   | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GFX1200

// --- CPU LLVM IR (host reference, strict mode) ---
// RUN: rocmlir-gen --arch gfx1100 --operation attention -rand 1 -current_seq_len=17 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --with-attn-bias -t bf16 -pv-strict \
// RUN:   | rocmlir-driver --host-pipeline=highlevel \
// RUN:   | rocmlir-driver -c \
// RUN:   | FileCheck %s --check-prefix=CPU

// GPU uses bf16 dot for the pre-softmax scale*QK+bias fusion (high-precision
// multiply-add with a single round to bf16 at the end).
// GFX1100: v_dot2_bf16_bf16
// GFX1200: v_dot2_bf16_bf16

// CPU first GEMM accumulates in f32, and (in strict mode) the fused
// scale*QK + bias chain also runs at f32 to match the GPU.
// CPU: llvm.func @host_naive_attention
// CPU: llvm.fmul {{.*}} : f32
// CPU: llvm.fadd {{.*}} : f32
