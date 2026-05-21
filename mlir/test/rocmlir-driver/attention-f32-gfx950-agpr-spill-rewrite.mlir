// Regression for an LLVM AMDGPU backend crash in
// AMDGPURewriteAGPRCopyMFMA on gfx950: without the cherry-picked
// fix in llvm-patches/patch4.patch, rocmlir-driver -c aborts on
// this attention shape. Compile-only; cross-compiles to gfx950 so
// the test runs on every CI host.
//
// Ticket: AIROCMLIR-826

// RUN: rocmlir-gen -operation attention -t f32 \
// RUN:   --arch gfx950:sramecc+:xnack- \
// RUN:   -g 8 -seq_len_q 1 -seq_len_k 349 -num_heads_q 128 -num_heads_kv 2 \
// RUN:   -head_dim_qk 233 -head_dim_v 236 \
// RUN:   -with-attn-scale=True -with-attn-bias=True \
// RUN:   -return_lse=True -split_kv=8 \
// RUN:   --perf_config=attn:v1:64,32,16,1,1,1,32,1,2,2,2 \
// RUN:   --current_seq_len=255,148,29,264,122,189,61,184 \
// RUN:   | rocmlir-driver -c | FileCheck %s

// CHECK: triton.hsaco
