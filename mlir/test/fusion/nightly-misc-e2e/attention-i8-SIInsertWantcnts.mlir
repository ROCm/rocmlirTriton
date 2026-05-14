// Regression for an LLVM AMDGPU backend abort in SIInsertWaitcnts on
// gfx950: without the cherry-picked fix in llvm-patches/patch6.patch,
// rocmlir-driver -c aborts inside SIInsertWaitcnts::run when
// WaitcntBrackets::mergeAsyncMarks reaches a control-flow join where
// one side has no async marks. MergeCount becomes 0 and the loop
// `for (auto Idx : seq_inclusive<unsigned>(1, MergeCount))` trips the
// `Begin <= End` assertion in llvm::iota_range (Sequence.h:275).
//
// Compile-only; cross-compiles to gfx950 so the test runs on every CI
// host with rocmlir-driver -c. The perf_config is pinned so the
// generated kernel is deterministic.
//
// Upstream fix: PR llvm/llvm-project#193499 (commit 81d618b6bc1e).

// RUN: rocmlir-gen -operation attention -t i8 --arch %arch --num_cu 256 --num_chiplets 8 -g 8 -seq_len_q 7 -seq_len_k 4096 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 16 -head_dim_v 16 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=True -transV=False -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:16,16,512,1,1,16,32,1,2,0,0 | rocmlir-driver -c | FileCheck %s

// CHECK: triton.hsaco
