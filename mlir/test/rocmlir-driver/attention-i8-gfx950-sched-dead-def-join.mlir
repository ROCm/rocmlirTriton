// Regression for an LLVM AMDGPU backend crash in the machine scheduler on
// gfx950: moving a dead subregister def upward retagged a following
// live-range segment to the same value number as the previous segment
// without joining the adjacent segments, tripping
// `Assertion \`LR.verify()' failed` in
// LiveIntervals::HMEditor::updateRange (via GCNSchedStage handleMove).
// Without the cherry-picked fix in llvm-patches/patch204648.patch,
// rocmlir-driver -c aborts on this attention shape. Compile-only;
// cross-compiles to gfx950 so the test runs on every CI host.
//
// Upstream fix: https://github.com/llvm/llvm-project/pull/204648

// RUN: rocmlir-gen -operation attention -t i8 \
// RUN:   --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 \
// RUN:   -g 8 -seq_len_q 7 -seq_len_k 4096 -num_heads_q 1 -num_heads_kv 1 \
// RUN:   -head_dim_qk 16 -head_dim_v 16 \
// RUN:   -with-attn-scale=False -with-attn-bias=False \
// RUN:   -transQ=False -transK=True -transV=False -transO=False \
// RUN:   -causal=False -return_lse=False -split_kv=1 \
// RUN:   --perf_config=attn:v6:128,128,0,16,1,1,1,16,1,3,0,0,-1,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver -c | FileCheck %s

// CHECK: triton.hsaco
