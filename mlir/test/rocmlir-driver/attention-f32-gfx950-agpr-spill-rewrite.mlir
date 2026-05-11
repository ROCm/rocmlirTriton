// Regression test for an LLVM AMDGPU backend crash in
// AMDGPURewriteAGPRCopyMFMA::eliminateSpillsOfReassignedVGPRs.
//
// Without the cherry-picked llvm/llvm-project#167347 (carried as
// llvm-patches/patch4.patch), this command aborts during
// TritonToHsacoPass with:
//   *** Bad machine code: Virtual register defs don't dominate all uses. ***
//   LLVM ERROR: Found 1 machine code errors.
// because the AGPR spill-elimination helper rewrites spill reloads that
// are not jointly dominated by spill stores, breaking SSA on the
// produced MachineFunction.
//
// The bug itself lives in a pass that runs on every MFMA-capable
// subtarget (gfx908/gfx90a/gfx94x/gfx95x via FeatureMAIInsts), but the
// only in-tree shape we have that actually trips the dominance
// violation is this one, and only on gfx950 -- the gfx950 register
// allocator places spill reloads in the pattern that exposes the
// missing dominance check, while gfx908/gfx90a/gfx942 happen to land
// on safer placements for the same source. Hence the hardcoded
// --arch; we cross-compile so the test runs on every CI host
// regardless of GPU. Compile-only, no GPU required.

// RUN: rocmlir-gen -operation attention -t f32 \
// RUN:   --arch gfx950:sramecc+:xnack- \
// RUN:   -g 8 -seq_len_q 1 -seq_len_k 349 -num_heads_q 128 -num_heads_kv 2 \
// RUN:   -head_dim_qk 233 -head_dim_v 236 \
// RUN:   -with-attn-scale=True -with-attn-bias=True \
// RUN:   -return_lse=True -split_kv=8 \
// RUN:   --perf_config=attn:v1:64,32,16,2,1,1,32,1,2,2,2 \
// RUN:   --current_seq_len=255,148,29,264,122,189,61,184 \
// RUN:   | rocmlir-driver -c | FileCheck %s

// CHECK: triton.hsaco
