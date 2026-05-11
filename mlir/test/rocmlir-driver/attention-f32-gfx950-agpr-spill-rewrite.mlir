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
// The arch is hardcoded because the crash only reproduces on
// AGPR-capable subtargets (gfx90a/gfx94x/gfx95x); the gfx95x AGPR
// allocator is what triggers the buggy code path here. Compile-only,
// no GPU required.

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
