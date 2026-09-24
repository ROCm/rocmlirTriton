//===----------------------------------------------------------------------===//
// Per-problem quick-tuning narrowing
//===----------------------------------------------------------------------===//

// quick-tuning-problem-key-hash.mlir pins the key's *computation*; this file
// pins its *reachability*, i.e. that a problem the shipped shards were
// generated for still finds its row. Without it a key change is caught only by
// a failing hash value, whose natural fix is to update the pinned number --
// leaving every shipped shard unreachable with nothing to say so, since a miss
// degrades silently into the set cover.
//
// The witness problem is gfx942 f32 GEMM 128x512x512, whose row lives in
// QuickTuningProblemMap/Gfx942GemmF32.inc. Its five perfconfigs are disjoint
// from the gfx942_gemm_f32 set cover, so one config from each side tells the
// two apart, and the pair below differs only in the K tile. Regenerating the
// shards can retire whichever config is named here: pick another member of the
// row rather than dropping the check.

// RUN: rocmlir-gen --arch gfx942 --operation=gemm -t f32 -g 1 -m 128 -n 512 -k 512 --num_cu=304 --emit-tuning-space=quick 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PER-PROBLEM \
// RUN:       --implicit-check-not='mPerBlock=16,nPerBlock=16,kPerBlock=512,'
// CHECK-PER-PROBLEM: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=256,kpack=1,numCTAs=1,numWaves=2,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,

// ROCMLIR_DISABLE_PER_PROBLEM_QUICK_TUNING turns off this layer alone, so the
// same problem falls back to the untouched set cover. The two runs must select
// each other's witness and nothing else, which is what proves the narrowing
// came from the per-problem row and not from filtering the set cover.
// RUN: ROCMLIR_DISABLE_PER_PROBLEM_QUICK_TUNING=1 rocmlir-gen --arch gfx942 --operation=gemm -t f32 -g 1 -m 128 -n 512 -k 512 --num_cu=304 --emit-tuning-space=quick 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SET-COVER \
// RUN:       --implicit-check-not='mPerBlock=16,nPerBlock=16,kPerBlock=256,'
// CHECK-SET-COVER: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=512,kpack=1,numCTAs=1,numWaves=2,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,

// The same guard for the other two operations that ship shards.
// getQuickTuningProblemKey branches per kernel type, so a key change breaks
// one operation at a time and a gemm-only check would let a conv or attention
// change strand its shards unnoticed -- as happened to attention, which is
// never split-K legal and so was stranded by a lookup that consulted the
// rankings only for split-K-capable callers. These name no perfconfig, only
// that the row is found and narrows, so regenerating does not churn them.
// DEFINE: %{conv} = rocmlir-gen --arch gfx942 --operation conv -t f32 --num_cu 304 --fil_layout k01gc --in_layout 01ngc --out_layout ngk01 --batchsize 1 --in_channels 3 --in_h 224 --in_w 224 --out_channels 64 --fil_h 7 --fil_w 7 --conv_stride_h 2 --conv_stride_w 2 --padding_h 3 --padding_w 3 --emit-tuning-space=quick
// RUN: %{conv} > %t.conv.narrowed
// RUN: ROCMLIR_DISABLE_PER_PROBLEM_QUICK_TUNING=1 %{conv} | not diff - %t.conv.narrowed

// DEFINE: %{attn} = rocmlir-gen --arch gfx942 --operation attention -t f16 --num_cu 304 -g 12 -seq_len_q 384 -seq_len_k 384 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 64 -head_dim_v 64 -transK=true --emit-tuning-space=quick
// RUN: %{attn} > %t.attn.narrowed
// RUN: ROCMLIR_DISABLE_PER_PROBLEM_QUICK_TUNING=1 %{attn} | not diff - %t.attn.narrowed

// A problem with no row of its own must come back unchanged: the per-problem
// layer deliberately has no key fallback, since a ranking only holds for the
// problem it was measured on. Shrinking M by one leaves the same lookup key
// (gfx942_gemm_f32) but a different problem hash, so both runs must agree.
// RUN: rocmlir-gen --arch gfx942 --operation=gemm -t f32 -g 1 -m 127 -n 512 -k 512 --num_cu=304 --emit-tuning-space=quick 2>&1 > %t.unmapped.quick
// RUN: ROCMLIR_DISABLE_PER_PROBLEM_QUICK_TUNING=1 rocmlir-gen --arch gfx942 --operation=gemm -t f32 -g 1 -m 127 -n 512 -k 512 --num_cu=304 --emit-tuning-space=quick 2>&1 \
// RUN:   | diff - %t.unmapped.quick
