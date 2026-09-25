// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A per-problem ranking is only reusable for the problem mode that was
// exhaustively tuned. The output fusion below was not represented when the
// shipped maps were generated, so quick tuning must diagnose it and use the
// no-split-K set cover. Dropping the fusion recovers the bare GEMM's
// per-problem row, including its split-K configs.
//
// The problem is gfx942 f32 GEMM 128x512x512, the same one
// rocmlir-gen/quick-tuning-per-problem.mlir uses, so its row is known to be
// reachable when the fusion is removed.
// Four of its five perfconfigs use splitKFactor=4.

// Relu cannot be applied to each split and then summed, so split-K is illegal.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=KEY-NO-SPLIT-K
// KEY-NO-SPLIT-K: -supportsSplitK false

// The fused mode has no compatible per-problem key until it is exhaustively
// tuned and the maps are regenerated. The bare GEMM retains its known hash.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | not rocmlir-gen --emit-quick-tuning-problem-key-hash - 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-FUSION
// RUN: sed -e '/migraphx.relu/d' -e 's/return %1 :/return %0 :/' %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-gen --emit-quick-tuning-problem-key-hash - | FileCheck %s --check-prefix=HASH
// UNSUPPORTED-FUSION: fields not represented by the shipped maps: output_fusions
// HASH: 8175943205932196350

// The fused mode warns and uses the no-split-K set cover.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-space=quick - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPACE-FUSED-FALLBACK \
// RUN:       --implicit-check-not='splitKFactor={{([2-9]|[1-9][0-9]+)}}'
// SPACE-FUSED-FALLBACK: warning: per-problem quick tuning does not represent the current problem's output_fusions
// SPACE-FUSED-FALLBACK: gemm:mPerBlock=256,nPerBlock=128,kPerBlock=16,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,

// Disabling the per-problem layer selects the same no-split-K set cover without
// trying to classify a per-problem key.
// RUN: ROCMLIR_DISABLE_PER_PROBLEM_QUICK_TUNING=1 rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | ROCMLIR_DISABLE_PER_PROBLEM_QUICK_TUNING=1 rocmlir-gen --emit-tuning-space=quick - \
// RUN:   | FileCheck %s --check-prefix=SPACE-SET-COVER \
// RUN:       --implicit-check-not='splitKFactor={{([2-9]|[1-9][0-9]+)}}'
// SPACE-SET-COVER: gemm:mPerBlock=256,nPerBlock=128,kPerBlock=16,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,

// Dropping the relu leaves a bare dot, which is split-K legal, so the same row
// is offered whole -- including the split-K configs the fused kernel lost.
// RUN: sed -e '/migraphx.relu/d' -e 's/return %1 :/return %0 :/' %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-gen --emit-tuning-space=quick - \
// RUN:   | FileCheck %s --check-prefix=SPACE-SPLIT-K \
// RUN:       --implicit-check-not='mPerBlock=256,nPerBlock=128,kPerBlock=16,'
// SPACE-SPLIT-K-DAG: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=256,kpack=1,numCTAs=1,numWaves=2,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,
// SPACE-SPLIT-K-DAG: gemm:mPerBlock=32,nPerBlock=32,kPerBlock=32,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=16,splitKFactor=4,numStages=3,

module {
  func.func private @mlir_dot_relu(%arg0: !migraphx.shaped<1x128x512xf32, 65536x512x1>,
                                   %arg1: !migraphx.shaped<1x512x512xf32, 262144x512x1>)
      -> (!migraphx.shaped<1x128x512xf32, 65536x512x1>)
      attributes {rock.kernel, rock.arch = "gfx942", rock.num_cu = 304 : i64} {
    %0 = migraphx.dot %arg0, %arg1 : <1x128x512xf32, 65536x512x1>, <1x512x512xf32, 262144x512x1> -> <1x128x512xf32, 65536x512x1>
    %1 = migraphx.relu %0 : <1x128x512xf32, 65536x512x1> -> <1x128x512xf32, 65536x512x1>
    return %1 : !migraphx.shaped<1x128x512xf32, 65536x512x1>
  }
}
