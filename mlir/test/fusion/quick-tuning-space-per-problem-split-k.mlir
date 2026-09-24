// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A per-problem ranking is measured with split-K allowed, so a kernel whose
// fusion forbids split-K can only run part of it. What makes that safe is a
// generator invariant: select_perfconfigs trades a row's last slot for the
// best measured splitKFactor=1 config, so every shipped row has at least one
// legal member. This test pins the two halves meeting -- the row is still
// consulted, the illegal members are dropped, and what survives is that
// reserved slot rather than an empty search space.
//
// The problem is gfx942 f32 GEMM 128x512x512, the same one
// rocmlir-gen/quick-tuning-per-problem.mlir uses, so its row is known to be
// reachable -- otherwise the checks below would pass for the wrong reason.
// Four of its five perfconfigs use splitKFactor=4.

// Relu cannot be applied to each split and then summed, so split-K is illegal.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=KEY-NO-SPLIT-K
// KEY-NO-SPLIT-K: -supportsSplitK false

// The fusion is not part of a problem's identity, so both spellings key the
// same row.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-quick-tuning-problem-key-hash - | FileCheck %s --check-prefix=HASH
// RUN: sed -e '/migraphx.relu/d' -e 's/return %1 :/return %0 :/' %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-gen --emit-quick-tuning-problem-key-hash - | FileCheck %s --check-prefix=HASH
// HASH: 8175943205932196350

// What is left is the row's reserved splitKFactor=1 slot. The negative on
// mPerBlock=256,nPerBlock=128,kPerBlock=16 -- a config only gfx942's
// no-split-K set cover has -- is what distinguishes a filtered row from a
// fallback to that set cover.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-space=quick - \
// RUN:   | FileCheck %s --check-prefix=SPACE-NO-SPLIT-K \
// RUN:       --implicit-check-not='splitKFactor={{([2-9]|[1-9][0-9]+)}}' \
// RUN:       --implicit-check-not='mPerBlock=256,nPerBlock=128,kPerBlock=16,'
// SPACE-NO-SPLIT-K: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=256,kpack=1,numCTAs=1,numWaves=2,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,

// Disabling the per-problem layer for the same fused kernel falls back to the
// no-split-K set cover, which is where that config does appear -- so the run
// above really did come from the row.
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
