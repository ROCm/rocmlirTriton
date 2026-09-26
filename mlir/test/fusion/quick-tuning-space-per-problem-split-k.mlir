// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Output fusions are not part of the per-problem key, so the fused kernel
// below maps to the bare GEMM's row. The relu makes split-K illegal, so the
// tuning space keeps only the row's split-K-free members. Dropping the fusion
// recovers the whole row, including its split-K configs.
//
// The problem is gfx942 f32 GEMM 128x512x512, the same one
// rocmlir-gen/quick-tuning-per-problem.mlir uses, so its row is known to be
// reachable when the fusion is removed.
// Four of its five perfconfigs use splitKFactor=4.

// Relu cannot be applied to each split and then summed, so split-K is illegal.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=KEY-NO-SPLIT-K
// KEY-NO-SPLIT-K: -supportsSplitK false

// The fused kernel and the bare GEMM share the known hash.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-quick-tuning-problem-key-hash - | FileCheck %s --check-prefix=HASH
// RUN: sed -e '/migraphx.relu/d' -e 's/return %1 :/return %0 :/' %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-gen --emit-quick-tuning-problem-key-hash - | FileCheck %s --check-prefix=HASH
// HASH: 8175943205932196350

// The fused kernel quietly uses the row's split-K-free members rather than the
// set cover.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-space=quick - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPACE-FUSED \
// RUN:       --implicit-check-not='splitKFactor={{([2-9]|[1-9][0-9]+)}}' --implicit-check-not=warning \
// RUN:       --implicit-check-not='mPerBlock=256,nPerBlock=128,kPerBlock=16,'
// SPACE-FUSED: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=256,kpack=1,numCTAs=1,numWaves=2,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,

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
