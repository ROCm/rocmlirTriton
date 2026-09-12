// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Regression test for the FP4 partial packed-upcast group crash on gfx950.
//
// verify-packed-upcast-layout.mlir covers the pass on hand-written TTGIR; this
// drives the original reproducer through the whole pipeline. The arch is pinned
// because the layouts are gfx950's, and both runs stop at compilation.

// RUN: rocmlir-gen --operation gemm -t f4E2M1FN -out_datatype f32 \
// RUN:   --arch gfx950 --num_cu 256 -g 1 -m 225 -k 1280 -n 10240 \
// RUN:   -transA=False -transB=False -transO=False \
// RUN:   --perf_config="gemm:mPerBlock=16,nPerBlock=16,kPerBlock=64,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0" \
// RUN: | not rocmlir-driver -c -o /dev/null 2>&1 | FileCheck %s --check-prefix=PARTIAL

// kPerBlock = 64 is not a whole `16x16x128` k-tile, so the dot decomposes and
// the layout leaves each thread two of the four packed values it needs.
// PARTIAL: 'ttg.fp4_to_fp' op upcast consumes 4 packed values per thread, but this layout provides 2
// PARTIAL: Lowering not applicable

// The same shape on the native scaled accelerator builds no upcast at all,
// which is what `staysOnNativeScaledPath` keeps the quick-tuning front() on.
// RUN: rocmlir-gen --operation gemm -t f4E2M1FN -out_datatype f32 \
// RUN:   --arch gfx950 --num_cu 256 -g 1 -m 225 -k 1280 -n 10240 \
// RUN:   -transA=False -transB=False -transO=False \
// RUN:   --perf_config="gemm:mPerBlock=64,nPerBlock=128,kPerBlock=128,kpack=1,numCTAs=1,numWaves=8,matrixInstrNonkdim=16,splitKFactor=1,numStages=3,wavesPerEU=0,gridGroupSize=0" \
// RUN: | rocmlir-driver -c -o /dev/null
