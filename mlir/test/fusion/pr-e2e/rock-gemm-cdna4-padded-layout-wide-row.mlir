// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Regression for composePaddedLayoutForAsyncCopyCDNA4 on dot operands whose
// rows are wider than a warp (triton-patches/patch11890.patch). With
// kPerBlock=2048 a row of A holds 2048 f16 elements, 256 vectors of 8, against
// a warp of 64 lanes. Before the patch the padded async-copy layout was
// staggered over rows the tile does not have, and
// PaddedSharedEncodingAttr::verify aborted the compiler. The first RUN is the
// config the weekly tuning hit on gfx950; the other two have 2 and 4 rows,
// fewer than the stagger. The layout is only used on gfx950, so lit.local.cfg
// limits the test to it.

// RUN: rocmlir-gen --arch %arch -operation gemm -t f16 -out_datatype f16 -g 1 -m 384 -k 3072 -n 768 -transA=False -transB=False -transO=False --perf_config=gemm:mPerBlock=1,nPerBlock=16,kPerBlock=2048,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// RUN: rocmlir-gen --arch %arch -operation gemm -t f16 -out_datatype f16 -g 1 -m 384 -k 3072 -n 768 -transA=False -transB=False -transO=False --perf_config=gemm:mPerBlock=2,nPerBlock=16,kPerBlock=2048,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// RUN: rocmlir-gen --arch %arch -operation gemm -t f16 -out_datatype f16 -g 1 -m 384 -k 3072 -n 768 -transA=False -transB=False -transO=False --perf_config=gemm:mPerBlock=4,nPerBlock=16,kPerBlock=1024,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
