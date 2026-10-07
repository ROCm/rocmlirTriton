// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Documents the Triton bug that keeps non-power-of-2 kPerBlock disabled on
// gfx950 (see rock::supportsNonPow2KPerBlock), reported upstream as
// https://github.com/triton-lang/triton/issues/12116.
//
// kPerBlock=36 splits the K loop into segments of 32 and 4, and kPerBlock=18
// into 16 and 2. With mPerBlock=16 or 32 the last segment copies a 16x4 or 32x2
// tile into LDS, fewer rows than a warp has lanes. CoalesceAsyncCopy then keeps
// a blocked order that puts consecutive lanes in different LDS columns,
// canLoadDirectToLDS rejects the copy, the buffer load to LDS is left unlowered
// and LLVM translation fails on a `builtin.unrealized_conversion_cast`.
//
// gfx950 takes this path by default for f32 GEMMs and convolutions. The first
// RUN is a plain GEMM that hits it, the second is the convolution the weekly
// tuning hit there. gfx950 now rejects these perf configs, so the test compiles
// them for gfx942 with useAsyncCopy=1 and useBf16x3ForF32=1, which takes the
// same path.
//
// The copy only goes wrong when its shared layout has a different order than
// its blocked layout. For the 16x4 tile of the first two RUNs that only happens
// when the f32 dot is split into BF16x3 (useBf16x3ForF32=1, the gfx950
// default), and with useBf16x3ForF32=0 they compile. The 32x2 tile of the third
// RUN fails with either value.
//
// Once this passes, remove the XFAIL and enable non-power-of-2 kPerBlock on
// gfx950 again.

// XFAIL: *

// RUN: rocmlir-gen --operation gemm -t f32 \
// RUN:   --arch gfx942:sramecc+:xnack- --num_cu 304 --num_chiplets 8 \
// RUN:   -g 1 -m 16 -n 16 -k 72 \
// RUN:   --perf_config="gemm:mPerBlock=16,nPerBlock=16,kPerBlock=36,kpack=1,numCTAs=1,numWaves=2,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=1" \
// RUN:   | rocmlir-driver -c --arch gfx942:sramecc+:xnack- | FileCheck %s

// RUN: rocmlir-gen --operation conv -t f32 \
// RUN:   --arch gfx942:sramecc+:xnack- --num_cu 304 --num_chiplets 8 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 512 --in_h 14 --in_w 14 --out_channels 512 \
// RUN:   --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 \
// RUN:   --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   --perf_config="gemm:mPerBlock=16,nPerBlock=16,kPerBlock=36,kpack=1,numCTAs=1,numWaves=2,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=1" \
// RUN:   | rocmlir-driver -c --arch gfx942:sramecc+:xnack- | FileCheck %s

// RUN: rocmlir-gen --operation conv -t f32 \
// RUN:   --arch gfx942:sramecc+:xnack- --num_cu 304 --num_chiplets 8 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 512 --in_h 7 --in_w 7 --out_channels 512 \
// RUN:   --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 \
// RUN:   --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   --perf_config="gemm:mPerBlock=32,nPerBlock=16,kPerBlock=18,kpack=1,numCTAs=1,numWaves=2,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=1" \
// RUN:   | rocmlir-driver -c --arch gfx942:sramecc+:xnack- | FileCheck %s

// CHECK: triton.hsaco
