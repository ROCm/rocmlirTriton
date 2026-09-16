// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Regression for an LLVM AMDGPU backend crash on gfx950. The scheduler's
// PreRARematStage refused to rematerialize a register with multiple users in
// the same region, leaving it spilled. AMDGPURewriteAGPRCopyMFMA then folded
// that VGPR spill slot back into a single vreg, producing overlapping live
// ranges that tripped
// `Assertion \`(i == Size || Traits::stopLess(b, start(i))) && "Overlapping
// insert"' failed` in LiveIntervalUnion::unify (via LiveRegMatrix::assign).
// Without the cherry-picked fix in llvm-patches/patch214725.patch,
// rocmlir-driver -c aborts on this conv shape and perf config. Compile-only;
// cross-compiles to gfx950 so the test runs on every CI host.
//
// JIRA: LCOMPILER-2224
// Upstream fix: https://github.com/llvm/llvm-project/pull/214725

// RUN: rocmlir-gen --operation conv -t f16 \
// RUN:   --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 768 --in_h 32 --in_w 32 \
// RUN:   --out_channels 383 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   --perf_config=gemm:mPerBlock=256,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=16,matrixInstrNonkdim=32,splitKFactor=3,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1 \
// RUN:   | rocmlir-driver -c | FileCheck %s

// CHECK: triton.hsaco
