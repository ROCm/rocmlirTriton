// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Regression for the AMDGPU backend's float-based expansion of 32-bit integer
// div/rem (llvm-patches/patch201186.patch). makeGroupedGridLayout computes
// m_block as bid % thisMBlocksPerGroup, with the grid-wide block id as the
// dividend and a runtime divisor. Without the patch AMDGPUCodeGenPrepare
// expands that through trunc(float(bid) * rcp(float(divisor))), which comes
// out one too large for some dividends past roughly 2^23, so m_block lands far
// outside A: wrong results, and occasionally a memory access fault.
//
// A 1x1 tile over 9216x1536 gives 14155776 workgroups. gridGroupSize=11 pins
// the divisor instead of deriving it from the CU and chiplet counts;
// v_rcp_f32(11) lies above 1/11 on gfx90a and gfx942, and without the patch
// about 238000 of the 14155776 output elements come out wrong there.

// RUN: rocmlir-gen --arch %arch -operation gemm -t f16 -out_datatype f16 -g 1 -m 9216 -k 512 -n 1536 -transA=True -transB=False -transO=False --perf_config=gemm:mPerBlock=1,nPerBlock=1,kPerBlock=32,kpack=1,numCTAs=1,numWaves=2,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=11,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
