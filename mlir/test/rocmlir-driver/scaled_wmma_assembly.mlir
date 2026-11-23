// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// gfx1250 scaled WMMA. Block 32 selects v_wmma_scale and block 16 selects
// v_wmma_scale16. Both forms of the 16x16x128 f8f6f4 instruction accept mixed
// f8/f4; f8E4M3FN scales are legal on fp4 x fp4 and on the fp4 side of a
// mixed pair.

// gfx1250 cannot pack a scaled fp4 operand along M/N, so B is transposed and
// both fp4 operands pack along K. -p fills host buffers through a layout that
// drops that K vectorization.

// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f8E4M3FN -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton \
// RUN:   | FileCheck %s --check-prefix=SCALE32
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f4E2M1FN -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   -scale_a_dtype f8E4M3FN -scale_b_dtype f8E4M3FN \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton \
// RUN:   | FileCheck %s --check-prefix=SCALE16
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f8E4M3FN_f4E2M1FN -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E4M3FN \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton \
// RUN:   | FileCheck %s --check-prefix=MIXED16

// SCALE32: llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4
// SCALE16: llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4
// MIXED16: llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4
