// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AIROCMLIR-1235: FMA dots that software-pipeline through LDS used to inherit
// the global-load order for the shared tile. A K-contiguous load (default A,
// or B with --transB) was then read from LDS strided. The pipeliner now
// re-orients that buffer to operand-major (M-contiguous for A, N-contiguous
// for B).
//
// The 8x32 tile is below the 16x16 MFMA/WMMA instruction, so accelerate-matmul
// keeps the FMA (blocked) encoding on every arch. numStages=2 is what makes
// the pipeliner allocate the shared buffer this layout applies to.

// NN: A is MxK (K-contiguous). The FMA pipeliner must flip A's LDS tile.
// RUN: rocmlir-gen -pv --operation gemm -t f32 --arch %arch \
// RUN:   -g 1 -m 64 -k 128 -n 128 --transA=false --transB=false \
// RUN:   --perf_config="gemm:v1:8,32,32,1,1,1,0,1,2,0,0" \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// NT: B is NxK (K-contiguous). The FMA pipeliner must flip B's LDS tile.
// RUN: rocmlir-gen -pv --operation gemm -t f32 --arch %arch \
// RUN:   -g 1 -m 64 -k 128 -n 128 --transA=false --transB=true \
// RUN:   --perf_config="gemm:v1:8,32,32,1,1,1,0,1,2,0,0" \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s --check-prefix=TRANSB

// CHECK: [1 1 1]
// TRANSB: [1 1 1]
