// Compile-only smoke test for gfx906 (Vega20 / GCN5_1).
//
// gfx906 has no MFMA/WMMA accelerator instructions, so rocmlirTriton lowers
// gemm/conv on it through Triton's non-accel codegen path (v_dot intrinsics
// for dot-eligible types). This test cross-compiles a small gemm and a small
// conv to gfx906 so that the GCN5_1 paths in AmdArchDb and Triton's AMD
// backend (deduceISAFamily / supportsVDot / etc.) keep compiling on every CI
// host, regardless of the host GPU.

// RUN: rocmlir-gen -operation gemm -t f32 -out_datatype f32 \
// RUN:   --arch gfx906 --num_cu 60 \
// RUN:   -g 1 -m 64 -k 64 -n 64 -transA=False -transB=False \
// RUN:   | rocmlir-driver -c | FileCheck %s --check-prefix=GEMM

// RUN: rocmlir-gen -operation gemm -t f16 -out_datatype f32 \
// RUN:   --arch gfx906 --num_cu 60 \
// RUN:   -g 1 -m 64 -k 64 -n 64 -transA=False -transB=False \
// RUN:   | rocmlir-driver -c | FileCheck %s --check-prefix=GEMM_F16

// RUN: rocmlir-gen -operation conv -t f32 \
// RUN:   --arch gfx906 --num_cu 60 \
// RUN:   -g 1 -batchsize 1 -in_channels 16 -out_channels 16 \
// RUN:   -in_h 8 -in_w 8 -fil_h 1 -fil_w 1 \
// RUN:   --dilation_h 1 --dilation_w 1 \
// RUN:   --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 0 --padding_w 0 \
// RUN:   | rocmlir-driver -c | FileCheck %s --check-prefix=CONV

// GEMM: triton.hsaco
// GEMM_F16: triton.hsaco
// CONV: triton.hsaco
