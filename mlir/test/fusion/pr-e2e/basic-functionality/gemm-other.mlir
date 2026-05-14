// gemm f32 -> f32, 1x1000x405x1024, pinned perf_config
// RUN: rocmlir-gen --arch %arch --operation gemm -t f32 -out_datatype f32 -transA=false -transB=false -g 1 -m 1000 -n 405 -k 1024 --perf_config=gemm:v1:16,32,32,2,1,4,32,1,2,0,0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// gemm fp8_fp8, 3x1024x769x1024
// RUN: rocmlir-gen --arch %arch --operation gemm -t fp8_fp8 -transA=false -g 3 -m 1024 -k 769 -n 1024 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// gemm f32, 3x1024x769x1024, pinned perf_config
// RUN: rocmlir-gen --arch %arch --operation gemm -t f32 -transA=false -g 3 -m 1024 -k 769 -n 1024 --perf_config=gemm:v1:256,128,32,1,1,4,16,4,1,0,0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// gemm i8, 1x4x128x4, pinned perf_config (small i8 shape that exercised a bug)
// RUN: rocmlir-gen --arch %arch --operation gemm -t i8 -transA=false -g 1 -m 4 -k 128 -n 4 --perf_config=gemm:v1:64,64,32,1,1,4,16,1,1,0,0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
