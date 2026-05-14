// gemm f16 -> f32, 1x64x256x128, pinned perf_config
// RUN: rocmlir-gen --arch %arch --operation gemm -t f16 -out_datatype f32 -g 1 -m 64 -k 256 -n 128 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// gemm f16 -> f32, 1x8x128x8, pinned perf_config (small-shape corner case)
// RUN: rocmlir-gen --arch %arch --operation gemm -t f16 -out_datatype f32 -g 1 -m 8 -k 128 -n 8 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// gemm f16 -> f16, 1x64x256x128, pinned perf_config
// RUN: rocmlir-gen --arch %arch --operation gemm -t f16 -out_datatype f16 -g 1 -m 64 -k 256 -n 128 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// gemm f16 -> f32, 1x64x256x128, default (tuner-picked) perf_config
// RUN: rocmlir-gen --arch %arch --operation gemm -t f16 -out_datatype f32 -g 1 -m 64 -k 256 -n 128 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// gemm f16 -> f32, 1x8x128x8, default perf_config
// RUN: rocmlir-gen --arch %arch --operation gemm -t f16 -out_datatype f32 -g 1 -m 8 -k 128 -n 8 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// gemm f16 -> f16, 1x64x256x128, default perf_config
// RUN: rocmlir-gen --arch %arch --operation gemm -t f16 -out_datatype f16 -g 1 -m 64 -k 256 -n 128 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
