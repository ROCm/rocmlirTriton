// Drives a tuning sweep through rocmlir-tuning-driver in
// `--compile-only` mode: the driver must compile every candidate perf config
// but never touch the GPU.

// RUN: rocmlir-gen --arch %arch -operation gemm -t f16 -out_datatype f32 -g 1 -m 1024 -k 1024 -n 1024 -transA=False -transB=False --perf_config= \
// RUN: | rocmlir-tuning-driver --tuning-space=quick --compile-only \
// RUN: | FileCheck %s

// CHECK: gemm:{{.*}}compiled
