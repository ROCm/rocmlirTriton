// RUN: rocmlir-gen -pv --operation gemm -t f32 --arch %arch -g 1 -m 128 -k 96 -n 128 --perf_config="gemm:v1:64,64,48,1,1,4,16,1,1,1,1" | rocmlir-driver -c | rocm-run | FileCheck %s
// A non-power-of-two kPerBlock (48) is requested via the GEMM's perf_config.

// CHECK: [1 1 1]
