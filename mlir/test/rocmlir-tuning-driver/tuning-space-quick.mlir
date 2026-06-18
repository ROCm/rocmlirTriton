// Drives a quick `gemm:v2:` tuning sweep through rocmlir-tuning-driver and
// only verifies that the driver returned at least one valid measurement (a
// line beginning with the `gemm:v2:` perf-config prefix). Catches regressions
// in the tuning loop itself; this is complementary to:
//   * benchmark-config.mlir          (single pinned perf_config measurement)
//   * lds-overflow-not-applicable.mlir (N/A classification on LDS overflow)

// RUN: rocmlir-gen --arch %arch -operation gemm -t f16 -out_datatype f32 -g 1 -m 1024 -k 1024 -n 1024 -transA=False -transB=False --perf_config= \
// RUN: | rocmlir-tuning-driver --tuning-space=quick --rep=10 --warmup=1 --sleep-us=100 --use-median --show-all-measurements=false \
// RUN: | FileCheck %s

// CHECK: gemm:v2:
