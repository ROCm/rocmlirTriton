// Verifies that when a perf config carries `kpack > 1` on an architecture
// where it is deprecated (CDNA4 / gfx950, gfx1250+), the tuning driver
// classifies the config as `NotApplicable`

// RUN: rocmlir-gen --operation gemm --arch gfx950 -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 --perf_config= \
// RUN: | not rocmlir-tuning-driver \
// RUN:     --benchmark-config="gemm:v1:128,128,128,2,1,4,16,1,2,0,0" 2>&1 \
// RUN: | FileCheck %s

// The arch-mismatched config is classified as N/A by the tuning driver.
// CHECK: gemm:v1:128,128,128,2,1,4,16,1,2,0,0{{[[:space:]]+}}N/A
