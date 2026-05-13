// Verifies that invalid perf configs carrying unsupported kpack values are
// classified as `NotApplicable` by the tuning driver.

// RUN: rocmlir-gen --operation gemm --arch gfx950 -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 --perf_config= \
// RUN: | not rocmlir-tuning-driver \
// RUN:     --benchmark-config="gemm:v1:128,128,128,2,1,4,16,1,2,0,0" 2>&1 \
// RUN: | FileCheck %s --check-prefix=TOO-LARGE \
// RUN:     --implicit-check-not="Diagnostic error:" \
// RUN:     --implicit-check-not="kpack=2 exceeds max" \
// RUN:     --implicit-check-not="Compilation pipeline failed for config:"

// RUN: rocmlir-gen --operation gemm --arch gfx950 -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 --perf_config= \
// RUN: | not rocmlir-tuning-driver \
// RUN:     --benchmark-config="gemm:v1:128,128,128,0,1,4,16,1,2,0,0" 2>&1 \
// RUN: | FileCheck %s --check-prefix=NON-POSITIVE \
// RUN:     --implicit-check-not="Diagnostic error:" \
// RUN:     --implicit-check-not="kpack=0 must be positive" \
// RUN:     --implicit-check-not="Compilation pipeline failed for config:"

// TOO-LARGE: gemm:v1:128,128,128,2,1,4,16,1,2,0,0{{[[:space:]]+}}N/A
// NON-POSITIVE: gemm:v1:128,128,128,0,1,4,16,1,2,0,0{{[[:space:]]+}}N/A
