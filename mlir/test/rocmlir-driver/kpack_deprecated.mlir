// Verifies that invalid user-provided kpack values emit clear diagnostics to
// stderr and the pipeline fails.

// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 \
// RUN:   --perf_config="gemm:v1:128,128,128,2,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-driver --kernel-pipeline=gpu 2>&1 \
// RUN: | FileCheck %s --check-prefix=TOO-LARGE-GFX950

// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 \
// RUN:   --perf_config="gemm:v1:128,128,128,2,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-driver --kernel-pipeline=gpu 2>&1 \
// RUN: | FileCheck %s --check-prefix=TOO-LARGE-GFX1250

// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 \
// RUN:   --perf_config="gemm:v1:128,128,128,0,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-driver --kernel-pipeline=gpu 2>&1 \
// RUN: | FileCheck %s --check-prefix=NON-POSITIVE

// TOO-LARGE-GFX950: error: kpack=2 exceeds max (1) for amdgcn-amd-amdhsa:gfx950
// TOO-LARGE-GFX1250: error: kpack=2 exceeds max (1) for amdgcn-amd-amdhsa:gfx1250
// NON-POSITIVE: error: kpack=0 must be positive
