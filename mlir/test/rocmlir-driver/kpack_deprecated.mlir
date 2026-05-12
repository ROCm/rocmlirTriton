// Verifies that when a user manually passes kpack > 1 validateKpack hook emits
// a clear, arch-aware diagnostic to stderr and the pipeline fails.

// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 \
// RUN:   --perf_config="gemm:v1:128,128,128,2,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-driver --kernel-pipeline=gpu 2>&1 \
// RUN: | FileCheck %s

// CHECK: error: kpack=2 exceeds max (1) for amdgcn-amd-amdhsa:gfx950
