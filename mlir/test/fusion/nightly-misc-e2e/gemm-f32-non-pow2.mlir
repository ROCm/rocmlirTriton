// Non-power-of-2 f32 gemm (m=1000, n=405) paired with a tiny-tile
// perf_config (mPerBlock=16). Exercises the padding path for awkwardly-
// sized shapes; the perf_config is f32-specific so this lives as a
// standalone test rather than as a GemmVariants.toml entry (which would
// multiply it across the dtype/trans sweep).

// RUN: rocmlir-gen --arch %arch --operation gemm -t f32 -out_datatype f32 -transA=false -transB=false -g 1 -m 1000 -n 405 -k 1024 --perf_config=gemm:v1:16,32,32,2,1,4,32,1,2,0,0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
