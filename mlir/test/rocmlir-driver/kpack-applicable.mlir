// Verifies that kpack values still accepted by the target pass through the
// driver pipeline.

// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 \
// RUN:   --perf_config="gemm:v1:128,128,128,1,1,4,16,1,2,0,0" \
// RUN: | rocmlir-driver --kernel-pipeline=gpu >/dev/null

// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 \
// RUN:   --perf_config="gemm:v1:128,128,128,2,1,4,16,1,2,0,0" \
// RUN: | rocmlir-driver --kernel-pipeline=gpu >/dev/null

// RUN: rocmlir-gen --arch gfx1200 --operation gemm -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 \
// RUN:   --perf_config="gemm:v1:128,128,128,1,1,4,16,1,2,0,0" \
// RUN: | rocmlir-driver --kernel-pipeline=gpu >/dev/null

// RUN: rocmlir-gen --arch gfx1200 --operation gemm -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 \
// RUN:   --perf_config="gemm:v1:128,128,128,2,1,4,16,1,2,0,0" \
// RUN: | rocmlir-driver --kernel-pipeline=gpu >/dev/null
