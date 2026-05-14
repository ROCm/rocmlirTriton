// Verifies that invalid perf configs carrying unsupported kpack values are
// classified as `NotApplicable` by the tuning driver. Covers both the MFMA
// (CDNA) and WMMA (RDNA) tuning paths, for the gemm (`getAccelRangeGemm`) and
// the attention (`getAccelRangeGemmGemm`) tuning-space generators.

// RUN: rocmlir-gen --operation gemm --arch gfx950 -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 --perf_config= \
// RUN: | not rocmlir-tuning-driver \
// RUN:     --benchmark-config="gemm:v1:128,128,128,2,1,4,16,1,2,0,0" 2>&1 \
// RUN: | FileCheck %s --check-prefix=GEMM-TOO-LARGE-GFX950 \
// RUN:     --implicit-check-not="Diagnostic error:" \
// RUN:     --implicit-check-not="kpack=2 exceeds max" \
// RUN:     --implicit-check-not="Compilation pipeline failed for config:"

// RUN: rocmlir-gen --operation gemm --arch gfx1250 -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 --perf_config= \
// RUN: | not rocmlir-tuning-driver \
// RUN:     --benchmark-config="gemm:v1:128,128,128,2,1,4,16,1,2,0,0" 2>&1 \
// RUN: | FileCheck %s --check-prefix=GEMM-TOO-LARGE-GFX1250 \
// RUN:     --implicit-check-not="Diagnostic error:" \
// RUN:     --implicit-check-not="kpack=2 exceeds max" \
// RUN:     --implicit-check-not="Compilation pipeline failed for config:"

// RUN: rocmlir-gen --operation gemm --arch gfx950 -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 --perf_config= \
// RUN: | not rocmlir-tuning-driver \
// RUN:     --benchmark-config="gemm:v1:128,128,128,0,1,4,16,1,2,0,0" 2>&1 \
// RUN: | FileCheck %s --check-prefix=GEMM-NON-POSITIVE \
// RUN:     --implicit-check-not="Diagnostic error:" \
// RUN:     --implicit-check-not="kpack=0 must be positive" \
// RUN:     --implicit-check-not="Compilation pipeline failed for config:"

// Attention (gemm-gemm) on a WMMA arch: gfx1250's max kpack is 1, so a
// benchmark-config with kpack=2 must be classified as `N/A` rather than a
// compilation failure. This exercises the `validRangeGemmGemmParamsWMMA`
// path in `getAccelRangeGemmGemm`, which fixes kPack to {1} for WMMA.

// RUN: rocmlir-gen --operation attention --arch gfx1250 -t f16 \
// RUN:   -seq_len_q 256 -seq_len_k 256 -head_dim_qk 64 -head_dim_v 64 \
// RUN:   --perf_config= \
// RUN: | not rocmlir-tuning-driver \
// RUN:     --benchmark-config="attn:v1:64,64,32,2,1,1,0,1,2,0,0" 2>&1 \
// RUN: | FileCheck %s --check-prefix=ATTN-TOO-LARGE-GFX1250 \
// RUN:     --implicit-check-not="Diagnostic error:" \
// RUN:     --implicit-check-not="kpack=2 exceeds max" \
// RUN:     --implicit-check-not="Compilation pipeline failed for config:"

// GEMM-TOO-LARGE-GFX950: gemm:v1:128,128,128,2,1,4,16,1,2,0,0{{[[:space:]]+}}N/A
// GEMM-TOO-LARGE-GFX1250: gemm:v1:128,128,128,2,1,4,16,1,2,0,0{{[[:space:]]+}}N/A
// GEMM-NON-POSITIVE: gemm:v1:128,128,128,0,1,4,16,1,2,0,0{{[[:space:]]+}}N/A
// ATTN-TOO-LARGE-GFX1250: attn:v1:64,64,32,2,1,1,0,1,2,0,0{{[[:space:]]+}}N/A
