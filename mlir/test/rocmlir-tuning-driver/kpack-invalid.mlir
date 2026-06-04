// Verifies that perf configs carrying unsupported kpack values are hard-failed
// (with a clear diagnostic) by the tuning driver rather than silently
// classified as `N/A`. Covers both the gemm and the attention tuning paths on
// arches where the maximum supported kpack is 1 (gfx950 and gfx1250).

// RUN: rocmlir-gen --operation gemm --arch gfx950 -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 --perf_config= \
// RUN: | not rocmlir-tuning-driver \
// RUN:     --benchmark-config="gemm:v1:128,128,128,2,1,4,16,1,2,0,0" 2>&1 \
// RUN: | FileCheck %s --check-prefix=GEMM-TOO-LARGE-GFX950 \
// RUN:     --implicit-check-not="N/A"

// RUN: rocmlir-gen --operation gemm --arch gfx1250 -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 --perf_config= \
// RUN: | not rocmlir-tuning-driver \
// RUN:     --benchmark-config="gemm:v1:128,128,128,2,1,4,16,1,2,0,0" 2>&1 \
// RUN: | FileCheck %s --check-prefix=GEMM-TOO-LARGE-GFX1250 \
// RUN:     --implicit-check-not="N/A"

// RUN: rocmlir-gen --operation gemm --arch gfx950 -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 --perf_config= \
// RUN: | not rocmlir-tuning-driver \
// RUN:     --benchmark-config="gemm:v1:128,128,128,0,1,4,16,1,2,0,0" 2>&1 \
// RUN: | FileCheck %s --check-prefix=GEMM-NON-POSITIVE \
// RUN:     --implicit-check-not="N/A"

// Attention (gemm-gemm) on a WMMA arch: gfx1250's max kpack is 1, so a
// benchmark-config with kpack=2 must be hard-failed rather than silently
// classified as `N/A`. This exercises `validatePerfConfig` reached via the
// `RockGemmGemmWrapperInterface` path in AffixTuningParameters.

// RUN: rocmlir-gen --operation attention --arch gfx1250 -t f16 \
// RUN:   -seq_len_q 256 -seq_len_k 256 -head_dim_qk 64 -head_dim_v 64 \
// RUN:   --perf_config= \
// RUN: | not rocmlir-tuning-driver \
// RUN:     --benchmark-config="attn:v1:64,64,32,2,1,1,0,1,2,0,0" 2>&1 \
// RUN: | FileCheck %s --check-prefix=ATTN-TOO-LARGE-GFX1250 \
// RUN:     --implicit-check-not="N/A"

// GEMM-TOO-LARGE-GFX950:  Compilation pipeline failed for config: gemm:v1:128,128,128,2,1,4,16,1,2,0,0
// GEMM-TOO-LARGE-GFX950:  Diagnostic error: kpack=2 exceeds max (1) for amdgcn-amd-amdhsa:gfx950

// GEMM-TOO-LARGE-GFX1250: Compilation pipeline failed for config: gemm:v1:128,128,128,2,1,4,16,1,2,0,0
// GEMM-TOO-LARGE-GFX1250: Diagnostic error: kpack=2 exceeds max (1) for amdgcn-amd-amdhsa:gfx1250

// GEMM-NON-POSITIVE:      Compilation pipeline failed for config: gemm:v1:128,128,128,0,1,4,16,1,2,0,0
// GEMM-NON-POSITIVE:      Diagnostic error: kpack=0 must be positive

// ATTN-TOO-LARGE-GFX1250: Compilation pipeline failed for config: attn:v1:64,64,32,2,1,1,0,1,2,0,0
// ATTN-TOO-LARGE-GFX1250: Diagnostic error: kpack=2 exceeds max (1) for amdgcn-amd-amdhsa:gfx1250
