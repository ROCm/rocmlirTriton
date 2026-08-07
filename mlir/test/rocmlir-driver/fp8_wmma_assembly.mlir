// Verify the public FP8/BF8 aliases select gfx117x's four OCP WMMA variants.
// Rock's GEMM lowering maps the generator's logical A/B buffers to the WMMA
// instruction's B/A operands, so the two mixed instruction suffixes are
// intentionally reversed relative to the generator type pair.

// RUN: rocmlir-gen --arch gfx1170 --operation gemm -t fp8_fp8 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=FP8-FP8
// RUN: rocmlir-gen --arch gfx1170 --operation gemm -t fp8_bf8 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=BF8-FP8
// RUN: rocmlir-gen --arch gfx1170 --operation gemm -t bf8_fp8 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=FP8-BF8
// RUN: rocmlir-gen --arch gfx1170 --operation gemm -t bf8_bf8 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=BF8-BF8

// FP8-FP8: v_wmma_f32_16x16x16_fp8_fp8
// FP8-BF8: v_wmma_f32_16x16x16_fp8_bf8
// BF8-FP8: v_wmma_f32_16x16x16_bf8_fp8
// BF8-BF8: v_wmma_f32_16x16x16_bf8_bf8
