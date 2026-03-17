// Verify that the Triton pipeline correctly propagates numCTAs from
// the perf_config into the ttg.num-ctas module attribute.

// gfx1250 with explicit perf_config setting numCTAs=2
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f16 -p --perf_config "gemm:v1:64,64,64,1,2,4,16,1,2,0,0" | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=GFX1250_CTA2

// gfx1250 with default numCTAs=1 for comparison
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f16 -p --perf_config "gemm:v1:64,64,64,1,1,4,16,1,2,0,0" | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=GFX1250_CTA1

// gfx942 with default numCTAs=1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=GFX942

// GFX1250_CTA2: "ttg.num-ctas" = 2 : i32
// GFX1250_CTA1: "ttg.num-ctas" = 1 : i32
// GFX942: "ttg.num-ctas" = 1 : i32
