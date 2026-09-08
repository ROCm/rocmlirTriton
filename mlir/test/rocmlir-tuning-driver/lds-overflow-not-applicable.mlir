// Verifies that when a perf config makes the kernel use more LDS than the
// hardware supports, the tuning driver classifies the failure as
// `NotApplicable` (printed as `N/A`) instead of a real compilation failure,
// and crucially does NOT surface the `ttg.shared (...) exceeds LDS limit`
// diagnostic emitted by the rock pass to stderr. This guards the
// per-context diagnostic buffering in rocmlir-tuning-driver: pass-level
// errors are captured per-config and only flushed on real failures, so that
// expected "this config doesn't fit" outcomes don't pollute tuning output.
//
// The chosen tile (mPerBlock=nPerBlock=256, kPerBlock=512, numStages=2, kpack=1,
// f32) makes Triton allocate 2097152 bytes (2 MiB) of shared memory, which
// exceeds the LDS budget on every currently supported architecture: gfx90a/942/1030/
// 1100/1200 have 64 KB, gfx950 has 160 KB, gfx1250 has 320 KB.
//
// The driver returns failure when no valid configs remain (only N/A here),
// so we wrap it in `not`.
//
// RUN: rocmlir-gen --operation gemm --arch %arch -t f32 -out_datatype f32 \
// RUN:   -transA=false -transB=false -g 1 -m 256 -n 256 -k 512 --perf_config= \
// RUN: | not rocmlir-tuning-driver \
// RUN:     --benchmark-config="gemm:v1:256,256,512,1,1,4,16,1,2,0,0" 2>&1 \
// RUN: | FileCheck %s \
// RUN:     --implicit-check-not="Diagnostic error:" \
// RUN:     --implicit-check-not="exceeds LDS limit" \
// RUN:     --implicit-check-not="Compilation pipeline failed for config:"

// CHECK: gemm:v1:256,256,512,1,1,4,16,1,2,0,0{{[[:space:]]+}}N/A
