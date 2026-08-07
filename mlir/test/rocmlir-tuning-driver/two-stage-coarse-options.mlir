// Verifies that the --coarse-* options actually take effect.

// RUN: rocmlir-gen --arch %arch -operation gemm -t f16 -out_datatype f32 -g 1 -m 1024 -k 1024 -n 1024 -transA=False -transB=False --perf_config= \
// RUN: | rocmlir-tuning-driver --tuning-space=quick --rep=10 --warmup=3 --sleep-us=100 --use-median --show-all-measurements=false \
// RUN:     --two-stage-topk=2 --coarse-rep-iters=1234 --coarse-min-rep-iters=77 \
// RUN:     --coarse-rel-sem-target=0.042 --coarse-chunk-iters=11 \
// RUN:     --coarse-warmup-iters=9 --coarse-warmup-floor-ms=3 \
// RUN: 2>&1 >/dev/null | FileCheck %s

// CHECK: Two-stage tuning budget: rep-iters=1234 min-rep-iters=77 rel-sem-target=0.042 chunk-iters=11 warmup-iters=9 warmup-floor-ms=3

// A warmup floor above --warmup would be silently capped away, so it is
// rejected up front rather than ignored.

// RUN: rocmlir-gen --arch %arch -operation gemm -t f16 -out_datatype f32 -g 1 -m 1024 -k 1024 -n 1024 -transA=False -transB=False --perf_config= \
// RUN: | not rocmlir-tuning-driver --tuning-space=quick --rep=10 --warmup=3 \
// RUN:     --two-stage-topk=2 --coarse-warmup-floor-ms=4 \
// RUN: 2>&1 >/dev/null | FileCheck %s --check-prefix=FLOOR

// FLOOR: coarse-warmup-floor-ms must not exceed warmup
