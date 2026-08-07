// RUN: rocmlir-gen --arch %arch -operation gemm -t f16 -out_datatype f32 -g 1 -m 1024 -k 1024 -n 1024 -transA=False -transB=False --perf_config= \
// RUN: | rocmlir-tuning-driver --tuning-space=quick --rep=10 --warmup=1 --sleep-us=100 --use-median --show-all-measurements=false --two-stage-topk=2 \
// RUN:     --coarse-warmup-floor-ms=0 \
// RUN: > %t.txt
// RUN: FileCheck %s < %t.txt
// RUN: FileCheck %s --check-prefix=OUTSIDETOPK < %t.txt

// We specified --two-stage-topk=2, so EXACTLY 2 configs
// should be measured:
// CHECK-COUNT-2: {{gemm:v5:[^[:space:]]+[\t ]+[0-9]}}
// CHECK-NOT: {{gemm:v5:[^[:space:]]+[\t ]+[0-9]}}

// The rest of the configs compiled and ran at the coarse budget but not in
// topK, so they must be reported as `Discarded`.

// OUTSIDETOPK: {{gemm:v5:[^[:space:]]+[\t ]+Discarded}}
