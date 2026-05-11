# End-to-end smoke test for tuningRunner.py: tune a tiny f32 GEMM with
# ``--tuning-space=quick``. Drives rocmlir-gen | rocmlir-tuning-driver and
# verifies the winning perf-config against the CPU reference, exercising
# every Python-side step (config parsing, subprocess orchestration, result
# parsing) end-to-end.
#
# RUN: tuningRunner.py --op gemm --tuning-space=quick \
# RUN:     --config='-g 1 -m 64 -n 64 -k 64 -t f32 -out_datatype f32 -transA 0 -transB 0' \
# RUN:     -q -o %t.tsv 2>&1 | FileCheck %s
#
# CHECK: Tuned and verified
# CHECK-SAME: gemm:v1:
