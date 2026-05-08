# End-to-end smoke test for perfRunner.py: run a tiny f32 GEMM in MLIR-only
# mode (``-b``) so we don't need MIOpen / hipBLASLt drivers. Validates the
# perfRunner pipeline (rocmlir-gen -ph + rocmlir-driver -c + rocprof
# parsing) without depending on external benchmark libraries.
#
# RUN: echo '-t f32 -out_datatype f32 -transA false -transB false -g 1 -m 64 -n 64 -k 64' > %t.cfg
# RUN: perfRunner.py --op gemm -b --configs_file=%t.cfg \
# RUN:     --data-type f32 -o %t.csv 2>&1 | FileCheck %s
#
# CHECK: Running MLIR Benchmark
# CHECK: TFlops
