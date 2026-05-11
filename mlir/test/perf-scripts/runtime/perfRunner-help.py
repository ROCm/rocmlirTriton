# Smoke test: ``perfRunner.py --help`` parses cleanly and advertises the op
# list. Lives under ``runtime/`` because perfRunner.main() calls
# ``get_arch()`` (hip device probe) before argparse, so even ``--help``
# needs a visible AMD GPU.
#
# RUN: perfRunner.py --help | FileCheck %s
#
# CHECK: usage: rocMLIR performance test runner
# CHECK-DAG: --op {{.*}}{conv,gemm,fusion,attention,gemm_gemm,conv_gemm}
# CHECK-DAG: --batch_mlir
# CHECK-DAG: --configs_file
# CHECK-DAG: --tuning_db
# CHECK-DAG: --mlir-build-dir
