# Smoke test: ``parameterSweeps.py --help`` advertises the right kinds and the
# common flags shared with attentionSweeps. Catches regressions in the
# positional choices and ``add_common_args`` flag set without needing a GPU.
#
# RUN: parameterSweeps.py --help | FileCheck %s
#
# CHECK: usage: parameterSweeps.py
# CHECK: positional arguments:
# CHECK:   {conv,gemm}
# CHECK-DAG: --debug
# CHECK-DAG: --quiet
# CHECK-DAG: --log-failures
# CHECK-DAG: --jobs
# CHECK-DAG: --samples
# CHECK-DAG: --seed
# CHECK-DAG: --test-timeout-sec
# CHECK-DAG: --max-timeouts
# CHECK-DAG: --mlir-build-dir
