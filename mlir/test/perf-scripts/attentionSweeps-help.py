# Smoke test: ``attentionSweeps.py --help`` advertises the right kinds. The
# spelling here matches ``rocmlir-gen --operation`` (``attention``,
# ``gemm_gemm``); a typo on either side would silently break the weekly sweep.
#
# RUN: attentionSweeps.py --help | FileCheck %s
#
# CHECK: usage: attentionSweeps.py
# CHECK: positional arguments:
# CHECK:   {attention,gemm_gemm}
# CHECK-DAG: --samples
# CHECK-DAG: --seed
# CHECK-DAG: --jobs
# CHECK-DAG: --log-failures
