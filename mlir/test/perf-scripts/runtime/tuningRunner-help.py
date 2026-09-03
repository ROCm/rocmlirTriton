# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Smoke test: ``tuningRunner.py --help`` parses cleanly and advertises the
# op + tuning-space lists. Lives under ``runtime/`` because
# tuningRunner.main() calls ``get_arch()`` before argparse, so even
# ``--help`` needs a visible AMD GPU.
#
# RUN: tuningRunner.py --help | FileCheck %s
#
# CHECK: usage: rocmlirTriton tuning runner
# CHECK-DAG: --op {{.*}}{conv,gemm,fusion,attention,gemm_gemm,conv_gemm}
# CHECK-DAG: --tuning-space {quick,full,exhaustive,lfbo,llm,llm-lfbo}
# CHECK-DAG: --search-effort {quick,full}
# CHECK-DAG: language-model search:
# CHECK-DAG: --llm-rounds N
# CHECK-DAG: --configs_file
# CHECK-DAG: --mlir-build-dir
# CHECK-DAG: -DLLVM_ENABLE_ZSTD=FORCE_ON
