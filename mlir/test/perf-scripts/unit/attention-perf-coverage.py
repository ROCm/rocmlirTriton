# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: rm -f %t.coverage
# RUN: env COVERAGE_FILE=%t.coverage PYTHONPATH=%S/../../../utils/performance %python -m coverage run --branch --source=attentionPerfUtils,filterAttentionConfigs,runAttentionBranchBenchmark,compareAttentionPerformance,generateAttentionPerformanceWorkbook -m unittest discover -s %S -p 'test_*.py' -v
# RUN: env COVERAGE_FILE=%t.coverage %python -m coverage report --show-missing --fail-under=100
