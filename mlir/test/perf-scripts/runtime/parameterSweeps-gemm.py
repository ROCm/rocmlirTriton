# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# GEMM companion to ``parameterSweeps-conv.py``: same harness, same
# rationale, exercises the GemmConfiguration code path.
#
# RUN: parameterSweeps.py gemm --samples 1 --seed 0 --jobs 1 --quiet \
# RUN:     --test-timeout-sec 20 > %t 2>&1 || true
# RUN: FileCheck %s < %t
#
# CHECK: Passed: {{[0-9]+}}, Not applicable: {{[0-9]+}}, Timed out: {{[0-9]+}}, Failed: {{[0-9]+}}
