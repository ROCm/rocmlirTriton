# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# End-to-end smoke test: drive one random conv config through
# rocmlir-gen | rocmlir-driver -c | rocm-run. ``--test-timeout-sec 20`` caps
# each sub-stage so the test fits in lit's per-test budget. ``> %t || true``
# decouples the script's exit code from FileCheck's verdict (lit pipes are
# pipefail, and timeouts/FAILs are an acceptable outcome here — we only
# care that ``run_config`` reached its summary line).
#
# RUN: parameterSweeps.py conv --samples 1 --seed 0 --jobs 1 --quiet \
# RUN:     --test-timeout-sec 20 > %t 2>&1 || true
# RUN: FileCheck %s < %t
#
# CHECK: Passed: {{[0-9]+}}, Not applicable: {{[0-9]+}}, Timed out: {{[0-9]+}}, Failed: {{[0-9]+}}
