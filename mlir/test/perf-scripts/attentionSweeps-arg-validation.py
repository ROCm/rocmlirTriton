# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Argument-validation smoke tests for attentionSweeps.py. attentionSweeps
# reuses ``add_common_args`` from parameterSweeps, so spotchecking one
# common flag here is enough to confirm it's still wired through; the
# positional choice list is its own thing and gets its own check.
#
# RUN: not attentionSweeps.py invalid-kind 2>&1 | FileCheck %s --check-prefix=BAD-KIND
# BAD-KIND: invalid choice: 'invalid-kind'
#
# RUN: not attentionSweeps.py --samples 0 attention 2>&1 | FileCheck %s --check-prefix=NONPOSITIVE-SAMPLES
# NONPOSITIVE-SAMPLES: must be > 0
