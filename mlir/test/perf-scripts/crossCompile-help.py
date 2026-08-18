# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Smoke test: ``crossCompile.py --help`` advertises the remote-host options and
# the tuning flags it forwards to tuningRunner.py. Also guards the import
# surface: crossCompile.py pulls in tuningArgumentUtils, so this fails if
# ci-performance-scripts stops deploying one of them next to the other.
#
# Doesn't need a GPU: crossCompile.py drives the compile host and never touches
# HIP itself.
#
# RUN: crossCompile.py --help | FileCheck %s
#
# CHECK: usage: crossCompile.py
# CHECK-DAG: --configs-file
# CHECK-DAG: --remote-host
# CHECK-DAG: --remote-user
# CHECK-DAG: --remote-artifacts-dir
# CHECK-DAG: --remote-repo-dir
# CHECK-DAG: --remote-build-dir
# CHECK-DAG: --remote-docker-container
# CHECK-DAG: --remote-setup-command
# CHECK-DAG: --local-build-dir
# CHECK-DAG: --local-artifacts-dir
# CHECK-DAG: --gpus
# CHECK-DAG: --gpu-run-timeout
# CHECK-DAG: --ssh
# CHECK-DAG: --ssh-option
# CHECK-DAG: --tar
# CHECK-DAG: --remote-ready-timeout
# CHECK-DAG: --remote-session-timeout
# CHECK-DAG: --dry-run
#
# Target identity is mandatory here: the compile host is GPU-less, so
# tuningRunner.py cannot discover it.
# CHECK-DAG: --target-arch
# CHECK-DAG: --target-num-cu
# CHECK-DAG: --target-num-chiplets
# CHECK-DAG: --tuning-space
# CHECK: full HIP
# CHECK-NEXT: gcnArchName including target features
# CHECK-NEXT: gfx950:sramecc+:xnack-
