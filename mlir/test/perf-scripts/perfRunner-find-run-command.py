#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for find_run_command.

The weekly `Tune Fusion` stage extracts its configs with find_run_command,
which stops at the first RUN line that runs the kernel. A RUN line it does not
recognize leaves `tuningRunner.py --op fusion` with no configs to load, so the
stage aborts. The fusion tests end in rocm-run; older ones end in mlir-runner.

# RUN: %python %s
"""

import glob
import os
import shutil
import sys
import tempfile
import unittest

# perfRunner.py is on PATH (lit's mlir_rock_tools_dir, populated by
# ci-performance-scripts). Import it from there rather than from the source
# tree: it depends on the compiled amd_arch_db binding, which only exists
# alongside the deployed scripts.
_script = shutil.which('perfRunner.py')
if _script is None:
    sys.exit("perfRunner.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

from perfRunner import find_run_command  # noqa: E402

TEST_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
# The directories the weekly `Tune Fusion` stage passes to --test-dir.
WEEKLY_FUSION_DIRS = ("fusion/resnet50-e2e", "xmir/bert-torch-tosa-e2e")

# Split so that lit does not take it for a RUN line of this test.
RUN_PREFIX = "// RUN" + ":"
GEN = "rocmlir-gen --clone-harness -arch %arch -fut test %s"
DRIVER = ("rocmlir-driver -kernel-pipeline migraphx,highlevel "
          "-host-pipeline migraphx,highlevel -arch %arch")


class FindRunCommandTest(unittest.TestCase):

    def find(self, run_line):
        with tempfile.NamedTemporaryFile('w', suffix='.mlir', delete=False) as f:
            f.write("// Copyright Advanced Micro Devices, Inc.\n")
            f.write(f"{RUN_PREFIX} {run_line}\n")
            path = f.name
        try:
            return find_run_command(path)
        finally:
            os.remove(path)

    def test_rocm_run_line(self):
        line = f"{GEN} | {DRIVER} | rocmlir-driver -c | rocm-run | FileCheck %s"
        self.assertEqual(self.find(line), (DRIVER, "test"))

    def test_mlir_runner_line(self):
        line = f"{GEN} | {DRIVER} | rocmlir-driver -c | mlir-runner --shared-libs=libfoo.so"
        self.assertEqual(self.find(line), (DRIVER, "test"))

    def test_line_without_runner(self):
        self.assertEqual(self.find(f"{GEN} | {DRIVER} | FileCheck %s"), (None, None))

    def test_weekly_fusion_dirs(self):
        """Every test the weekly fusion tuning reads yields a command."""
        for test_dir in WEEKLY_FUSION_DIRS:
            files = sorted(glob.glob(os.path.join(TEST_ROOT, test_dir, "*.mlir")))
            self.assertTrue(files, test_dir)
            for path in files:
                with self.subTest(path=os.path.relpath(path, TEST_ROOT)):
                    rocmlir_cmd, fut_name = find_run_command(path)
                    self.assertIsNotNone(rocmlir_cmd)
                    self.assertIsNotNone(fut_name)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
