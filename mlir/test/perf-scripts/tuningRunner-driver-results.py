#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""How tuningRunner.py turns rocmlir-tuning-driver output into a result.

find_best_perfconfig picks the winner out of the driver's
``<perfconfig>\\t<nanoseconds>`` lines, and tune_config classifies the driver's
exit status -- in particular GPU_TIMEOUT_EXIT_CODE, which means a kernel hung
on the device and the config must be recorded as timed out rather than slow.
The driver is stubbed out here, so no GPU is needed; ``runtime/`` holds the
test that really tunes a kernel.

# RUN: %python %s
"""

import os
import shutil
import sys
import unittest
from unittest.mock import MagicMock, patch

# tuningRunner.py is deployed next to perfRunner.py under ci-performance-scripts
# and depends on the compiled amd_arch_db binding in that directory.
_script = shutil.which('perfRunner.py')
if _script is None:
    sys.exit("perfRunner.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

import tuningRunner  # noqa: E402
from tuningRunner import NumaNodeLock, find_best_perfconfig, tune_config  # noqa: E402


class FindBestPerfconfigTest(unittest.TestCase):
    """Tests for find_best_perfconfig with a stubbed config."""

    @staticmethod
    def make_options():
        options = MagicMock()
        options.debug = False
        options.verify_all_perfconfigs = False
        return options

    def test_empty_lines_returns_none_winner(self):
        config = MagicMock()
        config.table_entry.return_value = {"TFlops": float("nan")}

        winner, tflops, entries = find_best_perfconfig([],
                                                       config,
                                                       MagicMock(),
                                                       self.make_options(),
                                                       gpu_id=0,
                                                       numa_lock=NumaNodeLock())

        self.assertIsNone(winner)
        self.assertIsNone(tflops)
        self.assertEqual(entries, [])

    def test_single_valid_line(self):
        config = MagicMock()
        config.table_entry.return_value = {"TFlops": 1.5}

        winner, tflops, entries = find_best_perfconfig(["perf_cfg_1\t12345"],
                                                       config,
                                                       MagicMock(),
                                                       self.make_options(),
                                                       gpu_id=0,
                                                       numa_lock=NumaNodeLock())

        self.assertEqual(winner, "perf_cfg_1")
        self.assertEqual(tflops, 1.5)
        self.assertEqual(len(entries), 1)


class StubConfiguration:
    """Config class whose instances only need to produce rocmlir-gen flags."""

    @staticmethod
    def from_command_line(command_line, arch, num_cu, num_chiplets):
        return StubConfiguration()

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags, kernel_repeats=None):
        return "--fake-rocmlir-gen-arg"


class TuneConfigTest(unittest.TestCase):
    """Tests for tune_config's handling of the driver's exit status."""

    @staticmethod
    def make_paths():
        paths = MagicMock()
        paths.mlir_paths.rocmlir_gen_path = "rocmlir-gen"
        paths.mlir_paths.rocmlir_tuning_driver_path = "rocmlir-tuning-driver"
        return paths

    @staticmethod
    def make_options():
        options = MagicMock()
        options.tuning_space_kind = "quick"
        options.debug = False
        options.verbose = False
        options.wait_for_compiles = False
        options.flush_last_level_cache = False
        options.gpu_run_timeout = 30
        options.perf_config_timeout = 0
        options.timeout = None
        options.arch = "gfx900"
        options.num_cu = 64
        options.num_chiplets = 1
        options.rocmlir_gen_flags = ""
        options.rep_ms = 100
        options.warmup_ms = 100
        options.two_stage_topk = 0
        return options

    def test_gpu_timeout_exit_code_marks_result_gpu_timed_out(self):

        def stub_run_pipeline(commands, env=None, timeout=None, cwd=None):
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0][0], "rocmlir-gen")
            self.assertEqual(commands[1][0], "rocmlir-tuning-driver")
            return tuningRunner.GPU_TIMEOUT_EXIT_CODE, "", "gpu timeout"

        with patch.object(tuningRunner, "_run_pipeline", stub_run_pipeline):
            result = tune_config("-g 1 -m 1024 -k 769 -n 512",
                                 StubConfiguration,
                                 self.make_paths(),
                                 self.make_options(),
                                 gpu_id=0,
                                 num_compile_threads=1,
                                 numa_lock=NumaNodeLock())

        self.assertFalse(result.success)
        self.assertTrue(result.gpu_timed_out)
        self.assertEqual(result.gpu_id, 0)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
