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

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

MLIR_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = MLIR_DIR / "utils" / "performance"
sys.path.insert(0, str(PERF_DIR))

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


class StubProcess:
    """Stand-in for a Popen object with a fixed exit status."""

    def __init__(self, returncode, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.pid = 1234
        self.stdout = MagicMock()
        self._output = (stdout, stderr)

    def communicate(self, timeout=None):
        return self._output

    def kill(self):
        pass

    def wait(self, timeout=None):
        return self.returncode


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
        options.wait_for_compiles = False
        options.gpu_run_timeout = 30
        options.timeout = None
        options.arch = "gfx900"
        options.num_cu = 64
        options.num_chiplets = 1
        options.rocmlir_gen_flags = ""
        return options

    def test_gpu_timeout_exit_code_marks_result_gpu_timed_out(self):
        rocmlir_gen = StubProcess(returncode=0)
        tuning_driver = StubProcess(returncode=tuningRunner.GPU_TIMEOUT_EXIT_CODE,
                                    stderr=b"gpu timeout")

        def stub_popen(command, **kwargs):
            if command[0] == "rocmlir-gen":
                return rocmlir_gen
            self.assertEqual(command[0], "rocmlir-tuning-driver")
            return tuning_driver

        with patch.object(tuningRunner.subprocess, "Popen", stub_popen):
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
