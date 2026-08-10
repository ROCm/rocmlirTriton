#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Command-line surface of tuningRunner.py: argparse contracts, error
formatting and the op-to-config-class mapping.

parse_arguments takes the GPU topology as an argument, so a stub topology
stands in for rocm-smi and these checks run without a GPU. The runtime
counterpart that actually tunes a kernel lives in ``runtime/``.

# RUN: %python %s
"""

import contextlib
import io
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

MLIR_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = MLIR_DIR / "utils" / "performance"
sys.path.insert(0, str(PERF_DIR))

import tuningRunner  # noqa: E402
from tuningRunner import (  # noqa: E402
    NumaTopology, Operation, format_error, get_config_class)


@dataclass(frozen=True)
class StubGpu:
    gpu_id: int
    sku: str
    numa_node: int


class StubGpuTopology:
    """The slice of GpuTopology parse_arguments uses, without rocm-smi."""

    def __init__(self, gpu_ids_and_skus):
        self.gpus = {
            gpu_id: StubGpu(gpu_id=gpu_id, sku=sku, numa_node=0) for gpu_id, sku in gpu_ids_and_skus
        }

    def get_numa_node(self, gpu_id: int) -> int:
        return self.gpus[gpu_id].numa_node

    def validate_homogeneity(self, gpu_ids) -> bool:
        if len(gpu_ids) <= 1:
            return True
        return len({self.gpus[gpu_id].sku for gpu_id in gpu_ids}) == 1


class FormatErrorTest(unittest.TestCase):
    """Tests for format_error."""

    def test_basic(self):
        self.assertIn("Something failed", format_error("Something failed"))

    def test_with_exit_code(self):
        self.assertIn("Exit code: 1", format_error("Failed", exit_code=1))

    def test_with_command_and_gpu(self):
        msg = format_error("Failed", command="rocmlir-gen -c", gpu_id=0)
        self.assertIn("ROCR_VISIBLE_DEVICES=0", msg)
        self.assertIn("rocmlir-gen", msg)

    def test_truncate_long_output(self):
        msg = format_error("Failed", stderr="line\n" * 20, max_lines=5)
        self.assertIn("omitted", msg)


class GetConfigClassTest(unittest.TestCase):
    """Tests for get_config_class."""

    def test_known_ops(self):
        self.assertEqual(get_config_class(Operation.CONV).__name__, "ConvConfiguration")
        self.assertEqual(get_config_class(Operation.GEMM).__name__, "GemmConfiguration")
        self.assertEqual(get_config_class(Operation.ATTENTION).__name__, "AttentionConfiguration")
        self.assertEqual(get_config_class(Operation.GEMM_GEMM).__name__, "GemmGemmConfiguration")
        self.assertEqual(get_config_class(Operation.CONV_GEMM).__name__, "ConvGemmConfiguration")

    def test_fusion_raises(self):
        with self.assertRaisesRegex(ValueError, "No config class"):
            get_config_class(Operation.FUSION)


class ParseArgumentsTest(unittest.TestCase):
    """Tests for parse_arguments with a stub GPU topology."""

    def parse(self, argv):
        topology = StubGpuTopology([(0, "gfx900")])
        return tuningRunner.parse_arguments(topology, [0], argv)

    def assert_rejects(self, argv):
        """Assert argparse exits, returning what it wrote to stderr."""
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            self.parse(argv)
        return stderr.getvalue()

    def test_required_op_and_config_group(self):
        self.assert_rejects(["--op", "gemm"])
        self.assert_rejects(["-c", "configs.txt"])

    def test_fusion_requires_test_dir(self):
        self.assert_rejects(["--op", "fusion", "-c", "dummy.txt"])

    def test_valid_gemm_single_config(self):
        parsed = self.parse(
            ["--op", "gemm", "--config", "-g 1 -m 1024 -k 769 -n 512 -t f32", "-o", "/tmp/out.tsv"])
        self.assertEqual(parsed.op, "gemm")
        self.assertEqual(parsed.config, "-g 1 -m 1024 -k 769 -n 512 -t f32")
        self.assertEqual(parsed.output, "/tmp/out.tsv")

    def test_tuning_space_choices(self):
        parsed = self.parse(
            ["--op", "gemm", "--config", "-g 1 -m 1024 -k 769 -n 512", "--tuning-space", "quick"])
        self.assertEqual(parsed.tuning_space, "quick")

    def test_verify_timeout_defaults_to_constant(self):
        parsed = self.parse(["--op", "gemm", "--config", "-g 1 -m 1024 -k 769 -n 512"])
        self.assertEqual(parsed.verify_timeout, tuningRunner.DEFAULT_VERIFY_TIMEOUT_SECONDS)

    def test_verify_timeout_override(self):
        parsed = self.parse(
            ["--op", "gemm", "--config", "-g 1 -m 1024 -k 769 -n 512", "--verify-timeout", "1800"])
        self.assertEqual(parsed.verify_timeout, 1800)

    def test_negative_gpu_run_timeout_rejected(self):
        stderr = self.assert_rejects(
            ["--op", "gemm", "--config", "-g 1 -m 1024 -k 769 -n 512", "--gpu-run-timeout", "-1"])
        self.assertIn("argument --gpu-run-timeout: must be non-negative", stderr)


class NumaTopologyParseCpuListTest(unittest.TestCase):
    """Tests for NumaTopology._parse_cpu_list (used when discovering NUMA)."""

    def test_single_range(self):
        self.assertEqual(NumaTopology._parse_cpu_list("0-3"), [0, 1, 2, 3])

    def test_comma_separated(self):
        self.assertEqual(NumaTopology._parse_cpu_list("0,2,4"), [0, 2, 4])

    def test_mixed(self):
        self.assertEqual(NumaTopology._parse_cpu_list("0-2,5,10-11"), [0, 1, 2, 5, 10, 11])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
