#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Data-independent tests for per-problem quick-tuning generation.

# RUN: %python %s
"""

from pathlib import Path
import sys
import unittest

import pandas as pd

MLIR_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MLIR_DIR / "utils" / "performance" / "analysis"))

from quickTuningProblemGen import generate, select_problem_configs  # noqa: E402


def perfconfig(tile, split_k):
    return (f"gemm:mPerBlock={tile},nPerBlock=64,kPerBlock=32,kpack=1,numCTAs=1,"
            f"numWaves=4,matrixInstrNonkdim=16,splitKFactor={split_k}")


class QuickTuningProblemGenTest(unittest.TestCase):

    def test_top_n_is_ordered_by_measurement(self):
        rows = pd.DataFrame({
            'PerfConfig': [perfconfig(32, 1), perfconfig(64, 1), perfconfig(128, 1)],
            'TFlops': [1.0, 3.0, 2.0],
        })
        configs, missing = select_problem_configs(rows, 'gemm', 2)
        self.assertEqual(configs, [perfconfig(64, 1), perfconfig(128, 1)])
        self.assertFalse(missing)

    def test_last_slot_is_reserved_for_non_split_k(self):
        rows = pd.DataFrame({
            'PerfConfig': [
                perfconfig(32, 8),
                perfconfig(64, 4),
                perfconfig(128, 2),
                perfconfig(256, 1),
            ],
            'TFlops': [4.0, 3.0, 2.0, 1.0],
        })
        configs, missing = select_problem_configs(rows, 'gemm', 3)
        self.assertEqual(
            configs,
            [perfconfig(32, 8), perfconfig(64, 4), perfconfig(256, 1)])
        self.assertFalse(missing)

    def test_missing_non_split_k_keeps_measured_leaders(self):
        rows = pd.DataFrame({
            'PerfConfig': [perfconfig(32, 4), perfconfig(64, 2)],
            'TFlops': [2.0, 1.0],
        })
        configs, missing = select_problem_configs(rows, 'conv', 2)
        self.assertEqual(configs, [perfconfig(32, 4), perfconfig(64, 2)])
        self.assertTrue(missing)

    def test_generated_fixture_is_sorted_and_interns_configs(self):
        shared = perfconfig(64, 1)
        content, summary = generate({
            'gfx000_gemm_f32': ('gemm', {
                'problem-b': [shared, perfconfig(128, 1)],
                'problem-a': [shared],
            })
        })
        self.assertLess(content.index('"problem-a"'), content.index('"problem-b"'))
        self.assertEqual(content.count(f'"{shared}"'), 1)
        self.assertEqual(summary[0][1:3], (2, 2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
