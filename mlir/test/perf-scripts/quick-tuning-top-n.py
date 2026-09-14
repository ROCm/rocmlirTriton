#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python tests for per-problem top-N selection.

# RUN: %python %s
"""

import contextlib
import io
from pathlib import Path
import sys
import types
import unittest

import pandas as pd

MLIR_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MLIR_DIR / "utils" / "performance" / "analysis"))
sys.modules.setdefault("pulp", types.SimpleNamespace())

from quickTuningGen import find_problem_topn  # noqa: E402


def config(name, split_k):
    return (f"gemm:mPerBlock={name},nPerBlock=64,kPerBlock=32,kpack=8,"
            f"numCTAs=1,numWaves=4,matrixInstrNonkdim=16,splitKFactor={split_k}")


def select(rows, top_n=3):
    frame = pd.DataFrame([{
        "DataType": "f16",
        "ProblemHash": "0x1234",
        "PerfConfig": perfconfig,
        "TFlops": tflops,
    } for perfconfig, tflops in rows])
    return find_problem_topn(frame, top_n)["f16"][0x1234]


class QuickTuningTopNTest(unittest.TestCase):

    def test_keeps_descending_top_n_when_non_split_k_is_present(self):
        configs = [config(128, 4), config(64, 2), config(32, 1), config(16, 1)]
        self.assertEqual(select(list(zip(configs, [100, 90, 80, 70]))), configs[:3])

    def test_best_non_split_k_displaces_nth_slot(self):
        configs = [config(128, 4), config(64, 2), config(32, 8), config(16, 1)]
        self.assertEqual(select(list(zip(configs, [100, 90, 80, 70]))),
                         [configs[0], configs[1], configs[3]])

    def test_warns_when_no_non_split_k_was_measured(self):
        configs = [config(128, 4), config(64, 2), config(32, 8)]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(select(list(zip(configs, [100, 90, 80]))), configs)
        self.assertIn("no splitKFactor=1 config measured", stdout.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
