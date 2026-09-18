#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for split-K-aware quick tuning lists.

A problem fused in a way that forbids split-K can only run a splitKFactor=1
config, so the quick list has to contain a good one for every problem shape --
not just for the shapes where split-K happened to lose.

# RUN: %python %s
"""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import pandas as pd

# quickTuningGen.py is an analysis helper that is not deployed, so it comes from
# the source tree. It pulls in perfCommonUtils via its own sys.path setup.
MLIR_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MLIR_DIR / "utils" / "performance" / "analysis"))


def stub_optional_pulp():
    """Let this coverage-only test import quickTuningGen without PuLP installed."""
    sys.modules.setdefault("pulp", types.SimpleNamespace())


stub_optional_pulp()

import quickTuningGen  # noqa: E402
from quickTuningGen import build_coverage, get_target_columns, threshold_for_priority  # noqa: E402

THRESHOLD = 0.93


def gemm_perfconfig(m_per_block, split_k):
    return (f"v2:mPerBlock={m_per_block},nPerBlock=64,kPerBlock=32,kpack=8,numCTAs=1,"
            f"numWaves=4,matrixInstrNonkdim=16,splitKFactor={split_k},numStages=1,"
            "wavesPerEU=0,gridGroupSize=0")


def make_df(op, rows, problem_id=0):
    """Build a debug-style frame for one problem from (perfConfig, TFlops) rows.

    The problem columns only matter for grouping, so they are placeholders; the
    last one carries ``problem_id`` to distinguish separate shapes.
    """
    columns = get_target_columns(op)
    problem = dict.fromkeys(columns, 0)
    problem[columns[-1]] = problem_id
    measurements = [dict(problem, PerfConfig=config, TFlops=tflops) for config, tflops in rows]
    return pd.DataFrame(measurements)


def coverage_for(op, rows, problem_id=0):
    df = make_df(op, rows, problem_id)
    return build_coverage(df, get_target_columns(op), op, THRESHOLD)


class QuickTuningSplitKCoverageTest(unittest.TestCase):

    def test_low_priority_relaxes_the_coverage_threshold(self):
        self.assertEqual(threshold_for_priority(1, THRESHOLD), 0.90)
        self.assertEqual(threshold_for_priority(2, THRESHOLD), 0.91)
        self.assertEqual(threshold_for_priority(3, THRESHOLD), 0.92)
        self.assertEqual(threshold_for_priority(4, THRESHOLD), THRESHOLD)
        self.assertEqual(threshold_for_priority(None, THRESHOLD), THRESHOLD)

    def test_low_priority_adds_nearby_configs_to_coverage(self):
        winner = gemm_perfconfig(64, 1)
        nearby = gemm_perfconfig(128, 1)
        df = make_df('gemm', [(winner, 100.0), (nearby, 91.0)])
        df['PerfPriority'] = 1

        strict = build_coverage(df, get_target_columns('gemm'), 'gemm', THRESHOLD)
        priority_aware = build_coverage(df,
                                        get_target_columns('gemm'),
                                        'gemm',
                                        THRESHOLD,
                                        use_perf_priority=True)

        self.assertEqual(list(strict.values()), [[winner]])
        self.assertEqual(list(priority_aware.values()), [[winner, nearby]])

    def test_find_ignores_priority_when_strict_coverage_fits(self):
        winner = gemm_perfconfig(64, 1)
        nearby = gemm_perfconfig(128, 1)
        df = make_df('gemm', [(winner, 100.0), (nearby, 91.0)])
        df['DataType'] = 'f32'
        df['PerfPriority'] = 1
        seen_coverage = []

        def solve(coverage, _dtype):
            seen_coverage.append(coverage)
            problems = sorted(coverage)
            configs = sorted({config for candidates in coverage.values() for config in candidates})
            config_idx = {config: i for i, config in enumerate(configs)}
            matrix = quickTuningGen.np.ones((len(problems), len(configs)), dtype=int)
            return [configs[0]], problems, configs, config_idx, matrix

        with mock.patch.object(quickTuningGen, 'solve_full_coverage', side_effect=solve):
            quickTuningGen.find_perfconfigs(df, 'gemm', THRESHOLD, max_configs=40)

        self.assertEqual(len(seen_coverage), 1)
        self.assertEqual(list(seen_coverage[0].values()), [[winner]])

    def test_priority_aggregation_uses_numeric_max(self):
        winner = gemm_perfconfig(64, 1)
        df = make_df('gemm', [(winner, 100.0), (winner, 100.0)])
        df['DataType'] = 'f32'
        df['PerfPriority'] = ['2', '10']
        seen_priorities = []
        original_build_coverage = quickTuningGen.build_coverage

        def capture_priority(typed, *args, **kwargs):
            seen_priorities.append(typed['PerfPriority'].iloc[0])
            return original_build_coverage(typed, *args, **kwargs)

        def solve(coverage, _dtype):
            problems = sorted(coverage)
            configs = sorted({config for candidates in coverage.values() for config in candidates})
            config_idx = {config: i for i, config in enumerate(configs)}
            matrix = quickTuningGen.np.ones((len(problems), len(configs)), dtype=int)
            return [configs[0]], problems, configs, config_idx, matrix

        with mock.patch.object(quickTuningGen,
                               'build_coverage',
                               side_effect=capture_priority), \
                mock.patch.object(quickTuningGen, 'solve_full_coverage', side_effect=solve):
            quickTuningGen.find_perfconfigs(df, 'gemm', THRESHOLD, max_configs=40)

        self.assertEqual(seen_priorities, [10])

    def test_bounded_weights_follow_coverage_keys(self):
        split_k_winner = gemm_perfconfig(128, 4)
        no_split_k_winner = gemm_perfconfig(64, 1)
        df = make_df('gemm', [(split_k_winner, 100.0), (no_split_k_winner, 90.0)])
        df['DataType'] = 'f32'
        df['PerfPriority'] = 7
        bounded_args = {}

        def solve(coverage, _dtype):
            problems = sorted(coverage)
            configs = sorted({config for candidates in coverage.values() for config in candidates})
            config_idx = {config: i for i, config in enumerate(configs)}
            matrix = quickTuningGen.np.ones((len(problems), len(configs)), dtype=int)
            return configs, problems, configs, config_idx, matrix

        def solve_bounded(problems, problem_weights, configs, _config_idx, _matrix, _max_configs):
            bounded_args['problems'] = problems
            bounded_args['problem_weights'] = problem_weights
            return [configs[0]]

        with mock.patch.object(quickTuningGen, 'solve_full_coverage', side_effect=solve), \
                mock.patch.object(quickTuningGen,
                                  'solve_bounded_coverage',
                                  side_effect=solve_bounded):
            quickTuningGen.find_perfconfigs(df, 'gemm', THRESHOLD, max_configs=1)

        self.assertEqual(set(bounded_args['problem_weights']), set(bounded_args['problems']))
        self.assertEqual(set(bounded_args['problem_weights'].values()), {7})

    def test_split_k_winner_gets_a_split_k_free_duplicate(self):
        split_k_winner = gemm_perfconfig(128, 4)
        best_without_split_k = gemm_perfconfig(64, 1)
        coverage = coverage_for('gemm', [
            (split_k_winner, 100.0),
            (best_without_split_k, 90.0),
            (gemm_perfconfig(256, 1), 50.0),
        ])

        problems = {problem for problem, _ in coverage}
        self.assertEqual(len(problems), 1)
        problem = problems.pop()
        self.assertEqual(coverage[problem, True], [split_k_winner])
        self.assertEqual(coverage[problem, False], [best_without_split_k])

    def test_conv_problems_are_duplicated_too(self):
        coverage = coverage_for('conv', [
            (gemm_perfconfig(128, 4), 100.0),
            (gemm_perfconfig(64, 1), 90.0),
        ])

        self.assertEqual(sorted(split_k_allowed for _, split_k_allowed in coverage), [False, True])

    def test_no_duplicate_when_split_k_does_not_win(self):
        winner = gemm_perfconfig(64, 1)
        coverage = coverage_for('gemm', [
            (winner, 100.0),
            (gemm_perfconfig(128, 4), 50.0),
        ])

        # The restricted candidate list would be identical, so constraining the
        # set cover with it twice buys nothing.
        self.assertEqual(list(coverage.values()), [[winner]])
        self.assertTrue(all(split_k_allowed for _, split_k_allowed in coverage))

    def test_problem_without_a_split_k_free_config_is_reported(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            coverage = coverage_for('gemm', [
                (gemm_perfconfig(128, 4), 100.0),
                (gemm_perfconfig(64, 2), 90.0),
            ])

        self.assertTrue(all(split_k_allowed for _, split_k_allowed in coverage))
        self.assertIn("no splitKFactor=1 config measured", stdout.getvalue())

    def test_attention_problems_are_not_duplicated(self):
        coverage = coverage_for('attention', [
            ("attn:mPerBlockG0=32,nPerBlockG0=256,splitKFactor=1", 100.0),
            ("attn:mPerBlockG0=64,nPerBlockG0=128,splitKFactor=1", 50.0),
        ])

        self.assertTrue(all(split_k_allowed for _, split_k_allowed in coverage))

    def test_each_problem_is_considered_separately(self):
        split_k_winner = gemm_perfconfig(128, 4)
        no_split_k_winner = gemm_perfconfig(64, 1)
        df = pd.concat([
            make_df('gemm', [(split_k_winner, 100.0), (no_split_k_winner, 90.0)], problem_id=0),
            make_df('gemm', [(no_split_k_winner, 100.0), (split_k_winner, 50.0)], problem_id=1),
        ],
                       ignore_index=True)

        coverage = build_coverage(df, get_target_columns('gemm'), 'gemm', THRESHOLD)

        # Only the split-K-won problem needs the extra constraint.
        self.assertEqual(len(coverage), 3)
        self.assertEqual(sum(1 for _, split_k_allowed in coverage if not split_k_allowed), 1)

    def test_no_splitk_output_is_separate(self):
        regular = {"f16": [gemm_perfconfig(128, 4), gemm_perfconfig(64, 1)]}
        no_splitk = {"f16": [gemm_perfconfig(64, 1)]}

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "QuickTuningPerfconfigs.inc"
            with mock.patch.object(quickTuningGen, "get_output_path", return_value=output):
                quickTuningGen.update_inc_file(regular, "gfx942", "gemm")
                quickTuningGen.update_inc_file(no_splitk, "gfx942", "gemm", no_splitk=True)
                once = output.read_text()
                quickTuningGen.update_inc_file(no_splitk, "gfx942", "gemm", no_splitk=True)

            self.assertEqual(once, output.read_text())
            self.assertIn("initParametersF16GemmGfx942[]", once)
            self.assertIn("initParametersF16GemmGfx942NoSplitK[]", once)
            self.assertIn("#ifdef Gemm_NOSPLITK_LOOKUP_TABLE_GEN", once)
            self.assertEqual(once.count('"gfx942_gemm_f16"'), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
