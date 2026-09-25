#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for per-problem quick-tuning map generation.

Within one lookup-key field schema, nothing downstream can tell a wrong problem
hash from a problem that was never measured, because both are a miss that falls
back to the set cover. So the properties the generator has to hold -- keying
every problem exactly once, reproducing byte-identical output, and inverting
``table_entry`` faithfully -- are pinned here rather than left to manual
inspection.

The compiler-side half (a shipped hash still reaching its row) is covered by
test/rocmlir-gen/quick-tuning-per-problem.mlir.

# RUN: %python %s
"""

import argparse
import contextlib
import io
import os
import re
import shutil
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

# perfRunner.py is on PATH (lit's mlir_rock_tools_dir, populated by
# ci-performance-scripts). Import it from there rather than from the source
# tree: it depends on the compiled amd_arch_db binding, which only exists
# alongside the deployed scripts.
_script = shutil.which('perfRunner.py')
if _script is None:
    sys.exit("perfRunner.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

# quickTuningGen.py is an analysis helper that is not deployed, so it still
# comes from the source tree.
MLIR_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MLIR_DIR / "utils" / "performance" / "analysis"))

# PuLP is only needed to solve the set cover, which per-problem generation does
# not use.
sys.modules.setdefault("pulp", types.SimpleNamespace())

import perfRunner  # noqa: E402
import quickTuningGen  # noqa: E402
from quickTuningGen import format_shard, get_target_columns  # noqa: E402
from quickTuningGen import per_problem_perfconfigs, positive_int  # noqa: E402
from quickTuningGen import select_perfconfigs  # noqa: E402

TOP_N = 5


def gemm_perfconfig(m_per_block, split_k=1):
    return (f"gemm:mPerBlock={m_per_block},nPerBlock=64,kPerBlock=32,kpack=1,numCTAs=1,"
            f"numWaves=4,matrixInstrNonkdim=16,splitKFactor={split_k},numStages=1,"
            "wavesPerEU=0,gridGroupSize=0")


def make_df(op, rows, problem_id=0):
    """Build a debug-style frame for one problem from (perfConfig, TFlops) rows."""
    columns = get_target_columns(op)
    problem = dict.fromkeys(columns, 0)
    problem[columns[-1]] = problem_id
    return pd.DataFrame(
        [dict(problem, PerfConfig=config, TFlops=tflops) for config, tflops in rows])


def rank(op, rows, hashes, top_n=TOP_N, version_hashes=None):
    """Run per_problem_perfconfigs over ``rows``, keying problems with ``hashes``.

    The compiler is the only speller of the key, so keying it here would mean
    running rocmlir-gen per problem; these tests are about what the generator
    does with the keys it gets back, so they supply them directly.
    """
    df = pd.concat([make_df(op, measurements, problem_id=i) for i, measurements in enumerate(rows)],
                   ignore_index=True)
    if version_hashes is None:
        version_hashes = [99] * len(hashes)
    keys = iter(zip(hashes, version_hashes))
    with mock.patch.object(quickTuningGen, "problem_key_and_version_hash",
                           lambda *_: next(keys)), \
            contextlib.redirect_stdout(io.StringIO()):
        return per_problem_perfconfigs(df, op, top_n, "rocmlir-gen")


class PerProblemSelectionTest(unittest.TestCase):

    def test_keeps_the_top_n_by_tflops(self):
        measurements = [(gemm_perfconfig(m), float(m)) for m in (16, 32, 64, 128, 256, 512)]
        best = pd.DataFrame(measurements, columns=["PerfConfig", "TFlops"])

        perfconfigs, missing = select_perfconfigs(best, "gemm", TOP_N)

        self.assertFalse(missing)
        self.assertEqual(perfconfigs, [gemm_perfconfig(m) for m in (512, 256, 128, 64, 32)])

    def test_trades_the_last_slot_for_a_legal_non_split_k_config(self):
        # A row of nothing but split-K configs is unusable by a kernel whose
        # fusion forbids split-K, so the weakest slot goes to the best measured
        # splitKFactor=1 config even when it ranks well below the cutoff.
        split_k = [(gemm_perfconfig(m, split_k=4), 100.0 - i) for i, m in enumerate((16, 32, 64))]
        legal = gemm_perfconfig(128, split_k=1)
        best = pd.DataFrame(split_k + [(legal, 1.0)], columns=["PerfConfig", "TFlops"])

        perfconfigs, missing = select_perfconfigs(best, "gemm", 3)

        self.assertFalse(missing)
        self.assertEqual(perfconfigs[-1], legal)
        self.assertEqual(len(perfconfigs), 3)

    def test_reports_a_problem_with_no_legal_non_split_k_config(self):
        best = pd.DataFrame([(gemm_perfconfig(16, split_k=4), 100.0)],
                            columns=["PerfConfig", "TFlops"])

        _, missing = select_perfconfigs(best, "gemm", TOP_N)

        self.assertTrue(missing)

    def test_rejects_a_top_n_select_perfconfigs_cannot_use(self):
        self.assertEqual(positive_int("1"), 1)
        for bad in ("0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                positive_int(bad)


class PerProblemRankingTest(unittest.TestCase):

    def test_colliding_problem_keys_are_rejected(self):
        # Two problems keyed alike means one silently replaces the other, which
        # would ship a ranking measured on a different shape. Generation has to
        # stop instead.
        with self.assertRaises(ValueError) as raised:
            rank("gemm", [[(gemm_perfconfig(16), 100.0)], [(gemm_perfconfig(32), 100.0)]],
                 hashes=[7, 7])

        self.assertIn("two problems share key 7", str(raised.exception))

    def test_mixed_lookup_key_versions_are_rejected(self):
        with self.assertRaises(ValueError) as raised:
            rank("gemm", [[(gemm_perfconfig(16), 100.0)], [(gemm_perfconfig(32), 100.0)]],
                 hashes=[7, 8],
                 version_hashes=[1, 2])

        self.assertIn("one shard cannot record multiple", str(raised.exception))

    def test_measurements_of_one_problem_are_kept_together(self):
        problems, _, _, short = rank(
            "gemm", [[(gemm_perfconfig(16), 100.0),
                      (gemm_perfconfig(32), 50.0)], [(gemm_perfconfig(64), 100.0)]],
            hashes=[11, 22])

        self.assertEqual(problems[11], [gemm_perfconfig(16), gemm_perfconfig(32)])
        self.assertEqual(problems[22], [gemm_perfconfig(64)])
        # Both rows are shorter than TOP_N; neither is padded.
        self.assertEqual(short, 2)

    def test_repeated_measurements_collapse_to_the_best(self):
        problems, _, _, _ = rank("gemm", [[(gemm_perfconfig(16), 10.0), (gemm_perfconfig(16), 90.0),
                                           (gemm_perfconfig(32), 50.0)]],
                                 hashes=[11])

        self.assertEqual(problems[11], [gemm_perfconfig(16), gemm_perfconfig(32)])


class ShardFormatTest(unittest.TestCase):

    def shard(self, problems):
        return format_shard("gfx942_gemm_f32", "gemm", problems, key_version_hash=1)

    def test_output_does_not_depend_on_insertion_order(self):
        # Regenerating has to be a no-op when the data has not changed,
        # otherwise every run shows up as a diff. groupby order is not the
        # emission order, so the shard sorts both the problems and the interned
        # strings.
        rows = {
            22: [gemm_perfconfig(64), gemm_perfconfig(16)],
            11: [gemm_perfconfig(32)],
        }
        reversed_rows = dict(reversed(list(rows.items())))

        self.assertEqual(self.shard(rows), self.shard(reversed_rows))

    def test_rows_are_variable_length_and_contiguous(self):
        shard = self.shard({
            11: [gemm_perfconfig(16), gemm_perfconfig(32),
                 gemm_perfconfig(64)],
            22: [gemm_perfconfig(16)],
        })

        refs = re.findall(r"\{(\d+)ULL, (\d+), (\d+)\}", shard)
        self.assertEqual(refs, [("11", "0", "3"), ("22", "3", "1")])

    def test_perfconfigs_are_interned(self):
        shared = gemm_perfconfig(16)
        shard = self.shard({11: [shared, gemm_perfconfig(32)], 22: [shared]})

        self.assertEqual(shard.count(f'"{shared}"'), 1)
        # Three slots in the ribbon, two distinct strings behind them.
        self.assertEqual(len(re.findall(r'"gemm:[^"]+"', shard)), 2)

    def test_lookup_entry_names_the_generated_arrays(self):
        shard = self.shard({11: [gemm_perfconfig(16)]})

        self.assertIn("#ifdef Gemm_PER_PROBLEM_DEFINITIONS_GEN", shard)
        self.assertIn(
            '{"gfx942_gemm_f32", QuickTuningProblemMap(1ULL, problemsGfx942GemmF32, '
            "perfConfigIndicesGfx942GemmF32, perfConfigsGfx942GemmF32)},", shard)

    def test_shard_records_the_table_lookup_key_version_hash(self):
        # A shard's hashes are only meaningful under the key they were computed
        # with, so the compiler needs the version to recognise a stale shard.
        shard = format_shard("gfx942_gemm_f32",
                             "gemm", {11: [gemm_perfconfig(16)]},
                             key_version_hash=7)

        self.assertIn('{"gfx942_gemm_f32", QuickTuningProblemMap(7ULL, problemsGfx942GemmF32, ',
                      shard)

    def test_attention_shards_land_in_the_gemm_gemm_section(self):
        shard = format_shard("gfx942_attention_f16",
                             "attention",
                             {11: ["attn:mPerBlockG0=32,nPerBlockG0=256,splitKFactor=1"]},
                             key_version_hash=1)

        self.assertIn("#ifdef GemmGemm_PER_PROBLEM_DEFINITIONS_GEN", shard)


class TableEntryRoundTripTest(unittest.TestCase):
    """``from_table_entry`` is what rebuilds the problem the key is computed on.

    If it drops or mangles a field, generation keys a different problem than
    the one measured and every affected row ships unreachable, so the inverse
    is pinned rather than assumed.
    """

    def assert_round_trips(self, config):
        row = config.table_entry(nanoseconds=1000.0)
        rebuilt = type(config).from_table_entry(row, config.arch, config.num_cu,
                                                config.num_chiplets)

        self.assertEqual(rebuilt.generate_problem_commandline(kernel_repeats=None),
                         config.generate_problem_commandline(kernel_repeats=None))

    def test_gemm_round_trips(self):
        self.assert_round_trips(
            perfRunner.GemmConfiguration(dtype="f32",
                                         out_dtype="f32",
                                         g=1,
                                         m=128,
                                         k=512,
                                         n=512,
                                         trans_a=False,
                                         trans_b=False,
                                         trans_o=False,
                                         arch="gfx942",
                                         num_cu=304,
                                         num_chiplets=1))

    def test_gemm_transposes_survive(self):
        # The transposes are part of the problem key, and table_bool is the
        # only thing standing between their string spelling and a silent False.
        self.assert_round_trips(
            perfRunner.GemmConfiguration(dtype="f16",
                                         out_dtype="f16",
                                         g=2,
                                         m=64,
                                         k=64,
                                         n=64,
                                         trans_a=True,
                                         trans_b=False,
                                         trans_o=True,
                                         arch="gfx942",
                                         num_cu=304,
                                         num_chiplets=1))

    def test_conv_round_trips(self):
        # Layouts, spatial dims, stride, dilation and padding all reach the
        # key. The group count deliberately does not survive -- the table has
        # no column for it -- which is safe only because the key is
        # group-invariant, as quick-tuning-problem-key-hash.mlir pins.
        self.assert_round_trips(
            perfRunner.ConvConfiguration.from_command_line(
                ("conv -F 1 -f N01GC -I 01NGC -O NGC01 -n 1 -c 3 -H 224 -W 224 -k 64 "
                 "-y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1").split(), "gfx942", 304, 1))

    def test_attention_round_trips(self):
        # Attention carries the most key fields of the three, including the
        # pre-softmax scale and bias flags that describe what was fused in.
        self.assert_round_trips(
            perfRunner.AttentionConfiguration.from_command_line(
                ("-t f16 -transQ false -transK true -transV false -transO false "
                 "-causal false -return_lse false -split_kv 1 -num_heads_q 1 "
                 "-num_heads_kv 1 -g 12 -seq_len_q 384 -seq_len_k 384 -head_dim_qk 64 "
                 "-head_dim_v 64 -with-attn-scale false -with-attn-bias false "
                 "-transBias false").split(), "gfx942", 304, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
