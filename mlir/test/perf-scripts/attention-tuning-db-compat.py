#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for attention tuning DB compatibility.

The tuning DB key for attention has grown optional boolean flags over time.
False-valued flags are identity cases, so old DB/debug rows that omit them
should still be readable. True-valued flags describe different generated
kernels and must not be silently matched against an old all-false row.

# RUN: %python %s %t
"""

from pathlib import Path
import os
import shutil
import sys
import types
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

# quickTuningGen.py is an analysis helper that is not deployed, so it still
# comes from the source tree.
MLIR_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = MLIR_DIR / "utils" / "performance"
sys.path.insert(0, str(PERF_DIR / "analysis"))


def stub_optional_pulp():
    """Let this parsing-only test import quickTuningGen without PuLP installed."""
    sys.modules.setdefault("pulp", types.SimpleNamespace())


stub_optional_pulp()

from perfRunner import AttentionConfiguration, lookup_tuning_db, read_tuning_db  # noqa: E402
from quickTuningGen import get_target_columns, load_data  # noqa: E402

ARCH = "gfx950:sramecc+:xnack-"
NUM_CU = 256
NUM_CHIPLETS = 8
PERFCONFIG = "attn:v4:32,256,32,1,1,4,16,1,1,0,0,-1,-1,-1,-1,-1,-1"
TMP_PREFIX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/attention-tuning-db-compat")


def make_config(extra_flags):
    """Build a canonical attention config with the requested optional flags."""
    key = ("-t f16 "
           "-transQ false -transK false -transV false -transO false "
           "-causal false -return_lse false -split_kv 1 -g 1 "
           "-seq_len_q 16 -seq_len_k 16 -num_heads_q 1 -num_heads_kv 1 "
           "-head_dim_qk 32 -head_dim_v 32 "
           f"{extra_flags}")
    return AttentionConfiguration.from_command_line(key.split(), ARCH, NUM_CU, NUM_CHIPLETS)


def write_tuning_db(path, test_vector):
    """Write a one-row tuning DB using the given attention problem key."""
    path.write_text("# arch\tnumCUs\tnumChiplets\ttestVector\tperfConfig\tTFlops\n"
                    f"{ARCH}\t{NUM_CU}\t{NUM_CHIPLETS}\t{test_vector}\t{PERFCONFIG}\t0.0\n")


def drop_flags(config_str, *flags):
    """Return a legacy key by removing selected false-valued flags."""
    for flag in flags:
        config_str = config_str.replace(flag, "")
    return config_str


class AttentionTuningDbCompatTest(unittest.TestCase):
    """Compatibility tests for attention tuning DB and debug TSV parsing."""

    def setUp(self):
        """Give each test its own temporary file prefix."""
        self.tmp_prefix = Path(f"{TMP_PREFIX}.{self._testMethodName}")

    def lookup_from_legacy_key(self, config, legacy_key):
        """Look up a current config against a DB row written with a legacy key."""
        path = Path(f"{self.tmp_prefix}.tsv")
        write_tuning_db(path, legacy_key)
        return lookup_tuning_db(read_tuning_db(str(path)), ARCH, config, config.to_command_line())

    def test_current_seq_len_is_runtime_only(self):
        """current_seq_len reaches rocmlir-gen without changing tuning identity."""
        config = make_config("-with-attn-scale false -with-attn-bias false -transBias false")
        config.current_seqlen = [4]

        self.assertNotIn("-current_seq_len", config.to_command_line())

        gen_args = config.generate_mlir_driver_commandline("", kernel_repeats=None).split()
        self.assertEqual(gen_args.count("-current_seq_len=4"), 1)

    def test_perf_runner_matches_legacy_all_false_attention_flags(self):
        """Old DB rows may omit all false-valued attention flags."""
        current_config = make_config(
            "-with-attn-scale false -with-attn-bias false -transBias false")
        legacy_key = drop_flags(current_config.to_command_line(), " -with-attn-scale false",
                                " -with-attn-bias false", " -transBias false")

        self.assertEqual(self.lookup_from_legacy_key(current_config, legacy_key), PERFCONFIG)

    def test_perf_runner_matches_pre_transbias_scale_bias_key(self):
        """Scale+bias DB rows from before transBias still match non-transposed bias."""
        current_config = make_config("-with-attn-scale true -with-attn-bias true -transBias false")
        legacy_key = drop_flags(current_config.to_command_line(), " -transBias false")

        self.assertEqual(self.lookup_from_legacy_key(current_config, legacy_key), PERFCONFIG)

    def test_perf_runner_keeps_true_scale_bias_distinct(self):
        """True scale/bias flags must not fall back to old all-false DB rows."""
        all_false_config = make_config(
            "-with-attn-scale false -with-attn-bias false -transBias false")
        legacy_all_false_key = drop_flags(all_false_config.to_command_line(),
                                          " -with-attn-scale false", " -with-attn-bias false",
                                          " -transBias false")

        scale_bias_config = make_config(
            "-with-attn-scale true -with-attn-bias true -transBias false")

        self.assertIsNone(self.lookup_from_legacy_key(scale_bias_config, legacy_all_false_key))

    def test_perf_runner_keeps_true_trans_bias_distinct(self):
        """Transposed bias must not fall back to an old all-false DB row."""
        all_false_config = make_config(
            "-with-attn-scale false -with-attn-bias false -transBias false")
        legacy_all_false_key = drop_flags(all_false_config.to_command_line(),
                                          " -with-attn-scale false", " -with-attn-bias false",
                                          " -transBias false")

        trans_bias_config = make_config(
            "-with-attn-scale false -with-attn-bias true -transBias true")

        self.assertIn("-transBias true", trans_bias_config.to_command_line())
        self.assertIsNone(self.lookup_from_legacy_key(trans_bias_config, legacy_all_false_key))

    def test_perf_runner_keeps_sliding_window_distinct(self):
        """A sliding-window kernel must not fall back to a row that lacks it.

        Unlike the boolean flags, sliding_window_size is only present in the key
        when > 0, so its distinctness relies on the key string, not on the
        false-flag stripping in lookup_tuning_db.
        """
        all_false_config = make_config(
            "-with-attn-scale false -with-attn-bias false -transBias false")
        legacy_all_false_key = drop_flags(all_false_config.to_command_line(),
                                          " -with-attn-scale false", " -with-attn-bias false",
                                          " -transBias false")

        sliding_window_config = make_config(
            "-sliding_window_size 8 -with-attn-scale false -with-attn-bias false -transBias false")

        self.assertIn("-sliding_window_size 8", sliding_window_config.to_command_line())
        self.assertIsNone(self.lookup_from_legacy_key(sliding_window_config, legacy_all_false_key))

    def test_perf_runner_matches_pre_transbias_sliding_window_key(self):
        """A sliding-window row from before transBias must still match.

        Exercises the reconciled columns together: stripping the false-valued
        transBias flag to reach a legacy row must not disturb the
        sliding_window_size column that sits earlier in the key.
        """
        current_config = make_config(
            "-sliding_window_size 8 -with-attn-scale false -with-attn-bias false -transBias false")
        legacy_key = drop_flags(current_config.to_command_line(), " -transBias false")

        self.assertIn("-sliding_window_size 8", legacy_key)
        self.assertNotIn("-transBias", legacy_key)
        self.assertEqual(self.lookup_from_legacy_key(current_config, legacy_key), PERFCONFIG)

    def test_quick_tuning_gen_defaults_missing_optional_columns(self):
        """Legacy debug TSV rows without TransBias/SlidingWindowSize get defaults."""
        debug_path = Path(f"{self.tmp_prefix}.debug")
        debug_path.write_text(
            "DataType\tChip\tnumCU\tnumChiplets\tTransQ\tTransK\tTransV\tTransO\t"
            "Causal\tReturnLSE\tSplitKV\tWithAttnScale\tWithAttnBias\tG\tSeqLenQ\t"
            "SeqLenK\tNumHeadsQ\tNumHeadsKV\tHeadDimQK\tHeadDimV\tPerfConfig\tTFlops\n"
            f"f16\tgfx950\t{NUM_CU}\t{NUM_CHIPLETS}\tFalse\tFalse\tFalse\tFalse\t"
            f"False\tFalse\t1\tTrue\tTrue\t1\t16\t16\t1\t1\t32\t32\t{PERFCONFIG}\t1.0\n")

        df = load_data([str(debug_path)], no_splitk=False)
        self.assertIn("TransBias", df.columns)
        self.assertTrue(df["TransBias"].eq(False).all())
        self.assertIn("SlidingWindowSize", df.columns)
        self.assertTrue(df["SlidingWindowSize"].eq(0).all())

        grouped = df.groupby(get_target_columns("attention") + ["PerfConfig"],
                             as_index=False)["TFlops"].max()
        self.assertFalse(grouped.empty)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
