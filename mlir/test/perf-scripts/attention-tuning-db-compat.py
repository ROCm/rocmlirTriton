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
import random
import shutil
import sys
import types
import unittest
from unittest import mock

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

import attentionSweeps  # noqa: E402
from perfRunner import AttentionConfiguration, lookup_tuning_db, read_tuning_db  # noqa: E402
from quickTuningGen import get_target_columns, load_data  # noqa: E402

ARCH = "gfx950:sramecc+:xnack-"
NUM_CU = 256
NUM_CHIPLETS = 8
# Canonical named perfConfig form. The legacy positional `attn:vN:` spelling is
# no longer emitted and parse_perfconfig() rejects it, so fixtures must not use it.
PERFCONFIG = ("attn:mPerBlockG0=32,nPerBlockG0=256,nPerBlockG1=32,kPerBlock=32,kpack=1,"
              "numCTAs=1,numWaves=4,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,"
              "wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,"
              "useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,"
              "useReductionLayout=-1,useOptimizeEpilogue=-1")
TMP_PREFIX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/attention-tuning-db-compat")


def make_config(extra_flags, g=1):
    """Build a canonical attention config with the requested optional flags."""
    key = ("-t f16 "
           "-transQ false -transK false -transV false -transO false "
           f"-causal false -return_lse false -split_kv 1 -g {g} "
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

    def test_last_valid_kv_index_is_runtime_only(self):
        """The runtime KV index reaches rocmlir-gen without changing tuning identity."""
        config = make_config("-last_valid_kv_index 4 "
                             "-with-attn-scale false -with-attn-bias false -transBias false")

        self.assertNotIn("-last_valid_kv_index", config.to_command_line())

        gen_args = config.generate_mlir_driver_commandline("", kernel_repeats=None).split()
        self.assertEqual(gen_args.count("-last_valid_kv_index=4"), 1)

    def test_perf_runner_accepts_last_valid_kv_index_endpoints(self):
        """Both inclusive KV-index endpoints are valid."""
        for index in (0, 15):
            with self.subTest(index=index):
                config = make_config(
                    f"-last_valid_kv_index {index} "
                    "-with-attn-scale false -with-attn-bias false -transBias false")
                self.assertEqual(config.last_valid_kv_index, [index])

    def test_perf_runner_rejects_invalid_last_valid_kv_indices(self):
        """KV indices outside [0, K - 1] are rejected."""
        for index in (-1, 16):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError,
                                                                   "0 <= P < seq_len_k"):
                make_config(f"-last_valid_kv_index {index} "
                            "-with-attn-scale false -with-attn-bias false -transBias false")

    def test_perf_runner_accepts_sliding_look_back_endpoints(self):
        """The disabled sentinel and maximum positive look-back are valid."""
        disabled = make_config("-sliding_window_look_back -1 "
                               "-with-attn-scale false -with-attn-bias false -transBias false")
        self.assertIsNone(disabled.sliding_window_look_back)

        maximum = make_config("-last_valid_kv_index 15 -sliding_window_look_back 15 "
                              "-with-attn-scale false -with-attn-bias false -transBias false")
        self.assertEqual(maximum.sliding_window_look_back, 15)

    def test_perf_runner_rejects_invalid_sliding_look_backs(self):
        """Look-backs outside {-1} union [1, K - 1] are rejected."""
        for look_back in (-2, 0, 16):
            with self.subTest(look_back=look_back), self.assertRaises(ValueError):
                make_config(f"-last_valid_kv_index 15 -sliding_window_look_back {look_back} "
                            "-with-attn-scale false -with-attn-bias false -transBias false")

    def test_attention_sweep_handles_single_key_kv_cache(self):
        """K=1 KV-cache samples use P=0 and disable sliding look-back."""
        with mock.patch.object(attentionSweeps, "MAX_TOKENS", 1):
            for seed in range(100):
                shape = attentionSweeps._sample_attn_shape(random.Random(seed), n_per_block=16)
                last_valid_kv_index = shape[-2]
                if last_valid_kv_index is not None:
                    self.assertEqual(shape[3], 1)
                    self.assertTrue(all(index == 0 for index in last_valid_kv_index))
                    self.assertIsNone(shape[-1])
                    break
            else:
                self.fail("No deterministic K=1 KV-cache sample found")

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

    def test_perf_runner_keeps_sliding_look_back_distinct(self):
        """A sliding-window kernel must not fall back to a row that lacks it.

        Unlike the boolean flags, sliding_window_look_back is only present in
        sliding keys, so its distinctness relies on the key string, not on the
        false-flag stripping in lookup_tuning_db.
        """
        all_false_config = make_config(
            "-with-attn-scale false -with-attn-bias false -transBias false")
        legacy_all_false_key = drop_flags(all_false_config.to_command_line(),
                                          " -with-attn-scale false", " -with-attn-bias false",
                                          " -transBias false")

        sliding_config = make_config(
            "-last_valid_kv_index 15 -sliding_window_look_back 8 "
            "-with-attn-scale false -with-attn-bias false -transBias false")

        self.assertIn("-sliding_window_look_back 8", sliding_config.to_command_line())
        self.assertIsNone(self.lookup_from_legacy_key(sliding_config, legacy_all_false_key))

    def test_perf_runner_matches_pre_transbias_sliding_look_back_key(self):
        """A sliding-window row from before transBias must still match.

        Exercises the reconciled columns together: stripping the false-valued
        transBias flag to reach a legacy row must not disturb the
        sliding_window_look_back column that sits earlier in the key.
        """
        current_config = make_config(
            "-last_valid_kv_index 15 -sliding_window_look_back 8 "
            "-with-attn-scale false -with-attn-bias false -transBias false")
        legacy_key = drop_flags(current_config.to_command_line(), " -transBias false")

        self.assertIn("-sliding_window_look_back 8", legacy_key)
        self.assertNotIn("-transBias", legacy_key)
        self.assertEqual(self.lookup_from_legacy_key(current_config, legacy_key), PERFCONFIG)

    def test_perf_runner_rejects_wrong_last_valid_kv_index_count(self):
        """Each attention group requires exactly one last-valid K/V index."""
        for indices in ("15", "15,14,13"):
            expected_count = len(indices.split(","))
            with self.subTest(indices=indices), self.assertRaisesRegex(
                    ValueError, rf"expected 2, got {expected_count}"):
                make_config(
                    f"-last_valid_kv_index {indices} "
                    "-with-attn-scale false -with-attn-bias false -transBias false",
                    g=2)

    def test_quick_tuning_gen_defaults_missing_optional_columns(self):
        """Debug TSV rows without TransBias/look-back columns get defaults."""
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
        self.assertIn("SlidingWindowLookBack", df.columns)
        self.assertTrue(df["SlidingWindowLookBack"].eq(-1).all())

        grouped = df.groupby(get_target_columns("attention") + ["PerfConfig"],
                             as_index=False)["TFlops"].max()
        self.assertFalse(grouped.empty)

    def test_quick_tuning_gen_fills_optional_columns_when_mixing_files(self):
        """Mixing TSVs must not drop rows without SlidingWindowLookBack.

        pd.concat keeps the SlidingWindowLookBack column from the newer file and
        fills the legacy row with NaN. Since groupby drops NaN keys by default,
        that row would silently disappear unless the NaN is backfilled to the
        disabled default.
        """
        legacy_header = ("DataType\tChip\tnumCU\tnumChiplets\tTransQ\tTransK\tTransV\tTransO\t"
                         "Causal\tReturnLSE\tSplitKV\tWithAttnScale\tWithAttnBias\tG\tSeqLenQ\t"
                         "SeqLenK\tNumHeadsQ\tNumHeadsKV\tHeadDimQK\tHeadDimV\tPerfConfig\tTFlops\t"
                         "TransBias\n")
        legacy_path = Path(f"{self.tmp_prefix}.legacy.debug")
        legacy_path.write_text(
            legacy_header + f"f16\tgfx950\t{NUM_CU}\t{NUM_CHIPLETS}\tFalse\tFalse\tFalse\tFalse\t"
            f"False\tFalse\t1\tTrue\tTrue\t1\t16\t16\t1\t1\t32\t32\t"
            f"{PERFCONFIG}\t1.0\tFalse\n")

        current_header = legacy_header.rstrip("\n") + "\tSlidingWindowLookBack\n"
        current_path = Path(f"{self.tmp_prefix}.current.debug")
        current_path.write_text(
            current_header + f"f16\tgfx950\t{NUM_CU}\t{NUM_CHIPLETS}\tFalse\tFalse\tFalse\tFalse\t"
            f"False\tFalse\t1\tTrue\tTrue\t1\t16\t16\t1\t1\t32\t32\t"
            f"{PERFCONFIG}\t1.0\tFalse\t-1\n")

        df = load_data([str(legacy_path), str(current_path)], no_splitk=False)
        self.assertFalse(df["SlidingWindowLookBack"].isna().any())

        grouped = df.groupby(get_target_columns("attention") + ["PerfConfig"],
                             as_index=False)["TFlops"].max()
        self.assertEqual(len(grouped), 1)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
