#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for split-K tuning-key metadata.

# RUN: %python %s
"""

from pathlib import Path
import sys
import tempfile
import types
import unittest

PERF_DIR = Path(__file__).resolve().parents[2] / "utils" / "performance"
sys.path.insert(0, str(PERF_DIR))

# These runtime-only dependencies are unnecessary for configuration parsing.
sys.modules.setdefault("amd_arch_db", types.SimpleNamespace())
hip_package = types.ModuleType("hip")
hip_package.hip = types.SimpleNamespace()
sys.modules.setdefault("hip", hip_package)

from perfRunner import (  # noqa: E402
    AttentionConfiguration, ConvConfiguration, ConvGemmConfiguration, GemmConfiguration,
    GemmGemmConfiguration, canonicalize_config, extract_tuning_key_metadata,
    lookup_fusion_tuning_config, lookup_tuning_db, read_tuning_db,
)

ARCH = "gfx900"
NUM_CU = 64
NUM_CHIPLETS = 1

SAMPLES = (
    (ConvConfiguration, "conv -F 1 -f GNC01 -I NGC01 -O NGC01 -n 1 -c 8 -H 16 -W 16 -k 16 "
     "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1"),
    (GemmConfiguration, "-t f32 -out_datatype f32 -transA false -transB false -transO false "
     "-g 1 -m 64 -n 64 -k 64"),
    (ConvGemmConfiguration, "-t f16 -f GNC01 -I NGC01 -transC false -transO false "
     "-n 1 -c 8 -H 16 -W 16 -k 16 -y 3 -x 3 -p 1 -q 1 "
     "-u 1 -v 1 -l 1 -j 1 -g 1 -gemmO 32"),
    (GemmGemmConfiguration, "-t f16 -transA false -transB false -transC false -transO false "
     "-g 1 -m 64 -k 64 -n 64 -gemmO 32"),
    (AttentionConfiguration, "-t f16 -transQ false -transK false -transV false -transO false "
     "-causal false -return_lse false -split_kv 1 -g 1 "
     "-seq_len_q 16 -seq_len_k 16 -num_heads_q 1 -num_heads_kv 1 "
     "-head_dim_qk 32 -head_dim_v 32 -with-attn-scale false "
     "-with-attn-bias false -transBias false"),
)


class SplitKTuningKeyTest(unittest.TestCase):

    def test_metadata_round_trips_for_all_problem_types(self):
        for config_class, raw in SAMPLES:
            with self.subTest(config_class=config_class.__name__):
                supports_split_k = config_class is not AttentionConfiguration
                canonical = canonicalize_config(
                    f"{raw} -supportsSplitK {str(supports_split_k).lower()}", config_class, ARCH,
                    NUM_CU, NUM_CHIPLETS)
                config = config_class.from_command_line(canonical.split(), ARCH, NUM_CU,
                                                        NUM_CHIPLETS)

                self.assertEqual(config.supports_split_k, supports_split_k)
                self.assertTrue(
                    canonical.endswith(f"-supportsSplitK {str(supports_split_k).lower()}"))
                self.assertEqual(
                    canonicalize_config(canonical, config_class, ARCH, NUM_CU, NUM_CHIPLETS),
                    canonical)
                self.assertNotIn("-supportsSplitK",
                                 config.generate_mlir_driver_commandline("", kernel_repeats=None))

    def test_missing_metadata_defaults_to_operation_split_k_support(self):
        for config_class, raw in SAMPLES:
            with self.subTest(config_class=config_class.__name__):
                canonical = canonicalize_config(raw, config_class, ARCH, NUM_CU, NUM_CHIPLETS)
                expected = "false" if config_class is AttentionConfiguration else "true"
                self.assertTrue(canonical.endswith(f"-supportsSplitK {expected}"))

    def test_attention_rejects_split_k_support(self):
        raw = SAMPLES[-1][1]
        with self.assertRaisesRegex(AssertionError, "attention does not support split-K"):
            AttentionConfiguration.from_command_line(f"{raw} -supportsSplitK true".split(), ARCH,
                                                     NUM_CU, NUM_CHIPLETS)

    def test_i8_base_key_defaults_to_split_k_support(self):
        raw = ("-t i8 -out_datatype i32 -transA false -transB false -transO false "
               "-g 1 -m 64 -n 64 -k 64")
        config = GemmConfiguration.from_command_line(raw.split(), ARCH, NUM_CU, NUM_CHIPLETS)

        self.assertTrue(config.supports_split_k)
        self.assertTrue(config.to_command_line().endswith("-supportsSplitK true"))

    def test_metadata_is_not_a_driver_option(self):
        argv, supports_split_k = extract_tuning_key_metadata(
            ["-t", "f32", "-supportsSplitK", "true", "-g", "1"])
        self.assertEqual(argv, ["-t", "f32", "-g", "1"])
        self.assertTrue(supports_split_k)

        with self.assertRaisesRegex(ValueError, "Missing value"):
            extract_tuning_key_metadata(["-supportsSplitK"])
        with self.assertRaisesRegex(ValueError, "Invalid value"):
            extract_tuning_key_metadata(["-supportsSplitK", "maybe"])

    def test_legacy_key_matches_unfused_default(self):
        config_class, raw = SAMPLES[1]
        supports_split_k = config_class.from_command_line(raw.split(), ARCH, NUM_CU, NUM_CHIPLETS)
        legacy_key = supports_split_k.to_command_line().replace(" -supportsSplitK true", "")
        tuning_db = {(ARCH, legacy_key): "perf_config"}

        self.assertEqual(
            lookup_tuning_db(tuning_db, ARCH, supports_split_k, supports_split_k.to_command_line()),
            "perf_config")

        no_split_k = config_class.from_command_line(f"{raw} -supportsSplitK false".split(), ARCH,
                                                    NUM_CU, NUM_CHIPLETS)
        self.assertIsNone(
            lookup_tuning_db(tuning_db, ARCH, no_split_k, no_split_k.to_command_line()))

    def test_legacy_attention_key_matches_no_split_k_default(self):
        config_class, raw = SAMPLES[-1]
        config = config_class.from_command_line(raw.split(), ARCH, NUM_CU, NUM_CHIPLETS)
        legacy_key = config.to_command_line().replace(" -supportsSplitK false", "")
        tuning_db = {(ARCH, legacy_key): "perf_config"}

        self.assertFalse(config.supports_split_k)
        self.assertEqual(lookup_tuning_db(tuning_db, ARCH, config, config.to_command_line()),
                         "perf_config")

    def test_fusion_lookup_falls_back_to_split_k_capable_base_key(self):
        config_class, raw = SAMPLES[1]
        no_split_k = config_class.from_command_line(f"{raw} -supportsSplitK false".split(), ARCH,
                                                    NUM_CU, NUM_CHIPLETS)
        capable_key = raw + " -supportsSplitK true"
        tuning_db = {(ARCH, capable_key): "v2:64,64,32,32,1,1,1"}

        perf_config = lookup_fusion_tuning_config(tuning_db, ARCH, no_split_k,
                                                  no_split_k.to_command_line())

        self.assertEqual(perf_config, "v2:64,64,32,32,1,1,1")
        self.assertFalse(no_split_k.supports_split_k)

    def test_tuning_db_distinguishes_split_k_support(self):
        raw = SAMPLES[1][1]
        split_k_key = canonicalize_config(f"{raw} -supportsSplitK true", GemmConfiguration, ARCH,
                                          NUM_CU, NUM_CHIPLETS)
        no_split_k_key = canonicalize_config(f"{raw} -supportsSplitK false", GemmConfiguration,
                                             ARCH, NUM_CU, NUM_CHIPLETS)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tuning.tsv"
            path.write_text(
                f"{ARCH}\t{NUM_CU}\t{NUM_CHIPLETS}\t{split_k_key}\tperf_split_k\n"
                f"{ARCH}\t{NUM_CU}\t{NUM_CHIPLETS}\t{no_split_k_key}\tperf_no_split_k\n")
            tuning_db = read_tuning_db(str(path))

        self.assertEqual(len(tuning_db), 2)
        self.assertEqual(tuning_db[ARCH, split_k_key], "perf_split_k")
        self.assertEqual(tuning_db[ARCH, no_split_k_key], "perf_no_split_k")

    def test_rocmlir_gen_disables_split_k_when_problem_key_does(self):
        _, raw = SAMPLES[1]
        capable = GemmConfiguration.from_command_line(f"{raw} -supportsSplitK true".split(), ARCH,
                                                    NUM_CU, NUM_CHIPLETS)
        incapable = GemmConfiguration.from_command_line(f"{raw} -supportsSplitK false".split(),
                                                        ARCH, NUM_CU, NUM_CHIPLETS)
        attention = AttentionConfiguration.from_command_line(SAMPLES[-1][1].split(), ARCH, NUM_CU,
                                                             NUM_CHIPLETS)

        capable_cmd = capable.generate_mlir_driver_commandline('')
        incapable_cmd = incapable.generate_mlir_driver_commandline('')
        attention_cmd = attention.generate_mlir_driver_commandline('')

        self.assertNotIn('-disable-split-k-for-tuning', capable_cmd.split())
        self.assertIn('-disable-split-k-for-tuning', incapable_cmd.split())
        self.assertIn('-disable-split-k-for-tuning', attention_cmd.split())

        merged = incapable.merge_rocmlir_gen_flags('-disable-split-k-for-tuning')
        self.assertEqual(merged.split().count('-disable-split-k-for-tuning'), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
