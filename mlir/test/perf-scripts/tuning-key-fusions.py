#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for the fusion words of a tuning key.

mlir::rock::getTuningProblemStr names the fusions around a kernel so that
MIGraphX, which looks tuning entries up by exact key, stops handing a fused
kernel the perf config tuned for the bare GEMM. This tooling only ever
generates standalone kernels, so its job is narrower: recognize the words and
drop them instead of failing to parse the key.

# RUN: %python %s
"""

from pathlib import Path
import sys
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
)

ARCH = "gfx942"
NUM_CU = 304
NUM_CHIPLETS = 8

GEMM_KEY = ("-t f32 -out_datatype f32 -transA false -transB false -transO false "
            "-g 1 -m 64 -n 64 -k 64")

# One sample per problem type, each with the fusion words where
# getTuningProblemStr puts them: after the problem and before -supportsSplitK.
SAMPLES = (
    (ConvConfiguration, "conv -F 1 -f GNC01 -I NGC01 -O NGC01 -n 1 -c 8 -H 16 -W 16 -k 16 "
     "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1", "-outputFusions=addf,maxnumf"),
    (GemmConfiguration, GEMM_KEY, "-inputFusions=mulf,addf -outputFusions=addf,erf"),
    (ConvGemmConfiguration, "-t f16 -f GNC01 -I NGC01 -transC false -transO false "
     "-n 1 -c 8 -H 16 -W 16 -k 16 -y 3 -x 3 -p 1 -q 1 "
     "-u 1 -v 1 -l 1 -j 1 -g 1 -gemmO 32", "-inputFusions=mulf"),
    (GemmGemmConfiguration, "-t f16 -transA false -transB false -transC false -transO false "
     "-g 1 -m 64 -k 64 -n 64 -gemmO 32", "-outputFusions=exp,mulf,addf"),
    (AttentionConfiguration, "-t f16 -transQ false -transK false -transV false -transO false "
     "-causal false -return_lse false -split_kv 1 -g 1 "
     "-seq_len_q 16 -seq_len_k 16 -num_heads_q 1 -num_heads_kv 1 "
     "-head_dim_qk 32 -head_dim_v 32 -with-attn-scale false "
     "-with-attn-bias false -transBias false", "-inputFusions=mulf"),
)


class FusionTuningKeyTest(unittest.TestCase):

    def test_fused_key_canonicalizes_to_the_standalone_problem(self):
        for config_class, raw, fusions in SAMPLES:
            with self.subTest(config_class=config_class.__name__):
                split_k = "false" if config_class is AttentionConfiguration else "true"
                unfused = f"{raw} -supportsSplitK {split_k}"
                fused = f"{raw} {fusions} -supportsSplitK {split_k}"

                self.assertEqual(
                    canonicalize_config(fused, config_class, ARCH, NUM_CU, NUM_CHIPLETS),
                    canonicalize_config(unfused, config_class, ARCH, NUM_CU, NUM_CHIPLETS))

    def test_fusions_are_not_driver_options(self):
        config = GemmConfiguration.from_command_line(
            f"{GEMM_KEY} -inputFusions=addf -outputFusions=addf -supportsSplitK true".split(), ARCH,
            NUM_CU, NUM_CHIPLETS)

        self.assertNotIn("Fusions", config.generate_mlir_driver_commandline("",
                                                                            kernel_repeats=None))
        self.assertNotIn("Fusions", config.generate_problem_commandline())

    def test_unfused_key_is_unchanged(self):
        """A problem without fusions keeps the key it had before the words existed,
        so an existing tuning database keeps serving the kernels it was tuned on."""
        config = GemmConfiguration.from_command_line(GEMM_KEY.split(), ARCH, NUM_CU, NUM_CHIPLETS)

        self.assertEqual(config.to_command_line(), f"{GEMM_KEY} -supportsSplitK true")

    def test_metadata_is_dropped_wherever_it_appears(self):
        argv, _ = extract_tuning_key_metadata(
            ["-t", "f32", "-inputFusions=addf,mulf", "-g", "1", "-outputFusions=maxnumf"])

        self.assertEqual(argv, ["-t", "f32", "-g", "1"])

        with self.assertRaisesRegex(ValueError, "Missing value"):
            extract_tuning_key_metadata(["-inputFusions="])


if __name__ == "__main__":
    unittest.main(verbosity=2)
