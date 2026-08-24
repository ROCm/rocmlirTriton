#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Canonical form of a test vector, for every op tuningRunner.py can tune.

perfRunner looks tuning-DB entries up by ``config.to_command_line()``, so the
``testVector`` column tuningRunner writes, its dedup/state-file keys and that
lookup key must all be the same string. Each op below is pinned with a raw
spelling (flags out of order, optional flags omitted) and the canonical form it
must collapse to, plus an already-canonical vector that has to survive
untouched. Pure string round-tripping, so no GPU is needed.

# RUN: %python %s
"""

import os
import shutil
import sys
import unittest

# perfRunner.py and tuningRunner.py are deployed together by ci-performance-scripts
# with the compiled amd_arch_db binding they depend on.
_script = shutil.which('perfRunner.py')
if _script is None:
    sys.exit("perfRunner.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

from perfRunner import (  # noqa: E402
    AttentionConfiguration, ConvConfiguration, ConvGemmConfiguration, GemmConfiguration,
    GemmGemmConfiguration, PerfConfiguration, canonicalize_config)
from tuningRunner import canonicalize_test_vector  # noqa: E402

ARCH = "gfx900"
NUM_CU = 64
NUM_CHIPLETS = 1

# op -> (conf_class, raw spelling, canonical form, already-canonical vector).
SAMPLE_TEST_VECTORS = {
    "gemm": (
        GemmConfiguration,
        "-g 1 -m 1024 -k 769 -n 512 -t f32 -out_datatype f32 -transA false -transB false",
        ("-t f32 -out_datatype f32 -transA false -transB false -transO false "
         "-g 1 -m 1024 -n 512 -k 769"),
        ("-t f16 -out_datatype f16 -transA false -transB true -transO false "
         "-g 1 -m 256 -n 128 -k 64"),
    ),
    "conv": (
        ConvConfiguration,
        ("convfp16 -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 "
         "-p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1"),
        ("convfp16 -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 "
         "-k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1"),
        ("convfp16 -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 "
         "-k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1"),
    ),
    "attention": (
        AttentionConfiguration,
        ("-g 1 -seq_len_q 256 -seq_len_k 256 -num_heads_q 8 -num_heads_kv 8 "
         "-head_dim_qk 64 -head_dim_v 64 -t f16 "
         "-transQ false -transK false -transV false -transO false "
         "-causal false -return_lse false -split_kv 1 "
         "-with-attn-scale false -with-attn-bias false"),
        ("-t f16 -transQ false -transK false -transV false -transO false "
         "-causal false -return_lse false -split_kv 1 -g 1 "
         "-seq_len_q 256 -seq_len_k 256 -num_heads_q 8 -num_heads_kv 8 "
         "-head_dim_qk 64 -head_dim_v 64 "
         "-with-attn-scale false -with-attn-bias false -transBias false"),
        ("-t f16 -transQ false -transK false -transV false -transO false "
         "-causal false -return_lse false -split_kv 1 -g 1 "
         "-seq_len_q 128 -seq_len_k 128 -num_heads_q 4 -num_heads_kv 4 "
         "-head_dim_qk 32 -head_dim_v 32 "
         "-with-attn-scale false -with-attn-bias false -transBias false"),
    ),
    "gemm_gemm": (
        GemmGemmConfiguration,
        ("-g 1 -m 64 -k 128 -n 256 -gemmO 32 -t f16 "
         "-transA false -transB false -transC false -transO false"),
        ("-t f16 -transA false -transB false -transC false -transO false "
         "-g 1 -m 64 -k 128 -n 256 -gemmO 32"),
        ("-t f16 -transA false -transB false -transC false -transO false "
         "-g 1 -m 32 -k 64 -n 128 -gemmO 16"),
    ),
    "conv_gemm": (
        ConvGemmConfiguration,
        ("-n 1 -c 64 -H 14 -W 14 -k 128 -y 3 -x 3 -gemmO 64 "
         "-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -f NCHW -I NCHW "
         "-t f16 -transC false -transO false"),
        ("-t f16 -f NCHW -I NCHW -transC false -transO false "
         "-n 1 -c 64 -H 14 -W 14 -k 128 -y 3 -x 3 "
         "-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -gemmO 64"),
        ("-t f16 -f NCHW -I NCHW -transC false -transO false "
         "-n 1 -c 64 -H 14 -W 14 -k 128 -y 3 -x 3 "
         "-p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -gemmO 64"),
    ),
}


def canonicalize(config_str, conf_class):
    return canonicalize_config(config_str, conf_class, ARCH, NUM_CU, NUM_CHIPLETS)


class CanonicalizeConfigTest(unittest.TestCase):
    """Tests for canonicalize_config across every supported op."""

    def test_reorders_flags(self):
        for op, (conf_class, raw, canonical, _) in SAMPLE_TEST_VECTORS.items():
            with self.subTest(op=op):
                self.assertEqual(canonicalize(raw, conf_class), canonical)

    def test_idempotent(self):
        for op, (conf_class, _, _, already_canonical) in SAMPLE_TEST_VECTORS.items():
            with self.subTest(op=op):
                self.assertEqual(canonicalize(already_canonical, conf_class), already_canonical)

    def test_round_trip_is_stable(self):
        for op, (conf_class, raw, _, _) in SAMPLE_TEST_VECTORS.items():
            with self.subTest(op=op):
                once = canonicalize(raw, conf_class)
                self.assertEqual(canonicalize(once, conf_class), once)

    def test_invalid_config_raises_valueerror(self):
        with self.assertRaisesRegex(ValueError, "Failed to parse"):
            canonicalize("not a valid config", GemmConfiguration)

    def test_wrong_op_raises_valueerror(self):
        gemm_tv = "-t f32 -out_datatype f32 -transA false -transB false -g 1 -m 64 -n 128 -k 256"
        with self.assertRaisesRegex(ValueError, "Failed to parse"):
            canonicalize(gemm_tv, ConvConfiguration)


class CanonicalizeFusionDispatchTest(unittest.TestCase):
    """Fusion tunes whatever the kernel contains, so it passes the
    PerfConfiguration base class and canonicalize_config picks the concrete
    class from the config string itself."""

    def test_dispatches_to_conv(self):
        raw = SAMPLE_TEST_VECTORS["conv"][1]
        self.assertEqual(canonicalize(raw, PerfConfiguration), canonicalize(raw, ConvConfiguration))

    def test_dispatches_to_gemm(self):
        raw = SAMPLE_TEST_VECTORS["gemm"][1]
        self.assertEqual(canonicalize(raw, PerfConfiguration), canonicalize(raw, GemmConfiguration))

    def test_invalid_raises_valueerror_naming_resolved_class(self):
        """Errors from fusion dispatch name the resolved concrete class, not the
        base class the caller passed."""
        with self.assertRaisesRegex(ValueError, "ConvConfiguration"):
            canonicalize("convfp16 not a real config", PerfConfiguration)
        with self.assertRaisesRegex(ValueError, "GemmConfiguration"):
            canonicalize("not a real config", PerfConfiguration)


class CanonicalizeTestVectorTest(unittest.TestCase):
    """Tests for canonicalize_test_vector (the tuningRunner-side wrapper)."""

    def test_mlir_path_passthrough(self):
        """.mlir test vectors are file paths handled via --emit-tuning-key."""
        path = "/some/test.mlir"
        self.assertEqual(
            canonicalize_test_vector(path, GemmConfiguration, ARCH, NUM_CU, NUM_CHIPLETS), path)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
