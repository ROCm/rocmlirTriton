#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for canonicalize_config on an op mismatch.

The weekly `Tune rocMLIR` stage points every op at one shared output TSV, so
the conv run reads back the rows the preceding gemm run wrote. Loading that
cache canonicalizes each row under the current op's class, which means
ConvConfiguration is handed a gemm test vector. That must surface as a
ValueError so TunedConfigsCache.from_output_file can skip the row; the parsers
otherwise leak whatever they raise first (a gemm vector leaves `datatype`
unassigned, i.e. UnboundLocalError) and abort the whole tuning run.

# RUN: %python %s
"""

import os
import shutil
import sys
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

from perfRunner import ConvConfiguration, GemmConfiguration  # noqa: E402
from perfRunner import PerfConfiguration, canonicalize_config  # noqa: E402

ARCH = "gfx90a:sramecc+:xnack-"
NUM_CU = 104
NUM_CHIPLETS = 1

# A gemm row as the tuning DB stores it, taken from a weekly tuning log.
GEMM_VECTOR = ("-t f16 -out_datatype f16 -transA false -transB false -transO false "
               "-g 1 -m 4096 -n 256 -k 128")


class CanonicalizeConfigOpMismatchTest(unittest.TestCase):
    """canonicalize_config must report parse failures as ValueError."""

    def canonicalize(self, config_str, conf_class):
        return canonicalize_config(config_str, conf_class, ARCH, NUM_CU, NUM_CHIPLETS)

    def test_gemm_vector_under_conv_class_raises_value_error(self):
        """A gemm row read by the conv tuning run is rejected, not fatal."""
        with self.assertRaises(ValueError) as caught:
            self.canonicalize(GEMM_VECTOR, ConvConfiguration)
        self.assertIn("ConvConfiguration", str(caught.exception))

    def test_gemm_vector_under_gemm_class_round_trips(self):
        """The same row still canonicalizes under its own op."""
        canonical = self.canonicalize(GEMM_VECTOR, GemmConfiguration)
        self.assertIn("-m 4096", canonical)
        self.assertEqual(canonical, self.canonicalize(canonical, GemmConfiguration))

    def test_fusion_catch_all_dispatches_by_prefix(self):
        """PerfConfiguration routes a non-conv vector to GemmConfiguration."""
        self.assertEqual(self.canonicalize(GEMM_VECTOR, PerfConfiguration),
                         self.canonicalize(GEMM_VECTOR, GemmConfiguration))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
