#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for the rocMLIR -> MIOpen conv layout translation.

MIOpenDriver only understands NCHW/NHWC, while rocMLIR configs use richer layout
names (e.g. GNC01, NGC01). perfRunner translates the -f/-I/-O layouts to NCHW/NHWC
when they map exactly and skips configs with no faithful MIOpen equivalent. These
tests pin that translate-or-skip behaviour so it can't regress; they mirror the
unit tests added alongside the same change in rocMLIR (PR #2422).

# RUN: %python %s
"""

import os
import shutil
import sys
import unittest

# perfRunner.py depends on the compiled amd_arch_db binding. Both are deployed
# together by ci-performance-scripts, so import that copy rather than the source
# file, where the binding is unavailable.
_script = shutil.which("perfRunner.py")
if _script is None:
    sys.exit("perfRunner.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

from perfRunner import rocmlir_layout_to_miopen, conv_commandline_to_miopen_layouts  # noqa: E402


def make_conv_commandline(fil, inp, out, group=1):
    """Build a minimal conv commandline (as a token list) with the given layouts."""
    return ("conv -F 1 -f {f} -I {i} -O {o} -n 1 -c 8 -H 16 -W 16 -k 8 "
            "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g {g}").format(f=fil, i=inp, o=out,
                                                                     g=group).split()


class RocmlirLayoutToMiopenTest(unittest.TestCase):
    """Tests for rocmlir_layout_to_miopen (single layout string -> MIOpen name)."""

    def test_channel_first_maps_to_nchw(self):
        """Dropping G and renaming 0/1 leaves the channel second, i.e. NCHW."""
        self.assertEqual(rocmlir_layout_to_miopen("NGC01"), "NCHW")
        self.assertEqual(rocmlir_layout_to_miopen("GNC01"), "NCHW")
        self.assertEqual(rocmlir_layout_to_miopen("NC0G1"), "NCHW")

    def test_channel_last_maps_to_nhwc(self):
        """A trailing channel dim maps to NHWC."""
        self.assertEqual(rocmlir_layout_to_miopen("N01GC"), "NHWC")
        self.assertEqual(rocmlir_layout_to_miopen("GN01C"), "NHWC")

    def test_already_miopen_layouts_pass_through(self):
        """NCHW/NHWC are returned unchanged."""
        self.assertEqual(rocmlir_layout_to_miopen("NCHW"), "NCHW")
        self.assertEqual(rocmlir_layout_to_miopen("NHWC"), "NHWC")

    def test_output_channel_letter_k_treated_as_c(self):
        """The output tensor spells the channel dim as K; MIOpen still wants NCHW/NHWC."""
        self.assertEqual(rocmlir_layout_to_miopen("NGK01"), "NCHW")
        self.assertEqual(rocmlir_layout_to_miopen("N01GK"), "NHWC")

    def test_unrepresentable_orderings_return_none(self):
        """Orderings that aren't NCHW/NHWC (channel or spatial in the wrong slot) skip."""
        self.assertIsNone(rocmlir_layout_to_miopen("G0NC1"))
        self.assertIsNone(rocmlir_layout_to_miopen("01NGC"))


class ConvCommandlineToMiopenLayoutsTest(unittest.TestCase):
    """Tests for conv_commandline_to_miopen_layouts (whole commandline translate-or-skip)."""

    def test_consistent_nchw_config_is_translated(self):
        """A config whose filter/input/output all map to NCHW is translated."""
        result = conv_commandline_to_miopen_layouts(make_conv_commandline(
            "GNC01", "NGC01", "NGC01"))
        self.assertIsNotNone(result)
        for flag in ("-f", "-I", "-O"):
            self.assertEqual(result[result.index(flag) + 1], "NCHW")

    def test_consistent_nhwc_config_is_translated(self):
        """A config whose filter/input/output all map to NHWC is translated."""
        result = conv_commandline_to_miopen_layouts(make_conv_commandline(
            "GN01C", "N01GC", "N01GC"))
        self.assertIsNotNone(result)
        for flag in ("-f", "-I", "-O"):
            self.assertEqual(result[result.index(flag) + 1], "NHWC")

    def test_group_conv_layout_is_still_translated(self):
        """Dropping G from the layout is valid; the group count rides on -g."""
        result = conv_commandline_to_miopen_layouts(
            make_conv_commandline("GNC01", "NGC01", "NGC01", group=2))
        self.assertIsNotNone(result)
        self.assertEqual(result[result.index("-g") + 1], "2")

    def test_unrepresentable_layout_is_skipped(self):
        """A layout with no NCHW/NHWC equivalent makes the whole config skip."""
        self.assertIsNone(
            conv_commandline_to_miopen_layouts(
                make_conv_commandline("G0NC1", "G0NC1", "NGC01", group=3)))

    def test_mixed_nchw_nhwc_config_is_skipped(self):
        """MIOpen has no solver for mixed filter/input/output layouts, so skip."""
        self.assertIsNone(
            conv_commandline_to_miopen_layouts(make_conv_commandline("GNC01", "NGC01", "N01GC")))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
