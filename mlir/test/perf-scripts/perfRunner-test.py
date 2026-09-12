#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python coverage for perfRunner.py.

perfRunner consumes tuning DBs written by tuningRunner, rocprof CSV output and
mlir-runner stdout, and it picks profiler flags per arch. All of that is
string/file handling that runs without a GPU, so it is pinned here rather than
left to the weekly Jenkins performance run. The runs that need a real GPU live
in ``runtime/``.

# RUN: %python %s %t
"""

import math
import os
import shutil
import sys
import unittest
from pathlib import Path

# perfRunner.py is on PATH (lit's mlir_rock_tools_dir, populated by
# ci-performance-scripts). Import it from there rather than from the source
# tree: it depends on the compiled amd_arch_db binding, which only exists
# alongside the deployed scripts.
_script = shutil.which('perfRunner.py')
if _script is None:
    sys.exit("perfRunner.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

import perfRunner  # noqa: E402
from perfRunner import GemmConfiguration, PerfConfiguration  # noqa: E402

TMP_PREFIX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/perfRunner-test")


class TempFileTestCase(unittest.TestCase):
    """Base class handing each test method its own file under lit's %t prefix."""

    def write_temp(self, suffix, contents):
        """Write contents to a per-test temporary file and return its path."""
        path = Path(f"{TMP_PREFIX}.{self._testMethodName}{suffix}")
        path.write_text(contents)
        return str(path)


class ParseTuningDbLineTest(unittest.TestCase):
    """Tests for parse_tuning_db_line (legacy, v2 and v3 formats).

    All three on-disk formats collapse to the same (arch, config, perfconfig)
    triple: the num_cu/num_chiplets columns are accepted for forward
    compatibility but dropped, because lookups here are keyed on
    (arch, config) alone.
    """

    def test_legacy_three_entries(self):
        out = perfRunner.parse_tuning_db_line(["gfx900", "config1", "perf1"])
        self.assertEqual(out, ("gfx900", "config1", "perf1"))

    def test_v2_four_entries_drops_num_cu(self):
        out = perfRunner.parse_tuning_db_line(["gfx900", "120", "config1", "perf1"])
        self.assertEqual(out, ("gfx900", "config1", "perf1"))

    def test_v3_five_entries_drops_num_cu_and_chiplets(self):
        out = perfRunner.parse_tuning_db_line(["gfx900", "120", "2", "config1", "perf1", "1.5"])
        self.assertEqual(out, ("gfx900", "config1", "perf1"))

    def test_v3_extra_columns_are_ignored(self):
        out = perfRunner.parse_tuning_db_line(
            ["gfx90x", "304", "8", "gemm -m 1024", "perf_x", "2.0", "extra"])
        self.assertEqual(out, ("gfx90x", "gemm -m 1024", "perf_x"))

    def test_non_numeric_third_column_is_read_as_v2(self):
        """A 5-column row whose third field is not a number is v2 plus a trailing
        metric, not v3; the config must come from column 3 rather than column 4."""
        out = perfRunner.parse_tuning_db_line(["gfx900", "120", "config1", "perf1", "1.5"])
        self.assertEqual(out, ("gfx900", "config1", "perf1"))

    def test_invalid_returns_none(self):
        self.assertIsNone(perfRunner.parse_tuning_db_line([]))
        self.assertIsNone(perfRunner.parse_tuning_db_line(["a"]))
        self.assertIsNone(perfRunner.parse_tuning_db_line(["a", "b"]))


class ReadTuningDbTest(TempFileTestCase):
    """Tests for read_tuning_db."""

    def test_read_empty_file(self):
        path = self.write_temp(".tsv", "")
        self.assertEqual(perfRunner.read_tuning_db(path, PerfConfiguration), {})

    def test_read_with_header_and_comments(self):
        # Written without -transO or -supportsSplitK; canonicalization adds both
        # (split-K defaults to true for gemm), and the key is the canonical form.
        gemm_a = ("-t f32 -out_datatype f32 -transA false -transB false "
                  "-g 1 -m 1024 -n 512 -k 769")
        gemm_b = ("-t f16 -out_datatype f16 -transA false -transB true "
                  "-g 1 -m 256 -n 128 -k 64")
        gemm_a_key = ("-t f32 -out_datatype f32 -transA false -transB false -transO false "
                      "-g 1 -m 1024 -n 512 -k 769 -supportsSplitK true")
        gemm_b_key = ("-t f16 -out_datatype f16 -transA false -transB true -transO false "
                      "-g 1 -m 256 -n 128 -k 64 -supportsSplitK true")
        path = self.write_temp(
            ".tsv", "# arch\tconfig\tperfconfig\n"
            f"gfx900\t{gemm_a}\tperf_1\n"
            "\n"
            f"gfx900\t{gemm_b}\tperf_2\n")

        db = perfRunner.read_tuning_db(path, GemmConfiguration, num_cu=120, num_chiplets=1)

        self.assertEqual(len(db), 2)
        self.assertEqual(db[("gfx900", gemm_a_key)], "perf_1")
        self.assertEqual(db[("gfx900", gemm_b_key)], "perf_2")

    def test_read_skips_unparseable_entries(self):
        """Entries that don't parse under the active conf_class -- different op,
        malformed, or .mlir keys from `tuningRunner --config foo.mlir` -- are
        skipped. They could never match perfRunner's canonical-string lookups
        anyway."""
        valid_gemm = ("-t f32 -out_datatype f32 -transA false -transB false "
                      "-g 1 -m 1024 -n 512 -k 769")
        valid_gemm_key = ("-t f32 -out_datatype f32 -transA false -transB false -transO false "
                          "-g 1 -m 1024 -n 512 -k 769 -supportsSplitK true")
        # A conv config: cannot be parsed under GemmConfiguration.
        conv_entry = ("convfp16 -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 "
                      "-k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1")
        # A truly malformed gemm config: missing required fields.
        malformed_gemm = "-g 1 -m 1024"
        # An .mlir path written by `tuningRunner --config foo.mlir`.
        mlir_path = "/path/to/fusion_kernel.mlir"
        path = self.write_temp(
            ".tsv", f"gfx900\t{valid_gemm}\tperf_ok\n"
            f"gfx900\t{conv_entry}\tperf_conv\n"
            f"gfx900\t{malformed_gemm}\tperf_bad\n"
            f"gfx900\t{mlir_path}\tperf_mlir\n")

        db = perfRunner.read_tuning_db(path, GemmConfiguration, num_cu=120, num_chiplets=1)

        self.assertEqual(len(db), 1)
        self.assertEqual(db[("gfx900", valid_gemm_key)], "perf_ok")

    def test_read_nonexistent_returns_none(self):
        self.assertIsNone(perfRunner.read_tuning_db("/nonexistent/path.tsv", PerfConfiguration))


class ParseDataTypesTest(unittest.TestCase):
    """Tests for parse_data_types (gemm data types)."""

    def test_empty_returns_defaults(self):
        dtypes, out_map = perfRunner.parse_data_types(None)
        self.assertIn("f32", dtypes)
        self.assertEqual(out_map.get("f32"), "f32")

    def test_single_type(self):
        dtypes, out_map = perfRunner.parse_data_types(["f16"])
        self.assertEqual(dtypes, ["f16"])
        self.assertEqual(out_map["f16"], "f16")

    def test_i8_maps_to_i32(self):
        dtypes, out_map = perfRunner.parse_data_types(["i8"])
        self.assertIn("i8", dtypes)
        self.assertEqual(out_map["i8"], "i32")

    def test_fp8_maps_to_f32(self):
        _dtypes, out_map = perfRunner.parse_data_types(["fp8"])
        self.assertEqual(out_map["fp8"], "f32")

    def test_pair_notation(self):
        dtypes, out_map = perfRunner.parse_data_types(["fp8_fp8"])
        self.assertIn("fp8", dtypes)
        self.assertEqual(out_map["fp8"], "fp8")


class LayoutHelpersTest(unittest.TestCase):
    """Tests for input/output/filter layout conversion."""

    def test_input_layouts(self):
        self.assertEqual(perfRunner.input_layouts("NCHW"), "nchw")

    def test_output_layouts(self):
        # OUTPUT_LAYOUT_MAP: C -> k, so NCHW -> nkhw
        self.assertEqual(perfRunner.output_layouts("NCHW"), "nkhw")

    def test_filter_layouts(self):
        # FILTER_LAYOUT_MAP: H -> y, W -> x, so NCHW -> kcyx
        self.assertEqual(perfRunner.filter_layouts("NCHW"), "kcyx")

    def test_inverse_roundtrip(self):
        layout = "NHWC"
        self.assertEqual(perfRunner.inverse_input_layouts(perfRunner.input_layouts(layout)), layout)
        self.assertEqual(perfRunner.inverse_output_layouts(perfRunner.output_layouts(layout)),
                         layout)
        self.assertEqual(perfRunner.inverse_filter_layouts(perfRunner.filter_layouts(layout)),
                         layout)


def make_conv_commandline(fil, inp, out, group=1):
    """Build a minimal conv commandline (as a token list) with the given layouts."""
    return ("conv -F 1 -f {f} -I {i} -O {o} -n 1 -c 8 -H 16 -W 16 -k 8 "
            "-y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g {g}").format(f=fil, i=inp, o=out,
                                                                     g=group).split()


class RocmlirLayoutToMiopenTest(unittest.TestCase):
    """Tests for rocmlir_layout_to_miopen (single layout string -> MIOpen name).

    MIOpenDriver only accepts NCHW/NHWC, so a rocMLIR layout is only usable once the
    group dim is dropped (MIOpen passes the group count via -g) and the spatial dims
    are renamed 0->H, 1->W. Anything else has no faithful MIOpen equivalent.
    """

    def test_channel_first_maps_to_nchw(self):
        """Dropping G and renaming 0/1 leaves the channel second, i.e. NCHW."""
        self.assertEqual(perfRunner.rocmlir_layout_to_miopen("NGC01"), "NCHW")
        self.assertEqual(perfRunner.rocmlir_layout_to_miopen("GNC01"), "NCHW")
        self.assertEqual(perfRunner.rocmlir_layout_to_miopen("NC0G1"), "NCHW")

    def test_channel_last_maps_to_nhwc(self):
        """A trailing channel dim maps to NHWC."""
        self.assertEqual(perfRunner.rocmlir_layout_to_miopen("N01GC"), "NHWC")
        self.assertEqual(perfRunner.rocmlir_layout_to_miopen("GN01C"), "NHWC")

    def test_already_miopen_layouts_pass_through(self):
        """NCHW/NHWC are returned unchanged."""
        self.assertEqual(perfRunner.rocmlir_layout_to_miopen("NCHW"), "NCHW")
        self.assertEqual(perfRunner.rocmlir_layout_to_miopen("NHWC"), "NHWC")

    def test_output_channel_letter_k_treated_as_c(self):
        """The output tensor spells the channel dim as K; MIOpen still wants NCHW/NHWC."""
        self.assertEqual(perfRunner.rocmlir_layout_to_miopen("NGK01"), "NCHW")
        self.assertEqual(perfRunner.rocmlir_layout_to_miopen("N01GK"), "NHWC")

    def test_unrepresentable_orderings_return_none(self):
        """Orderings that aren't NCHW/NHWC (channel or spatial in the wrong slot) skip."""
        self.assertIsNone(perfRunner.rocmlir_layout_to_miopen("G0NC1"))
        self.assertIsNone(perfRunner.rocmlir_layout_to_miopen("01NGC"))


class ConvCommandlineToMiopenLayoutsTest(unittest.TestCase):
    """Tests for conv_commandline_to_miopen_layouts (whole commandline translate-or-skip)."""

    def test_consistent_nchw_config_is_translated(self):
        """A config whose filter/input/output all map to NCHW is translated."""
        result = perfRunner.conv_commandline_to_miopen_layouts(
            make_conv_commandline("GNC01", "NGC01", "NGC01"))
        self.assertIsNotNone(result)
        for flag in ("-f", "-I", "-O"):
            self.assertEqual(result[result.index(flag) + 1], "NCHW")

    def test_consistent_nhwc_config_is_translated(self):
        """A config whose filter/input/output all map to NHWC is translated."""
        result = perfRunner.conv_commandline_to_miopen_layouts(
            make_conv_commandline("GN01C", "N01GC", "N01GC"))
        self.assertIsNotNone(result)
        for flag in ("-f", "-I", "-O"):
            self.assertEqual(result[result.index(flag) + 1], "NHWC")

    def test_group_conv_layout_is_still_translated(self):
        """Dropping G from the layout is valid; the group count rides on -g."""
        result = perfRunner.conv_commandline_to_miopen_layouts(
            make_conv_commandline("GNC01", "NGC01", "NGC01", group=2))
        self.assertIsNotNone(result)
        self.assertEqual(result[result.index("-g") + 1], "2")

    def test_unrepresentable_layout_is_skipped(self):
        """A layout with no NCHW/NHWC equivalent makes the whole config skip."""
        self.assertIsNone(
            perfRunner.conv_commandline_to_miopen_layouts(
                make_conv_commandline("G0NC1", "G0NC1", "NGC01", group=3)))

    def test_mixed_nchw_nhwc_config_is_skipped(self):
        """MIOpen has no solver for mixed filter/input/output layouts, so skip."""
        self.assertIsNone(
            perfRunner.conv_commandline_to_miopen_layouts(
                make_conv_commandline("GNC01", "NGC01", "N01GC")))


class GetNanosecondsTest(TempFileTestCase):
    """Tests for get_nanoseconds (reads the CSV rocprof leaves behind)."""

    def test_missing_file_returns_nan(self):
        self.assertTrue(math.isnan(perfRunner.get_nanoseconds("/nonexistent/path.csv")))

    def test_valid_csv(self):
        path = self.write_temp(".csv", "KernelName,AverageNs,SomeOther\n"
                               "kern1,1000,0\n"
                               "kern2,2000,0\n")
        self.assertEqual(perfRunner.get_nanoseconds(path), 3000)


class GetProfilerOutputPathTest(unittest.TestCase):
    """Tests for get_profiler_output_path (arch-dependent path)."""

    def test_gfx950_returns_base(self):
        self.assertEqual(perfRunner.get_profiler_output_path("gfx950", "results.csv"),
                         "results.csv")

    def test_other_arch_returns_pmc_subdir(self):
        self.assertEqual(perfRunner.get_profiler_output_path("gfx900", "results.csv"),
                         os.path.join("pmc_1", "results.csv"))


class GetMetricArgsForRocprofTest(unittest.TestCase):
    """Tests for get_metric_args_for_rocprof."""

    def test_gfx950_no_metrics(self):
        self.assertEqual(perfRunner.get_metric_args_for_rocprof("gfx950"), [])

    def test_other_arch_uses_metrics_file(self):
        args = perfRunner.get_metric_args_for_rocprof("gfx900")
        self.assertIn("-i", args)
        self.assertTrue(any("rocmlir_metrics" in str(x) for x in args))


class GetMilisecondsTest(unittest.TestCase):
    """Tests for get_miliseconds (kernel time parsing)."""

    def test_match(self):
        self.assertEqual(perfRunner.get_miliseconds(b"some output\nkernel time: 1.234\n"), 1.234)

    def test_no_match_returns_nan(self):
        self.assertTrue(math.isnan(perfRunner.get_miliseconds(b"no kernel time here")))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
