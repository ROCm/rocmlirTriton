#!/usr/bin/env python3
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Resume bookkeeping for tuningRunner.py: the ``.state`` sidecar file, the
cache of already-tuned configs read back from an output TSV, and the debug TSV
writer.

A tuning run that is interrupted (or killed by a hang) must resume without
re-tuning what it already finished, which makes these files part of the
contract between two separate invocations. Everything here is file I/O and
state transitions, so no GPU is involved.

# RUN: %python %s %t
"""

import json
import sys
import unittest
from pathlib import Path

MLIR_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = MLIR_DIR / "utils" / "performance"
sys.path.insert(0, str(PERF_DIR))

from perfRunner import GemmConfiguration, canonicalize_config  # noqa: E402
from tuningRunner import (  # noqa: E402
    ConfigState, DebugFileWriter, Options, TunedConfigsCache, TuningResult, TuningState,
    TuningStateFile, get_git_commit_hash, get_state_filepath)

TMP_PREFIX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/tuningRunner-state")

ARCH = "gfx900"
NUM_CU = 64
NUM_CHIPLETS = 1
OUTPUT_HEADER = ("# arch\tnumCUs\tnumChiplets\ttestVector\tperfConfig\tTFlops\ttuningSpace\t"
                 "commitId\ttimestamp\tdurationSec\n")


def make_options(output, arch=ARCH, num_cu=NUM_CU, num_chiplets=NUM_CHIPLETS, tuning_space="full"):
    """Build the Options bundle TunedConfigsCache.from_output_file reads."""
    return Options(
        chip=arch,
        arch=arch,
        num_cu=num_cu,
        num_chiplets=num_chiplets,
        debug=False,
        debug_quick_tune_data=False,
        quiet=False,
        verbose=False,
        tuning_space_kind=tuning_space,
        rocmlir_gen_flags="",
        disable_verify_winning_config=True,
        verify_all_perfconfigs=False,
        output=output,
        abort_on_error=False,
        retune=False,
        retry_states=frozenset(),
        gpu_ids=[0],
        num_cpus=None,
        wait_for_compiles=False,
        flush_last_level_cache=False,
        timeout=None,
        verify_timeout=600,
        perf_config_timeout=0,
        gpu_run_timeout=0,
    )


class TempFileTestCase(unittest.TestCase):
    """Base class handing each test method its own file under lit's %t prefix."""

    def temp_path(self, suffix, contents=None):
        """Return a per-test temporary path, optionally seeded with contents."""
        path = Path(f"{TMP_PREFIX}.{self._testMethodName}{suffix}")
        if contents is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(contents)
        return path


class GetStateFilepathTest(unittest.TestCase):
    """Tests for get_state_filepath."""

    def test_stdout_returns_none(self):
        self.assertIsNone(get_state_filepath("-"))

    def test_normal_path(self):
        self.assertEqual(get_state_filepath("out.tsv"), "out.tsv.state")


class TuningStateTest(unittest.TestCase):
    """Tests for TuningState (in-memory state transitions)."""

    def test_empty_should_skip_returns_false_for_unknown(self):
        self.assertFalse(TuningState().should_skip("config1"))

    def test_failed_should_skip_without_retry(self):
        state = TuningState()
        state.set_failed("config1")
        self.assertTrue(state.should_skip("config1"))
        self.assertFalse(state.should_skip("config1", retry_states=frozenset({ConfigState.FAILED})))

    def test_timed_out_and_crashed_skipped(self):
        state = TuningState()
        state.set_timed_out("c1")
        state.set_failed("c2")
        self.assertEqual(state.timed_out_count(), 1)
        self.assertEqual(state.failed_count(), 1)

    def test_promote_running_to_interrupted(self):
        state = TuningState()
        state.set_running("c1")
        self.assertEqual(state.promote_running_to_interrupted(), 1)
        self.assertEqual(state.configs["c1"], ConfigState.INTERRUPTED)

    def test_remove_clears_config(self):
        state = TuningState()
        state.set_failed("c1")
        state.remove("c1")
        self.assertTrue(state.is_empty())


class TuningStateFileTest(TempFileTestCase):
    """Tests for TuningStateFile (state persisted next to the output TSV)."""

    _TV_A = ("-t f32 -out_datatype f32 -transA false -transB false -transO false "
             "-g 1 -m 1024 -n 512 -k 769")
    _TV_B = ("-t f16 -out_datatype f16 -transA false -transB true -transO false "
             "-g 1 -m 256 -n 128 -k 64")
    _CONTEXT_KEY = f"{ARCH}/{NUM_CU}/{NUM_CHIPLETS}/full"

    def make_state_file(self, filepath):
        return TuningStateFile(filepath,
                               chip=ARCH,
                               arch=ARCH,
                               num_cu=NUM_CU,
                               num_chiplets=NUM_CHIPLETS,
                               tuning_space="full",
                               conf_class=GemmConfiguration)

    def test_no_filepath_is_noop(self):
        state_file = self.make_state_file(None)
        state_file.set_running(self._TV_A)
        state_file.set_failed(self._TV_A)
        self.assertEqual(state_file.state.failed_count(), 1)

    def test_save_and_load(self):
        path = self.temp_path(".state", json.dumps({"contexts": {}}))
        state_file = self.make_state_file(str(path))
        state_file.set_failed(self._TV_A)
        state_file.set_timed_out(self._TV_B)

        reloaded = self.make_state_file(str(path))

        self.assertEqual(reloaded.state.configs.get(self._TV_A), ConfigState.FAILED)
        self.assertEqual(reloaded.state.configs.get(self._TV_B), ConfigState.TIMED_OUT)

    def test_running_becomes_crashed_on_load(self):
        path = self.temp_path(
            ".state", json.dumps({"contexts": {
                self._CONTEXT_KEY: {
                    self._TV_A: "running"
                }
            }}))

        state_file = self.make_state_file(str(path))

        self.assertEqual(state_file.state.configs.get(self._TV_A), ConfigState.CRASHED)

    def test_old_state_file_configs_are_canonicalized(self):
        """Non-canonical test vectors in a state file are canonicalized on load."""
        non_canonical = ("-g 1 -m 1024 -k 769 -n 512 -t f32 -out_datatype f32 "
                         "-transA false -transB false")
        self.assertNotEqual(non_canonical, self._TV_A)
        path = self.temp_path(
            ".state", json.dumps({"contexts": {
                self._CONTEXT_KEY: {
                    non_canonical: "failed"
                }
            }}))

        state_file = self.make_state_file(str(path))

        self.assertEqual(state_file.state.configs.get(self._TV_A), ConfigState.FAILED)
        self.assertIsNone(state_file.state.configs.get(non_canonical))


class TunedConfigsCacheTest(TempFileTestCase):
    """Tests for TunedConfigsCache.from_output_file (parsing only, no GPU)."""

    def test_missing_file_returns_empty_cache(self):
        cache = TunedConfigsCache.from_output_file(make_options("/nonexistent/out.tsv"),
                                                   GemmConfiguration)
        self.assertEqual(cache.count(), 0)

    def test_stdout_output_returns_empty(self):
        cache = TunedConfigsCache.from_output_file(make_options("-"), GemmConfiguration)
        self.assertEqual(cache.count(), 0)

    def test_parse_new_format_tsv(self):
        test_vector = ("-t f32 -out_datatype f32 -transA false -transB false -transO false "
                       "-g 1 -m 1024 -n 512 -k 769")
        path = self.temp_path(
            ".tsv", OUTPUT_HEADER + f"{ARCH}\t{NUM_CU}\t{NUM_CHIPLETS}\t{test_vector}\t"
            "perf_best\t1.5\tfull\tabc123\t2025-01-01T00:00:00Z\t10.0\n")

        cache = TunedConfigsCache.from_output_file(make_options(str(path)), GemmConfiguration)

        self.assertEqual(cache.count(), 1)
        result = cache.get(test_vector)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertEqual(result.winning_config, "perf_best")
        self.assertEqual(result.max_tflops, 1.5)

    def test_arch_mismatch_not_loaded(self):
        path = self.temp_path(
            ".tsv", OUTPUT_HEADER + "gfx1030\t72\t1\t-g 1 -m 1024\tperf_x\t1.0\tfull\tx\t"
            "2025-01-01T00:00:00Z\t5.0\n")

        cache = TunedConfigsCache.from_output_file(make_options(str(path), arch="gfx900"),
                                                   GemmConfiguration)

        self.assertEqual(cache.count(), 0)

    def test_cache_loaded_with_canonical_key(self):
        """from_output_file canonicalizes test vectors so cache lookups match the
        keys perfRunner and the state file use."""
        raw = "-g 1 -m 1024 -k 769 -n 512 -t f32 -out_datatype f32 -transA false -transB false"
        canonical = canonicalize_config(raw, GemmConfiguration, ARCH, NUM_CU, NUM_CHIPLETS)
        path = self.temp_path(
            ".tsv", OUTPUT_HEADER + f"{ARCH}\t{NUM_CU}\t{NUM_CHIPLETS}\t{raw}\tperf_best\t1.5\t"
            "full\tabc123\t2025-01-01T00:00:00Z\t10.0\n")

        cache = TunedConfigsCache.from_output_file(make_options(str(path)), GemmConfiguration)

        self.assertEqual(cache.count(), 1)
        self.assertIsNone(cache.get(raw), "raw (non-canonical) key should not match")
        result = cache.get(canonical)
        self.assertIsNotNone(result, "canonical key should match")
        self.assertEqual(result.winning_config, "perf_best")


class DebugFileWriterTest(TempFileTestCase):
    """DebugFileWriter rejects appending rows whose schema would not match an
    existing header, since quickTuningGen reads these files as one table."""

    @staticmethod
    def make_result(entries):
        return TuningResult(test_vector="-g 1 -m 1 -n 1 -k 1",
                            success=True,
                            gpu_id=0,
                            duration_seconds=1.0,
                            timestamp="2026-01-01T00:00:00Z",
                            winning_config="cfg",
                            max_tflops=1.0,
                            entries=entries)

    def test_fresh_file_writes_header(self):
        path = self.temp_path(".tsv.debug")
        row = {"M": 1, "N": 2, "PerfConfig": "p", "TFlops": 1.0}

        with DebugFileWriter(str(path)) as writer:
            writer.write_result(self.make_result([row]))

        contents = path.read_text().splitlines()

        self.assertEqual(contents[0], "M\tN\tPerfConfig\tTFlops")
        self.assertEqual(contents[1], "1\t2\tp\t1.0")

    def test_append_with_same_schema_skips_header(self):
        path = self.temp_path(".tsv.debug")
        first_row = {"M": 1, "N": 2, "PerfConfig": "p1", "TFlops": 1.0}
        second_row = {"M": 3, "N": 4, "PerfConfig": "p2", "TFlops": 2.0}

        with DebugFileWriter(str(path)) as writer:
            writer.write_result(self.make_result([first_row]))
        with DebugFileWriter(str(path)) as writer:
            writer.write_result(self.make_result([second_row]))

        contents = path.read_text().splitlines()

        # One header, two data rows: the second open must not re-emit the header.
        self.assertEqual(contents, [
            "M\tN\tPerfConfig\tTFlops",
            "1\t2\tp1\t1.0",
            "3\t4\tp2\t2.0",
        ])

    def test_append_with_different_schema_raises(self):
        path = self.temp_path(".tsv.debug")
        gemm_row = {"M": 1, "N": 2, "K": 3, "PerfConfig": "p", "TFlops": 1.0}
        conv_row = {"H": 1, "W": 2, "PerfConfig": "p", "TFlops": 1.0}

        with DebugFileWriter(str(path)) as writer:
            writer.write_result(self.make_result([gemm_row]))

        with DebugFileWriter(str(path)) as writer:
            with self.assertRaisesRegex(ValueError, "schema that does not match"):
                writer.write_result(self.make_result([conv_row]))

    def test_empty_existing_file_treated_as_fresh(self):
        path = self.temp_path(".tsv.debug", "")  # file exists but is empty
        row = {"M": 1, "N": 2, "PerfConfig": "p", "TFlops": 1.0}

        with DebugFileWriter(str(path)) as writer:
            writer.write_result(self.make_result([row]))

        self.assertEqual(path.read_text().splitlines()[0], "M\tN\tPerfConfig\tTFlops")


class GetGitCommitHashTest(unittest.TestCase):
    """The commitId column comes from get_git_commit_hash, which must degrade to
    a placeholder rather than raise when git isn't usable."""

    def test_returns_nonempty_string(self):
        commit_hash = get_git_commit_hash()
        self.assertIsInstance(commit_hash, str)
        self.assertTrue(commit_hash)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
