"""Resume coverage for ``tuningRunner.py --benchmark-artifacts``.

The benchmark half of a cross-compiled tuning run is the long one: it walks
every problem in an artifact bundle on a single GPU, so an interruption hours
in used to mean starting over. It now keeps the same two records in-place
tuning keeps -- the output TSV for the problems that succeeded, a ``.state``
file for the ones that did not -- and skips both on the next run.

These tests pin that bookkeeping: which failure lands in which state, what an
interrupt leaves behind, and which problems a resumed run re-attempts. Getting
it wrong is expensive and quiet, either re-benchmarking hours of finished work
or dropping problems from the output TSV that quickTuningGen then can't see.

Doesn't need a GPU: the tuning driver subprocess is stubbed, so no test here
compiles, launches, or times anything.

unittest's exit code is the verdict, so there is nothing to FileCheck; lit
prints the failing test's traceback on a non-zero exit.

# RUN: %python %s
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

# tuningRunner.py is on PATH (lit's mlir_rock_tools_dir, populated by
# ci-performance-scripts). Resolve it and add its directory to sys.path so we
# can import it as a module instead of shelling out to it.
_script = shutil.which('tuningRunner.py')
if _script is None:
    sys.exit("tuningRunner.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

import perfRunner  # noqa: E402
import tuningRunner  # noqa: E402
from tuningRunner import ConfigState  # noqa: E402

ARCH = "gfx1170"
CHIP = "gfx1170"
NUM_CU = 4
NUM_CHIPLETS = 1
TUNING_SPACE = "exhaustive"

# Stands in for this checkout's HEAD on both sides of the build-commit
# guardrail, so the tests neither depend on git nor have to waive the check.
COMMIT = "a" * 40

# Three problems is the smallest bundle that can show a partial run: one before
# the interesting one, one after.
VECTORS = [
    "-t f32 -F 1 -n 1",
    "-t f32 -F 1 -n 2",
    "-t f32 -F 1 -n 3",
]

# Two perf configs, the faster one second, so a winner has to be chosen rather
# than fallen into.
DRIVER_OUTPUT = "cfg-slow\t200.0\ncfg-fast\t100.0\n"


class FakeConfig:
    """Stand-in for a PerfConfiguration that round-trips its test vector.

    Only the surface ``run_benchmark_artifacts`` and its callees touch:
    construction from a command line, canonicalization back to one, and a table
    entry whose TFlops picks the winner.
    """

    def __init__(self, test_vector):
        self.test_vector = test_vector
        self.perfconfig = None

    @classmethod
    def from_command_line(cls, command_line, arch, num_cu, num_chiplets):
        return cls(" ".join(command_line))

    def to_command_line(self):
        return self.test_vector

    def set_perfconfig(self, perfconfig):
        self.perfconfig = perfconfig

    def table_entry(self, nanoseconds):
        return {
            "TestVector": self.test_vector,
            "PerfConfig": self.perfconfig,
            "TFlops": math.nan if math.isnan(nanoseconds) else 1000.0 / nanoseconds,
        }


class FakeDriver:
    """Stubbed rocmlir-tuning-driver with a scripted outcome per problem.

    Stands in for ``_run_pipeline``, recording which problems were benchmarked
    so a resumed run can be checked for what it did *not* re-run.
    """

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def __call__(self, commands, env=None, timeout=None):
        flag = next(part for part in commands[0] if part.startswith("--benchmark-artifacts="))
        problem_hash = os.path.basename(flag.split("=", 1)[1])
        self.calls.append(problem_hash)

        outcome = self.outcomes.get(problem_hash, "ok")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(cmd=commands[0], timeout=timeout or 1)
        if outcome == "interrupt":
            raise KeyboardInterrupt()
        if outcome == "gpu-timeout":
            return tuningRunner.GPU_TIMEOUT_EXIT_CODE, "", "GPU hung"
        if outcome == "fail":
            return 1, "", "driver exploded"
        if outcome == "no-winner":
            # Every perf config unmeasurable: applies to nothing on this target.
            return 0, f"cfg-slow\t{tuningRunner.NOT_APPLICABLE_STATUS}\n", ""
        return 0, DRIVER_OUTPUT, ""


def make_options(output, artifacts_dir, **overrides):
    """A frozen Options for the benchmark phase, overridable per test."""
    fields = dict(
        debug=False,
        debug_quick_tune_data=True,
        tuning_space_kind=TUNING_SPACE,
        quiet=True,
        verbose=False,
        chip=CHIP,
        arch=ARCH,
        num_cu=NUM_CU,
        num_chiplets=NUM_CHIPLETS,
        rocmlir_gen_flags="",
        verify_winning_config=False,
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
        perf_config_timeout=60,
        gpu_run_timeout=60,
        rep_ms=1,
        warmup_ms=1,
        two_stage_topk=0,
        coarse_rep_iters=0,
        coarse_warmup_iters=0,
        coarse_warmup_floor_ms=1,
        coarse_rel_sem_target=0.0,
        coarse_chunk_iters=1,
        coarse_min_rep_iters=1,
        benchmark_artifacts_dir=artifacts_dir,
        allow_commit_mismatch=False,
    )
    fields.update(overrides)
    return tuningRunner.Options(**fields)


def make_context(configs, options):
    """A TuningContext pointing at tools that are never actually executed."""
    nowhere = "/nonexistent"
    paths = perfRunner.Paths(configuration_file_path="",
                             mlir_paths=perfRunner.MLIRPaths(
                                 rocmlir_gen_path=f"{nowhere}/rocmlir-gen",
                                 rocmlir_driver_path=f"{nowhere}/rocmlir-driver",
                                 rocmlir_opt_path=f"{nowhere}/rocmlir-opt",
                                 rocm_run_path=f"{nowhere}/rocm-run",
                                 rocmlir_tuning_driver_path=f"{nowhere}/rocmlir-tuning-driver"))
    topology = tuningRunner.GpuTopology(
        gpus={0: tuningRunner.Gpu(gpu_id=0, sku="fake", numa_node=0)})
    return tuningRunner.TuningContext(configs=list(configs),
                                      conf_class=FakeConfig,
                                      paths=paths,
                                      options=options,
                                      gpu_topology=topology,
                                      numa_topology=tuningRunner.NumaTopology.discover())


class BenchmarkArtifactsResumeTest(unittest.TestCase):
    """Drives run_benchmark_artifacts over a fabricated artifact bundle."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.artifacts = os.path.join(self.tmpdir, "artifacts")
        self.output = os.path.join(self.tmpdir, "results.tsv")

        commit = mock.patch.object(tuningRunner, "get_git_commit_hash", return_value=COMMIT)
        commit.start()
        self.addCleanup(commit.stop)

        # _problem_hash keys off the target options, which are fixed across
        # every run in this file, so one throwaway Options builds the bundle.
        options = make_options(self.output, self.artifacts)
        self.hashes = {tv: tuningRunner._problem_hash(tv, options) for tv in VECTORS}
        self._write_bundle(options)

    def _write_bundle(self, options):
        """A bundle with a directory per problem, as --compile-only leaves it."""
        problems = {}
        for test_vector, problem_hash in self.hashes.items():
            os.makedirs(os.path.join(self.artifacts, "problems", problem_hash))
            problems[problem_hash] = {
                "testVector": test_vector,
                "commitId": COMMIT,
                "runId": "unit-test",
            }
        index = {
            "arch": ARCH,
            "numCUs": NUM_CU,
            "numChiplets": NUM_CHIPLETS,
            "tuningSpace": TUNING_SPACE,
            "rocmlirGenFlags": options.rocmlir_gen_flags,
            "problems": problems,
        }
        with open(os.path.join(self.artifacts, "index.json"), "w") as index_file:
            json.dump(index, index_file)

    def hash_of(self, test_vector):
        return self.hashes[test_vector]

    def run_benchmark(self, outcomes=None, status_only=False, **overrides):
        """One --benchmark-artifacts run; returns its verdict and the driver."""
        options = make_options(self.output, self.artifacts, **overrides)
        ctx = make_context(VECTORS, options)
        driver = FakeDriver(outcomes or {})
        with mock.patch.object(tuningRunner, "_run_pipeline", driver):
            succeeded = tuningRunner.run_benchmark_artifacts(ctx, status_only=status_only)
        return succeeded, driver

    def read_state(self):
        """The state file's entries for this target: test vector -> state."""
        path = f"{self.output}.state"
        if not os.path.exists(path):
            return {}
        with open(path) as state_file:
            contexts = json.load(state_file)["contexts"]
        key = f"{CHIP}/{NUM_CU}/{NUM_CHIPLETS}/{TUNING_SPACE}"
        return {tv: ConfigState(state) for tv, state in contexts.get(key, {}).items()}

    def read_column(self, path, column):
        """One column of a TSV, in file order.

        Handles both writers: OutputFileWriter comments its header out, while
        the debug file is pandas output whose header reappears verbatim every
        time a run reopens the file, so headers are recognized by name.
        """
        if not os.path.exists(path):
            return []
        values = []
        header = None
        with open(path) as tsv:
            for line in tsv:
                fields = line.rstrip("\n").split("\t")
                if not any(fields):
                    continue
                if fields[0].startswith("#") or fields[0] == column:
                    header = [field.lstrip("#").strip() for field in fields]
                    continue
                values.append(fields[header.index(column)])
        return values

    def tuned_vectors(self):
        return self.read_column(self.output, "testVector")

    def debug_vectors(self):
        return self.read_column(f"{self.output}.debug", "TestVector")

    def write_result_row(self, test_vector, *, arch=ARCH, num_cu=NUM_CU):
        """Append a finished result to the output TSV, as a past run would."""
        row = [
            arch,
            str(num_cu),
            str(NUM_CHIPLETS), test_vector, "cfg-fast", "10.0", TUNING_SPACE, COMMIT,
            "2026-01-01T00:00:00Z", "1.0"
        ]
        write_header = not os.path.exists(self.output)
        with open(self.output, "a") as tsv:
            if write_header:
                print("# " + "\t".join(tuningRunner.OUTPUT_HEADER_COLUMNS), file=tsv)
            print("\t".join(row), file=tsv)

    def test_first_run_benchmarks_every_problem(self):
        succeeded, driver = self.run_benchmark()

        self.assertTrue(succeeded)
        self.assertCountEqual(driver.calls, self.hashes.values())
        self.assertCountEqual(self.tuned_vectors(), VECTORS)
        # Nothing outstanding, so nothing for the next run to resume.
        self.assertEqual(self.read_state(), {})

    def test_winner_is_the_fastest_perfconfig(self):
        self.run_benchmark()

        self.assertEqual(set(self.read_column(self.output, "perfConfig")), {"cfg-fast"})

    def test_failure_is_recorded_then_skipped_on_resume(self):
        failed = VECTORS[1]
        succeeded, _ = self.run_benchmark(outcomes={self.hash_of(failed): "fail"})

        self.assertFalse(succeeded)
        self.assertEqual(self.read_state(), {failed: ConfigState.FAILED})
        self.assertCountEqual(self.tuned_vectors(), [VECTORS[0], VECTORS[2]])

        # A plain resume has nothing left to do: two problems are in the TSV and
        # the third is a known failure, which needs --retry to be re-attempted.
        succeeded, driver = self.run_benchmark()

        self.assertTrue(succeeded)
        self.assertEqual(driver.calls, [])
        self.assertCountEqual(self.tuned_vectors(), [VECTORS[0], VECTORS[2]])

    def test_retry_reattempts_only_the_failure(self):
        failed = VECTORS[1]
        self.run_benchmark(outcomes={self.hash_of(failed): "fail"})

        succeeded, driver = self.run_benchmark(retry_states=frozenset({ConfigState.FAILED}))

        self.assertTrue(succeeded)
        self.assertEqual(driver.calls, [self.hash_of(failed)])
        self.assertCountEqual(self.tuned_vectors(), VECTORS)
        self.assertEqual(self.read_state(), {})

    def test_retry_ignores_unrelated_states(self):
        failed = VECTORS[1]
        self.run_benchmark(outcomes={self.hash_of(failed): "fail"})

        # The one outstanding problem failed, not timed out, so a timeout-only
        # retry leaves it alone.
        succeeded, driver = self.run_benchmark(retry_states=frozenset({ConfigState.TIMED_OUT}))

        self.assertTrue(succeeded)
        self.assertEqual(driver.calls, [])
        self.assertEqual(self.read_state(), {failed: ConfigState.FAILED})

    def test_each_failure_kind_gets_its_own_state(self):
        outcomes = {
            self.hash_of(VECTORS[0]): "timeout",
            self.hash_of(VECTORS[1]): "gpu-timeout",
            self.hash_of(VECTORS[2]): "no-winner",
        }
        succeeded, _ = self.run_benchmark(outcomes=outcomes)

        self.assertFalse(succeeded)
        self.assertEqual(
            self.read_state(), {
                VECTORS[0]: ConfigState.TIMED_OUT,
                VECTORS[1]: ConfigState.GPU_TIMED_OUT,
                VECTORS[2]: ConfigState.FAILED,
            })
        self.assertEqual(self.tuned_vectors(), [])

    def test_bundle_faults_stay_unrecorded_and_retryable(self):
        # A missing problem directory is a broken bundle, not a bad problem: it
        # must not be recorded as a failure, or --retry would be needed to get
        # past a fault that a repaired bundle fixes on its own.
        missing = VECTORS[1]
        shutil.rmtree(os.path.join(self.artifacts, "problems", self.hash_of(missing)))

        succeeded, driver = self.run_benchmark()

        self.assertFalse(succeeded)
        self.assertNotIn(self.hash_of(missing), driver.calls)
        self.assertEqual(self.read_state(), {})

        succeeded, driver = self.run_benchmark()

        self.assertEqual(driver.calls, [])

    def test_interrupt_leaves_the_problem_retryable(self):
        interrupted = VECTORS[1]
        outcomes = {self.hash_of(interrupted): "interrupt"}

        with self.assertRaises(KeyboardInterrupt):
            self.run_benchmark(outcomes=outcomes)

        # Ctrl-C is nobody's fault, so the in-flight problem is parked as
        # interrupted rather than failed, and a plain resume picks it back up.
        self.assertEqual(self.read_state(), {interrupted: ConfigState.INTERRUPTED})
        self.assertEqual(self.tuned_vectors(), [VECTORS[0]])

        succeeded, driver = self.run_benchmark()

        self.assertTrue(succeeded)
        self.assertCountEqual(driver.calls, [self.hash_of(v) for v in VECTORS[1:]])
        self.assertCountEqual(self.tuned_vectors(), VECTORS)

    def test_resume_extends_the_output_files(self):
        interrupted = VECTORS[1]
        with self.assertRaises(KeyboardInterrupt):
            self.run_benchmark(outcomes={self.hash_of(interrupted): "interrupt"})
        self.run_benchmark()

        # quickTuningGen consumes the .debug file, so a resumed run has to add
        # to it rather than replace what the interrupted run already measured.
        self.assertCountEqual(self.tuned_vectors(), VECTORS)
        self.assertCountEqual(set(self.debug_vectors()), VECTORS)

    def test_retune_restarts_from_scratch(self):
        self.run_benchmark(outcomes={self.hash_of(VECTORS[1]): "fail"})

        succeeded, driver = self.run_benchmark(retune=True)

        self.assertTrue(succeeded)
        self.assertCountEqual(driver.calls, self.hashes.values())
        # The previous run's rows are discarded, not appended to: one row per
        # problem, no duplicates from the first attempt.
        self.assertCountEqual(self.tuned_vectors(), VECTORS)
        self.assertEqual(self.read_state(), {})

    def test_retune_forgets_the_previous_runs_failures(self):
        self.run_benchmark(outcomes={self.hash_of(VECTORS[1]): "fail"})
        self.assertEqual(self.read_state(), {VECTORS[1]: ConfigState.FAILED})

        # Abort on the first problem so this run never reaches the one that
        # failed last time. Whatever is left in the state file is then exactly
        # what --retune carried over, and it should be nothing.
        self.run_benchmark(outcomes={self.hash_of(VECTORS[0]): "fail"},
                           retune=True,
                           abort_on_error=True)

        self.assertEqual(self.read_state(), {VECTORS[0]: ConfigState.FAILED})

    def test_status_reports_without_benchmarking(self):
        self.run_benchmark(outcomes={self.hash_of(VECTORS[1]): "fail"})
        before = self.tuned_vectors()

        succeeded, driver = self.run_benchmark(status_only=True)

        self.assertTrue(succeeded)
        self.assertEqual(driver.calls, [])
        self.assertEqual(self.tuned_vectors(), before)
        self.assertEqual(self.read_state(), {VECTORS[1]: ConfigState.FAILED})

    def test_status_reports_a_retune_without_carrying_it_out(self):
        # Asking what a --retune would do must not be the thing that does it.
        self.run_benchmark(outcomes={self.hash_of(VECTORS[1]): "fail"})
        before = self.tuned_vectors()

        succeeded, driver = self.run_benchmark(status_only=True, retune=True)

        self.assertTrue(succeeded)
        self.assertEqual(driver.calls, [])
        self.assertEqual(self.tuned_vectors(), before)
        self.assertEqual(self.read_state(), {VECTORS[1]: ConfigState.FAILED})

    def test_abort_on_error_stops_at_the_first_failure(self):
        succeeded, driver = self.run_benchmark(outcomes={self.hash_of(VECTORS[0]): "fail"},
                                               abort_on_error=True)

        self.assertFalse(succeeded)
        self.assertEqual(driver.calls, [self.hash_of(VECTORS[0])])
        # Aborting still records the failure, so the next run knows why it stopped.
        self.assertEqual(self.read_state(), {VECTORS[0]: ConfigState.FAILED})

    def test_results_from_another_target_are_not_treated_as_done(self):
        # One TSV accumulates rows for every target it was ever pointed at.
        # Rows measured on other hardware say nothing about this GPU, so
        # counting them as finished work would publish timings from the wrong
        # machine for problems this run never benchmarked.
        for test_vector in VECTORS:
            self.write_result_row(test_vector, arch="gfx942:sramecc+:xnack-", num_cu=304)

        succeeded, driver = self.run_benchmark()

        self.assertTrue(succeeded)
        self.assertCountEqual(driver.calls, self.hashes.values())

    def test_a_bundle_built_for_another_target_is_refused(self):
        # The target is part of the problem hash, so without this check a
        # mismatched run would read as a bundle full of missing problems
        # instead of the configuration error it is.
        with self.assertRaises(tuningRunner.TuningError) as refusal:
            self.run_benchmark(num_cu=NUM_CU + 1)

        self.assertIn("--target-num-cu", str(refusal.exception))
        self.assertEqual(self.tuned_vectors(), [])

    def test_artifacts_from_another_commit_are_refused(self):
        # The two hosts build their own tools; a bundle from a different commit
        # can disagree with this binary about bundle format or grid logic.
        with mock.patch.object(tuningRunner, "get_git_commit_hash", return_value="b" * 40):
            succeeded, driver = self.run_benchmark()

        self.assertFalse(succeeded)
        self.assertEqual(driver.calls, [])
        # The drift belongs to the bundle, not to the problems, so nothing is
        # recorded against them: rebuild and resume, no --retry needed.
        self.assertEqual(self.read_state(), {})
        self.assertEqual(self.tuned_vectors(), [])

    def test_allow_commit_mismatch_benchmarks_them_anyway(self):
        with mock.patch.object(tuningRunner, "get_git_commit_hash", return_value="b" * 40):
            succeeded, driver = self.run_benchmark(allow_commit_mismatch=True)

        self.assertTrue(succeeded)
        self.assertCountEqual(driver.calls, self.hashes.values())

    def test_a_failure_reports_how_to_reproduce_it(self):
        # The benchmark host is often somebody else's machine reached over ssh,
        # so the log is all a reader gets. It has to name the shape, the bundle
        # directory, and the command, without a trip back to index.json.
        failed = VECTORS[1]
        with self.assertLogs(tuningRunner.logger, level="ERROR") as captured:
            self.run_benchmark(outcomes={self.hash_of(failed): "fail"})
        report = "\n".join(captured.output)

        self.assertIn(failed, report)
        self.assertIn(self.hash_of(failed), report)
        self.assertIn("[2/3]", report)
        self.assertIn("Reproduce:", report)
        self.assertIn(f"--benchmark-artifacts={self.artifacts}", report)
        self.assertIn("Exit code: 1", report)
        self.assertIn("driver exploded", report)

    def test_the_closing_summary_says_what_to_retry(self):
        with self.assertLogs(tuningRunner.logger, level="INFO") as captured:
            self.run_benchmark(outcomes={
                self.hash_of(VECTORS[0]): "gpu-timeout",
                self.hash_of(VECTORS[1]): "fail",
            })
        report = "\n".join(captured.output)

        self.assertIn("1 benchmarked, 2 unsuccessful", report)
        self.assertIn(f"{self.output}.state", report)
        self.assertIn("--retry failed gpu_timed_out", report)

    def test_the_summary_still_lands_when_the_run_is_interrupted(self):
        # The run that gets cut short is exactly the one whose reader needs to
        # know how much was banked before it stopped.
        with self.assertLogs(tuningRunner.logger, level="INFO") as captured:
            with self.assertRaises(KeyboardInterrupt):
                self.run_benchmark(outcomes={self.hash_of(VECTORS[1]): "interrupt"})

        self.assertIn("1 benchmarked", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
