# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Unit tests for ``crossCompile.py``, the local-compile / remote-benchmark
wrapper around tuningRunner.py.

crossCompile.py is almost entirely command-line and shell-script construction:
a mistake in a flag name, in the local/remote split of the passthrough flags,
or in the ordering of the remote session script only shows up hours into a real
cross-compile run against a remote GPU host. These tests pin that construction
directly, stubbing every subprocess boundary.

Doesn't need a GPU, and never spawns ssh, tar, docker, or tuningRunner.py.

unittest's exit code is the verdict, so there is nothing to FileCheck; lit
prints the failing test's traceback on a non-zero exit.

# RUN: %python %s
"""

import argparse
import contextlib
import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

# crossCompile.py is on PATH (lit's mlir_rock_tools_dir, populated by
# ci-performance-scripts). Resolve it and add its directory to sys.path so we
# can import it as a module instead of shelling out to it.
_script = shutil.which('crossCompile.py')
if _script is None:
    sys.exit("crossCompile.py not on PATH; did you run "
             "`ninja ci-performance-scripts`?")
sys.path.insert(0, os.path.dirname(_script))

import crossCompile  # noqa: E402

REMOTE_ARTIFACTS = "/remote/artifacts"
REMOTE_REPO = "/remote/home/rocmlirTriton"
REMOTE_BUILD = REMOTE_REPO + "/build"
CONTAINER = "tuning-container"
TARGET_ARCH = "gfx950:sramecc+:xnack-"


def flag_value(cmd, flag):
    """The argument following ``flag``, e.g. --op gemm -> 'gemm'."""
    return cmd[cmd.index(flag) + 1]


class FakeStream:
    """Minimal stand-in for ``Popen.stdout``; b"" means EOF."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.closed = False

    def readline(self):
        return self._lines.pop(0) if self._lines else b""

    def close(self):
        self.closed = True


class BlockingStream:
    """Stand-in for the stdout of a remote that is alive but silent.

    ``readline`` blocks until the process is killed, mirroring a real pipe whose
    write end only closes when ssh dies.
    """

    def __init__(self, released):
        self._released = released
        self.closed = False

    def readline(self):
        self._released.wait()
        return b""

    def close(self):
        self.closed = True


class FakeProcess:
    """Minimal stand-in for ``subprocess.Popen``."""

    def __init__(self, lines=(), returncode=0, alive=True, wait_times_out=False, silent=False):
        self._released = threading.Event()
        self.stdout = BlockingStream(self._released) if silent else FakeStream(lines)
        self.stdin = None
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self._alive = alive
        self._wait_times_out = wait_times_out

    def poll(self):
        return None if self._alive else self.returncode

    def wait(self, timeout=None):
        if timeout is not None and self._wait_times_out:
            self._wait_times_out = False
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        self._alive = False
        return self.returncode

    def terminate(self):
        self.terminated = True
        self._released.set()

    def kill(self):
        self.killed = True
        self._alive = False


class CrossCompileTestCase(unittest.TestCase):
    """Base fixture: a temp dir holding a real config file and output path.

    ``parse_args`` requires --configs-file to exist and creates the parent of
    --output, so everything it touches has to live under a temp dir.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.configs_file = self.tmp / "configs.txt"
        self.configs_file.write_text("-g 1 -m 64 -n 64 -k 64 -t f32\n")
        self.output = self.tmp / "results" / "out.tsv"
        self.local_artifacts = self.tmp / "artifacts"

    def base_argv(self):
        return [
            "--op", "gemm",
            "--tuning-space", "quick",
            "--target-arch", TARGET_ARCH,
            "--target-num-cu", "256",
            "--target-num-chiplets", "8",
            "--configs-file", str(self.configs_file),
            "--output", str(self.output),
            "--local-artifacts-dir", str(self.local_artifacts),
            "--remote-host", "gpuhost",
            "--remote-artifacts-dir", REMOTE_ARTIFACTS,
            "--remote-repo-dir", REMOTE_REPO,
            "--remote-docker-container", CONTAINER,
        ]  # yapf: disable

    def make_build_dir(self, name="build"):
        """A directory that passes --local-build-dir's rocMLIR build tree check."""
        build_dir = self.tmp / name
        (build_dir / "bin").mkdir(parents=True)
        (build_dir / "bin" / "rocmlir-gen").touch()
        return build_dir

    def parse(self, *extra):
        return crossCompile.parse_args(self.base_argv() + list(extra))

    def parse_expecting_error(self, *extra):
        """Run parse() expecting argparse to exit; returns the stderr text."""
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                self.parse(*extra)
        return stderr.getvalue()


class TestPathHelpers(CrossCompileTestCase):

    def test_logical_absolute_keeps_absolute_paths(self):
        absolute = Path("/already/absolute")
        self.assertEqual(crossCompile.logical_absolute(absolute), absolute)

    def test_logical_absolute_prefers_pwd_over_getcwd(self):
        # PWD preserves symlinked paths that os.getcwd() would resolve away,
        # which matters because the remote side is handed these paths verbatim.
        with mock.patch.dict(os.environ, {"PWD": "/logical/dir"}):
            self.assertEqual(crossCompile.logical_absolute(Path("configs.txt")),
                             Path("/logical/dir/configs.txt"))

    def test_existing_file_accepts_a_file(self):
        self.assertEqual(crossCompile.existing_file(str(self.configs_file)), self.configs_file)

    def test_existing_file_rejects_missing_path(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            crossCompile.existing_file(str(self.tmp / "nope.txt"))

    def test_existing_file_rejects_directory(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            crossCompile.existing_file(str(self.tmp))

    def test_absolute_remote_path_accepts_absolute(self):
        self.assertEqual(crossCompile.absolute_remote_path("/opt/x", "--remote-artifacts-dir"),
                         "/opt/x")

    def test_absolute_remote_path_rejects_relative(self):
        # A relative remote path would land wherever the SSH session starts.
        with self.assertRaises(argparse.ArgumentTypeError):
            crossCompile.absolute_remote_path("relative/x", "--remote-artifacts-dir")


class TestSshTarget(unittest.TestCase):

    def test_omits_user_by_default(self):
        self.assertEqual(crossCompile.ssh_target("gpuhost", None), "gpuhost")

    def test_uses_explicit_remote_user(self):
        self.assertEqual(crossCompile.ssh_target("gpuhost", "alice"), "alice@gpuhost")

    def test_passes_through_user_at_host(self):
        self.assertEqual(crossCompile.ssh_target("alice@gpuhost", None), "alice@gpuhost")

    def test_rejects_two_ways_of_saying_the_user(self):
        with self.assertRaises(ValueError):
            crossCompile.ssh_target("alice@gpuhost", "bob")


class TestArgumentPostProcessing(CrossCompileTestCase):

    def test_derives_remote_paths_from_the_artifacts_dir(self):
        args = self.parse()
        self.assertEqual(args.remote_target, "gpuhost")
        self.assertEqual(args.remote_configs_file, "/remote/artifacts/configs.txt")
        self.assertEqual(args.remote_output, "/remote/artifacts/out.tsv")

    def test_strips_trailing_slash_from_the_artifacts_dir(self):
        args = self.parse("--remote-artifacts-dir", "/remote/artifacts/")
        self.assertEqual(args.remote_artifacts_dir, "/remote/artifacts")
        self.assertEqual(args.remote_configs_file, "/remote/artifacts/configs.txt")

    def test_keeps_root_as_the_artifacts_dir(self):
        args = self.parse("--remote-artifacts-dir", "/")
        self.assertEqual(args.remote_artifacts_dir, "/")
        self.assertEqual(args.remote_configs_file, "/configs.txt")

    def test_defaults_the_ready_timeout_to_ten_minutes(self):
        # Has to cover interactive SSH auth plus a cold container start.
        self.assertEqual(self.parse().remote_ready_timeout, 600)

    def test_defaults_the_remote_session_timeout_to_six_hours(self):
        self.assertEqual(self.parse().remote_session_timeout, 6 * 60 * 60)

    def test_creates_the_local_output_directory(self):
        self.assertFalse(self.output.parent.exists())
        args = self.parse()
        self.assertTrue(args.output.parent.is_dir())

    def test_rejects_stdout_as_output(self):
        # tuningRunner.py accepts -o -, but results here are copied back as a
        # file, so there is nothing to stream to.
        self.assertIn("--output must be a file path", self.parse_expecting_error("--output", "-"))

    def test_rejects_an_output_that_would_delete_a_remote_input(self):
        # The compiled bundle, the config file and the result TSV share one
        # remote directory, and the session clears the result files out of it
        # before benchmarking.
        collisions = {
            "index.json": "the artifact index",
            "problems": "the compiled bundles",
            "configs.txt": "the config file",
        }
        for name, deleted in collisions.items():
            with self.subTest(output=name):
                self.assertIn(f"would make the remote delete {deleted}",
                              self.parse_expecting_error("--output", str(self.tmp / name)))

    def test_rejects_an_output_whose_state_file_would_delete_a_remote_input(self):
        # <output>.state is cleared alongside the TSV, so it needs the same check.
        state_named_config = self.tmp / "out.tsv.state"
        state_named_config.write_text("-g 1 -m 64 -n 64 -k 64 -t f32\n")
        self.assertIn(
            "would make the remote delete the config file",
            self.parse_expecting_error("--configs-file", str(state_named_config), "--output",
                                       str(self.tmp / "out.tsv")))

    def test_rejects_relative_remote_artifacts_dir(self):
        self.assertIn("must be an absolute remote path",
                      self.parse_expecting_error("--remote-artifacts-dir", "artifacts"))

    def test_requires_remote_repo_dir(self):
        argv = self.base_argv()
        option_index = argv.index("--remote-repo-dir")
        del argv[option_index:option_index + 2]

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                crossCompile.parse_args(argv)
        self.assertIn("--remote-repo-dir", stderr.getvalue())

    def test_rejects_missing_configs_file(self):
        self.assertIn("is not a file",
                      self.parse_expecting_error("--configs-file", str(self.tmp / "nope.txt")))

    def test_rejects_ssh_options_that_request_a_pty(self):
        conflicting_options = [
            ["--ssh-option=-t"],
            ["--ssh-option=-tt"],
            ["--ssh-option=-oRequestTTY=force"],
            ["--ssh-option=-o", "--ssh-option=RequestTTY=yes"],
            ["--ssh-option=-o", "--ssh-option=RequestTTY auto"],
        ]
        for options in conflicting_options:
            with self.subTest(options=options):
                self.assertIn("cannot request a PTY", self.parse_expecting_error(*options))

    def test_allows_explicit_request_tty_no(self):
        args = self.parse("--ssh-option=-oRequestTTY=no")
        self.assertIn("-oRequestTTY=no", args.ssh_option)

    def test_rejects_more_than_one_gpu_before_compiling(self):
        # The remote benchmark phase takes a single device. Rejecting it here
        # rather than on the remote saves a compile that can run for hours.
        self.assertIn("exactly one GPU", self.parse_expecting_error("--gpus", "0", "1"))

    def test_accepts_a_single_gpu(self):
        self.assertEqual(self.parse("--gpus", "3").gpus, [3])


class TestPassthroughFlags(CrossCompileTestCase):
    """The local compile phase and the remote benchmark phase get disjoint
    subsets of the tuning flags; forwarding a GPU flag to the GPU-less compile
    host (or a CPU flag to the benchmark host) makes tuningRunner.py exit."""

    SHARED = ["--debug", "--debug-quick-tune-data", "--verbose", "--timeout=60"]
    GPU_ONLY = [
        "--gpus", "2", "--verify-winning-config", "--verify-all-perfconfigs",
        "--flush-last-level-cache", "--gpu-run-timeout", "45", "--allow-commit-mismatch"
    ]
    # Forwarded to both phases whether or not the user named them, so the two
    # hosts cannot expand the config file differently.
    DEFAULT_DATA_TYPES = ["--data-type", "f32", "f16", "i8"]

    def passthrough(self, remote, *extra):
        args = self.parse(*extra)
        cmd = []
        crossCompile.add_passthrough_flags(cmd, args, remote=remote)
        return cmd

    def test_shared_flags_reach_both_phases(self):
        for remote in (False, True):
            with self.subTest(remote=remote):
                cmd = self.passthrough(remote, *self.SHARED)
                self.assertIn("--debug", cmd)
                self.assertIn("--debug-quick-tune-data", cmd)
                self.assertIn("--verbose", cmd)
                self.assertIn("--timeout=60", cmd)

    def test_gpu_flags_only_reach_the_remote_phase(self):
        remote_cmd = self.passthrough(True, *self.GPU_ONLY)
        self.assertEqual(
            remote_cmd, self.DEFAULT_DATA_TYPES + [
                "--gpus", "2", "--verify-winning-config", "--verify-all-perfconfigs",
                "--flush-last-level-cache", "--gpu-run-timeout=45", "--allow-commit-mismatch"
            ])
        self.assertEqual(self.passthrough(False, *self.GPU_ONLY), self.DEFAULT_DATA_TYPES)

    def test_num_cpus_only_reaches_the_local_phase(self):
        self.assertEqual(self.passthrough(False, "--num-cpus", "8"),
                         self.DEFAULT_DATA_TYPES + ["--num-cpus=8"])
        self.assertEqual(self.passthrough(True, "--num-cpus", "8"), self.DEFAULT_DATA_TYPES)

    def test_only_the_config_expansion_flags_are_forwarded_by_default(self):
        self.assertEqual(self.passthrough(False), self.DEFAULT_DATA_TYPES)
        self.assertEqual(self.passthrough(True), self.DEFAULT_DATA_TYPES)

    def test_data_and_scale_types_reach_both_phases(self):
        # A divergence here changes the problem hash, so the benchmark host
        # would look for artifact directories the compile host never wrote.
        expanded = ["--data-type", "bf16", "fp8", "--scale-type", "f8E8M0FNU"]
        for remote in (False, True):
            with self.subTest(remote=remote):
                cmd = self.passthrough(remote, *expanded)
                self.assertEqual(cmd[:len(expanded)], expanded)

    def test_scale_type_is_omitted_when_unset(self):
        self.assertNotIn("--scale-type", self.passthrough(True))


class TestLocalCompileCommand(CrossCompileTestCase):

    def test_runs_tuning_runner_in_compile_only_mode(self):
        cmd = crossCompile.build_local_compile_command(self.parse())
        self.assertEqual(cmd[0], sys.executable)
        self.assertTrue(cmd[1].endswith("tuningRunner.py"), cmd[1])
        self.assertEqual(flag_value(cmd, "--compile-only"), str(self.local_artifacts))
        self.assertEqual(flag_value(cmd, "--op"), "gemm")
        self.assertEqual(flag_value(cmd, "--tuning-space"), "quick")
        self.assertEqual(flag_value(cmd, "--configs-file"), str(self.configs_file))

    def test_pins_the_target_identity(self):
        # The compile host is GPU-less, so tuningRunner.py cannot discover any
        # of this and refuses to run without all three.
        cmd = crossCompile.build_local_compile_command(self.parse())
        self.assertEqual(flag_value(cmd, "--target-arch"), TARGET_ARCH)
        self.assertEqual(flag_value(cmd, "--target-num-cu"), "256")
        self.assertEqual(flag_value(cmd, "--target-num-chiplets"), "8")

    def test_omits_gpu_and_benchmark_flags(self):
        cmd = crossCompile.build_local_compile_command(self.parse("--gpus", "1"))
        self.assertNotIn("--gpus", cmd)
        self.assertNotIn("--benchmark-artifacts", cmd)

    def test_forwards_rocmlir_gen_flags_only_when_set(self):
        self.assertNotIn("--rocmlir-gen-flags",
                         crossCompile.build_local_compile_command(self.parse()))
        cmd = crossCompile.build_local_compile_command(
            self.parse("--rocmlir-gen-flags", "-mfma=on -atomic_add=on"))
        self.assertEqual(flag_value(cmd, "--rocmlir-gen-flags"), "-mfma=on -atomic_add=on")


class TestLocalBuildDir(CrossCompileTestCase):
    """The compile phase is told which build tree to use, like the remote one is.

    perfRunner.find_mlir_build_dir()'s only cheap candidate is ./bin/rocmlir-gen,
    so a wrapper started from anywhere but the build tree would otherwise fall
    back to searching the checkout for one -- and would do it after the SSH
    session has already been opened.
    """

    def test_pins_an_explicit_build_dir(self):
        build_dir = self.make_build_dir()
        cmd = crossCompile.build_local_compile_command(
            self.parse("--local-build-dir", str(build_dir)))
        self.assertEqual(flag_value(cmd, "--mlir-build-dir"), str(build_dir))

    def test_defaults_to_the_build_tree_this_script_was_installed_into(self):
        build_dir = self.make_build_dir()
        with mock.patch.object(crossCompile, "installed_build_dir", return_value=build_dir):
            cmd = crossCompile.build_local_compile_command(self.parse())
        self.assertEqual(flag_value(cmd, "--mlir-build-dir"), str(build_dir))

    def test_leaves_discovery_alone_for_a_source_tree_copy(self):
        # A source checkout has no built tools beside it, and needs no help
        # there: tuningRunner.py locates <repo>/build from its own path.
        with mock.patch.object(crossCompile, "installed_build_dir", return_value=None):
            cmd = crossCompile.build_local_compile_command(self.parse())
        self.assertNotIn("--mlir-build-dir", cmd)

    def test_rejects_a_directory_holding_no_tools(self):
        self.assertIn("not a rocMLIR build directory",
                      self.parse_expecting_error("--local-build-dir", str(self.tmp)))

    def test_rejects_a_missing_directory(self):
        self.assertIn("not a rocMLIR build directory",
                      self.parse_expecting_error("--local-build-dir", str(self.tmp / "nope")))

    def test_finds_the_build_tree_from_an_installed_copy(self):
        build_dir = self.make_build_dir()
        with mock.patch.object(crossCompile, "__file__",
                               str(build_dir / "bin" / "crossCompile.py")):
            self.assertEqual(crossCompile.installed_build_dir(), build_dir)

    def test_finds_no_build_tree_from_a_source_tree_copy(self):
        script_dir = self.tmp / "performance"
        script_dir.mkdir()
        with mock.patch.object(crossCompile, "__file__", str(script_dir / "crossCompile.py")):
            self.assertIsNone(crossCompile.installed_build_dir())


class TestArtifactTarCommand(CrossCompileTestCase):

    def test_bundles_artifacts_and_the_config_file(self):
        # Artifacts are archived at the root of the tar so the remote side can
        # extract straight into its own artifacts dir; the config file rides
        # along by name from its own parent directory.
        cmd = crossCompile.build_artifact_tar_command(self.parse())
        self.assertEqual(cmd, [
            "tar", "-C",
            str(self.local_artifacts), "-cf", "-", ".", "-C",
            str(self.tmp), "configs.txt"
        ])

    def test_honours_a_custom_tar(self):
        cmd = crossCompile.build_artifact_tar_command(self.parse("--tar", "gtar"))
        self.assertEqual(cmd[0], "gtar")


class TestRemoteBenchmarkCommand(CrossCompileTestCase):

    def test_runs_tuning_runner_inside_the_container(self):
        cmd = crossCompile.build_remote_benchmark_args(self.parse())
        self.assertEqual(cmd[:3], ["docker", "exec", "-w"])
        # perfRunner.find_mlir_build_dir()'s only cheap candidate is
        # ./bin/rocmlir-gen, so run from the build dir.
        self.assertEqual(cmd[3], REMOTE_BUILD)
        self.assertEqual(cmd[4], CONTAINER)
        runner = cmd.index("python3")
        self.assertEqual(cmd[runner + 1], REMOTE_BUILD + "/bin/tuningRunner.py")
        self.assertEqual(flag_value(cmd, "--mlir-build-dir"), REMOTE_BUILD)

    def test_honours_a_custom_remote_build_dir(self):
        cmd = crossCompile.build_remote_benchmark_args(
            self.parse("--remote-build-dir", "/opt/rocmlir-build"))
        self.assertEqual(cmd[3], "/opt/rocmlir-build")
        runner = cmd.index("python3")
        self.assertEqual(cmd[runner + 1], "/opt/rocmlir-build/bin/tuningRunner.py")
        self.assertEqual(flag_value(cmd, "--mlir-build-dir"), "/opt/rocmlir-build")

    def test_rejects_a_relative_remote_build_dir(self):
        self.assertIn("must be an absolute remote path",
                      self.parse_expecting_error("--remote-build-dir", "build"))

    def test_bounds_the_benchmark_inside_the_container(self):
        # Terminating the local ssh leaves a docker exec'd benchmark running and
        # holding the GPU, so the deadline has to hold on the remote as well.
        cmd = crossCompile.build_remote_benchmark_args(self.parse("--remote-session-timeout",
                                                                  "900"))
        self.assertEqual(cmd[5:10], ["timeout", "-k", "30", "900", "python3"])

    def test_omits_the_container_timeout_when_the_session_waits_forever(self):
        cmd = crossCompile.build_remote_benchmark_args(self.parse("--remote-session-timeout", "0"))
        self.assertEqual(cmd[5], "python3")
        self.assertNotIn("timeout", cmd)

    def test_benchmarks_the_streamed_artifacts(self):
        cmd = crossCompile.build_remote_benchmark_args(self.parse())
        self.assertEqual(flag_value(cmd, "--benchmark-artifacts"), REMOTE_ARTIFACTS)
        self.assertEqual(flag_value(cmd, "--configs-file"), "/remote/artifacts/configs.txt")
        self.assertEqual(flag_value(cmd, "--output"), "/remote/artifacts/out.tsv")
        self.assertNotIn("--compile-only", cmd)

    def test_repeats_the_target_identity(self):
        # Must match the compile phase or the manifest check rejects the bundle.
        cmd = crossCompile.build_remote_benchmark_args(self.parse())
        self.assertEqual(flag_value(cmd, "--target-arch"), TARGET_ARCH)
        self.assertEqual(flag_value(cmd, "--target-num-cu"), "256")
        self.assertEqual(flag_value(cmd, "--target-num-chiplets"), "8")

    def test_forwards_the_gpu_selection(self):
        cmd = crossCompile.build_remote_benchmark_args(self.parse("--gpus", "3"))
        self.assertEqual(flag_value(cmd, "--gpus"), "3")

    def test_forwards_rocmlir_gen_flags_for_artifact_lookup(self):
        self.assertNotIn("--rocmlir-gen-flags",
                         crossCompile.build_remote_benchmark_args(self.parse()))
        cmd = crossCompile.build_remote_benchmark_args(
            self.parse("--rocmlir-gen-flags", "-mfma=on -atomic_add=on"))
        self.assertEqual(flag_value(cmd, "--rocmlir-gen-flags"), "-mfma=on -atomic_add=on")


class TestRemoteSessionScript(CrossCompileTestCase):

    def test_ssh_command_forces_non_pty_mode(self):
        args = self.parse("--ssh-option=-v")
        cmd = crossCompile.build_remote_session_command(args)
        self.assertIn("-T", cmd)
        self.assertLess(cmd.index("-v"), cmd.index("-T"))
        self.assertLess(cmd.index("-T"), cmd.index(args.remote_target))

    def test_reserves_stdout_for_the_result_stream(self):
        script = crossCompile.build_remote_session_script(self.parse())
        # fd 3 becomes the real stdout and everything else is pushed to stderr,
        # so remote chatter cannot corrupt the tar stream coming back.
        self.assertTrue(script.startswith("set -eo pipefail; exec 3>&1; exec 1>&2;"), script)

    def test_announces_readiness_before_the_long_compile(self):
        script = crossCompile.build_remote_session_script(self.parse())
        # The marker goes to fd 3, the only stream the local side reads.
        marker = f"echo {shlex.quote(crossCompile.REMOTE_READY_MARKER)} >&3"
        self.assertIn(marker, script)
        # It must come after the docker probe (so an unusable container fails
        # fast) but before the artifact extraction that waits on stdin.
        self.assertLess(script.index("docker exec"), script.index(marker))
        self.assertLess(script.index(marker), script.index("-xf -"))

    def test_probes_the_container_in_the_build_dir(self):
        # A container that cannot see the build tree must fail at the probe,
        # not after the whole artifact bundle has been streamed to it.
        script = crossCompile.build_remote_session_script(self.parse())
        self.assertIn(f"docker exec -w {REMOTE_BUILD} {CONTAINER} true", script)

    def test_orders_extract_benchmark_and_result_tar(self):
        script = crossCompile.build_remote_session_script(self.parse())
        self.assertLess(script.index("-xf -"), script.index("--benchmark-artifacts"))
        self.assertLess(script.index("--benchmark-artifacts"),
                        script.index('-cf - "${files[@]}" >&3'))

    def test_removes_previous_results_before_benchmarking(self):
        # The state file has to go with the TSV it describes. tuningRunner.py
        # resumes from the pair, so a state file that outlived its TSV would make
        # the benchmark skip problems that failed in an earlier run while the
        # rows recording them are gone -- silently short results, exit code 0.
        args = self.parse()
        script = crossCompile.build_remote_session_script(args)
        cleanup = shlex.join([
            "rm", "-f", "--", args.remote_output, f"{args.remote_output}.debug",
            f"{args.remote_output}.state"
        ])
        self.assertIn(cleanup, script)
        self.assertLess(script.index("-xf -"), script.index(cleanup))
        self.assertLess(script.index(cleanup), script.index("--benchmark-artifacts"))

    def test_creates_the_remote_artifacts_dir(self):
        script = crossCompile.build_remote_session_script(self.parse())
        self.assertIn(f"mkdir -p -- {REMOTE_ARTIFACTS}", script)

    def test_returns_whichever_result_files_the_remote_produced(self):
        script = crossCompile.build_remote_session_script(self.parse())
        self.assertIn("files=()", script)
        self.assertIn("if [ -f out.tsv ]; then files+=(out.tsv); fi", script)
        self.assertIn("if [ -f out.tsv.debug ]; then files+=(out.tsv.debug); fi", script)
        # tar -cf on a missing member is an error, so a run that wrote nothing
        # must skip the archive rather than fail on top of its own failure.
        self.assertIn('if [ ${#files[@]} -gt 0 ]; then', script)

    def test_ships_results_even_when_a_problem_failed(self):
        # tuningRunner.py exits non-zero if any single problem failed, but the
        # rows it already wrote are still worth having; `set -e` must not abort
        # before the result tar runs.
        script = crossCompile.build_remote_session_script(self.parse())
        benchmark = shlex.join(
            [str(part) for part in crossCompile.build_remote_benchmark_args(self.parse())])
        self.assertIn(f"{benchmark} || benchmark_rc=$?", script)
        self.assertLess(script.index("benchmark_rc=0"), script.index(benchmark))
        # The status is still what the session reports, just after the results.
        self.assertTrue(script.endswith("; exit $benchmark_rc"), script)
        self.assertLess(script.index('-cf - "${files[@]}" >&3'), script.index("exit $benchmark_rc"))

    def test_runs_the_setup_command_before_the_ready_marker(self):
        script = crossCompile.build_remote_session_script(
            self.parse("--remote-setup-command", "module load rocm"))
        self.assertIn("module load rocm", script)
        self.assertLess(script.index("module load rocm"),
                        script.index(crossCompile.REMOTE_READY_MARKER))

    def test_quotes_paths_with_shell_metacharacters(self):
        # These strings are interpolated into a `bash -lc` script, so anything
        # unquoted here is remote command injection.
        script = crossCompile.build_remote_session_script(
            self.parse("--remote-artifacts-dir", "/remote/a b;touch pwned"))
        self.assertIn("'/remote/a b;touch pwned'", script)
        self.assertNotIn("-- /remote/a b", script)


class TestTerminateProcess(unittest.TestCase):

    def test_ignores_none(self):
        crossCompile.terminate_process(None)

    def test_leaves_an_exited_process_alone(self):
        process = FakeProcess(alive=False)
        crossCompile.terminate_process(process)
        self.assertFalse(process.terminated)

    def test_terminates_a_running_process(self):
        process = FakeProcess()
        crossCompile.terminate_process(process)
        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)

    def test_kills_a_process_that_ignores_terminate(self):
        process = FakeProcess(wait_times_out=True)
        crossCompile.terminate_process(process)
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)


class TestWaitForRemoteReady(unittest.TestCase):

    def wait(self, process, timeout=None):
        """Run wait_for_remote_ready, returning whatever it echoed locally."""
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            crossCompile.wait_for_remote_ready(process, ["ssh"], timeout)
        return stderr.getvalue()

    def test_accepts_the_ready_marker(self):
        process = FakeProcess([crossCompile.REMOTE_READY_MARKER.encode() + b"\n"])
        self.assertEqual(self.wait(process), "")
        self.assertFalse(process.terminated)

    def test_skips_login_shell_noise_before_the_marker(self):
        # bash -lc sources /etc/profile.d, conda, direnv, ... before the session
        # script's `exec 1>&2` can take effect, so their output arrives here.
        process = FakeProcess([
            b"(base) conda activated\n",
            b"direnv: loading .envrc\n",
            crossCompile.REMOTE_READY_MARKER.encode() + b"\n",
        ])
        printed = self.wait(process)
        self.assertIn("conda activated", printed)
        self.assertIn("direnv: loading", printed)
        self.assertFalse(process.terminated)

    def test_accepts_a_marker_glued_to_a_banner(self):
        # A profile script that prints without a trailing newline shares a line
        # with the marker, so the match cannot be an equality check.
        process = FakeProcess([b"(base) " + crossCompile.REMOTE_READY_MARKER.encode() + b"\n"])
        self.assertEqual(self.wait(process), "")

    def test_reports_a_session_that_died_before_the_marker(self):
        # EOF means ssh itself failed (bad host, auth failure, ...); surface the
        # remote exit code rather than blocking on a stream nobody will write.
        process = FakeProcess([b"ssh: connect to host gpuhost: No route\n"], returncode=255)
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.wait(process)
        self.assertEqual(caught.exception.returncode, 255)

    def test_never_reports_a_markerless_session_as_success(self):
        # main() returns CalledProcessError.returncode, so a zero here would
        # report success for a run that produced no results.
        process = FakeProcess([], returncode=0)
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.wait(process)
        self.assertNotEqual(caught.exception.returncode, 0)

    def test_gives_up_on_a_remote_that_is_alive_but_silent(self):
        # An unresponsive docker daemon or a stuck mount never reaches the
        # marker and never closes the pipe, so only the watchdog ends the wait.
        process = FakeProcess(silent=True)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.wait(process, timeout=0.05)
        self.assertTrue(process.terminated)

    def test_zero_timeout_waits_indefinitely(self):
        process = FakeProcess([crossCompile.REMOTE_READY_MARKER.encode() + b"\n"])
        with mock.patch("threading.Timer", side_effect=AssertionError("started a watchdog")):
            self.assertEqual(self.wait(process, timeout=0), "")


class TestWaitForRemoteResults(unittest.TestCase):

    def test_waits_for_transfer_remote_and_reader(self):
        future = Future()
        future.set_result(None)
        result = crossCompile.wait_for_remote_results(FakeProcess(),
                                                      FakeProcess(),
                                                      future, ["ssh"],
                                                      timeout=60)
        self.assertEqual(result, (0, 0))

    def test_reports_transfer_timeout_as_remote_session_timeout(self):
        future = Future()
        future.set_result(None)
        with mock.patch.object(crossCompile.time, "monotonic", side_effect=[0, 61]), \
                self.assertRaises(subprocess.TimeoutExpired) as caught:
            crossCompile.wait_for_remote_results(FakeProcess(),
                                                 FakeProcess(),
                                                 future, ["ssh", "host"],
                                                 timeout=60)
        self.assertEqual(caught.exception.cmd, ["ssh", "host"])
        self.assertEqual(caught.exception.timeout, 60)

    def test_times_out_if_result_reader_does_not_finish(self):
        future = Future()
        with self.assertRaises(subprocess.TimeoutExpired):
            crossCompile.wait_for_remote_results(FakeProcess(),
                                                 FakeProcess(),
                                                 future, ["ssh"],
                                                 timeout=0.01)

    def test_propagates_result_reader_exception(self):
        error = OSError("disk full")
        future = Future()
        future.set_exception(error)
        with self.assertRaises(OSError) as caught:
            crossCompile.wait_for_remote_results(FakeProcess(),
                                                 FakeProcess(),
                                                 future, ["ssh"],
                                                 timeout=60)
        self.assertIs(caught.exception, error)

    def test_copy_stream_future_captures_reader_exception(self):
        future = Future()
        error = OSError("quota exceeded")
        with mock.patch.object(crossCompile, "copy_stream_to_file", side_effect=error):
            crossCompile.copy_stream_to_future(mock.Mock(), Path("unused"), future)
        with self.assertRaises(OSError) as caught:
            future.result()
        self.assertIs(caught.exception, error)

    def test_remote_failure_takes_precedence_over_artifact_sigpipe(self):
        remote_cmd = ["ssh", "gpuhost"]
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            crossCompile.raise_for_session_errors(-signal.SIGPIPE, 42, ["tar"], remote_cmd)
        self.assertEqual(caught.exception.returncode, 42)
        self.assertEqual(caught.exception.cmd, remote_cmd)

    def test_non_sigpipe_artifact_failure_remains_primary(self):
        artifact_cmd = ["tar", "-cf", "-"]
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            crossCompile.raise_for_session_errors(2, 42, artifact_cmd, ["ssh", "gpuhost"])
        self.assertEqual(caught.exception.returncode, 2)
        self.assertEqual(caught.exception.cmd, artifact_cmd)


class TestExtractReturnedResults(CrossCompileTestCase):
    """The returned TSV must survive a run in which some problems failed.

    tuningRunner.py --benchmark-artifacts exits non-zero if any problem failed,
    so extraction cannot wait until the session status has been cleared or a
    single bad problem would discard the whole run's results.
    """

    def extract(self, *, contents=b"tar bytes", session_failed=False, tar_fails=False):
        """Run extract_returned_results, returning the tar command it issued."""
        args = self.parse()
        tar_path = self.tmp / "results.tar"
        tar_path.write_bytes(contents)
        error = subprocess.CalledProcessError(2, ["tar"]) if tar_fails else None
        runner = mock.Mock(side_effect=error)
        stderr = io.StringIO()
        with mock.patch.object(crossCompile, "run_command", runner):
            with contextlib.redirect_stderr(stderr):
                crossCompile.extract_returned_results(args, tar_path, session_failed=session_failed)
        return runner, stderr.getvalue()

    def test_extracts_into_the_output_directory(self):
        runner, _ = self.extract()
        cmd = runner.call_args.args[1]
        self.assertEqual(cmd[:3], [self.parse().tar, "-C", str(self.output.parent)])
        self.assertEqual(cmd[3], "-xf")

    def test_extracts_partial_results_from_a_failed_session(self):
        runner, _ = self.extract(session_failed=True)
        runner.assert_called_once()

    def test_skips_extraction_when_the_remote_returned_nothing(self):
        # A session that died before the result tar leaves an empty file; running
        # tar on it would only bury the real failure under a bogus tar error.
        runner, _ = self.extract(contents=b"", session_failed=True)
        runner.assert_not_called()

    def test_does_not_let_a_bad_archive_mask_the_session_failure(self):
        _, printed = self.extract(session_failed=True, tar_fails=True)
        self.assertIn("could not unpack", printed)

    def test_reports_an_extraction_failure_on_a_successful_session(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.extract(tar_fails=True)


class TestCountCompiledProblems(CrossCompileTestCase):
    """Decides whether a failed local compile produced anything worth shipping."""

    def count(self, contents=None, previous_run_id="run-0"):
        artifacts = self.tmp / "bundle"
        artifacts.mkdir(exist_ok=True)
        if contents is not None:
            (artifacts / "index.json").write_text(contents)
        return crossCompile.count_compiled_problems(artifacts, previous_run_id)

    def test_counts_the_problems_this_run_recorded(self):
        self.assertEqual(
            self.count('{"lastRunId": "run-1", '
                       '"problems": {"aaa": {"runId": "run-1"}, "bbb": {"runId": "run-1"}}}'), 2)

    def test_ignores_problems_left_by_an_earlier_run(self):
        # --local-artifacts-dir is reused, so a bundle compiled for some other
        # config file must not be mistaken for this run's output.
        self.assertEqual(
            self.count('{"lastRunId": "run-1", '
                       '"problems": {"aaa": {"runId": "run-0"}, "bbb": {"runId": "run-1"}}}'), 1)

    def test_counts_nothing_when_the_compile_never_claimed_the_bundle(self):
        # A compile that died before recording a problem leaves the id alone.
        self.assertEqual(
            self.count('{"lastRunId": "run-0", "problems": {"aaa": {"runId": "run-0"}}}'), 0)

    def test_treats_a_missing_index_as_nothing_compiled(self):
        self.assertEqual(self.count(), 0)

    def test_treats_an_unreadable_index_as_nothing_compiled(self):
        self.assertEqual(self.count("not json at all"), 0)

    def test_treats_a_malformed_index_as_nothing_compiled(self):
        self.assertEqual(self.count('{"lastRunId": "run-1", "problems": "not a dict"}'), 0)
        self.assertEqual(self.count('{"arch": "gfx950"}'), 0)
        # An index without run ids predates them; fail closed rather than ship.
        self.assertEqual(self.count('{"problems": {"aaa": {}}}'), 0)

    def test_reads_back_the_run_that_wrote_the_bundle(self):
        artifacts = self.tmp / "bundle"
        artifacts.mkdir(exist_ok=True)
        self.assertIsNone(crossCompile.read_bundle_run_id(artifacts))
        (artifacts / "index.json").write_text('{"lastRunId": "run-7"}')
        self.assertEqual(crossCompile.read_bundle_run_id(artifacts), "run-7")


class TestPartialLocalCompile(CrossCompileTestCase):
    """A compile where some problems failed still has bundles worth benchmarking.

    tuningRunner.py --compile-only exits non-zero if any single problem failed,
    so aborting on that status would throw away every problem that did compile
    -- the opposite of what the remote side does with a partial benchmark run.
    """

    def write_index(self, artifacts_dir, run_id, problems):
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "index.json").write_text(
            json.dumps({
                "lastRunId": run_id,
                "problems": {
                    f"{run_id}-{i}": {
                        "runId": run_id
                    } for i in range(problems)
                },
            }))

    def run_session(self, compile_rc, compiled_problems, stale_problems=0):
        args = self.parse()
        args.local_artifacts_dir.mkdir(parents=True)
        if stale_problems:
            self.write_index(args.local_artifacts_dir, "stale-run", stale_problems)

        def fake_compile(*_args, **_kwargs):
            # The bundle only picks up this run's id once the compile records a
            # problem, so a compile that produced nothing leaves whatever an
            # earlier run wrote untouched.
            if compiled_problems:
                self.write_index(args.local_artifacts_dir, "this-run", compiled_problems)
            return mock.Mock(returncode=compile_rc)

        remote = FakeProcess(lines=[crossCompile.REMOTE_READY_MARKER.encode() + b"\n"])
        remote.stdin = mock.Mock()
        stderr = io.StringIO()
        error = None
        with mock.patch.object(crossCompile, "subprocess") as fake_subprocess, \
             mock.patch.object(crossCompile, "wait_for_remote_results", return_value=(0, 0)), \
             mock.patch.object(crossCompile, "extract_returned_results"), \
             mock.patch.object(crossCompile, "terminate_process"):
            fake_subprocess.Popen.return_value = remote
            fake_subprocess.run.side_effect = fake_compile
            fake_subprocess.CalledProcessError = subprocess.CalledProcessError
            with contextlib.redirect_stderr(stderr):
                try:
                    crossCompile.run_cross_compile_session(args)
                except subprocess.CalledProcessError as raised:
                    error = raised
        return error, stderr.getvalue(), fake_subprocess

    def test_benchmarks_the_problems_that_did_compile(self):
        error, printed, fake_subprocess = self.run_session(compile_rc=1, compiled_problems=3)
        self.assertIn("benchmarking the 3 problem(s) that did compile", printed)
        # The artifact tar still ran, so the partial bundle reached the remote.
        self.assertEqual(fake_subprocess.Popen.call_count, 2)
        # ...but the compile status is still the run's verdict.
        self.assertIsNotNone(error)
        self.assertEqual(error.returncode, 1)

    def test_aborts_when_nothing_compiled(self):
        error, printed, fake_subprocess = self.run_session(compile_rc=1, compiled_problems=0)
        self.assertNotIn("did compile", printed)
        # Only the ssh session was spawned; no point streaming an empty bundle.
        self.assertEqual(fake_subprocess.Popen.call_count, 1)
        self.assertIsNotNone(error)
        self.assertEqual(error.returncode, 1)

    def test_ignores_a_bundle_left_by_an_earlier_run(self):
        # --local-artifacts-dir defaults to a fixed path and is never cleaned,
        # so bundles compiled for some other config file can still be sitting
        # there. They are no reason to ship anything for this run.
        error, printed, fake_subprocess = self.run_session(compile_rc=1,
                                                           compiled_problems=0,
                                                           stale_problems=3)
        self.assertNotIn("did compile", printed)
        self.assertEqual(fake_subprocess.Popen.call_count, 1)
        self.assertEqual(error.returncode, 1)

    def test_a_clean_compile_is_not_reported_as_a_failure(self):
        error, printed, _ = self.run_session(compile_rc=0, compiled_problems=3)
        self.assertIsNone(error)
        self.assertNotIn("did compile", printed)


class TestDryRun(CrossCompileTestCase):

    def test_prints_the_flow_without_spawning_anything(self):
        args = self.parse("--dry-run")
        stderr = io.StringIO()
        with mock.patch("subprocess.Popen", side_effect=AssertionError("spawned a process")), \
             mock.patch("subprocess.run", side_effect=AssertionError("spawned a process")):
            with contextlib.redirect_stderr(stderr):
                crossCompile.run_cross_compile_session(args)

        printed = stderr.getvalue()
        self.assertIn(crossCompile.REMOTE_READY_MARKER, printed)
        self.assertIn("--compile-only", printed)
        self.assertIn("--benchmark-artifacts", printed)
        self.assertIn(args.remote_target, printed)


class TestMain(CrossCompileTestCase):

    def argv(self, *extra):
        return [
            "--op", "gemm",
            "--tuning-space", "quick",
            "--target-arch", TARGET_ARCH,
            "--target-num-cu", "256",
            "--target-num-chiplets", "8",
            "--configs-file", str(self.configs_file),
            "--output", str(self.output),
            "--remote-host", "gpuhost",
            "--remote-artifacts-dir", REMOTE_ARTIFACTS,
            "--remote-repo-dir", REMOTE_REPO,
            "--remote-docker-container", CONTAINER,
        ] + list(extra)  # yapf: disable

    def run_main(self, *extra, session=None):
        stderr = io.StringIO()
        with mock.patch.object(crossCompile, "run_cross_compile_session", session or mock.Mock()):
            with contextlib.redirect_stderr(stderr):
                return crossCompile.main(self.argv(*extra)), stderr.getvalue()

    def test_returns_zero_on_success(self):
        code, _ = self.run_main()
        self.assertEqual(code, 0)

    def test_propagates_the_failing_command_exit_code(self):
        failure = subprocess.CalledProcessError(42, ["tuningRunner.py"])
        code, printed = self.run_main(session=mock.Mock(side_effect=failure))
        self.assertEqual(code, 42)
        self.assertIn("exit code 42", printed)
        self.assertIn("--verbose", printed)

    def test_reports_a_ready_timeout(self):
        failure = subprocess.TimeoutExpired(["ssh"], 600)
        code, printed = self.run_main(session=mock.Mock(side_effect=failure))
        self.assertEqual(code, 2)
        self.assertIn("timed out", printed)
        self.assertIn("--remote-ready-timeout", printed)
        self.assertIn("--remote-session-timeout", printed)

    def test_reports_usage_errors_as_exit_two(self):
        # ssh_target() raises ValueError for a doubly-specified user.
        code, printed = self.run_main("--remote-user", "bob", "--remote-host", "alice@gpuhost")
        self.assertEqual(code, 2)
        self.assertIn("--remote-user cannot be used", printed)


if __name__ == "__main__":
    unittest.main()
