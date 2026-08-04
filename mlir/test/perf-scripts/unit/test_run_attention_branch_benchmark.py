# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: env PYTHONPATH=%S/../../../utils/performance %python %s

import argparse
import contextlib
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import perfRunner
import runAttentionBranchBenchmark

runner = runAttentionBranchBenchmark


def make_args(root):
    return types.SimpleNamespace(base_source=root / "base",
                                 base_build=root / "base-build",
                                 candidate_source=root / "candidate",
                                 candidate_build=root / "candidate-build",
                                 configs=root / "configs",
                                 output_dir=root / "output",
                                 gpu=None,
                                 samples=2,
                                 retries=1,
                                 retry_backoff=0,
                                 perf_config_timeout=10,
                                 gpu_run_timeout=5,
                                 tuning_timeout=20,
                                 benchmark_timeout=30,
                                 idle_samples=1,
                                 idle_interval=0)


class BranchAndTreeTest(unittest.TestCase):

    def test_branch_paths_and_git_helpers(self):
        branch = runner.BranchRun("base", Path("/s"), Path("/b"), "sha", Path("/o"))
        self.assertEqual(branch.tuning_db, Path("/o/quick-tuning.tsv"))
        self.assertEqual(branch.results, Path("/o/performance.csv"))
        self.assertEqual(branch.log, Path("/o/run.log"))
        with mock.patch.object(runner.subprocess, "check_output", return_value="output") as check:
            self.assertEqual(runner._check_output(["command"]), "output")
        self.assertEqual(check.call_args.kwargs["timeout"], 60)
        with mock.patch.object(runner, "_check_output", side_effect=[" sha\n", "dirty\n", ""]):
            self.assertEqual(runner.git_revision(Path("/s")), "sha")
            self.assertTrue(runner.git_is_dirty(Path("/s")))
            self.assertFalse(runner.git_is_dirty(Path("/s")))

    def test_validate_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            build = root / "build"
            with self.assertRaisesRegex(ValueError, "source directory"):
                runner.validate_tree("base", source, build)
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "build directory"):
                runner.validate_tree("base", source, build)
            build.mkdir()

            with mock.patch.object(runner, "git_is_dirty", return_value=True):
                with self.assertRaisesRegex(ValueError, "dirty"):
                    runner.validate_tree("base", source, build)

            with mock.patch.object(runner, "git_is_dirty", return_value=False):
                with self.assertRaisesRegex(ValueError, "tuningRunner"):
                    runner.validate_tree("base", source, build)

                scripts = source / "mlir" / "utils" / "performance"
                scripts.mkdir(parents=True)
                for script in ("tuningRunner.py", "perfRunner.py"):
                    (scripts / script).touch()
                with self.assertRaisesRegex(ValueError, "rocmlir-gen"):
                    runner.validate_tree("base", source, build)

                (build / "bin").mkdir()
                for binary in ("rocmlir-gen", "rocmlir-driver", "rocmlir-tuning-driver"):
                    (build / "bin" / binary).write_text(binary, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "configured from"):
                    runner.validate_tree("base", source, build)
                (build / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n",
                                                      encoding="utf-8")
                with mock.patch.object(runner, "git_revision", return_value="revision"):
                    with self.assertRaisesRegex(ValueError, "source stamp"):
                        runner.validate_tree("base", source, build)
                    (build / "rocmlir-source-sha").write_text("revision\n", encoding="utf-8")
                    self.assertEqual(runner.validate_tree("base", source, build), "revision")
                self.assertEqual(runner.cmake_source_directory(build), source)
                self.assertEqual(len(runner.build_fingerprint(source, build)), 64)
                (build / "CMakeCache.txt").write_text("OTHER=value\n", encoding="utf-8")
                self.assertIsNone(runner.cmake_source_directory(build))

    def test_validate_branch_unchanged(self):
        branch = runner.BranchRun("base", Path("/source"), Path("/build"), "sha", Path("/out"))
        with mock.patch.object(runner, "validate_tree", return_value="sha"), \
                mock.patch.object(runner, "build_fingerprint", return_value="fingerprint"):
            runner.validate_branch_unchanged(branch, "fingerprint")
        with mock.patch.object(runner, "validate_tree", return_value="other"):
            with self.assertRaisesRegex(RuntimeError, "source revision"):
                runner.validate_branch_unchanged(branch, "fingerprint")
        with mock.patch.object(runner, "validate_tree", return_value="sha"), \
                mock.patch.object(runner, "build_fingerprint", return_value="other"):
            with self.assertRaisesRegex(RuntimeError, "build artifacts"):
                runner.validate_branch_unchanged(branch, "fingerprint")


class GpuSelectionTest(unittest.TestCase):

    def test_parse_gpu_status(self):
        status_json = json.dumps({
            "card0": {
                "GFX Version": "gfx90a",
                "Card SKU": "SKU",
                "GPU use (%)": "0",
                "GPU Memory Allocated (VRAM%)": "0"
            },
            "unexpected": {}
        })
        process_json = json.dumps(
            {"system": {
                "PID100": "name, 0, 4",
                "PIDbad": "name, 0",
                "OTHER": "ignored",
            }})
        with mock.patch.object(runner.os, "getpid", return_value=999), \
                mock.patch.object(runner.Path, "exists", return_value=True):
            status, processes = runner.parse_gpu_status(status_json, process_json)
        self.assertEqual(status[0]["arch"], "gfx90a")
        self.assertEqual(processes, {0: [100]})
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            runner.parse_gpu_status('{"card0":{}}', '{"system":{}}')
        with self.assertRaisesRegex(RuntimeError, "any GPU"):
            runner.parse_gpu_status('{"unexpected":{}}', '{"system":{}}')
        with self.assertRaisesRegex(RuntimeError, "process table"):
            runner.parse_gpu_status(status_json, '{}')
        with self.assertRaisesRegex(RuntimeError, "process entry"):
            runner.parse_gpu_status(status_json, '{"system":{"PID1":"short"}}')
        with self.assertRaisesRegex(RuntimeError, "GPU index"):
            runner.parse_gpu_status(status_json, '{"system":{"PID1":"name, bad"}}')
        process_json = '{"system":{"PID999":"name, 0","PID101":"name, 4"}}'
        with mock.patch.object(runner.os, "getpid", return_value=999), \
                mock.patch.object(runner.Path, "exists", return_value=False):
            _, processes = runner.parse_gpu_status(status_json, process_json)
        self.assertEqual(processes, {0: []})

    def test_query_gpu_status(self):
        with mock.patch.object(runner.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "rocm-smi"):
                runner.query_gpu_status()
        responses = [
            '{"card0":{"GFX Version":"gfx90a","Card SKU":"SKU","GPU use (%)":"0",'
            '"GPU Memory Allocated (VRAM%)":"0"}}', '{"system":{}}'
        ]
        with mock.patch.object(runner.shutil, "which", return_value="/bin/rocm-smi"), \
                mock.patch.object(runner, "_check_output", side_effect=responses):
            status, processes = runner.query_gpu_status()
        self.assertIn(0, status)
        self.assertEqual(processes[0], [])

    def test_idle_selection(self):
        status = {
            0: {
                "use": 10,
                "memory": 0,
                "arch": "gfx"
            },
            1: {
                "use": 0,
                "memory": 0,
                "arch": "gfx"
            },
            2: {
                "use": 0,
                "memory": 1,
                "arch": "gfx"
            },
        }
        processes = {0: [], 1: [], 2: []}
        self.assertFalse(runner.is_gpu_idle(9, status, processes))
        self.assertEqual(runner.select_idle_gpu(None, status, processes), 1)
        self.assertEqual(runner.select_idle_gpu(1, status, processes), 1)
        with self.assertRaisesRegex(RuntimeError, "not reported"):
            runner.select_idle_gpu(9, status, processes)
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            runner.select_idle_gpu(0, status, processes)
        with self.assertRaisesRegex(RuntimeError, "No idle"):
            runner.select_idle_gpu(None, {0: status[0]}, {0: []})

    def test_stable_idle(self):
        idle = ({0: {"use": 0, "memory": 0, "arch": "gfx950"}}, {0: []})
        with mock.patch.object(runner, "query_gpu_status", return_value=idle), \
                mock.patch.object(runner.time, "sleep") as sleep:
            self.assertEqual(runner.verify_stably_idle(0, 2, 1), "gfx950")
            sleep.assert_called_once_with(1)
        busy = ({0: {"use": 1, "memory": 0, "arch": "gfx"}}, {0: []})
        with mock.patch.object(runner, "query_gpu_status", return_value=busy):
            with self.assertRaisesRegex(RuntimeError, "became busy"):
                runner.verify_stably_idle(0, 1, 0)
        with self.assertRaises(AssertionError):
            runner.verify_stably_idle(0, 0, 0)

    def test_gpu_lock(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner, "Path",
                                  side_effect=lambda value: Path(directory) / Path(value).name):
            with runner.gpu_lock(2):
                pass
            with mock.patch.object(runner.os, "chmod", side_effect=PermissionError):
                with runner.gpu_lock(2):
                    pass
            with mock.patch.object(runner.fcntl, "flock", side_effect=BlockingIOError):
                with self.assertRaisesRegex(RuntimeError, "reserved"):
                    with runner.gpu_lock(2):
                        pass

    def test_output_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with runner.output_lock(output):
                self.assertTrue((output / ".attention-perf.lock").exists())
            with mock.patch.object(runner.fcntl, "flock", side_effect=BlockingIOError):
                with self.assertRaisesRegex(RuntimeError, "another run"):
                    with runner.output_lock(output):
                        pass


class CommandAndResultTest(unittest.TestCase):

    def test_run_with_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            failed = mock.Mock(pid=10)
            failed.wait.return_value = 1
            passed = mock.Mock(pid=11)
            passed.wait.return_value = 0
            before = mock.Mock()
            with mock.patch.object(runner.subprocess, "Popen", side_effect=[failed, passed]), \
                    mock.patch.object(runner.time, "sleep") as sleep:
                runner.run_with_retries(["command", "arg"], {},
                                        log,
                                        1,
                                        2,
                                        Path(directory),
                                        before_attempt=before)
            sleep.assert_called_once_with(2)
            self.assertEqual(before.call_count, 2)
            self.assertIn("$ command arg", log.read_text(encoding="utf-8"))
            failed.wait.return_value = 2
            with mock.patch.object(runner.subprocess, "Popen", return_value=failed):
                with self.assertRaisesRegex(RuntimeError, "failed after 1"):
                    runner.run_with_retries(["false"], {}, log, 0, 0, Path(directory))

            timed_out = mock.Mock(pid=12)
            timed_out.wait.side_effect = [subprocess.TimeoutExpired(["hung"], 1), 0]
            with mock.patch.object(runner.subprocess, "Popen", return_value=timed_out), \
                    mock.patch.object(runner.os, "killpg") as kill:
                with self.assertRaisesRegex(RuntimeError, "failed after 1"):
                    runner.run_with_retries(["hung"], {}, log, 0, 0, Path(directory), timeout=1)
            kill.assert_called_once_with(12, runner.signal.SIGKILL)

    def test_command_construction(self):
        args = types.SimpleNamespace(perf_config_timeout=30, gpu_run_timeout=10)
        branch = runner.BranchRun("base", Path("/source"), Path("/build"), "sha", Path("/out"))
        tune = runner.tuning_command(branch, "-t f16 -g 1", "-current_seq_len=3", 3, args)
        self.assertIn("--disable-verify-winning-config", tune)
        self.assertEqual(tune[tune.index("--gpus") + 1], "3")
        self.assertIn("--rocmlir-gen-flags=-current_seq_len=3", tune)
        benchmark = runner.benchmark_command(branch, "-t f16 -g 1", "-current_seq_len=3",
                                             Path("/tmp.csv"))
        self.assertEqual(benchmark[-4:], ["--", "-t", "f16", "-g", "1"][-4:])
        self.assertIn(str(branch.tuning_db), benchmark)
        self.assertIn("--rocmlir_gen_flags=-current_seq_len=3", benchmark)
        tune = runner.tuning_command(branch, "-t f16", "", 3, args)
        benchmark = runner.benchmark_command(branch, "-t f16", "", Path("/tmp.csv"))
        self.assertFalse(any(value.startswith("--rocmlir-gen-flags") for value in tune))
        self.assertFalse(any(value.startswith("--rocmlir_gen_flags") for value in benchmark))

    def test_tuned_perf_runner_preserves_runtime_flags(self):
        config = mock.Mock(perfconfig="attn:v4:test")
        config.generate_mlir_driver_commandline.return_value = "-operation attention"
        paths = types.SimpleNamespace(mlir_paths=types.SimpleNamespace(
            rocmlir_gen_path="rocmlir-gen", rocmlir_tuning_driver_path="rocmlir-tuning-driver"))
        with mock.patch.object(perfRunner.os.path, "exists", return_value=False), \
                mock.patch.object(perfRunner, "run_pipeline", return_value=("time 10", True)):
            result = perfRunner.run_config_with_mlir(config,
                                                     paths,
                                                     "gfx90a",
                                                     "-current_seq_len=7",
                                                     debug=False)
        self.assertEqual(result, 10)
        config.generate_mlir_driver_commandline.assert_called_once_with("-current_seq_len=7", None)

    def test_prepare_config(self):
        config, flags = runner.prepare_config("-g 2 -seq_len_q 1 -seq_len_k 8")
        self.assertEqual(config, "-g 2 -seq_len_q 1 -seq_len_k 8")
        self.assertEqual(flags, "-current_seq_len=7,7")
        config, flags = runner.prepare_config(
            "-g 1 -seq_len_q 1 -seq_len_k 8 -current_seq_len=4 -prefix_offset 2")
        self.assertNotIn("current_seq_len", config)
        self.assertNotIn("prefix_offset", config)
        self.assertEqual(flags, "-current_seq_len=4 -prefix_offset=2")
        with self.assertRaisesRegex(ValueError, "current_seq_len"):
            runner.prepare_config("-g 2 -seq_len_q 1 -seq_len_k 8 -current_seq_len=4")
        with self.assertRaisesRegex(ValueError, "current_seq_len"):
            runner.prepare_config("-g 1 -seq_len_q 1 -seq_len_k 8 -current_seq_len=8")
        self.assertEqual(runner.remove_config_options("--g=1", set()), "--g 1")

    def test_perf_row_and_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "perf.csv"
            csv_path.write_text("Chip,PerfConfig,TFlops\ngfx90a,p,12.5\n", encoding="utf-8")
            self.assertEqual(runner.read_perf_row(csv_path)["Chip"], "gfx90a")
            csv_path.write_text("Chip,PerfConfig,TFlops\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Expected one"):
                runner.read_perf_row(csv_path)
            csv_path.write_text("Chip,PerfConfig,TFlops\ngfx,p,nan\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Invalid"):
                runner.read_perf_row(csv_path)
            self.assertEqual(runner.read_results(root / "missing"), [])
            rows = [{
                "RunLabel": "base",
                "SourceSha": "sha",
                "Chip": "gfx",
                "Config": "-g 1",
                "RocmlirGenFlags": "",
                "Sample": "1",
                "PerfConfig": "p",
                "TFlops": "2",
            }]
            runner.write_results(csv_path, rows)
            self.assertEqual(runner.read_results(csv_path), rows)

    def test_tuning_and_paired_benchmark_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = make_args(root)
            base = runner.BranchRun("base", root, root, "base-sha", root / "base")
            candidate = runner.BranchRun("candidate", root, root, "candidate-sha",
                                         root / "candidate")
            with mock.patch.object(runner, "run_with_retries"), \
                    mock.patch.object(runner, "verify_stably_idle"), \
                    mock.patch.object(runner,
                                      "read_perf_row",
                                      return_value={
                                          "Chip": "gfx",
                                          "PerfConfig": "p",
                                          "TFlops": "2.5"
                                      }):
                runner.quick_tune_branch(candidate, ["-g 1"], 0, args)
                runner.quick_tune_branch(base, ["-g 1"], 0, args)
                runner.benchmark_paired(base, candidate, ["-g 1"], "gfx", 0, args)
                self.assertEqual(len(runner.read_results(base.results)), 2)
                self.assertEqual(len(runner.read_results(candidate.results)), 2)
                runner.benchmark_paired(base, candidate, ["-g 1"], "gfx", 0, args)
                self.assertEqual(len(runner.read_results(base.results)), 2)

            rows = runner.read_results(base.results)
            rows[0]["SourceSha"] = "stale"
            runner.write_results(base.results, rows)
            with self.assertRaisesRegex(RuntimeError, "Stale"):
                runner.resume_state(base, "gfx", args.samples)
            rows[0]["SourceSha"] = base.sha
            rows.append(dict(rows[0]))
            runner.write_results(base.results, rows)
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                runner.resume_state(base, "gfx", args.samples)

            with mock.patch.object(runner, "run_with_retries"), \
                    mock.patch.object(runner,
                                      "read_perf_row",
                                      return_value={
                                          "Chip": "other",
                                          "PerfConfig": "p",
                                          "TFlops": "2"
                                      }):
                with self.assertRaisesRegex(RuntimeError, "expected gfx"):
                    runner.benchmark_sample(candidate, "-g 2", 1, "gfx", 0, args, [], set())


class WorkflowTest(unittest.TestCase):

    def test_manifest_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            self.assertIsNone(runner._load_manifest(path))
            path.write_text('{"base_sha":"a"}', encoding="utf-8")
            self.assertEqual(runner._load_manifest(path), {"base_sha": "a"})
        runner.verify_manifest(None, {})
        runner.verify_manifest({"base_sha": "a"}, {"base_sha": "a"})
        with self.assertRaisesRegex(RuntimeError, "base_sha"):
            runner.verify_manifest({"base_sha": "a"}, {"base_sha": "b"})

    def test_run_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = make_args(root)
            args.configs.write_text("-g 1\n", encoding="utf-8")
            idle = ({2: {"use": 0, "memory": 0, "arch": "gfx1201", "sku": "SKU"}}, {2: []})
            with mock.patch.object(runner,
                                   "validate_tree",
                                   side_effect=["base-sha", "candidate-sha", "base-sha",
                                                "candidate-sha"]), \
                    mock.patch.object(runner,
                                      "build_fingerprint",
                                      side_effect=["base-fp", "candidate-fp", "base-fp",
                                                   "candidate-fp"]), \
                    mock.patch.object(runner, "query_gpu_status", return_value=idle), \
                    mock.patch.object(runner, "verify_stably_idle", return_value="gfx1201"), \
                    mock.patch.object(runner, "validate_branch_unchanged"), \
                    mock.patch.object(runner, "quick_tune_branch") as tune, \
                    mock.patch.object(runner, "benchmark_paired") as benchmark, \
                    mock.patch.object(runner, "gpu_lock", return_value=contextlib.nullcontext()), \
                    mock.patch("builtins.print"):
                self.assertEqual(runner.run_workflow(args), 0)
                self.assertEqual(runner.run_workflow(args), 0)
            self.assertEqual([call.args[0].label for call in tune.call_args_list],
                             ["candidate", "base", "candidate", "base"])
            self.assertEqual(benchmark.call_count, 2)
            manifest = json.loads(
                (args.output_dir / "gfx1201" / "run-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")

            with mock.patch.object(runner, "validate_tree", return_value="same"):
                with self.assertRaisesRegex(ValueError, "identical"):
                    runner.run_workflow(args)
            args.configs.write_text("", encoding="utf-8")
            with mock.patch.object(runner, "validate_tree", side_effect=["base", "candidate"]):
                with self.assertRaisesRegex(ValueError, "empty"):
                    runner.run_workflow(args)

            args.base_source = args.candidate_source
            with self.assertRaisesRegex(ValueError, "distinct"):
                runner.run_workflow(args)

    def test_workflow_rejects_orphaned_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = make_args(root)
            args.configs.write_text("-g 1\n", encoding="utf-8")
            artifact = args.output_dir / "gfx90a" / "base" / "performance.csv"
            artifact.parent.mkdir(parents=True)
            artifact.touch()
            idle = ({0: {"use": 0, "memory": 0, "arch": "gfx90a", "sku": "SKU"}}, {0: []})
            with mock.patch.object(runner,
                                   "validate_tree",
                                   side_effect=["base", "candidate"]), \
                    mock.patch.object(runner,
                                      "build_fingerprint",
                                      side_effect=["base-fp", "candidate-fp"]), \
                    mock.patch.object(runner, "query_gpu_status", return_value=idle), \
                    mock.patch.object(runner, "verify_stably_idle", return_value="gfx90a"), \
                    mock.patch.object(runner, "gpu_lock", return_value=contextlib.nullcontext()):
                with self.assertRaisesRegex(RuntimeError, "without a matching"):
                    runner.run_workflow(args)

    def test_workflow_rejects_heterogeneous_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = make_args(root)
            args.configs.write_text("-g 1\n", encoding="utf-8")
            status = {
                0: {
                    "use": 1,
                    "memory": 0,
                    "arch": "gfx90a",
                    "sku": "A"
                },
                1: {
                    "use": 0,
                    "memory": 0,
                    "arch": "gfx950",
                    "sku": "B"
                },
            }
            with mock.patch.object(runner,
                                   "validate_tree",
                                   side_effect=["base", "candidate"]), \
                    mock.patch.object(runner, "query_gpu_status", return_value=(status, {
                        0: [],
                        1: []
                    })):
                with self.assertRaisesRegex(RuntimeError, "non-first"):
                    runner.run_workflow(args)

    def test_detach_and_main(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = types.SimpleNamespace(pid=123)
            with mock.patch.object(runner.subprocess, "Popen", return_value=process) as popen, \
                    mock.patch("builtins.print"):
                self.assertEqual(runner.launch_detached(["--detach"], root), 0)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual((root / "detached.pid").read_text(encoding="utf-8"), "123\n")

            args = make_args(root)
            argv = [
                "--base-source",
                str(args.base_source), "--base-build",
                str(args.base_build), "--candidate-source",
                str(args.candidate_source), "--candidate-build",
                str(args.candidate_build), "--configs",
                str(args.configs), "--output-dir",
                str(args.output_dir)
            ]
            with mock.patch.object(runner, "run_workflow", return_value=7):
                self.assertEqual(runner.main(argv), 7)
            with mock.patch.object(runner, "launch_detached", return_value=8):
                self.assertEqual(runner.main(argv + ["--detach"]), 8)
            with mock.patch.object(runner.sys, "argv", ["tool", *argv]), \
                    mock.patch.object(runner, "run_workflow", return_value=9):
                self.assertEqual(runner.main(), 9)
            with self.assertRaisesRegex(ValueError, "intervals"):
                runner.main(argv + ["--retry-backoff", "-1"])
            with self.assertRaisesRegex(ValueError, "intervals"):
                runner.main(argv + ["--idle-interval", "-1"])

    def test_integer_argument_types(self):
        self.assertEqual(runner.positive_integer("2"), 2)
        self.assertEqual(runner.nonnegative_integer("0"), 0)
        with self.assertRaises(argparse.ArgumentTypeError):
            runner.positive_integer("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            runner.nonnegative_integer("-1")


if __name__ == "__main__":
    unittest.main()
