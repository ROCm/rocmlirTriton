#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Compile tuning artifacts locally, benchmark them on a remote GPU host.

This is a thin orchestration wrapper around tuningRunner.py:

The rocmlirTriton repo must already exist on both machines, and the local
and remote checkouts should be at the same commit so artifact metadata matches
the benchmarking binaries. You also need to have the docker image/container
already setup on the remote host. The remote build tree is assumed to be
<--remote-repo-dir>/build as the container sees it; pass --remote-build-dir if
it lives anywhere else. The local build tree is the one this script was
installed into, which ci-performance-scripts makes <build>/bin; pass
--local-build-dir when running a copy that sits outside a build tree.

Artifact bundles are zstd-framed, so both builds must be configured with
-DLLVM_ENABLE_ZSTD=FORCE_ON, which needs libzstd headers (libzstd-dev) and not
just the runtime library. Without it step 2 below fails before spending any
compile time, and step 4 refuses the bundle.

--target-arch must be the full gcnArchName reported by HIP on the benchmark
host, including target features (for example, gfx950:sramecc+:xnack-). A bare
chip name describes a different code-generation target and will be rejected
when the bundle is compared with the live GPU. The full string is also written
to the result TSV, matching normal local tuning database keys.

  1. Open one SSH session before a long compile starts. The remote side waits
     for the artifact tar stream, so password prompts happen up front without
     creating a reusable OpenSSH control socket.
  2. Run tuningRunner.py --compile-only on the local machine. A compile where
     some problems failed still ships the bundles it did produce; the compile
     status is reported after the results come back.
  3. Stream artifacts and the config file into the waiting remote session.
  4. Run tuningRunner.py --benchmark-artifacts on the remote host.
  5. Stream the result TSV back to the local output path over the same session.
     A run where some problems failed still returns the rows it did produce, and
     the remote's exit status is reported afterwards.
"""

import argparse
import json
import os
import posixpath
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import List, Optional, Sequence

from tuningArgumentUtils import ADAPTIVE_TUNING_SPACES, add_common_tuning_arguments

# The remote script writes this after SSH authentication and remote directory
# setup complete. The local process waits for it before starting a long compile,
# so password prompts happen up front instead of hours later.
REMOTE_READY_MARKER = "__ROCMLIR_CROSS_COMPILE_READY__"


def logical_absolute(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(os.environ.get("PWD", os.getcwd())) / path


def ssh_target(remote_host: str, remote_user: Optional[str]) -> str:
    if "@" in remote_host:
        if remote_user:
            raise ValueError(
                "--remote-user cannot be used when --remote-host already contains user@")
        return remote_host
    if remote_user:
        return f"{remote_user}@{remote_host}"
    return remote_host


def print_command(label: str, cmd: Sequence[str], verbose: bool) -> None:
    if not verbose:
        return
    print(f"\n# {label}", file=sys.stderr)
    print(shlex.join([str(part) for part in cmd]), file=sys.stderr)


def run_command(label: str,
                cmd: Sequence[str],
                dry_run: bool = False,
                verbose: bool = False,
                check: bool = True) -> int:
    """Run a wrapper command and return its exit status.

    With check=True a non-zero status raises CalledProcessError, as
    subprocess.run does; pass check=False to inspect the status instead.
    """
    print_command(label, cmd, verbose or dry_run)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=check).returncode


def load_bundle_index(artifacts_dir: Path) -> dict:
    """Contents of a --compile-only bundle index, empty when unusable."""
    try:
        with open(artifacts_dir / "index.json") as index_file:
            index = json.load(index_file)
    except (OSError, json.JSONDecodeError):
        return {}
    return index if isinstance(index, dict) else {}


def read_bundle_run_id(artifacts_dir: Path) -> Optional[str]:
    """Id of the last --compile-only run to record a problem in the bundle."""
    run_id = load_bundle_index(artifacts_dir).get("lastRunId")
    return run_id if isinstance(run_id, str) else None


def count_compiled_problems(artifacts_dir: Path, previous_run_id: Optional[str]) -> int:
    """Number of problems the compile that just ran left in the bundle.

    --local-artifacts-dir is reused across runs and never cleaned, so problems
    an earlier run recorded for a different config file are still sitting
    there. Count only what carries this run's id, and nothing at all when the
    compile died before claiming the bundle.
    """
    index = load_bundle_index(artifacts_dir)
    run_id = index.get("lastRunId")
    if not isinstance(run_id, str) or run_id == previous_run_id:
        return 0
    problems = index.get("problems")
    if not isinstance(problems, dict):
        return 0
    return sum(1 for problem in problems.values()
               if isinstance(problem, dict) and problem.get("runId") == run_id)


def terminate_process(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def copy_stream_to_file(stream, path: Path) -> None:
    with open(path, "wb") as output:
        shutil.copyfileobj(stream, output)


def copy_stream_to_future(stream, path: Path, future: Future) -> None:
    """Drain a stream and preserve any reader exception for the main thread."""
    try:
        copy_stream_to_file(stream, path)
    except BaseException as error:
        future.set_exception(error)
    else:
        future.set_result(None)


def add_passthrough_flags(cmd: List[str], args: argparse.Namespace, *, remote: bool) -> None:
    if args.debug:
        cmd.append("--debug")
    if args.debug_quick_tune_data:
        cmd.append("--debug-quick-tune-data")
    if args.verbose:
        cmd.append("--verbose")
    if args.timeout is not None:
        cmd.append(f"--timeout={args.timeout}")

    # Both phases expand the config file into test vectors, and the expansion
    # feeds the hash that names each problem's artifact directory. Pass these
    # explicitly rather than letting each side fall back to its own default, so
    # the remote cannot end up looking for bundles the compile host never made.
    cmd.append("--data-type")
    cmd.extend(args.data_type)
    if args.scale_type:
        cmd.append("--scale-type")
        cmd.extend(args.scale_type)

    if remote:
        if args.gpus:
            cmd.append("--gpus")
            cmd.extend(str(gpu) for gpu in args.gpus)
        if args.verify_winning_config:
            cmd.append("--verify-winning-config")
        if args.verify_all_perfconfigs:
            cmd.append("--verify-all-perfconfigs")
        if args.flush_last_level_cache:
            cmd.append("--flush-last-level-cache")
        if args.gpu_run_timeout:
            cmd.append(f"--gpu-run-timeout={args.gpu_run_timeout}")
        if args.allow_commit_mismatch:
            cmd.append("--allow-commit-mismatch")
    else:
        if args.num_cpus is not None:
            cmd.append(f"--num-cpus={args.num_cpus}")


def build_local_compile_command(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(logical_absolute(Path(__file__)).parent / "tuningRunner.py"),
    ]
    if args.local_build_dir:

        cmd += ["--mlir-build-dir", str(args.local_build_dir)]
    cmd += [
        "--op",
        args.op,
        "--configs-file",
        str(args.configs_file),
        "--tuning-space",
        args.tuning_space,
        "--target-arch",
        args.target_arch,
        "--target-num-cu",
        str(args.target_num_cu),
        "--target-num-chiplets",
        str(args.target_num_chiplets),
        "--compile-only",
        str(args.local_artifacts_dir),
        "--perf-config-timeout",
        str(args.perf_config_timeout),
    ]
    if args.rocmlir_gen_flags:
        cmd.extend(["--rocmlir-gen-flags", args.rocmlir_gen_flags])
    add_passthrough_flags(cmd, args, remote=False)
    return cmd


def build_artifact_tar_command(args: argparse.Namespace) -> List[str]:
    config_parent = args.configs_file.parent
    config_name = args.configs_file.name
    return [
        args.tar,
        "-C",
        str(args.local_artifacts_dir),
        "-cf",
        "-",
        ".",
        "-C",
        str(config_parent),
        config_name,
    ]


def build_remote_benchmark_args(args: argparse.Namespace) -> List[str]:
    remote_runner = posixpath.join(args.remote_build_dir, "bin/tuningRunner.py")
    remote_docker_workdir = args.remote_build_dir
    remote_cmd = [
        "docker",
        "exec",
        "-w",
        remote_docker_workdir,
        args.remote_docker_container,
    ]
    if args.remote_session_timeout:
        # Terminating the local ssh does not terminate a docker exec'd process,
        # so the local deadline alone would leave the benchmark running and
        # holding the GPU for the next run to trip over. Bound it inside the
        # container as well, where nothing about the local side's health
        # matters. SIGKILL follows if tuningRunner.py does not unwind promptly.
        remote_cmd += ["timeout", "-k", "30", str(args.remote_session_timeout)]
    remote_cmd += [
        "python3",
        remote_runner,
        "--mlir-build-dir",
        args.remote_build_dir,
        "--op",
        args.op,
        "--configs-file",
        args.remote_configs_file,
        "--tuning-space",
        args.tuning_space,
        "--target-arch",
        args.target_arch,
        "--target-num-cu",
        str(args.target_num_cu),
        "--target-num-chiplets",
        str(args.target_num_chiplets),
        "--benchmark-artifacts",
        args.remote_artifacts_dir,
        "--output",
        args.remote_output,
    ]
    if args.rocmlir_gen_flags:
        # Benchmarking does not invoke rocmlir-gen, but tuningRunner includes
        # these flags in the problem hash used to locate compiled artifacts.
        remote_cmd.extend(["--rocmlir-gen-flags", args.rocmlir_gen_flags])
    add_passthrough_flags(remote_cmd, args, remote=True)
    return remote_cmd


def build_remote_session_script(args: argparse.Namespace) -> str:
    remote_output_dir = posixpath.dirname(args.remote_output) or "."
    remote_output_name = posixpath.basename(args.remote_output)
    remote_debug_name = remote_output_name + ".debug"
    remote_docker_workdir = args.remote_build_dir

    # stdout is reserved for the final result tar stream. Redirect normal remote
    # command output to stderr so setup or benchmark output cannot corrupt it.
    script_parts = [
        "set -eo pipefail",
        "exec 3>&1",
        "exec 1>&2",
        f"mkdir -p -- {shlex.quote(args.remote_artifacts_dir)}",
        f"cd {shlex.quote(args.remote_repo_dir)}",
    ]
    if args.remote_setup_command:
        script_parts.append(args.remote_setup_command)
    script_parts.extend([
        shlex.join(
            ["docker", "exec", "-w", remote_docker_workdir, args.remote_docker_container, "true"]),
        f"echo {shlex.quote(REMOTE_READY_MARKER)} >&3",
        f"{shlex.quote(args.tar)} -C {shlex.quote(args.remote_artifacts_dir)} -xf -",
        shlex.join(["rm", "-f", "--", args.remote_output, f"{args.remote_output}.debug"]),
        # tuningRunner.py exits non-zero when any single problem fails, but the
        # problems that already succeeded are in the TSV. Keep the status rather
        # than letting `set -e` abort here, so the rows still get shipped back;
        # the status is reported at the end instead.
        "benchmark_rc=0",
        shlex.join([str(part) for part in build_remote_benchmark_args(args)]) +
        " || benchmark_rc=$?",
        f"cd {shlex.quote(remote_output_dir)}",
        # A run that died before writing anything leaves no files behind, and
        # `tar -cf` on a missing member is itself an error, so build the member
        # list out of what actually exists.
        "files=()",
        f"if [ -f {shlex.quote(remote_output_name)} ]; then "
        f"files+=({shlex.quote(remote_output_name)}); fi",
        f"if [ -f {shlex.quote(remote_debug_name)} ]; then "
        f"files+=({shlex.quote(remote_debug_name)}); fi",
        f"if [ ${{#files[@]}} -gt 0 ]; then "
        f"{shlex.quote(args.tar)} -cf - \"${{files[@]}}\" >&3; fi",
        "exit $benchmark_rc",
    ])
    return "; ".join(script_parts)


def build_remote_session_command(args: argparse.Namespace) -> List[str]:
    """Build an SSH command whose stdout remains a binary-safe channel."""
    return [args.ssh] + args.ssh_option + [
        "-T",
        args.remote_target,
        "bash -lc " + shlex.quote(build_remote_session_script(args)),
    ]


def print_streaming_flow(args: argparse.Namespace) -> None:
    remote_session_cmd = build_remote_session_command(args)
    result_extract_cmd = [args.tar, "-C", str(args.output.parent), "-xf", "-"]

    print_command("Open remote SSH session and wait for artifacts", remote_session_cmd, True)
    print(f"# Wait for remote ready marker: {REMOTE_READY_MARKER}", file=sys.stderr)
    print_command("Compile artifacts locally", build_local_compile_command(args), True)
    print_command("Stream artifacts into remote session", build_artifact_tar_command(args), True)
    print(f"> {args.remote_target} stdin", file=sys.stderr)
    print("\n# Extract returned results locally", file=sys.stderr)
    print(f"{args.remote_target} stdout | {shlex.join(result_extract_cmd)}", file=sys.stderr)


def wait_for_remote_ready(remote_session: subprocess.Popen,
                          cmd: Sequence[str],
                          timeout: Optional[float] = None) -> None:
    assert remote_session.stdout is not None

    # A remote that stays alive but silent (unresponsive docker daemon, stuck
    # mount) would block the run forever. Killing the session closes the pipe,
    # which is what unblocks the read below; polling the stream directly is not
    # an option because it is handed to the result-tar reader afterwards.
    timed_out = threading.Event()

    def give_up() -> None:
        timed_out.set()
        terminate_process(remote_session)

    watchdog = threading.Timer(timeout, give_up) if timeout else None
    if watchdog:
        watchdog.start()

    try:
        while True:
            line = remote_session.stdout.readline()
            if not line:
                if timed_out.is_set():
                    raise subprocess.TimeoutExpired(cmd, timeout)
                # A clean exit without the marker is still a failed session, so
                # never report it as success.
                raise subprocess.CalledProcessError(remote_session.wait() or 1, cmd)

            text = line.decode("utf-8", errors="replace")
            if REMOTE_READY_MARKER in text:
                return
            sys.stderr.write(text)
    finally:
        if watchdog:
            watchdog.cancel()


def wait_for_remote_results(artifact_tar: subprocess.Popen,
                            remote_session: subprocess.Popen,
                            result_future: Future,
                            cmd: Sequence[str],
                            timeout: Optional[float] = None) -> tuple[int, int]:
    """Wait for transfer, remote benchmark, and result streaming under one deadline."""
    deadline = time.monotonic() + timeout if timeout else None

    def remaining() -> Optional[float]:
        if deadline is None:
            return None
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise subprocess.TimeoutExpired(cmd, timeout)
        return seconds

    def wait_for_process(process: subprocess.Popen) -> int:
        while True:
            if result_future.done():
                result_future.result()
            wait_timeout = remaining()
            wait_timeout = min(wait_timeout, 0.1) if wait_timeout is not None else 0.1
            try:
                return process.wait(timeout=wait_timeout)
            except subprocess.TimeoutExpired:
                continue

    try:
        artifact_rc = wait_for_process(artifact_tar)
        remote_rc = wait_for_process(remote_session)
        try:
            result_future.result(timeout=remaining())
        except FutureTimeoutError:
            raise subprocess.TimeoutExpired(cmd, timeout) from None
    except subprocess.TimeoutExpired:
        raise subprocess.TimeoutExpired(cmd, timeout) from None

    return artifact_rc, remote_rc


def extract_returned_results(args: argparse.Namespace, result_tar_path: Path, *,
                             session_failed: bool) -> None:
    """Unpack the result TSV the remote streamed back, if it sent one.

    The remote returns the rows it managed to write even when some problems
    failed, so this runs before the session status is reported; otherwise a
    single bad problem would discard a whole benchmark run. A session that died
    mid-stream leaves nothing or a truncated archive, which is fallout from the
    original failure and must not replace it as the reported error.
    """
    if not result_tar_path.stat().st_size and session_failed:
        return

    try:
        run_command(
            "Extract returned results locally",
            [args.tar, "-C", str(args.output.parent), "-xf",
             str(result_tar_path)],
            verbose=args.verbose)
    except subprocess.CalledProcessError:
        if not session_failed:
            raise
        print(f"crossCompile.py: could not unpack the results returned in {result_tar_path}",
              file=sys.stderr)


def raise_for_session_errors(artifact_rc: int, remote_rc: int, artifact_cmd: Sequence[str],
                             remote_cmd: Sequence[str]) -> None:
    # If the remote session dies while tar is writing artifacts, tar is killed
    # by SIGPIPE. That failure is only fallout from the remote error, so report
    # the remote command and its useful exit status instead.
    if remote_rc != 0 and artifact_rc in (0, -signal.SIGPIPE):
        raise subprocess.CalledProcessError(remote_rc, remote_cmd)
    if artifact_rc != 0:
        raise subprocess.CalledProcessError(artifact_rc, artifact_cmd)
    if remote_rc != 0:
        raise subprocess.CalledProcessError(remote_rc, remote_cmd)


def run_cross_compile_session(args: argparse.Namespace) -> None:
    if args.dry_run:
        print_streaming_flow(args)
        return

    remote_session_cmd = build_remote_session_command(args)

    print_command("Open remote SSH session and wait for artifacts", remote_session_cmd,
                  args.verbose)
    remote_session = subprocess.Popen(remote_session_cmd,
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE)
    artifact_tar = None
    result_reader = None
    result_future = None
    result_tar_path = None
    try:
        print("Waiting for remote SSH session to become ready...", file=sys.stderr)
        wait_for_remote_ready(remote_session, remote_session_cmd, args.remote_ready_timeout)
        previous_run_id = read_bundle_run_id(args.local_artifacts_dir)
        compile_cmd = build_local_compile_command(args)
        compile_rc = run_command("Compile artifacts locally",
                                 compile_cmd,
                                 verbose=args.verbose,
                                 check=False)
        if compile_rc != 0:
            # tuningRunner.py exits non-zero when any single problem fails, but
            # the bundles it did produce are still benchmarkable. Mirror what
            # the remote side does with a partial benchmark run: ship what
            # exists and report the compile status once the results are back.
            compiled = count_compiled_problems(args.local_artifacts_dir, previous_run_id)
            if not compiled:
                raise subprocess.CalledProcessError(compile_rc, compile_cmd)
            print(
                f"crossCompile.py: local compile failed with exit code {compile_rc}; "
                f"benchmarking the {compiled} problem(s) that did compile",
                file=sys.stderr)

        assert remote_session.stdin is not None
        assert remote_session.stdout is not None

        with tempfile.NamedTemporaryFile(prefix="rocmlir-cross-results-",
                                         suffix=".tar",
                                         delete=False) as result_tar:
            result_tar_path = Path(result_tar.name)
        result_future = Future()
        result_reader = threading.Thread(target=copy_stream_to_future,
                                         args=(remote_session.stdout, result_tar_path,
                                               result_future),
                                         daemon=True)
        result_reader.start()

        artifact_tar_cmd = build_artifact_tar_command(args)
        artifact_tar = subprocess.Popen(artifact_tar_cmd, stdout=remote_session.stdin)
        remote_session.stdin.close()

        artifact_rc, remote_rc = wait_for_remote_results(artifact_tar, remote_session,
                                                         result_future, remote_session_cmd,
                                                         args.remote_session_timeout)

        extract_returned_results(args,
                                 result_tar_path,
                                 session_failed=artifact_rc != 0 or remote_rc != 0)
        raise_for_session_errors(artifact_rc, remote_rc, artifact_tar_cmd, remote_session_cmd)
        if compile_rc != 0:
            raise subprocess.CalledProcessError(compile_rc, compile_cmd)
    except BaseException:
        if remote_session.stdin:
            remote_session.stdin.close()
        if remote_session.stdout:
            remote_session.stdout.close()
        terminate_process(artifact_tar)
        terminate_process(remote_session)
        raise
    finally:
        if result_reader and result_reader.is_alive():
            result_reader.join(timeout=5)
        if result_tar_path:
            result_tar_path.unlink(missing_ok=True)


def existing_file(path: str) -> Path:
    result = Path(path).expanduser()
    if not result.is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a file")
    return result


def build_dir_marker(build_dir: Path) -> Path:
    """Tool whose presence identifies a directory as a rocMLIR build tree.

    perfRunner.find_mlir_build_dir() keys off this same path, so accepting
    exactly what it accepts keeps --local-build-dir from rejecting a build tree
    tuningRunner.py would have been happy to discover on its own.
    """
    return build_dir / "bin" / "rocmlir-gen"


def existing_build_dir(path: str) -> Path:
    result = logical_absolute(Path(path).expanduser())
    if not build_dir_marker(result).is_file():
        raise argparse.ArgumentTypeError(f"{path} is not a rocMLIR build directory: "
                                         "it has no bin/rocmlir-gen")
    return result


def installed_build_dir() -> Optional[Path]:
    """Build tree this script was installed into, or None in a source checkout.

    ci-performance-scripts copies these scripts next to the built tools in
    <build>/bin, so the installed copy can name its own build directory rather
    than leave tuningRunner.py to guess one from the current directory. A source
    checkout has no such neighbour, and needs no help: tuningRunner.py finds
    <repo>/build from its own location in that case.
    """
    script_dir = logical_absolute(Path(__file__)).parent
    return script_dir.parent if (script_dir / "rocmlir-gen").is_file() else None


def absolute_remote_path(path: str, arg_name: str) -> str:
    if not path.startswith("/"):
        raise argparse.ArgumentTypeError(f"{arg_name} must be an absolute remote path")
    return path


def ssh_options_request_pty(options: Sequence[str]) -> bool:
    """Return True if SSH options explicitly request terminal allocation."""
    pending_o_value = False
    for option in options:
        if option.startswith("-") and option[1:] and set(option[1:]) == {"t"}:
            return True

        setting = None
        if pending_o_value:
            setting = option
            pending_o_value = False
        elif option == "-o":
            pending_o_value = True
        elif option.startswith("-o"):
            setting = option[2:]

        if setting is None:
            continue
        setting = setting.strip()
        if "=" in setting:
            key, value = setting.split("=", 1)
        else:
            parts = setting.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
        if key.lower() == "requesttty" and value.strip().lower() != "no":
            return True
    return False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-compile rocMLIR tuning artifacts locally and benchmark them remotely.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_tuning_arguments(parser,
                                target_required=True,
                                tuning_space_required=True,
                                perf_config_timeout_default=250)

    parser.add_argument("-c",
                        "--configs-file",
                        required=True,
                        type=existing_file,
                        help="Problem config file to tune.")
    parser.add_argument("--remote-host",
                        required=True,
                        help="Remote GPU host or user@host to benchmark on.")
    parser.add_argument("--remote-user",
                        default=None,
                        help="SSH user. Omit to let SSH choose the user, or when "
                        "--remote-host is user@host.")
    parser.add_argument("--local-build-dir",
                        default=None,
                        type=existing_build_dir,
                        help="rocMLIR build directory on this machine; holds bin/rocmlir-gen "
                        "and the tuning driver the compile phase runs. Defaults to the build "
                        "tree this script was installed into, and otherwise leaves "
                        "tuningRunner.py to discover one.")
    parser.add_argument("--local-artifacts-dir",
                        type=Path,
                        default=Path("cross_compile_artifacts"),
                        help="Local directory for tuningRunner.py --compile-only output.")
    parser.add_argument("--remote-artifacts-dir",
                        required=True,
                        type=lambda path: absolute_remote_path(path, "--remote-artifacts-dir"),
                        help="Remote directory for artifacts, config file, and remote result TSV.")
    parser.add_argument("--remote-repo-dir",
                        required=True,
                        type=lambda path: absolute_remote_path(path, "--remote-repo-dir"),
                        help="rocmlirTriton checkout directory on the remote host.")
    parser.add_argument("--remote-build-dir",
                        default=None,
                        type=lambda path: absolute_remote_path(path, "--remote-build-dir"),
                        help="rocMLIR build directory as seen inside the remote container; "
                        "holds bin/tuningRunner.py and the tuning driver. Defaults to "
                        "<--remote-repo-dir>/build.")
    parser.add_argument("-o",
                        "--output",
                        type=Path,
                        default=Path("tuning_results_cross.tsv"),
                        help="Local result TSV copied back from the remote host.")
    parser.add_argument("--remote-docker-container",
                        required=True,
                        help="Docker container name or id to run remote benchmarking in.")
    parser.add_argument("--remote-setup-command",
                        default=None,
                        help="Optional shell command run before remote tuningRunner.py.")

    parser.add_argument("--gpus",
                        type=int,
                        nargs="+",
                        default=None,
                        help="Remote GPU IDs passed to tuningRunner.py.")
    parser.add_argument("--gpu-run-timeout",
                        type=int,
                        default=0,
                        metavar="SECONDS",
                        help="Per-config GPU-run timeout forwarded to the remote benchmark "
                        "(0 disables the timeout).")
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="Print wrapper commands and forward --verbose to tuningRunner.py.")
    parser.add_argument("--ssh", default="ssh", help="SSH executable.")
    parser.add_argument("--ssh-option",
                        action="append",
                        default=[],
                        help="Extra SSH option. Repeat for multiple arguments. PTY allocation "
                        "options are rejected because stdout carries a binary tar stream.")
    parser.add_argument("--tar", default="tar", help="tar executable on both hosts.")
    parser.add_argument(
        "--remote-ready-timeout",
        type=int,
        default=600,
        metavar="SECONDS",
        help="Give up if the remote session does not report readiness within this "
        "many seconds; covers interactive SSH auth and container startup. 0 waits forever.")
    parser.add_argument(
        "--remote-session-timeout",
        type=int,
        default=21600,  # 6 hours
        metavar="SECONDS",
        help="Give up if artifact transfer, remote benchmarking, and result streaming do "
        "not finish within this shared deadline after local compilation. The same budget "
        "also bounds the benchmark inside the container, because terminating the local SSH "
        "would otherwise leave it running and holding the GPU; a benchmark stopped that way "
        "reports exit code 124. 0 waits forever on both sides.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running compile, transfer, or benchmark steps.")

    args = parser.parse_args(argv)
    if ssh_options_request_pty(args.ssh_option):
        parser.error("--ssh-option cannot request a PTY because SSH stdout carries a binary tar")
    # The local phase compiles everything up front with --compile-only, which a
    # searched space cannot do: it picks each batch from the previous batch's
    # timings, and there is no GPU on this side to produce them.
    if args.tuning_space in ADAPTIVE_TUNING_SPACES:
        parser.error(f"--tuning-space={args.tuning_space} cannot be cross-compiled: it needs "
                     "benchmark timings to decide what to compile next")
    # tuningRunner.py --benchmark-artifacts benchmarks one problem at a time on
    # a single device and rejects anything else. Catch it here rather than
    # letting the remote reject it after a compile that can run for hours.
    if args.gpus is not None and len(args.gpus) != 1:
        parser.error("--gpus must name exactly one GPU: the remote benchmark phase runs on a "
                     "single device")
    args.remote_target = ssh_target(args.remote_host, args.remote_user)
    if args.remote_build_dir is None:
        args.remote_build_dir = posixpath.join(args.remote_repo_dir, "build")
    if args.local_build_dir is None:
        args.local_build_dir = installed_build_dir()
    args.configs_file = logical_absolute(args.configs_file)
    args.local_artifacts_dir = args.local_artifacts_dir.expanduser()
    args.output = args.output.expanduser()
    if str(args.output) == "-":
        parser.error("--output must be a file path because results are copied back locally")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config_name = args.configs_file.name

    # The bundle, the config file, and the result TSV all land in the same
    # remote directory, and the remote session deletes <output> and
    # <output>.debug there before benchmarking so a rerun cannot append to a
    # stale TSV. An --output naming one of the inputs would delete that input.
    remote_inputs = {
        config_name: "the config file",
        "index.json": "the artifact index",
        "problems": "the compiled bundles",
    }
    for name in (args.output.name, f"{args.output.name}.debug"):
        if name in remote_inputs:
            parser.error(f"--output would make the remote delete {remote_inputs[name]}: "
                         f"'{name}' is unpacked into --remote-artifacts-dir")

    args.remote_artifacts_dir = args.remote_artifacts_dir.rstrip("/") or "/"
    args.remote_configs_file = posixpath.join(args.remote_artifacts_dir, config_name)
    args.remote_output = posixpath.join(args.remote_artifacts_dir, args.output.name)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        run_cross_compile_session(args)
    except subprocess.CalledProcessError as error:
        print(f"crossCompile.py: command failed with exit code {error.returncode}", file=sys.stderr)
        if error.returncode in (124, 137):
            print(
                "crossCompile.py: that usually means the remote benchmark exceeded "
                "--remote-session-timeout and was stopped inside the container",
                file=sys.stderr)
        print("crossCompile.py: rerun with -v/--verbose to print wrapper commands", file=sys.stderr)
        return error.returncode
    except subprocess.TimeoutExpired as error:
        print(f"crossCompile.py: {error}", file=sys.stderr)
        print(
            "crossCompile.py: rerun with --remote-ready-timeout or "
            "--remote-session-timeout to wait longer",
            file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"crossCompile.py: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
