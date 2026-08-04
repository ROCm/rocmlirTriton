# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Quick-tune and benchmark attention configs on two isolated source trees."""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from attentionPerfUtils import atomic_write_text, canonical_config, normalize_option, parse_config
from attentionPerfUtils import read_config_lines, remove_flag, write_json

RESULT_FIELDS = [
    "RunLabel", "SourceSha", "Chip", "Config", "RocmlirGenFlags", "Sample", "PerfConfig", "TFlops"
]
PROCESS_GPU_INDEX = 1
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class BranchRun:
    label: str
    source: Path
    build: Path
    sha: str
    output_dir: Path

    @property
    def tuning_db(self) -> Path:
        return self.output_dir / "quick-tuning.tsv"

    @property
    def results(self) -> Path:
        return self.output_dir / "performance.csv"

    @property
    def log(self) -> Path:
        return self.output_dir / "run.log"


def _check_output(command: Sequence[str]) -> str:
    return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=60)


def git_revision(source: Path) -> str:
    return _check_output(["git", "-C", str(source), "rev-parse", "HEAD"]).strip()


def git_is_dirty(source: Path) -> bool:
    return bool(_check_output(["git", "-C", str(source), "status", "--porcelain"]).strip())


def build_fingerprint(source: Path, build: Path) -> str:
    digest = hashlib.sha256()
    artifacts = [
        build / "CMakeCache.txt",
        build / "bin" / "rocmlir-gen",
        build / "bin" / "rocmlir-driver",
        build / "bin" / "rocmlir-tuning-driver",
        build / "bin" / "rocm-run",
        build / "lib" / "libconv-validation-wrappers.so",
        source / "external" / "triton" / "llvm-project" / "build" / "bin" / "mlir-runner",
    ]
    runtime_dir = source / "external" / "triton" / "llvm-project" / "build" / "lib"
    artifacts.extend(sorted(runtime_dir.glob("libmlir_*_runtime.so")))
    for path in artifacts:
        digest.update(str(path).encode("utf-8"))
        if not path.is_file():
            digest.update(b"<missing>")
            continue
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def cmake_source_directory(build: Path) -> Optional[Path]:
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        return None
    with cache.open("r", encoding="utf-8") as stream:
        for line in stream:
            prefix = "CMAKE_HOME_DIRECTORY:INTERNAL="
            if line.startswith(prefix):
                return Path(line[len(prefix):].strip()).resolve()
    return None


def validate_tree(label: str, source: Path, build: Path) -> str:
    if not source.is_dir():
        raise ValueError(f"{label} source directory does not exist: {source}")
    if not build.is_dir():
        raise ValueError(f"{label} build directory does not exist: {build}")
    if git_is_dirty(source):
        raise ValueError(f"{label} source tree is dirty: {source}")
    required_source = ("tuningRunner.py", "perfRunner.py")
    for script in required_source:
        path = source / "mlir" / "utils" / "performance" / script
        if not path.is_file():
            raise ValueError(f"{label} source is missing {path}")
    for binary in ("rocmlir-gen", "rocmlir-driver", "rocmlir-tuning-driver"):
        path = build / "bin" / binary
        if not path.is_file():
            raise ValueError(f"{label} build is missing {path}")
    configured_source = cmake_source_directory(build)
    if configured_source != source.resolve():
        raise ValueError(
            f"{label} build was configured from {configured_source}, expected {source.resolve()}")
    revision = git_revision(source)
    stamp = build / "rocmlir-source-sha"
    if not stamp.is_file() or stamp.read_text(encoding="utf-8").strip() != revision:
        raise ValueError(f"{label} build source stamp does not match {revision}: {stamp}")
    return revision


def validate_branch_unchanged(branch: BranchRun, expected_fingerprint: str) -> None:
    if validate_tree(branch.label, branch.source, branch.build) != branch.sha:
        raise RuntimeError(f"{branch.label} source revision changed during the run")
    if build_fingerprint(branch.source, branch.build) != expected_fingerprint:
        raise RuntimeError(f"{branch.label} build artifacts changed during the run")


def _integer(value) -> int:
    return int(str(value).rstrip("%"))


def parse_gpu_status(status_json: str,
                     process_json: str) -> Tuple[Dict[int, Dict], Dict[int, List[int]]]:
    status = {}
    raw_status = json.loads(status_json)
    for card, values in raw_status.items():
        match = re.fullmatch(r"card(\d+)", card)
        if match is None:
            continue
        required = {"GFX Version", "Card SKU", "GPU use (%)", "GPU Memory Allocated (VRAM%)"}
        if not required.issubset(values):
            raise RuntimeError(f"Incomplete rocm-smi status for {card}")
        gpu = int(match.group(1))
        status[gpu] = {
            "arch": values["GFX Version"],
            "sku": values["Card SKU"],
            "use": _integer(values["GPU use (%)"]),
            "memory": _integer(values["GPU Memory Allocated (VRAM%)"]),
        }
    if not status:
        raise RuntimeError("rocm-smi did not report any GPU cards")

    processes = {gpu: [] for gpu in status}
    process_data = json.loads(process_json)
    if not isinstance(process_data.get("system"), dict):
        raise RuntimeError("rocm-smi did not report a process table")
    raw_processes = process_data["system"]
    for process_name, description in raw_processes.items():
        pid_match = re.fullmatch(r"PID(\d+)", process_name)
        fields = [field.strip() for field in str(description).split(",")]
        if pid_match is None:
            continue
        if len(fields) <= PROCESS_GPU_INDEX:
            raise RuntimeError(f"Malformed rocm-smi process entry: {process_name}")
        try:
            gpu = int(fields[PROCESS_GPU_INDEX])
        except ValueError as error:
            raise RuntimeError(f"Malformed GPU index for {process_name}") from error
        pid = int(pid_match.group(1))
        if gpu in processes and pid != os.getpid() and Path(f"/proc/{pid}").exists():
            processes[gpu].append(pid)
    return status, processes


def query_gpu_status() -> Tuple[Dict[int, Dict], Dict[int, List[int]]]:
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi is None:
        raise RuntimeError("rocm-smi is required to prove that a GPU is idle")
    status = _check_output([rocm_smi, "--showproductname", "--showuse", "--showmemuse", "--json"])
    processes = _check_output([rocm_smi, "--showpids", "--json"])
    return parse_gpu_status(status, processes)


def is_gpu_idle(gpu: int, status: Mapping[int, Dict], processes: Mapping[int, List[int]]) -> bool:
    values = status.get(gpu)
    return values is not None and values["use"] == 0 and values["memory"] == 0 and not processes[gpu]


def select_idle_gpu(requested: Optional[int], status: Mapping[int, Dict],
                    processes: Mapping[int, List[int]]) -> int:
    if requested is not None:
        if requested not in status:
            raise RuntimeError(f"Requested GPU {requested} was not reported by rocm-smi")
        if not is_gpu_idle(requested, status, processes):
            raise RuntimeError(f"Requested GPU {requested} is not idle")
        return requested
    for gpu in sorted(status):
        if is_gpu_idle(gpu, status, processes):
            return gpu
    raise RuntimeError("No idle GPU is available")


def verify_stably_idle(gpu: int, samples: int, interval: float) -> str:
    arch = None
    for sample in range(samples):
        status, processes = query_gpu_status()
        if not is_gpu_idle(gpu, status, processes):
            raise RuntimeError(f"GPU {gpu} became busy during idle verification")
        arch = status[gpu]["arch"]
        if sample + 1 < samples:
            time.sleep(interval)
    assert arch is not None
    return arch


@contextmanager
def gpu_lock(gpu: int):
    lock_path = Path(f"/tmp/rocmlir-attention-perf-gpu{gpu}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        os.chmod(lock_path, 0o666)
    except PermissionError:
        pass
    with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"GPU {gpu} is reserved by another benchmark process") from error
        yield


@contextmanager
def output_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".attention-perf.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Output directory is used by another run: {output_dir}") from error
        yield


def run_with_retries(command: Sequence[str],
                     env: Mapping[str, str],
                     log_path: Path,
                     retries: int,
                     backoff: float,
                     cwd: Path,
                     timeout: Optional[float] = None,
                     before_attempt: Optional[Callable[[], None]] = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        if before_attempt is not None:
            before_attempt()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {shlex.join(command)}\n")
            log.flush()
            process = subprocess.Popen(command,
                                       cwd=str(cwd),
                                       env=dict(env),
                                       stdout=log,
                                       stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return_code = -signal.SIGKILL
                log.write(f"Command timed out after {timeout} seconds\n")
        if return_code == 0:
            return
        if attempt == retries:
            raise RuntimeError(
                f"Command failed after {attempt + 1} attempts: {shlex.join(command)}")
        time.sleep(backoff * (2**attempt))
        attempt += 1


def remove_config_options(config: str, names: Set[str]) -> str:
    tokens = shlex.split(config)
    retained = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        option = token.split("=", 1)[0]
        if normalize_option(option) in names:
            index += 1 if "=" in token else 2
            continue
        retained.append(token)
        if "=" not in token:
            retained.append(tokens[index + 1])
            index += 2
        else:
            retained[-1:] = token.split("=", 1)
            index += 1
    return shlex.join(retained)


def prepare_config(config: str) -> Tuple[str, str]:
    options = parse_config(config)
    runtime_flags = []
    current_seq_len = options.get("current_seq_len")
    if current_seq_len is None and options.get("seq_len_q") == "1":
        group = int(options["g"])
        seq_len_k = int(options["seq_len_k"])
        current_seq_len = ",".join([str(seq_len_k - 1)] * group)
    if current_seq_len is not None:
        values = [int(value) for value in current_seq_len.split(",")]
        group = int(options["g"])
        seq_len_k = int(options["seq_len_k"])
        if len(values) != group or any(value < 1 or value >= seq_len_k for value in values):
            raise ValueError("current_seq_len must contain G values in [1, SeqLenK)")
        runtime_flags.append(f"-current_seq_len={current_seq_len}")
    if "prefix_offset" in options:
        runtime_flags.append(f"-prefix_offset={options['prefix_offset']}")
    runner_config = remove_config_options(config, {"current_seq_len", "prefix_offset"})
    return runner_config, " ".join(runtime_flags)


def tuning_command(branch: BranchRun, config: str, runtime_flags: str, gpu: int, args) -> List[str]:
    runner = branch.source / "mlir" / "utils" / "performance" / "tuningRunner.py"
    command = [
        sys.executable,
        str(runner),
        "--op",
        "attention",
        "--config",
        config,
        "--output",
        str(branch.tuning_db),
        "--mlir-build-dir",
        str(branch.build),
        "--tuning-space",
        "quick",
        "--disable-verify-winning-config",
        "--gpus",
        str(gpu),
        "--retry",
        "failed",
        "timed_out",
        "gpu_timed_out",
        "crashed",
        "--perf-config-timeout",
        str(args.perf_config_timeout),
        "--gpu-run-timeout",
        str(args.gpu_run_timeout),
    ]
    if runtime_flags:
        command.append(f"--rocmlir-gen-flags={runtime_flags}")
    return command


def benchmark_command(branch: BranchRun, config: str, runtime_flags: str,
                      temporary_csv: Path) -> List[str]:
    runner = branch.source / "mlir" / "utils" / "performance" / "perfRunner.py"
    command = [
        sys.executable,
        str(runner),
        "--op",
        "attention",
        "--mlir-build-dir",
        str(branch.build),
        "--tuning_db",
        str(branch.tuning_db),
        "-o",
        str(temporary_csv),
    ]
    if runtime_flags:
        command.append(f"--rocmlir_gen_flags={runtime_flags}")
    command.extend(["--", *shlex.split(config)])
    return command


def read_perf_row(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one benchmark row in {path}, found {len(rows)}")
    tflops = float(rows[0]["TFlops"])
    if not math.isfinite(tflops) or tflops <= 0:
        raise RuntimeError(f"Invalid TFlops value in {path}: {rows[0]['TFlops']}")
    return rows[0]


def read_results(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_results(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def branch_environment(gpu: int) -> Dict[str, str]:
    environment = os.environ.copy()
    environment["ROCR_VISIBLE_DEVICES"] = str(gpu)
    environment.pop("HIP_VISIBLE_DEVICES", None)
    return environment


def quick_tune_branch(branch: BranchRun, configs: Sequence[str], gpu: int, args) -> None:
    branch.output_dir.mkdir(parents=True, exist_ok=True)
    environment = branch_environment(gpu)
    for config in configs:
        runner_config, runtime_flags = prepare_config(config)
        run_with_retries(tuning_command(branch, runner_config, runtime_flags, gpu, args),
                         environment,
                         branch.log,
                         args.retries,
                         args.retry_backoff,
                         branch.source,
                         timeout=args.tuning_timeout,
                         before_attempt=lambda: verify_stably_idle(gpu, 1, 0))


def resume_state(branch: BranchRun, arch: str,
                 samples: int) -> Tuple[List[Dict[str, str]], Set[Tuple[str, int]]]:
    rows = read_results(branch.results)
    completed = set()
    for row in rows:
        sample = int(row["Sample"])
        expected_flags = prepare_config(row["Config"])[1]
        if (row["RunLabel"] != branch.label or row["SourceSha"] != branch.sha or
                row["Chip"] != arch or row["RocmlirGenFlags"] != expected_flags or sample < 1 or
                sample > samples):
            raise RuntimeError(f"Stale or inconsistent resumed row in {branch.results}")
        key = (canonical_config(row["Config"]), sample)
        if key in completed:
            raise RuntimeError(f"Duplicate resumed row in {branch.results}: {key}")
        completed.add(key)
    return rows, completed


def benchmark_sample(branch: BranchRun, config: str, sample: int, arch: str, gpu: int, args,
                     rows: List[Dict[str, str]], completed: Set[Tuple[str, int]]) -> None:
    config_id = canonical_config(config)
    if (config_id, sample) in completed:
        return
    runner_config, runtime_flags = prepare_config(config)
    temporary = branch.output_dir / f".perf-{os.getpid()}-{sample}.csv"
    try:
        temporary.unlink(missing_ok=True)
        run_with_retries(benchmark_command(branch, runner_config, runtime_flags, temporary),
                         branch_environment(gpu),
                         branch.log,
                         args.retries,
                         args.retry_backoff,
                         branch.source,
                         timeout=args.benchmark_timeout,
                         before_attempt=lambda: verify_stably_idle(gpu, 1, 0))
        perf_row = read_perf_row(temporary)
        if perf_row["Chip"] != arch:
            raise RuntimeError(f"Benchmark reported {perf_row['Chip']}, expected {arch}")
        rows.append({
            "RunLabel": branch.label,
            "SourceSha": branch.sha,
            "Chip": perf_row["Chip"],
            "Config": config,
            "RocmlirGenFlags": runtime_flags,
            "Sample": str(sample),
            "PerfConfig": perf_row["PerfConfig"],
            "TFlops": perf_row["TFlops"],
        })
        write_results(branch.results, rows)
        completed.add((config_id, sample))
    finally:
        temporary.unlink(missing_ok=True)


def benchmark_paired(base: BranchRun, candidate: BranchRun, configs: Sequence[str], arch: str,
                     gpu: int, args) -> None:
    states = {
        base.label: resume_state(base, arch, args.samples),
        candidate.label: resume_state(candidate, arch, args.samples),
    }
    for config_index, config in enumerate(configs):
        for sample in range(1, args.samples + 1):
            order = [candidate, base] if (config_index + sample) % 2 else [base, candidate]
            for branch in order:
                rows, completed = states[branch.label]
                benchmark_sample(branch, config, sample, arch, gpu, args, rows, completed)


def _load_manifest(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def verify_manifest(previous: Optional[Dict], expected: Mapping) -> None:
    if previous is None:
        return
    keys = (
        "version",
        "arch",
        "gpu",
        "base_sha",
        "candidate_sha",
        "base_source",
        "candidate_source",
        "base_build",
        "candidate_build",
        "base_build_fingerprint",
        "candidate_build_fingerprint",
        "config_file",
        "config_ids",
        "runtime_flags",
        "samples",
        "retries",
        "retry_backoff",
        "perf_config_timeout",
        "gpu_run_timeout",
        "tuning_timeout",
        "benchmark_timeout",
    )
    for key in keys:
        if previous.get(key) != expected.get(key):
            raise RuntimeError(f"Existing run manifest does not match {key}")


def run_workflow(args) -> int:
    base_source = args.base_source.resolve()
    base_build = args.base_build.resolve()
    candidate_source = args.candidate_source.resolve()
    candidate_build = args.candidate_build.resolve()
    if base_source == candidate_source or base_build == candidate_build:
        raise ValueError("Base and candidate require distinct source and build directories")
    base_sha = validate_tree("base", base_source, base_build)
    candidate_sha = validate_tree("candidate", candidate_source, candidate_build)
    if base_sha == candidate_sha:
        raise ValueError("Base and candidate source revisions are identical")

    configs = [config for _, config in read_config_lines(args.configs)]
    if not configs:
        raise ValueError("The filtered config file is empty")

    status, processes = query_gpu_status()
    gpu = select_idle_gpu(args.gpu, status, processes)
    first_gpu = min(status)
    selected_model = (status[gpu]["arch"], status[gpu]["sku"])
    first_model = (status[first_gpu]["arch"], status[first_gpu]["sku"])
    if gpu != first_gpu and selected_model != first_model:
        raise RuntimeError(
            "Selecting a non-first physical GPU requires the same architecture and SKU as the "
            "first GPU because tuningRunner initializes HIP before applying --gpus")
    arch = verify_stably_idle(gpu, args.idle_samples, args.idle_interval)
    output_dir = args.output_dir.resolve()
    base = BranchRun("base", base_source, base_build, base_sha, output_dir / arch / "base")
    candidate = BranchRun("candidate", candidate_source, candidate_build, candidate_sha,
                          output_dir / arch / "candidate")
    manifest_path = output_dir / arch / "run-manifest.json"
    manifest = {
        "version": MANIFEST_VERSION,
        "status": "running",
        "arch": arch,
        "gpu": gpu,
        "base_sha": base_sha,
        "candidate_sha": candidate_sha,
        "base_source": str(base_source),
        "candidate_source": str(candidate_source),
        "base_build": str(base_build),
        "candidate_build": str(candidate_build),
        "base_build_fingerprint": build_fingerprint(base_source, base_build),
        "candidate_build_fingerprint": build_fingerprint(candidate_source, candidate_build),
        "config_file": str(args.configs.resolve()),
        "config_ids": [canonical_config(config) for config in configs],
        "runtime_flags": [prepare_config(config)[1] for config in configs],
        "samples": args.samples,
        "retries": args.retries,
        "retry_backoff": args.retry_backoff,
        "perf_config_timeout": args.perf_config_timeout,
        "gpu_run_timeout": args.gpu_run_timeout,
        "tuning_timeout": args.tuning_timeout,
        "benchmark_timeout": args.benchmark_timeout,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    base_fingerprint = manifest["base_build_fingerprint"]
    candidate_fingerprint = manifest["candidate_build_fingerprint"]
    with gpu_lock(gpu), output_lock(output_dir / arch):
        previous = _load_manifest(manifest_path)
        artifacts = (base.tuning_db, base.results, candidate.tuning_db, candidate.results)
        if previous is None and any(path.exists() for path in artifacts):
            raise RuntimeError("Benchmark artifacts exist without a matching run manifest")
        verify_manifest(previous, manifest)
        if previous is not None:
            manifest["started_at"] = previous["started_at"]
        write_json(manifest_path, manifest)
        verify_stably_idle(gpu, 1, 0)
        validate_branch_unchanged(candidate, candidate_fingerprint)
        quick_tune_branch(candidate, configs, gpu, args)
        validate_branch_unchanged(candidate, candidate_fingerprint)
        validate_branch_unchanged(base, base_fingerprint)
        quick_tune_branch(base, configs, gpu, args)
        validate_branch_unchanged(base, base_fingerprint)
        validate_branch_unchanged(candidate, candidate_fingerprint)
        benchmark_paired(base, candidate, configs, arch, gpu, args)
        validate_branch_unchanged(base, base_fingerprint)
        validate_branch_unchanged(candidate, candidate_fingerprint)
        manifest["status"] = "complete"
        manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(manifest_path, manifest)
    print(f"Completed {arch} benchmark results in {output_dir / arch}")
    return 0


def launch_detached(argv: Sequence[str], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "detached.log"
    command = [sys.executable, str(Path(__file__).resolve()), *remove_flag(argv, "--detach")]
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command,
                                   stdin=subprocess.DEVNULL,
                                   stdout=log,
                                   stderr=subprocess.STDOUT,
                                   start_new_session=True,
                                   close_fds=True)
    atomic_write_text(output_dir / "detached.pid", f"{process.pid}\n")
    print(f"Started PID {process.pid}; log: {log_path}")
    return 0


def positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def nonnegative_integer(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--base-build", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=nonnegative_integer)
    parser.add_argument("--samples", type=positive_integer, default=3)
    parser.add_argument("--retries", type=nonnegative_integer, default=2)
    parser.add_argument("--retry-backoff", type=float, default=5.0)
    parser.add_argument("--perf-config-timeout", type=nonnegative_integer, default=300)
    parser.add_argument("--gpu-run-timeout", type=nonnegative_integer, default=120)
    parser.add_argument("--tuning-timeout", type=positive_integer, default=3600)
    parser.add_argument("--benchmark-timeout", type=positive_integer, default=600)
    parser.add_argument("--idle-samples", type=positive_integer, default=2)
    parser.add_argument("--idle-interval", type=float, default=2.0)
    parser.add_argument("--detach", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_arguments(effective_argv)
    if args.retry_backoff < 0 or args.idle_interval < 0:
        raise ValueError("time intervals must be nonnegative")
    if args.detach:
        return launch_detached(effective_argv, args.output_dir)
    return run_workflow(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
