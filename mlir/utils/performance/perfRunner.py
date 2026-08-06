#!/usr/bin/env python3

import csv
from collections import OrderedDict
import getopt
import os
import subprocess
import signal
import tempfile
import sys
import time
import math
import itertools
from datetime import date
from pathlib import Path
import glob
import argparse
import re

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

import reportUtils
from perfCommonUtils import GEMMLibrary, Operation, SPLITK_IDX

# Hard dependency, copied next to the scripts by ci-performance-scripts.
import amd_arch_db

# Rock treats a WGP as the effective compute unit on architectures that support
# WGP mode. Set this before importing HIP so multiProcessorCount always reports
# WGPs there, even if the caller requested CU mode in its environment.
if os.environ.get("GPU_ENABLE_WGP_MODE") == "0":
    print(
        "WARNING: GPU_ENABLE_WGP_MODE=0 is overridden to 1 because perfRunner "
        "requires WGP mode.",
        file=sys.stderr)
os.environ["GPU_ENABLE_WGP_MODE"] = "1"

from hip import hip  # noqa: E402

# global variables.
# Honor ROCM_PATH so the scripts work with relocatable/SDK ROCm installs
# instead of assuming the system path at /opt/rocm.
ROCM_PATH = os.environ.get('ROCM_PATH', '/opt/rocm')
ROCPROF = f'{ROCM_PATH}/bin/rocprofv3'
MIOPENDRIVER = f'{ROCM_PATH}/bin/MIOpenDriver'
BENCHMARKING_RESULT_FILE_NAME = 'results'
BENCHMARKING_STATS_FILE_NAME = 'results_kernel_stats.csv'
BENCHMARKING_METRICS_FILE_NAME = 'results_counter_collection.csv'
ROCMLIR_INPUT_METRICS_FILE_NAME = 'rocmlir_metrics.txt'
DIRECTIONS = ['-F 1', '-F 2', '-F 4']
LAYOUTS = ['NHWC', 'NCHW']

DATA_TYPES_GEMM = ['f32', 'f16', 'bf16', 'i8', 'fp8', 'f4E2M1FN']
DATA_TYPES_GEMM_SCALES = ['f32', 'f8E8M0FNU']
DATA_TYPES_ATTENTION = ['i8', 'f32', 'f16', 'bf16']
DATA_TYPES_GEMM_GEMM = ['f32', 'f16', 'bf16']
DATA_TYPES_CONV_GEMM = ['f32', 'f16', 'bf16']
# Canonical mapping from a (rocmlir-gen) conv dtype to its MIOpen-style argv[0]
# prefix used in legacy MIOpen config files (see ConvConfiguration.from_command_line).
DATA_TYPES_CONV_TO_MIOPEN = {
    'f32': 'conv',
    'f16': 'convfp16',
    'bf16': 'convbfp16',
    'i8': 'convint8',
    'fp8': 'convfp8',
    'fp8_fp8': 'convfp8',
}
DATA_TYPES_CONV = list(DATA_TYPES_CONV_TO_MIOPEN.keys())
DATA_TYPES_CONV_MIOPEN = sorted(set(DATA_TYPES_CONV_TO_MIOPEN.values()))
OUTPUT_DATA_TYPES_MAP = {
    'f32': 'f32',
    'f16': 'f16',
    'bf16': 'bf16',
    'i8': 'i32',
    'fp8': 'f32',
    'fp8_fp8': 'f32',
    'fp8_bf8': 'f32',
    'bf8_fp8': 'f32',
    'bf8_bf8': 'f32',
    'f4E2M1FN': 'f32'
}
# rocmlir-gen host-harness kernel repeat count (--kernel-repeats, used with -ph).
MLIR_N_REPEATS = 100

# Time budgets (ms) for the tuning-driver benchmark. The number of warmup and
# measured iterations is derived from these budgets and the estimated per-launch
# runtime (Triton do_bench style). These mirror Triton's do_bench defaults.
# tuningRunner imports these so that inference (benchmark) numbers stay consistent
# with tuning numbers for the same perfConfig.
TUNE_WARMUP_MS = 25
TUNE_REP_MS = 100
SLEEP_US = 100  # 0.1 ms

FILTER_LAYOUT_MAP = {'N': 'k', 'C': 'c', 'H': 'y', 'W': 'x', 'G': 'g', '0': '0', '1': '1'}
INPUT_LAYOUT_MAP = {'N': 'n', 'C': 'c', 'H': 'h', 'W': 'w', 'G': 'g', '0': '0', '1': '1'}
OUTPUT_LAYOUT_MAP = {'N': 'n', 'C': 'k', 'H': 'h', 'W': 'w', 'G': 'g', '0': '0', '1': '1'}

# Compiled regexp object used for extracting elapsed time from MIOpenDriver's output
ELAPSED_TIME_RE = re.compile(r"Elapsed: ([0-9\.]*) ms")
# Compiled regexp object used for extracting target chip from arch
GFX_CHIP_RE = re.compile(r"gfx[0-9a-z]+")


def input_layouts(input_layout):
    return "".join(INPUT_LAYOUT_MAP[char] for char in input_layout)


def output_layouts(output_layout):
    return "".join(OUTPUT_LAYOUT_MAP[char] for char in output_layout)


def filter_layouts(filter_layout):
    return "".join(FILTER_LAYOUT_MAP[char] for char in filter_layout)


def inverse_output_layouts(output_layout):
    map = {v: k for k, v in OUTPUT_LAYOUT_MAP.items()}
    return "".join(map[char] for char in output_layout)


def inverse_input_layouts(input_layout):
    map = {v: k for k, v in INPUT_LAYOUT_MAP.items()}
    return "".join(map[char] for char in input_layout)


def inverse_filter_layouts(filter_layout):
    map = {v: k for k, v in FILTER_LAYOUT_MAP.items()}
    return "".join(map[char] for char in filter_layout)


@dataclass
class MLIRPaths:
    rocmlir_gen_path: str
    rocmlir_driver_path: str
    rocmlir_opt_path: str
    rocm_run_path: str
    rocmlir_tuning_driver_path: str
    ck_gemm_benchmark_driver_path: Optional[str] = None
    hipblaslt_benchmark_driver_path: Optional[str] = None


@dataclass
class Paths:
    """This structure is used to hold paths needed to perform the tests"""
    configuration_file_path: str
    mlir_paths: Optional[MLIRPaths] = None


def find_mlir_build_dir() -> str:
    """
    Finds mlir build dir searching either WORKSPACE dir
    or home dir
    """
    rocmlir_gen_path = None
    candidate_paths = [
        # if the script is run from build dir
        Path('./bin/rocmlir-gen'),
        # if the script is run from source
        Path(__file__).parent.parent.parent.parent / 'build' / 'bin' / 'rocmlir-gen'
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            rocmlir_gen_path = candidate_path

    if not rocmlir_gen_path:
        try:
            # Prioritize the search in the current repo first.
            search_root = str(
                subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
        except subprocess.CalledProcessError:
            # Else look in the home or WORKSPACE directory
            search_root = os.environ.get('WORKSPACE', str(Path.home()))
            assert search_root, "Cant find WORKSPACE env arg or home directory"

        rocmlir_gen_path = glob.glob(search_root + '/**/bin/rocmlir-gen', recursive=True)
        if len(rocmlir_gen_path) != 1:
            # rocmlir_gen not available or ambiguous
            return None
        rocmlir_gen_path = rocmlir_gen_path[0]

    build_dir = Path(rocmlir_gen_path).parent.parent
    return str(build_dir)


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def iter_device_props():
    for device in range(hip_check(hip.hipGetDeviceCount())):
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, device))
        yield props


def get_arch() -> str:
    agents = set()
    for props in iter_device_props():
        agent = props.gcnArchName.decode('utf-8')
        agents.add(agent)
    if (len(agents) > 1):
        print(
            f"WARNING: Found {len(agents)} different kinds of agents on the same machine :  {', '.join(agents)}"
        )
        print(
            "WARNING: Using the first agent by default. If you want to use a different agent, please set the HIP_VISIBLE_DEVICES environment variable."
        )
    # select first agent by default
    return list(agents)[0]


def get_chip():
    arch = get_arch()
    chip = GFX_CHIP_RE.search(arch).group(0)
    return chip


def create_paths(config_file_path, mlir_build_dir_path) -> Paths:
    """Creates the composite Paths structure using build dir paths"""

    mlir_paths = None
    if mlir_build_dir_path:
        mlir_bin_dir_path = (Path(mlir_build_dir_path) / 'bin').resolve()
        mlir_bin_dir = str(mlir_bin_dir_path)
        ck_gemm_benchmark_driver_location = mlir_bin_dir_path / 'ck-gemm-benchmark-driver'
        hipblaslt_benchmark_driver_location = mlir_bin_dir_path / 'hipblaslt-benchmark-driver'

        mlir_paths = MLIRPaths(
            rocmlir_gen_path=mlir_bin_dir + '/rocmlir-gen',
            rocmlir_driver_path=mlir_bin_dir + '/rocmlir-driver',
            rocmlir_opt_path=mlir_bin_dir + '/rocmlir-opt',
            rocm_run_path=mlir_bin_dir + '/rocm-run',
            rocmlir_tuning_driver_path=mlir_bin_dir + '/rocmlir-tuning-driver',
            ck_gemm_benchmark_driver_path=(str(ck_gemm_benchmark_driver_location)
                                           if ck_gemm_benchmark_driver_location.exists() else None),
            hipblaslt_benchmark_driver_path=(str(hipblaslt_benchmark_driver_location)
                                             if hipblaslt_benchmark_driver_location.exists() else
                                             None))

    return Paths(config_file_path, mlir_paths)


# utility functions.
def get_nanoseconds(filename):
    if not os.path.exists(filename):
        return np.nan
    with open(filename, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        result = 0
        for row in reader:
            result += int(float(row['AverageNs']))
        csv_file.close()
        return result


# Architectures where rocprof hardware-counter collection (``-i <metrics>``) is
# skipped. gfx950's counters are unsupported; on gfx1170 the counter-collection
# path wedges rocprofv3 in an uninterruptible state (the metrics we request,
# e.g. LDSBankConflict, are not yet wired up for this arch). These metrics are
# diagnostic-only -- benchmark timing comes from ``--kernel-trace --stats`` -- so
# skipping them keeps benchmarking working without the (optional) bank-conflict
# stats.
ROCPROF_METRICS_UNSUPPORTED_CHIPS = ["gfx950", "gfx1170"]


def get_profiler_output_path(arch: str, base_out_path):
    chip = GFX_CHIP_RE.search(arch).group(0)
    # rocprof only emits the ``pmc_1/`` subdirectory when a hardware-counter
    # (pmc) pass runs, i.e. when metrics collection is enabled. Arches that skip
    # metrics (see ROCPROF_METRICS_UNSUPPORTED_CHIPS) write the stats file
    # directly, so must not look under ``pmc_1/``.
    if (chip not in ROCPROF_METRICS_UNSUPPORTED_CHIPS):
        return os.path.join('pmc_1', base_out_path)
    return base_out_path


def get_metric_args_for_rocprof(arch: str):
    chip = GFX_CHIP_RE.search(arch).group(0)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(current_dir, ROCMLIR_INPUT_METRICS_FILE_NAME)
    metrics = []
    if (chip not in ROCPROF_METRICS_UNSUPPORTED_CHIPS):
        metrics = ['-i', metrics_path]
    return metrics


# Bank conflict functions.The percentage of GPUTime LDS is stalled by bank
# conflicts. Value range: 0% (optimal) to 100% (bad).
def get_bank_conflict(filename):
    if not os.path.exists(filename):
        result = "NaN"
        return result
    with open(filename, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        header = reader.fieldnames
        if 'Counter_Name' not in header or 'Counter_Value' not in header:
            return np.nan

        result = []
        for row in reader:
            if row['Counter_Name'] == 'LDSBankConflict':
                result.append(float(row['Counter_Value']))
        csv_file.close()
        result_average = sum(result) / len(result)
        return result_average


# Tuning databases
MaybeTuningDb = Optional[Dict[Tuple[str, str], str]]


def parse_tuning_db_line(entries: list) -> Optional[Tuple[str, str, str]]:
    """
    Parse a tuning database line and return (arch, config, perfconfig) tuple.
    Returns None if the line format is not recognized.

    Supported formats:
    - Legacy (3 entries): arch, config, perfconfig
    - v2 (4+ entries): arch, num_cu, config, perfconfig, [tflops, ...]
    - v3 (5+ entries): arch, num_cu, num_chiplets, config, perfconfig, [tflops, ...]
    """
    n = len(entries)

    if n == 3:
        # Legacy: arch, config, perfconfig
        return tuple(entries)

    if n >= 5 and entries[2].isdigit():
        # v3: arch, num_cu, num_chiplets, config, perfconfig, [optional...]
        arch, _num_cu, _num_chiplets, config, perfconfig = entries[:5]
        return (arch, config, perfconfig)

    if n >= 4:
        # v2: arch, num_cu, config, perfconfig, [optional...]
        arch, _num_cu, config, perfconfig = entries[:4]
        return (arch, config, perfconfig)

    return None


def read_tuning_db(path: Optional[str]) -> MaybeTuningDb:
    try:
        ret = {}
        with open(path, 'r') as db_file:
            for line in db_file:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                entries = line.split('\t')

                parsed = parse_tuning_db_line(entries)
                if parsed is None:
                    print(f"Warning: Malformed tuning database entry: {line}")
                    continue

                arch, config, perfconfig = parsed
                ret[arch, config] = perfconfig
        return ret
    except FileNotFoundError:
        if path:
            print(f"Warning: Failed to find tuning database: {path}")
        return None


def get_miliseconds(output):
    result = re.search(r"kernel time: (.*)", output.decode("utf-8"))
    if not result:
        return float('NaN')

    return float(result.group(1))


def _kill_proc(proc):
    """Best-effort terminate a subprocess and reap it."""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
    except Exception:
        pass


def run_command_pipeline(commands, *, env=None, cwd=None, timeout=None):
    """Run a shell-style pipeline: commands[0] | commands[1] | ... | commands[-1].

    This is the single implementation of pipeline spawning shared across the
    perf tooling; callers that want a different return shape (e.g. run_pipeline's
    (stdout, ok) tuple, or decoded strings) wrap this.

    ``timeout`` is one shared budget for spawning and waiting for the complete
    pipeline, not a fresh allowance for each stage.

    All pipes are drained so no stage can deadlock on a full OS pipe buffer:
      * Each stage's stderr goes to its own temp file, so a chatty stage can
        never block writing stderr into a PIPE that is not read until after
        ``wait()`` (the classic pipeline deadlock -- e.g. rocprof emitting many
        KiB of trace/stats output, or the tuning driver's benchmark logs).
      * Every stage still uses ``stdout=PIPE`` to wire ``stage[i]`` into
        ``stage[i+1]``, but the parent closes each intermediate read-end right
        after wiring it, so only the final stage's stdout is read by the parent
        -- drained with ``communicate()`` (which reads concurrently), letting
        the whole chain make progress before any stage is reaped.
      * Because each intermediate read-end is closed in the parent, upstream
        stages observe EOF/SIGPIPE and exit.
    """
    if not commands:
        raise ValueError("Pipeline must contain at least one command")

    procs = []
    prev_stdout = None
    stderr_files = []
    deadline = time.monotonic() + timeout if timeout is not None else None

    def remaining_timeout():
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(commands, timeout)
        return remaining

    try:
        for cmd in commands:
            errf = tempfile.TemporaryFile()
            stderr_files.append(errf)
            proc = subprocess.Popen(
                cmd,
                stdin=prev_stdout if prev_stdout is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=errf,
                env=env,
                cwd=cwd)
            if prev_stdout is not None:
                prev_stdout.close()
            procs.append(proc)
            prev_stdout = proc.stdout

        out, _ = procs[-1].communicate(timeout=remaining_timeout())
        for proc in procs[:-1]:
            try:
                proc.wait(timeout=remaining_timeout())
            except subprocess.TimeoutExpired:
                _kill_proc(proc)
                raise

        rc = 0
        failing_idx = None
        for idx, proc in enumerate(procs):
            # A non-final stage killed by SIGPIPE (-13) just means a downstream
            # stage exited first; the real failure (if any) is reported
            # downstream. The final stage has no downstream reader (its stdout
            # is drained here via communicate()), so a SIGPIPE there is a real
            # failure and must not be swallowed.
            allow_sigpipe = idx < len(procs) - 1 and proc.returncode == -signal.SIGPIPE
            if proc.returncode != 0 and not allow_sigpipe:
                rc = proc.returncode
                failing_idx = idx
                break

        err_idx = failing_idx if failing_idx is not None else len(stderr_files) - 1
        stderr_files[err_idx].seek(0)
        err = stderr_files[err_idx].read()
        return rc, out, err
    finally:
        for proc in procs:
            _kill_proc(proc)
        for errf in stderr_files:
            errf.close()


def run_pipeline(proc_specs):
    """Run a pipeline and return (final_stdout_bytes, ok).

    Thin back-compat wrapper over run_command_pipeline for callers that only
    need the final stdout and a success flag.
    """
    pipeline_str = " | ".join(" ".join(spec) for spec in proc_specs)
    try:
        rc, outs, errs = run_command_pipeline(proc_specs)
    except Exception as err:
        # Spawning or driving the pipeline failed (e.g. a missing executable,
        # or an error while draining). Callers key off the returned bool rather
        # than catching exceptions, so report context and fail gracefully
        # instead of crashing the whole runner.
        print(f"Error:  {err}")
        print(f"Failing pipeline:  {pipeline_str}")
        return b"", False
    if rc != 0:
        print(f"Error:  {errs.decode('utf-8', 'replace')}")
        print(f"Failing pipeline:  {pipeline_str}")
        return outs, False
    return outs, True


class PerfConfiguration:
    TABLE_COLUMNS = []

    def compute_tflops(self, ns: int) -> float:
        raise NotImplementedError()

    def table_entry(self, nanoseconds):
        raise NotImplementedError()

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags):
        raise NotImplementedError()

    def set_perfconfig(self, perf_config):
        raise NotImplementedError()

    @classmethod
    def from_command_line(cls, argv, arch, num_cu, num_chiplets):
        raise NotImplementedError()

    def to_command_line(self):
        raise NotImplementedError()

    @classmethod
    def benchmark_external(cls, commandline, paths: Paths, arch, num_cu, num_chiplets):
        raise NotImplementedError()

    EXTERNAL_NAME = "unknown"

    def __repr__(self):
        attrs = ', '.join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


# convolution configurations.
def get_conv_configurations(filename, target_chip: Optional[str] = None):
    configs = []
    chip = target_chip
    if filename:
        with open(filename, 'r') as config_file:
            lines = config_file.readlines()
            # All combinations of conv direction, type and layouts
            for direction, datatype, layout, line in \
                    itertools.product(DIRECTIONS, DATA_TYPES_CONV_MIOPEN, LAYOUTS, lines):
                line = line.strip()

                # Skip empty lines
                if len(line) == 0 or line[0] == '#':
                    continue

                # Skip unsupported datatypes
                if datatype == 'convfp8':
                    unsupported_chips = {'gfx908', 'gfx90a', 'gfx942', 'gfx1030', 'gfx1101'}
                    if chip is None:
                        chip = get_chip()
                    if chip in unsupported_chips:
                        continue

                # Skip int8 non-fwd convolutions
                if (datatype == 'convint8' or datatype == 'convfp8') and direction != '-F 1':
                    continue

                # Skip datatype if already in
                datatype = f"{datatype} "
                # check for the presense of a positional arg
                if line[0][0] != "-":
                    datatype = ""

                # Skip direction if already in
                direction = f"{direction} "
                if "-F" in line:
                    direction = ""

                # Skip filter layout if already in
                filter_layout = f"-f {layout} "
                if "-f" in line:
                    filter_layout = ""

                # Skip input layout if already in
                input_layout = f"-I {layout} "
                if "-I" in line:
                    input_layout = ""

                # Skip output layout if already in
                output_layout = f"-O {layout} "
                if "-O" in line:
                    output_layout = ""

                one_config = f"{datatype}{direction}{filter_layout}{input_layout}{output_layout}{line}"
                if one_config not in configs:
                    configs.append(one_config)
    return configs


class ConvConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.CONV_TEST_PARAMETERS + ['LDSBankConflict'] + ['TFlops']
    EXTERNAL_NAME = "MIOpen"
    SWEEP_KIND = "conv"

    def compute_tflops(self, ns):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        assert (self.k % self.group == 0)
        assert (self.c % self.group == 0)
        return (2.0 * self.n * (self.c // self.group) * self.k * self.ho * self.wo * self.y *
                self.x) / (float(ns) * 1e-9) / 1e12

    def table_entry(self, nanoseconds):
        # Future(kdrewnia): This can just be a dict literal on Python 3.7+
        bank_conflict = get_bank_conflict(
            get_profiler_output_path(self.arch, BENCHMARKING_METRICS_FILE_NAME))
        result = OrderedDict()
        values = [
            self.direction, self.datatype, self.chip, self.num_cu, self.num_chiplets,
            self.filter_layout, self.input_layout, self.output_layout, self.n, self.c, self.hi,
            self.wi, self.k, self.y, self.x, self.dilation_h, self.dilation_w, self.conv_stride_h,
            self.conv_stride_w, self.padding_hl, self.padding_wl, self.perfconfig, bank_conflict,
            self.compute_tflops(nanoseconds)
        ]
        assert (len(self.TABLE_COLUMNS) == len(values))

        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def set_perfconfig(self, perf_config):
        self.perfconfig = perf_config

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags, kernel_repeats=MLIR_N_REPEATS):
        direction = {
            'fwd': '--operation conv',
            'bwd': '--operation conv_bwd_data',
            'wrw': '--operation conv_bwd_weight'
        }[self.direction]

        # Pick symmetric or asymmetric padding cmdline form per axis.
        # rocmlir-gen errors out when --padding_h and --padding_h_l/_h_r
        # disagree on the same axis (validatePadding in rocmlir-gen.cpp).
        padding_args = []
        if self.padding_hl == self.padding_hr:
            padding_args += ['--padding_h', str(self.padding_hl)]
        else:
            padding_args += [
                '--padding_h_l',
                str(self.padding_hl), '--padding_h_r',
                str(self.padding_hr)
            ]
        if self.padding_wl == self.padding_wr:
            padding_args += ['--padding_w', str(self.padding_wl)]
        else:
            padding_args += [
                '--padding_w_l',
                str(self.padding_wl), '--padding_w_r',
                str(self.padding_wr)
            ]

        result = ' '.join([
            direction, '-t', self.datatype, '--arch', self.arch, '--num_cu',
            str(self.num_cu), '--num_chiplets',
            str(self.num_chiplets), '--fil_layout', self.filter_layout, '--in_layout',
            self.input_layout, '--out_layout', self.output_layout, '--batchsize',
            str(self.n), '--in_channels',
            str(self.c), '--in_h',
            str(self.hi), '--in_w',
            str(self.wi), '--out_channels',
            str(self.k), '--fil_h',
            str(self.y), '--fil_w',
            str(self.x), '--dilation_h',
            str(self.dilation_h), '--dilation_w',
            str(self.dilation_w), '--conv_stride_h',
            str(self.conv_stride_h), '--conv_stride_w',
            str(self.conv_stride_w), *padding_args, '--groupsize',
            str(self.group),
            *(['--kernel-repeats', str(kernel_repeats)] if kernel_repeats is not None else []),
            f"--perf_config={self.perfconfig}"
        ])
        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    @classmethod
    def from_command_line(cls, argv, arch, num_cu, num_chiplets):
        # determine datatype from argv[1]
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        if argv[0] == 'conv':
            datatype = 'f32'
        elif argv[0] == 'convfp16':
            datatype = 'f16'
        elif argv[0] == 'convbfp16':
            datatype = 'bf16'
        elif argv[0] == 'convint8':
            datatype = 'i8'
        elif argv[0] == 'convfp8_fp8':
            datatype = 'fp8_fp8'
        elif argv[0] == 'convfp8':
            datatype = 'fp8'
        elif argv[0] == 'convfp8_bf8':
            datatype = 'fp8_bf8'
        elif argv[0] == 'convbf8_fp8':
            datatype = 'bf8_fp8'
        elif argv[0] == 'convbf8_bf8':
            datatype = 'bf8_bf8'

        try:
            # TBD:
            # implement -m ?
            # implement -t ?
            opts, _ = getopt.getopt(argv[1:], "F:f:I:O:n:c:H:W:k:y:x:p:q:l:j:u:v:g:m:t:")
        except getopt.GetoptError:
            print('getopt error')
            sys.exit(1)

        for opt, arg in opts:
            if opt == '-F':
                # -F
                # 1 fwd only
                # 2 bwd only
                # 4 wrw only
                # TBD:
                # 0 fwd+bwd+wrw
                # 3 fwd+bwd
                # 5 fwd+wrw
                # 6 bwd+wrw
                if int(arg) == 1:
                    direction = 'fwd'
                elif int(arg) == 2:
                    direction = 'bwd'
                elif int(arg) == 4:
                    direction = 'wrw'
            elif opt == '-f':
                filter_layout = arg
            elif opt == '-I':
                input_layout = arg
            elif opt == '-O':
                output_layout = arg
            elif opt == "-n":
                n = int(arg)
            elif opt == '-c':
                c = int(arg)
            elif opt == '-H':
                hi = int(arg)
            elif opt == '-W':
                wi = int(arg)
            elif opt == '-k':
                k = int(arg)
            elif opt == '-y':
                y = int(arg)
            elif opt == '-x':
                x = int(arg)
            elif opt == '-u':
                conv_stride_h = int(arg)
            elif opt == '-v':
                conv_stride_w = int(arg)
            elif opt == '-p':
                padding_h = int(arg)
            elif opt == '-q':
                padding_w = int(arg)
            elif opt == '-l':
                dilation_h = int(arg)
            elif opt == '-j':
                dilation_w = int(arg)
            elif opt == '-g':
                group = int(arg)
            else:
                continue

        # MIOpen only supports symmetric padding; replicate to both sides.
        return cls(datatype, direction, filter_layout, input_layout, output_layout, n, c, hi, wi, k,
                   y, x, conv_stride_h, conv_stride_w, padding_h, padding_h, padding_w, padding_w,
                   dilation_h, dilation_w, group, arch, num_cu, num_chiplets)

    def to_command_line(self):
        return (
            f"conv{dict(f32='', f16='fp16', bf16='bfp16', i8='int8', fp8_fp8='fp8_fp8', fp8='fp8')[self.datatype]} "
            + f"-F {dict(fwd=1, bwd=2, wrw=4)[self.direction]} " +
            f"-f {inverse_filter_layouts(self.filter_layout)} -I {self.input_layout.upper()} " +
            f"-O {inverse_output_layouts(self.output_layout)} " +
            f"-n {self.n} -c {self.c} -H {self.hi} -W {self.wi} -k {self.k} " +
            # MIOpen only supports symmetric padding; use the left side as
            # the single value (asymmetric configs are produced by the sweep
            # script, not benchmarked against MIOpen).
            f"-y {self.y} -x {self.x} -p {self.padding_hl} -q {self.padding_wl} " +
            f"-u {self.conv_stride_h} -v {self.conv_stride_w} -l {self.dilation_h} " +
            f"-j {self.dilation_w} -m conv -g {self.group} -t 1")

    def __init__(self,
                 dtype: str,
                 direction: str,
                 filter_layout: str,
                 input_layout: str,
                 output_layout: str,
                 n: int,
                 c: int,
                 hi: int,
                 wi: int,
                 k: int,
                 y: int,
                 x: int,
                 conv_stride_h: int,
                 conv_stride_w: int,
                 padding_hl: int,
                 padding_hr: int,
                 padding_wl: int,
                 padding_wr: int,
                 dilation_h: int,
                 dilation_w: int,
                 group: int,
                 arch: str,
                 num_cu: int,
                 num_chiplets: int,
                 perf_config: str = ''):
        if dtype not in DATA_TYPES_CONV:
            raise ValueError(f"Invalid datatype: {dtype}")
        if direction not in {"fwd", "bwd", "wrw"}:
            raise ValueError(f"Invalid direction: {direction}")

        self.datatype = dtype
        self.direction = direction

        self.filter_layout = filter_layouts(filter_layout)
        self.input_layout = input_layouts(input_layout)
        self.output_layout = output_layouts(output_layout)

        self.n = n
        self.c = c
        self.hi = hi
        self.wi = wi
        self.k = k
        self.y = y
        self.x = x

        self.conv_stride_h = conv_stride_h
        self.conv_stride_w = conv_stride_w
        # Per-side padding (rocmlir-gen --padding_h_l / --padding_h_r etc.).
        # Symmetric callers (e.g. MIOpen-style ``from_command_line``) pass
        # the same value for both sides.
        self.padding_hl = padding_hl
        self.padding_hr = padding_hr
        self.padding_wl = padding_wl
        self.padding_wr = padding_wr
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w

        self.group = group
        self.arch = arch
        self.num_cu = num_cu
        self.num_chiplets = num_chiplets
        self.chip = GFX_CHIP_RE.search(arch).group(0)

        # Per-side padding gives the correct formula in both symmetric
        # (hl + hr == 2 * padding_h) and asymmetric cases.
        self.ho = math.floor((self.hi + self.padding_hl + self.padding_hr -
                              (self.y - 1) * self.dilation_h - 1) / self.conv_stride_h) + 1
        self.wo = math.floor((self.wi + self.padding_wl + self.padding_wr -
                              (self.x - 1) * self.dilation_w - 1) / self.conv_stride_w) + 1

        self.perfconfig = perf_config

    @classmethod
    def benchmark_external(cls, commandline, paths: Paths, arch, num_cu, num_chiplets):
        if os.path.exists(get_profiler_output_path(arch, BENCHMARKING_METRICS_FILE_NAME)):
            os.remove(get_profiler_output_path(arch, BENCHMARKING_METRICS_FILE_NAME))
        config = cls.from_command_line(commandline, arch, num_cu, num_chiplets)
        miopen_driver_cmd = [MIOPENDRIVER, *commandline, '-V', '0', '-t', '1']
        print("Running MIOpen Benchmark: ", ' '.join(commandline))
        # invoke MIOpenDriver.
        outs, noerr = run_pipeline([miopen_driver_cmd])
        nanoseconds = np.nan
        if noerr:
            # convert bytes to str
            outs = outs.decode('utf-8')
            # Extract Elapsed time in ms from the output of MIOpenDriver
            # Use regular expression to match the contents between
            # "Elasped: " (note the space at the end) and "ms"
            elapsed_time_in_ms = ELAPSED_TIME_RE.search(outs).group(1)
            nanoseconds = float(elapsed_time_in_ms) * 1.0e6

        return config.table_entry(nanoseconds)


def get_gemm_configurations(filename,
                            datatypes=DATA_TYPES_GEMM,
                            out_dtype_map=OUTPUT_DATA_TYPES_MAP,
                            scale_types=DATA_TYPES_GEMM_SCALES,
                            target_chip: Optional[str] = None):
    configs = []
    chip = target_chip

    if filename:
        with open(filename, 'r') as config_file:
            lines = config_file.readlines()

            # All combinations of types and transposition (A, B and O)
            for datatype, trans_a, trans_b, trans_o, line in \
                    itertools.product(DATA_TYPES_GEMM, ['false', 'true'], ['false', 'true'], ['false', 'true'], lines):
                line = line.strip()

                # Skip empty lines
                if len(line) == 0 or line[0] == '#':
                    continue
                if datatype not in datatypes:
                    continue

                # Skip unsupported datatypes
                if datatype == 'f4E2M1FN':
                    # TODO: use information from AMDArchDB when it becomes available to determine supported chips
                    supported_chips = {'gfx950'}
                    if chip is None:
                        chip = get_chip()
                    if chip not in supported_chips:
                        continue

                if datatype == 'fp8':
                    unsupported_chips = {'gfx908', 'gfx90a', 'gfx942', 'gfx1030', 'gfx1101'}
                    if chip is None:
                        chip = get_chip()
                    if chip in unsupported_chips:
                        continue

                # We need trailing spaces here to account for the concat below
                # Skip type if already in
                datatype_string = ""
                if "-t " not in line:
                    datatype_string = f"-t {datatype} "

                # Skip trans_a if already in
                trans_a_string = ""
                if "-transA " not in line:
                    trans_a_string = f"-transA {trans_a} "

                # Skip trans_b if already in
                trans_b_string = ""
                if "-transB " not in line:
                    trans_b_string = f"-transB {trans_b} "

                # Skip trans_o if already in
                trans_o_string = ""
                if "-transO " not in line:
                    trans_o_string = f"-transO {trans_o} "

                # Skip out_datatype if already in
                out_dtype_string = ""
                if "-out_datatype" not in line:
                    out_dtype_string = "-out_datatype " + out_dtype_map.get(datatype,
                                                                            datatype) + " "

                # Handle scale types for scaled GEMM
                is_scaled_gemm = "-scaledGemm" in line
                if is_scaled_gemm:
                    # Generate all combinations of scale types for scaled GEMM
                    for scale_a_dtype, scale_b_dtype in itertools.product(scale_types, scale_types):
                        # Skip if scale types are already specified in the line
                        scale_a_string = ""
                        scale_b_string = ""
                        if "-scale_a_dtype" not in line:
                            scale_a_string = f"-scale_a_dtype {scale_a_dtype} "
                        if "-scale_b_dtype" not in line:
                            scale_b_string = f"-scale_b_dtype {scale_b_dtype} "

                        # Strip to avoid spurious spaces
                        one_config = f"{datatype_string}{out_dtype_string}{trans_a_string}{trans_b_string}{trans_o_string}{scale_a_string}{scale_b_string}{line}".strip(
                        )
                        if one_config not in configs:
                            configs.append(one_config)
                else:
                    # Strip to avoid spurious spaces
                    one_config = f"{datatype_string}{out_dtype_string}{trans_a_string}{trans_b_string}{trans_o_string}{line}".strip(
                    )
                    if one_config not in configs:
                        configs.append(one_config)
    return configs


def get_conv_gemm_configurations(filename):
    bool_space = ['false', 'true']
    default_test_space = {
        "-t": DATA_TYPES_CONV_GEMM,
        "-f": LAYOUTS,
        "-I": LAYOUTS,
        "-transC": bool_space,
        "-transO": bool_space,
    }
    configs = []
    if filename:
        with open(filename, 'r') as config_file:
            lines = config_file.readlines()
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if len(line) == 0 or line[0] == '#':
                    continue
                test_space = []
                args = []
                for arg in default_test_space.keys():
                    """
                    Next condition checks if a flag is not present in the line. Check with re.search(...)
                    ensures flags are matched exactly and not as substring.

                    - (?<!\S) ensures that flag is not part of another token (e.g. that -t is not part of -transQ)
                    - (?!\S) ensures that flag is followed by a space or line end.
                    - re.escape(arg) ensures that flag, in case it contains special character(s), is matched as it is.
                    """
                    if not re.search(rf"(?<!\S){re.escape(arg)}(?!\S)", line):
                        test_space.append(default_test_space[arg])
                        args.append(arg)
                for test_vector in itertools.product(*test_space):
                    # Strip to avoid spurious spaces
                    one_config = line.strip()
                    for arg, value in zip(args, test_vector):
                        one_config = f"{arg} {value} {one_config}"
                    if one_config not in configs:
                        configs.append(one_config)
    return configs


def get_gemm_gemm_configurations(filename):
    bool_space = ['false', 'true']
    default_test_space = {
        "-t": DATA_TYPES_GEMM_GEMM,
        "-transA": bool_space,
        "-transB": bool_space,
        "-transC": bool_space,
        "-transO": bool_space,
    }
    configs = []
    if filename:
        with open(filename, 'r') as config_file:
            lines = config_file.readlines()
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if len(line) == 0 or line[0] == '#':
                    continue
                test_space = []
                args = []
                for arg in default_test_space.keys():
                    """
                    Next condition checks if a flag is not present in the line. Check with re.search(...)
                    ensures flags are matched exactly and not as substring.

                    - (?<!\S) ensures that flag is not part of another token (e.g. that -t is not part of -transQ)
                    - (?!\S) ensures that flag is followed by a space or line end.
                    - re.escape(arg) ensures that flag, in case it contains special character(s), is matched as it is.
                    """
                    if not re.search(rf"(?<!\S){re.escape(arg)}(?!\S)", line):
                        test_space.append(default_test_space[arg])
                        args.append(arg)
                for test_vector in itertools.product(*test_space):
                    # Strip to avoid spurious spaces
                    one_config = line.strip()
                    for arg, value in zip(args, test_vector):
                        one_config = f"{arg} {value} {one_config}"
                    if one_config not in configs:
                        configs.append(one_config)
    return configs


def get_attn_configurations(filename):
    bool_space = ['false', 'true']
    # if not defined, set it to false
    default_to_false = ['false']
    default_test_space = {
        "-t": DATA_TYPES_ATTENTION,
        "-transQ": bool_space,
        "-transK": bool_space,
        "-transV": bool_space,
        "-transO": bool_space,
        "-causal": default_to_false,
        "-return_lse": default_to_false,
        "-with-attn-scale": default_to_false,
        "-with-attn-bias": default_to_false,
        "-transBias": default_to_false
    }

    configs = []
    if filename:
        with open(filename, 'r') as config_file:
            lines = config_file.readlines()
            for line in lines:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue

                test_space = []
                args = []
                for arg in default_test_space.keys():
                    """
                    Next condition checks if a flag is not present in the line. Check with re.search(...)
                    ensures flags are matched exactly and not as substring.

                    - (?<!\S) ensures that flag is not part of another token (e.g. that -t is not part of -transQ)
                    - (?!\S) ensures that flag is followed by a space or line end.
                    - re.escape(arg) ensures that flag, in case it contains special character(s), is matched as it is.
                    """
                    if not re.search(rf"(?<!\S){re.escape(arg)}(?!\S)", line):
                        test_space.append(default_test_space[arg])
                        args.append(arg)

                for test_vector in itertools.product(*test_space):
                    # Strip to avoid spurious spaces
                    one_config = line.strip()
                    for arg, value in zip(args, test_vector):
                        one_config = f"{arg} {value} {one_config}"

                    # Check for valid dtypes
                    found_dtype = re.search(r"-t\s+(\w+)", one_config)
                    if not found_dtype or found_dtype.group(1) not in DATA_TYPES_ATTENTION:
                        continue

                    if one_config not in configs:
                        configs.append(one_config)

    return configs


class GemmConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.GEMM_TEST_PARAMETERS + ['LDSBankConflict'] + ['TFlops']
    SWEEP_KIND = "gemm"

    def compute_tflops(self, ns):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        return (2.0 * self.g * self.m * self.k * self.n) / (float(ns) * 1e-9) / 1e12

    def table_entry(self, nanoseconds):
        # Future(kdrewnia): This can just be a dict literal on Python 3.7+
        bank_conflict = get_bank_conflict(
            get_profiler_output_path(self.arch, BENCHMARKING_METRICS_FILE_NAME))
        result = OrderedDict()
        values = [
            self.datatype, self.out_dtype, self.chip, self.num_cu, self.num_chiplets, self.trans_a,
            self.trans_b, self.trans_o, self.g, self.m, self.k, self.n, self.scaled_gemm,
            self.scale_a_dtype, self.scale_b_dtype, self.trans_scale_a, self.trans_scale_b,
            self.perfconfig, bank_conflict,
            self.compute_tflops(nanoseconds)
        ]
        assert (len(self.TABLE_COLUMNS) == len(values))

        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def set_perfconfig(self, perf_config):
        self.perfconfig = perf_config

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags, kernel_repeats=MLIR_N_REPEATS):
        result = ' '.join([
            '-operation', 'gemm', '-t', self.datatype, '-out_datatype', self.out_dtype, '--arch',
            self.arch, '--num_cu',
            str(self.num_cu), '--num_chiplets',
            str(self.num_chiplets), '-g',
            str(self.g), '-m',
            str(self.m), '-k',
            str(self.k), '-n',
            str(self.n), f"-transA={self.trans_a}", f"-transB={self.trans_b}",
            f"-transO={self.trans_o}",
            *(['--kernel-repeats', str(kernel_repeats)] if kernel_repeats is not None else []),
            f"--perf_config={self.perfconfig}"
        ])

        if self.scaled_gemm:
            result += ' -scaledGemm'
        if self.scale_a_dtype:
            result += f' -scale_a_dtype {self.scale_a_dtype}'
        if self.scale_b_dtype:
            result += f' -scale_b_dtype {self.scale_b_dtype}'
        if self.trans_scale_a:
            result += f' -transScaleA {str(self.trans_scale_a)}'
        if self.trans_scale_b:
            result += f' -transScaleB {str(self.trans_scale_b)}'

        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    @classmethod
    def from_command_line(cls, argv, arch, num_cu, num_chiplets):
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        dtype = None
        g = None
        m = None
        k = None
        n = None
        trans_a = None
        trans_b = None
        trans_o = False
        out_dtype = None
        perf_config = ''
        scaled_gemm = False
        scale_a_dtype = None
        scale_b_dtype = None
        trans_scale_a = False
        trans_scale_b = False
        i = 0
        while i < len(argv):
            opt = argv[i]
            # Handle flags without values
            if opt == '-scaledGemm':
                scaled_gemm = True
                i += 1
                continue
            # Handle flags with values
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for argument {opt}")
            val = argv[i + 1]
            if opt == '-t':
                dtype = val
            elif opt == '-g':
                g = int(val)
            elif opt == '-m':
                m = int(val)
            elif opt == '-k':
                k = int(val)
            elif opt == '-n':
                n = int(val)
            elif opt.endswith("-transA"):
                trans_a = (val.lower() in ["1", "true"])
            elif opt.endswith("-transB"):
                trans_b = (val.lower() in ["1", "true"])
            elif opt.endswith("-transO"):
                trans_o = (val.lower() in ["1", "true"])
            elif opt.endswith("-out_datatype"):
                out_dtype = val.lower()
            elif opt.endswith("-perf_config"):
                perf_config = val
            elif opt == '-scale_a_dtype':
                scale_a_dtype = val
            elif opt == '-scale_b_dtype':
                scale_b_dtype = val
            elif opt.endswith("-transScaleA"):
                trans_scale_a = (val.lower() in ["1", "true"])
            elif opt.endswith("-transScaleB"):
                trans_scale_b = (val.lower() in ["1", "true"])
            else:
                raise ValueError(f"Unknown GEMM config argument {opt} -> {val}")
            i += 2
        for v in [dtype, out_dtype, g, m, k, n, trans_a, trans_b]:
            if v is None:
                raise ValueError("Incomplete GEMM configuration")

        return cls(dtype, out_dtype, g, m, k, n, trans_a, trans_b, trans_o, scaled_gemm,
                   scale_a_dtype, scale_b_dtype, trans_scale_a, trans_scale_b, arch, num_cu,
                   num_chiplets, perf_config)

    def to_command_line(self):
        result = (f"-t {self.datatype} -out_datatype {self.out_dtype} " +
                  f"-transA {str(self.trans_a).lower()} -transB {str(self.trans_b).lower()} " +
                  f"-transO {str(self.trans_o).lower()} " +
                  f"-g {self.g} -m {self.m} -n {self.n} -k {self.k}")
        if self.scaled_gemm:
            result += " -scaledGemm"
        if self.scale_a_dtype:
            result += f" -scale_a_dtype {self.scale_a_dtype}"
        if self.scale_b_dtype:
            result += f" -scale_b_dtype {self.scale_b_dtype}"
        if self.trans_scale_a:
            result += f" -transScaleA {str(self.trans_scale_a).lower()}"
        if self.trans_scale_b:
            result += f" -transScaleB {str(self.trans_scale_b).lower()}"
        return result

    def __init__(self,
                 dtype: str,
                 out_dtype: str,
                 g: int,
                 m: int,
                 k: int,
                 n: int,
                 trans_a: bool,
                 trans_b: bool,
                 trans_o: bool = False,
                 scaled_gemm: bool = False,
                 scale_a_dtype: str = None,
                 scale_b_dtype: str = None,
                 trans_scale_a: bool = False,
                 trans_scale_b: bool = False,
                 arch: str = '',
                 num_cu: int = 0,
                 num_chiplets: int = 0,
                 perf_config: str = ''):
        if dtype not in DATA_TYPES_GEMM:
            raise ValueError(f"Invalid datatype: {dtype}")

        if scale_a_dtype is not None and scale_a_dtype not in DATA_TYPES_GEMM_SCALES:
            raise ValueError(
                f"Invalid scale_a_dtype: {scale_a_dtype}. Must be one of {DATA_TYPES_GEMM_SCALES}")

        if scale_b_dtype is not None and scale_b_dtype not in DATA_TYPES_GEMM_SCALES:
            raise ValueError(
                f"Invalid scale_b_dtype: {scale_b_dtype}. Must be one of {DATA_TYPES_GEMM_SCALES}")

        self.datatype = dtype
        self.out_dtype = out_dtype
        self.g = g
        self.m = m
        self.k = k
        self.n = n
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_o = trans_o
        self.perfconfig = perf_config
        self.scaled_gemm = scaled_gemm
        self.scale_a_dtype = scale_a_dtype
        self.scale_b_dtype = scale_b_dtype
        self.trans_scale_a = trans_scale_a
        self.trans_scale_b = trans_scale_b
        self.arch = arch
        self.chip = GFX_CHIP_RE.search(arch).group(0)
        self.num_cu = num_cu
        self.num_chiplets = num_chiplets


class ConvGemmConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.CONV_GEMM_TEST_PARAMETERS + ['TFlops']

    def __init__(self,
                 dtype: str,
                 filter_layout: str,
                 input_layout: str,
                 trans_c: bool,
                 trans_o: bool,
                 n: int,
                 c: int,
                 hi: int,
                 wi: int,
                 k: int,
                 y: int,
                 x: int,
                 o: int,
                 conv_stride_h: int,
                 conv_stride_w: int,
                 padding_h: int,
                 padding_w: int,
                 dilation_h: int,
                 dilation_w: int,
                 group: int,
                 arch: str,
                 num_cu: int,
                 num_chiplets: int,
                 perf_config: str = ''):
        if dtype not in DATA_TYPES_CONV_GEMM:
            raise ValueError(f"Invalid datatype for a: {dtype}")

        self.datatype = dtype

        self.filter_layout = filter_layouts(filter_layout)
        self.input_layout = input_layouts(input_layout)
        self.trans_c = trans_c
        self.trans_o = trans_o

        self.n = n
        self.c = c
        self.hi = hi
        self.wi = wi
        self.k = k
        self.y = y
        self.x = x
        self.o = o

        self.conv_stride_h = conv_stride_h
        self.conv_stride_w = conv_stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w

        self.group = group
        self.arch = arch
        self.chip = GFX_CHIP_RE.search(arch).group(0)
        self.num_cu = num_cu
        self.num_chiplets = num_chiplets
        self.perfconfig = perf_config

        self.ho = math.floor((self.hi + self.padding_h * 2 -
                              (self.y - 1) * self.dilation_h - 1) / self.conv_stride_h) + 1
        self.wo = math.floor((self.wi + self.padding_w * 2 -
                              (self.x - 1) * self.dilation_w - 1) / self.conv_stride_w) + 1

    def compute_tflops(self, ns):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        assert (self.k % self.group == 0)
        assert (self.c % self.group == 0)

        first_conv_flops = 2.0 * self.n * (
            self.c // self.group) * self.k * self.ho * self.wo * self.y * self.x
        first_gemm_m = self.k
        first_gemm_n = self.n * self.ho * self.wo
        batch_second_gemm = 1.0
        second_matmul_flops = 2.0 * batch_second_gemm * first_gemm_m * first_gemm_n * self.o
        total_flops = first_conv_flops + second_matmul_flops

        return total_flops / (float(ns) * 1e-9) / 1e12

    def table_entry(self, nanoseconds):
        result = {}
        values = [
            self.datatype, self.chip, self.num_cu, self.num_chiplets, self.filter_layout,
            self.input_layout, self.trans_c, self.trans_o, self.n, self.c, self.hi, self.wi, self.k,
            self.y, self.x, self.o, self.dilation_h, self.dilation_w, self.conv_stride_h,
            self.conv_stride_w, self.padding_h, self.padding_w, self.perfconfig,
            self.compute_tflops(nanoseconds)
        ]
        assert (len(self.TABLE_COLUMNS) == len(values))
        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def set_perfconfig(self, perf_config):
        self.perfconfig = perf_config

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags, kernel_repeats=MLIR_N_REPEATS):
        result = ' '.join([
            '-operation', 'conv_gemm', '-t', self.datatype, '--arch', self.arch,
            f'--num_cu={self.num_cu}', f'--num_chiplets={self.num_chiplets}',
            f'--fil_layout={self.filter_layout}', f'--in_layout={self.input_layout}',
            f'--transC={self.trans_c}', f'--transO={self.trans_o}', f'--batchsize={self.n}',
            f'--in_channels={self.c}', f'--in_h={self.hi}', f'--in_w={self.wi}',
            f'--out_channels={self.k}', f'--fil_h={self.y}', f'--fil_w={self.x}',
            f'--dilation_h={self.dilation_h}', f'--dilation_w={self.dilation_w}',
            f'--conv_stride_h={self.conv_stride_h}', f'--conv_stride_w={self.conv_stride_w}',
            f'--padding_h={self.padding_h}', f'--padding_w={self.padding_w}',
            f'--groupsize={self.group}', f'--gemmO={self.o}',
            *(['--kernel-repeats', str(kernel_repeats)] if kernel_repeats is not None else []),
            f"--perf_config={self.perfconfig}"
        ])
        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    @classmethod
    def from_command_line(cls, argv, arch, num_cu, num_chiplets):
        # optional defaults
        perf_config = ''
        dtype = None
        n = None
        c = None
        hi = None
        wi = None
        k = None
        y = None
        x = None
        o = None
        conv_stride_h = None
        conv_stride_w = None
        padding_h = None
        padding_w = None
        dilation_h = None
        dilation_w = None
        group = None
        filter_layout = None
        input_layout = None
        trans_c = False
        trans_o = False
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        for i in range(0, len(argv), 2):
            opt = argv[i]
            val = argv[i + 1]
            if opt.endswith("-t"):
                dtype = val
            elif opt.endswith("-n"):
                n = int(val)
            elif opt.endswith("-c"):
                c = int(val)
            elif opt.endswith("-H"):
                hi = int(val)
            elif opt.endswith("-W"):
                wi = int(val)
            elif opt.endswith("-k"):
                k = int(val)
            elif opt.endswith("-y"):
                y = int(val)
            elif opt.endswith("-x"):
                x = int(val)
            elif opt.endswith("-gemmO"):
                o = int(val)
            elif opt == '-u':
                conv_stride_h = int(val)
            elif opt == '-v':
                conv_stride_w = int(val)
            elif opt == '-p':
                padding_h = int(val)
            elif opt == '-q':
                padding_w = int(val)
            elif opt == '-l':
                dilation_h = int(val)
            elif opt == '-j':
                dilation_w = int(val)
            elif opt == '-g':
                group = int(val)
            elif opt == '-f':
                filter_layout = val
            elif opt == '-I':
                input_layout = val
            elif opt.endswith("-transC"):
                trans_c = (val.lower() in ["1", "true"])
            elif opt.endswith("-transO"):
                trans_o = (val.lower() in ["1", "true"])
            elif opt.endswith("-perf_config"):
                perf_config = val
            else:
                raise ValueError(f"Unknown conv+gemm config argument {opt} -> {val}")
        for v in [
                dtype, n, c, hi, wi, k, y, x, o, conv_stride_h, conv_stride_w, padding_h, padding_w,
                dilation_h, dilation_w, group, filter_layout, input_layout, trans_c, trans_o
        ]:
            if v is None:
                raise ValueError("Incomplete conv+gemm configuration")

        return cls(dtype, filter_layout, input_layout, trans_c, trans_o, n, c, hi, wi, k, y, x, o,
                   conv_stride_h, conv_stride_w, padding_h, padding_w, dilation_h, dilation_w,
                   group, arch, num_cu, num_chiplets, perf_config)

    def to_command_line(self):
        return (f"-t {self.datatype} " +
                f"-f {inverse_filter_layouts(self.filter_layout)} -I {self.input_layout.upper()} " +
                f"-transC {str(self.trans_c).lower()} -transO {str(self.trans_o).lower()} " +
                f"-n {self.n} -c {self.c} -H {self.hi} -W {self.wi} -k {self.k} " +
                f"-y {self.y} -x {self.x} -p {self.padding_h} -q {self.padding_w} " +
                f"-u {self.conv_stride_h} -v {self.conv_stride_w} -l {self.dilation_h} " +
                f"-j {self.dilation_w} -g {self.group} -gemmO {str(self.o)}")


class GemmGemmConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.GEMM_GEMM_TEST_PARAMETERS + ['TFlops']
    SWEEP_KIND = "gemm_gemm"

    def __init__(self,
                 dtype: str,
                 g: int,
                 m: int,
                 k: int,
                 n: int,
                 o: int,
                 trans_a: bool,
                 trans_b: bool,
                 trans_c: bool,
                 trans_o: bool,
                 arch: str,
                 num_cu: int,
                 num_chiplets: int,
                 perf_config: str = ''):
        if dtype not in DATA_TYPES_GEMM_GEMM:
            raise ValueError(f"Invalid datatype for a: {dtype}")

        self.datatype = dtype
        self.g = g
        self.m = m
        self.k = k
        self.n = n
        self.o = o
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.trans_o = trans_o

        self.arch = arch
        self.chip = GFX_CHIP_RE.search(arch).group(0)
        self.num_cu = num_cu
        self.num_chiplets = num_chiplets
        self.perfconfig = perf_config

    def compute_tflops(self, ns):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        first_matmul_flops = 2.0 * self.g * self.m * self.k * self.n
        second_matmul_flops = 2.0 * self.g * self.m * self.n * self.o
        total_flops = first_matmul_flops + second_matmul_flops

        return total_flops / (float(ns) * 1e-9) / 1e12

    def table_entry(self, nanoseconds):
        result = {}
        values = [
            self.datatype, self.chip, self.num_cu, self.num_chiplets, self.trans_a, self.trans_b,
            self.trans_c, self.trans_o, self.g, self.m, self.k, self.n, self.o, self.perfconfig,
            self.compute_tflops(nanoseconds)
        ]
        assert (len(self.TABLE_COLUMNS) == len(values))
        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def set_perfconfig(self, perf_config):
        self.perfconfig = perf_config

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags, kernel_repeats=MLIR_N_REPEATS):
        result = ' '.join([
            '-operation', 'gemm_gemm', '-t', self.datatype, '--arch', self.arch, '--num_cu',
            str(self.num_cu), '--num_chiplets',
            str(self.num_chiplets), '-g',
            str(self.g), '-m',
            str(self.m), '-k',
            str(self.k), '-n',
            str(self.n), '-gemmO',
            str(self.o), f"-transA={self.trans_a}", f"-transB={self.trans_b}",
            f"-transC={self.trans_c}", f"-transO={self.trans_o}",
            *(['--kernel-repeats', str(kernel_repeats)] if kernel_repeats is not None else []),
            f"--perf_config={self.perfconfig}"
        ])
        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    @classmethod
    def from_command_line(cls, argv, arch, num_cu, num_chiplets):
        # optional defaults
        perf_config = ''
        dtype = None
        g = None
        m = None
        k = None
        n = None
        o = None
        trans_a = False
        trans_b = False
        trans_c = False
        trans_o = False
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        for i in range(0, len(argv), 2):
            opt = argv[i]
            val = argv[i + 1]
            if opt.endswith("-t"):
                dtype = val
            elif opt.endswith("-g"):
                g = int(val)
            elif opt.endswith("-m"):
                m = int(val)
            elif opt.endswith("-k"):
                k = int(val)
            elif opt.endswith("-n"):
                n = int(val)
            elif opt.endswith("-gemmO"):
                o = int(val)
            elif opt.endswith("-transA"):
                trans_a = (val.lower() in ["1", "true"])
            elif opt.endswith("-transB"):
                trans_b = (val.lower() in ["1", "true"])
            elif opt.endswith("-transC"):
                trans_c = (val.lower() in ["1", "true"])
            elif opt.endswith("-transO"):
                trans_o = (val.lower() in ["1", "true"])
            elif opt.endswith("-perf_config"):
                perf_config = val
            else:
                raise ValueError(f"Unknown gemm+gemm config argument {opt} -> {val}")
        for v in [dtype, g, m, k, n, o, trans_a, trans_b, trans_c, trans_o]:
            if v is None:
                raise ValueError("Incomplete gemm+gemm configuration")

        return cls(dtype, g, m, k, n, o, trans_a, trans_b, trans_c, trans_o, arch, num_cu,
                   num_chiplets, perf_config)

    def to_command_line(self):
        return (f"-t {self.datatype} " +
                f"-transA {str(self.trans_a).lower()} -transB {str(self.trans_b).lower()} " +
                f"-transC {str(self.trans_c).lower()} -transO {str(self.trans_o).lower()} " +
                f"-g {self.g} " +
                f"-m {str(self.m)} -k {str(self.k)} -n {str(self.n)} -gemmO {str(self.o)}")


class AttentionConfiguration(PerfConfiguration):
    TABLE_COLUMNS = reportUtils.ATTN_TEST_PARAMETERS + ['TFlops']
    SWEEP_KIND = "attn"

    def __init__(self,
                 dtype: str,
                 g: int,
                 seq_len_q: int,
                 seq_len_k: int,
                 num_heads_q: int,
                 num_heads_kv: int,
                 head_dim_qk: int,
                 head_dim_v: int,
                 with_attn_scale: bool,
                 with_attn_bias: bool,
                 trans_q: bool,
                 trans_k: bool,
                 trans_v: bool,
                 trans_o: bool,
                 causal: bool,
                 return_lse: bool,
                 split_kv: int,
                 arch: str,
                 num_cu: int,
                 num_chiplets: int,
                 perf_config: str = '',
                 current_seqlen: Optional[List[int]] = None,
                 trans_bias: bool = False):
        if dtype not in DATA_TYPES_ATTENTION:
            raise ValueError(f"Invalid datatype for a: {dtype}")
        if trans_bias and not with_attn_bias:
            raise ValueError("--transBias requires --with-attn-bias")

        self.datatype = dtype
        self.g = g
        self.seq_len_q = seq_len_q
        self.seq_len_k = seq_len_k
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.with_attn_scale = with_attn_scale
        self.with_attn_bias = with_attn_bias
        self.trans_bias = trans_bias
        self.trans_q = trans_q
        self.trans_k = trans_k
        self.trans_v = trans_v
        self.trans_o = trans_o
        self.causal = causal
        self.return_lse = return_lse
        self.split_kv = split_kv
        # Only set in KV-cache mode (seq_len_q == 1). This is a runtime input,
        # so generate_mlir_driver_commandline emits it while to_command_line
        # intentionally omits it from the tuning problem identity.
        self.current_seqlen = current_seqlen

        self.arch = arch
        self.chip = GFX_CHIP_RE.search(arch).group(0)
        self.num_cu = num_cu
        self.num_chiplets = num_chiplets
        self.perfconfig = perf_config

    def compute_tflops(self, ns, only_matmul_flops=True):
        # NaN will propagate as expected
        # Repeats are handled by the fact that we're using avarageNs
        # GQA broadcasts so that both num_heads_q == num_heads_kv
        g = self.g * max(self.num_heads_q, self.num_heads_kv)
        first_matmul_flops = 2.0 * g * self.seq_len_q * self.head_dim_qk * self.seq_len_k
        # max, sub, exp, sum, div
        softmax_flops = 5.0 * g * self.seq_len_q * self.seq_len_k
        second_matmul_flops = 2.0 * g * self.seq_len_q * self.seq_len_k * self.head_dim_v
        total_flops = first_matmul_flops + second_matmul_flops
        # Weirdly, triton does not account for flops coming from
        # non matmul operations as per FA2 paper. Hence not including
        # by default
        # References:
        # 1) https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
        # 2) Flash-Attention 2 : https://arxiv.org/abs/2307.08691
        if not only_matmul_flops:
            total_flops += softmax_flops
            if self.with_attn_scale:
                total_flops += g * self.seq_len_q * self.seq_len_k
            if self.with_attn_bias:
                total_flops += g * self.seq_len_q * self.seq_len_k
        return total_flops / (float(ns) * 1e-9) / 1e12

    def table_entry(self, nanoseconds):
        result = {}
        values = [
            self.datatype, self.chip, self.num_cu, self.num_chiplets, self.trans_q, self.trans_k,
            self.trans_v, self.trans_o, self.causal, self.return_lse, self.split_kv,
            self.with_attn_scale, self.with_attn_bias, self.trans_bias, self.g, self.seq_len_q,
            self.seq_len_k, self.num_heads_q, self.num_heads_kv, self.head_dim_qk, self.head_dim_v,
            self.perfconfig,
            self.compute_tflops(nanoseconds)
        ]
        assert (len(self.TABLE_COLUMNS) == len(values))
        for k, v in zip(self.TABLE_COLUMNS, values):
            result[k] = v
        return result

    def set_perfconfig(self, perf_config):
        self.perfconfig = perf_config

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags, kernel_repeats=MLIR_N_REPEATS):
        result = ' '.join([
            '-operation', 'attention', '-t', self.datatype, '--arch', self.arch, '--num_cu',
            str(self.num_cu), '--num_chiplets',
            str(self.num_chiplets), '-g',
            str(self.g), '-seq_len_q',
            str(self.seq_len_q), '-seq_len_k',
            str(self.seq_len_k), '-num_heads_q',
            str(self.num_heads_q), '-num_heads_kv',
            str(self.num_heads_kv), '-head_dim_qk',
            str(self.head_dim_qk), '-head_dim_v',
            str(self.head_dim_v), f"-with-attn-scale={self.with_attn_scale}",
            f"-with-attn-bias={self.with_attn_bias}", f"-transBias={self.trans_bias}",
            f"-transQ={self.trans_q}", f"-transK={self.trans_k}", f"-transV={self.trans_v}",
            f"-transO={self.trans_o}", f"-causal={self.causal}", f"-return_lse={self.return_lse}",
            f"-split_kv={self.split_kv}",
            *([f"-current_seq_len={','.join(map(str, self.current_seqlen))}"]
              if self.current_seqlen else []),
            *(['--kernel-repeats', str(kernel_repeats)] if kernel_repeats is not None else []),
            f"--perf_config={self.perfconfig}"
        ])
        result += ' '
        if rocmlir_gen_flags != '':
            result += ' '.join(rocmlir_gen_flags.split())
        return result

    @classmethod
    def from_command_line(cls, argv, arch, num_cu, num_chiplets):
        # optional defaults
        perf_config = ''
        dtype = None
        g = None
        seq_len_q = None
        seq_len_k = None
        num_heads_q = 1
        num_heads_kv = 1
        head_dim_qk = None
        head_dim_v = None
        trans_q = False
        trans_k = False
        trans_v = False
        trans_o = False
        causal = False
        return_lse = False
        split_kv = 1
        with_attn_scale = False
        with_attn_bias = False
        trans_bias = False
        # Please keep this in sync with mlir::rock::getTuningProblemStr()
        for i in range(0, len(argv), 2):
            opt = argv[i]
            val = argv[i + 1]
            if opt.endswith("-t"):
                dtype = val
            elif opt.endswith("-g"):
                g = int(val)
            elif opt.endswith("-seq_len_q"):
                seq_len_q = int(val)
            elif opt.endswith("-seq_len_k"):
                seq_len_k = int(val)
            elif opt.endswith("-num_heads_q"):
                num_heads_q = int(val)
            elif opt.endswith("-num_heads_kv"):
                num_heads_kv = int(val)
            elif opt.endswith("-head_dim_qk"):
                head_dim_qk = int(val)
            elif opt.endswith("-head_dim_v"):
                head_dim_v = int(val)
            elif opt.endswith("-with-attn-scale"):
                with_attn_scale = (val.lower() in ["1", "true"])
            elif opt.endswith("-with-attn-bias"):
                with_attn_bias = (val.lower() in ["1", "true"])
            elif opt.endswith("-transBias"):
                trans_bias = (val.lower() in ["1", "true"])
            elif opt.endswith("-transQ"):
                trans_q = (val.lower() in ["1", "true"])
            elif opt.endswith("-transK"):
                trans_k = (val.lower() in ["1", "true"])
            elif opt.endswith("-transV"):
                trans_v = (val.lower() in ["1", "true"])
            elif opt.endswith("-transO"):
                trans_o = (val.lower() in ["1", "true"])
            elif opt.endswith("-causal"):
                causal = (val.lower() in ["1", "true"])
            elif opt.endswith("-return_lse"):
                return_lse = (val.lower() in ["1", "true"])
            elif opt.endswith("-split_kv"):
                split_kv = int(val)
            elif opt.endswith("-perf_config"):
                perf_config = val
            else:
                raise ValueError(f"Unknown Attention config argument {opt} -> {val}")
        for v in [
                dtype, g, seq_len_q, seq_len_k, num_heads_q, num_heads_kv, head_dim_qk, head_dim_v,
                with_attn_scale, with_attn_bias, trans_q, trans_k, trans_v, trans_o, causal,
                return_lse, split_kv
        ]:
            if v is None:
                raise ValueError("Incomplete Attention configuration")

        return cls(dtype,
                   g,
                   seq_len_q,
                   seq_len_k,
                   num_heads_q,
                   num_heads_kv,
                   head_dim_qk,
                   head_dim_v,
                   with_attn_scale,
                   with_attn_bias,
                   trans_q,
                   trans_k,
                   trans_v,
                   trans_o,
                   causal,
                   return_lse,
                   split_kv,
                   arch,
                   num_cu,
                   num_chiplets,
                   perf_config,
                   trans_bias=trans_bias)

    def to_command_line(self):
        return (
            f"-t {self.datatype} " +
            f"-transQ {str(self.trans_q).lower()} -transK {str(self.trans_k).lower()} " +
            f"-transV {str(self.trans_v).lower()} -transO {str(self.trans_o).lower()} " +
            f"-causal {str(self.causal).lower()} " +
            f"-return_lse {str(self.return_lse).lower()} " + f"-split_kv {str(self.split_kv)} " +
            f"-g {self.g} " +
            f"-seq_len_q {str(self.seq_len_q)} -seq_len_k {str(self.seq_len_k)} -num_heads_q {str(self.num_heads_q)} -num_heads_kv {str(self.num_heads_kv)} -head_dim_qk {str(self.head_dim_qk)} -head_dim_v {str(self.head_dim_v)} "
            + f"-with-attn-scale {str(self.with_attn_scale).lower()} " +
            f"-with-attn-bias {str(self.with_attn_bias).lower()} " +
            f"-transBias {str(self.trans_bias).lower()}")


def auto_precision_flags_att(config: PerfConfiguration) -> List[str]:
    """Return precision-aware rocmlir-gen flags for verification.

    Verification compares the GPU output against the host CPU reference. With
    long ``seq_length`` attention (f32/bf16), kernel error accumulates due to
    reduction drift, masking the GPU's actual precision.

    ``--pv-f64`` promotes the host kernel's interior to f64, eliminating
    the reference-side drift. It implies ``--pv-strict`` and is only valid
    for non-quantized attention (rocmlir-gen errors out for non-attention
    or i8 attention), so we only emit it for f32/bf16 attention here.

    Shared between the tuner (``tuningRunner``) and the parameter sweeps
    (``attentionSweeps``); keep the only definition here.
    """
    flags: List[str] = []
    if not isinstance(config, AttentionConfiguration):
        return flags

    # CPU drift observed for f32 attention at long seq_len_k > 1024.
    if config.datatype == 'f32' and config.seq_len_k > 1024:
        flags.append('--pv-f64')
    # CPU drift observed for bf16 attention at long seq_len_k > 70.
    if config.datatype == 'bf16' and config.seq_len_k > 70:
        flags.append('--pv-f64')

    return flags


class HipBLASLtGemmConfig(GemmConfiguration):
    EXTERNAL_NAME = "hipBLASLt"

    @classmethod
    def benchmark_external(cls, commandline, paths: Paths, arch, num_cu, num_chiplets):
        config = cls.from_command_line(commandline, arch, num_cu, num_chiplets)
        if not paths.mlir_paths.hipblaslt_benchmark_driver_path:
            raise ValueError("hipblaslt-benchmark-driver not built")
        benchmark_args = config.generate_mlir_driver_commandline("")
        # remove the result file generated by rocprof in previous benchmarking
        if os.path.exists(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME)):
            os.remove(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME))
        print(f"Running hipBLASLt benchmark {config!r}")
        profiler_cmd = [paths.mlir_paths.hipblaslt_benchmark_driver_path] + \
            benchmark_args.split()
        outs, noerr = run_pipeline([profiler_cmd])
        nanoseconds = np.nan
        if noerr:
            miliseconds = get_miliseconds(outs)
            nanoseconds = miliseconds * 1e6

        return config.table_entry(nanoseconds)


class CKGemmConfig(GemmConfiguration):
    EXTERNAL_NAME = "CK"

    @classmethod
    def benchmark_external(cls, commandline, paths: Paths, arch, num_cu, num_chiplets):
        config = cls.from_command_line(commandline, arch, num_cu, num_chiplets)
        if not paths.mlir_paths.ck_gemm_benchmark_driver_path:
            raise ValueError("ck-gemm-benchmark-driver not built")
        benchmark_args = config.generate_mlir_driver_commandline("")

        print(f"Running CK benchmark {config!r}")

        if arch == "gfx1030" and config.g > 1:
            return config.table_entry(float('NaN'))

        profiler_cmd = [paths.mlir_paths.ck_gemm_benchmark_driver_path] + \
            benchmark_args.split()
        outs, noerr = run_pipeline([profiler_cmd])
        nanoseconds = np.nan
        if noerr:
            miliseconds = get_miliseconds(outs)
            nanoseconds = miliseconds * 1e6

        return config.table_entry(nanoseconds)


def run_config_with_mlir(config: PerfConfiguration,
                         paths: Paths,
                         arch,
                         rocmlir_gen_flags,
                         use_rocprof=False,
                         flush_last_level_cache=False,
                         debug=True):
    # remove the result file generated by rocprof in previous benchmarking
    if os.path.exists(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME)):
        os.remove(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME))
    use_tuning_driver = (not use_rocprof) and bool(config.perfconfig)
    use_host_harness = not use_tuning_driver

    rocmlir_gen_flags = rocmlir_gen_flags + ' -ph' if use_host_harness else ''
    # We want to use kernel_repeats only if we are passing ' -ph' to rocmlir-gen, otherwise we use None.
    # This is because the kernel-repeats flag is only supported with host harness or CPU validation.
    kernel_repeats = MLIR_N_REPEATS if use_host_harness else None

    commandline_options = config.generate_mlir_driver_commandline(rocmlir_gen_flags, kernel_repeats)
    rocmlir_gen_cmd = paths.mlir_paths.rocmlir_gen_path + ' ' + commandline_options
    if debug:
        print("Running MLIR Benchmark: ", repr(config))

    nanoseconds = np.nan

    # Use HIP timing via tuning-driver if rocprof is disabled and perfconfig is present
    if use_tuning_driver:
        if debug:
            print("Using HIP timing for benchmarking")
        tuning_driver_command = [
            paths.mlir_paths.rocmlir_tuning_driver_path, f'--benchmark-config={config.perfconfig}',
            f'--rep={TUNE_REP_MS}', f'--warmup={TUNE_WARMUP_MS}', f'--sleep-us={SLEEP_US}',
            '--use-median'
        ]
        if flush_last_level_cache:
            tuning_driver_command.append("--flush-last-level-cache")
        tuning_driver_command.append('-')
        outs, noerr = run_pipeline([rocmlir_gen_cmd.split(), tuning_driver_command])
        if noerr:
            try:
                _, time = outs.split()
                if time != "N/A":
                    nanoseconds = float(time)
            except ValueError:
                if debug:
                    print(f"Failed to parse timing result: {outs}")
    else:
        if debug:
            print("Using rocprof for benchmarking")
        if flush_last_level_cache:
            print(
                "Warning: --flush-last-level-cache is ignored when using rocprof for benchmarking")
        rocmlir_driver_cmd = [paths.mlir_paths.rocmlir_driver_path, '-c']
        profiler_cmd = [ROCPROF] + get_metric_args_for_rocprof(arch) + [
            '--kernel-trace', '--stats', '-f', 'csv', '-o', BENCHMARKING_RESULT_FILE_NAME, '--',
            paths.mlir_paths.rocm_run_path
        ]

        outs, noerr = run_pipeline([rocmlir_gen_cmd.split(), rocmlir_driver_cmd, profiler_cmd])
        if noerr:
            nanoseconds = get_nanoseconds(
                get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME))

    return nanoseconds


def canonicalize_config(config_str: str, conf_class: type, arch: str, num_cu: int,
                        num_chiplets: int) -> str:
    """Canonicalize a config by round-tripping it through
    ``conf_class.from_command_line`` / ``to_command_line``.

    perfRunner resolves tuned perf-configs from the tuning DB by
    ``config.to_command_line()`` (see ``benchmark_mlir``), so every producer and
    consumer of a config string must agree on this canonical form. Running each
    config string through here makes a raw test vector (e.g. one missing the
    ``-m conv ... -t 1`` MIOpen suffix, or spelling a layout differently) match
    the key perfRunner looks up by.

    ``PerfConfiguration`` is the fusion catch-all and dispatches by
    positional-arg prefix: a ``conv*`` first token routes to
    ``ConvConfiguration``, otherwise to ``GemmConfiguration``.

    Raises ``ValueError`` if ``conf_class`` cannot parse ``config_str``.
    """
    resolved_class = conf_class
    if resolved_class is PerfConfiguration:
        resolved_class = (ConvConfiguration
                          if config_str.lstrip().startswith('conv') else GemmConfiguration)
    config = resolved_class.from_command_line(config_str.split(), arch, num_cu, num_chiplets)
    return config.to_command_line()


def lookup_tuning_db(tuning_db: MaybeTuningDb, arch: str, config: PerfConfiguration,
                     config_str: str) -> Optional[str]:
    """Return the perf config for ``config_str``, including legacy attention keys.

    Attention tuning keys gained optional identity flags over time. If an exact
    lookup misses, try legacy keys with only false-valued attention flags
    removed so older DB rows continue to match equivalent kernels.
    """
    if not tuning_db:
        return None

    if (arch, config_str) in tuning_db:
        return tuning_db[arch, config_str]

    if isinstance(config, AttentionConfiguration):
        false_flags = []
        if not config.with_attn_scale:
            false_flags.append(" -with-attn-scale false")
        if not config.with_attn_bias:
            false_flags.append(" -with-attn-bias false")
        if not config.trans_bias:
            false_flags.append(" -transBias false")

        # Older tuning DB rows may predate one or more false-valued attention
        # identity flags. Never strip a true-valued flag: those describe
        # different generated kernels and need separate tuning entries.
        for num_stripped in range(1, len(false_flags) + 1):
            for flag_group in itertools.combinations(false_flags, num_stripped):
                legacy_config_str = config_str
                for flag in flag_group:
                    legacy_config_str = legacy_config_str.replace(flag, "")
                if (arch, legacy_config_str) in tuning_db:
                    return tuning_db[arch, legacy_config_str]

    return None


# Benchmarking function.
def benchmark_mlir(commandline,
                   conf_class,
                   paths: Paths,
                   arch,
                   num_cu,
                   num_chiplets,
                   tuning_db: MaybeTuningDb,
                   rocmlir_gen_flags,
                   use_rocprof=False,
                   flush_last_level_cache=False):
    config = conf_class.from_command_line(commandline, arch, num_cu, num_chiplets)
    config_str = config.to_command_line()
    if tuning_db:
        perf_config = lookup_tuning_db(tuning_db, arch, config, config_str)
        if perf_config is not None:
            config.set_perfconfig(perf_config)
        else:  # Tuning DB present but doesn't contain config, return N/A
            return config.table_entry(np.nan)

    nanoseconds = run_config_with_mlir(config, paths, arch, rocmlir_gen_flags, use_rocprof,
                                       flush_last_level_cache)
    return config.table_entry(nanoseconds)


# Generate MLIR vs. MIOpen or hipBLASLt performance results
def generate_performance_results(configs,
                                 conf_class,
                                 paths: Paths,
                                 arch,
                                 num_cu,
                                 num_chiplets,
                                 tuning_db: MaybeTuningDb,
                                 quick_tuning_db: MaybeTuningDb,
                                 rocmlir_gen_flags,
                                 use_rocprof=False,
                                 flush_last_level_cache=False):
    # Never pass tuning DB to this run
    mlir_df = pd.DataFrame(
        benchmark_mlir(test_vector.split(sep=' '), conf_class, paths, arch, num_cu, num_chiplets,
                       None, rocmlir_gen_flags, use_rocprof, flush_last_level_cache)
        for test_vector in configs)
    tuned_df = None
    if tuning_db:
        tuned_df = pd.DataFrame(
            benchmark_mlir(test_vector.split(
                sep=' '), conf_class, paths, arch, num_cu, num_chiplets, tuning_db,
                           rocmlir_gen_flags, use_rocprof, flush_last_level_cache)
            for test_vector in configs)
    quick_tuned_df = None
    if quick_tuning_db:
        quick_tuned_df = pd.DataFrame(
            benchmark_mlir(test_vector.split(
                sep=' '), conf_class, paths, arch, num_cu, num_chiplets, quick_tuning_db,
                           rocmlir_gen_flags, use_rocprof, flush_last_level_cache)
            for test_vector in configs)

    external_df = pd.DataFrame(
        conf_class.benchmark_external(test_vector.split(sep=' '), paths, arch, num_cu, num_chiplets)
        for test_vector in configs)

    external_name = conf_class.EXTERNAL_NAME
    df = mlir_df.merge(external_df,
                       on=conf_class.TABLE_COLUMNS[:-2],
                       suffixes=('', f" ({external_name})"))
    external_tflops_col = f"{external_name} TFlops (no MLIR Kernels)"
    df.rename(columns={
        'TFlops': 'MLIR TFlops',
        f"TFlops ({external_name})": external_tflops_col
    },
              inplace=True)
    #     if tuned_df is None and quick_tuned_df is None:
    #         df.drop(columns=['PerfConfig'], inplace=True)
    if tuned_df is not None:
        # No need for suffixes, the conflicting columns have been renamed
        # Also note that we're ignoring PerfConfig with the -3
        df = df.merge(tuned_df, on=conf_class.TABLE_COLUMNS[:-3], suffixes=('', ' (tuned)'))
        df.drop(columns=['PerfConfig'], inplace=True)
        df.rename(columns={
            'TFlops': 'Tuned MLIR TFlops',
            'PerfConfig (tuned)': 'PerfConfig'
        },
                  inplace=True)
    if quick_tuned_df is not None:
        # No need for suffixes, the conflicting columns have been renamed
        # Also note that we're ignoring PerfConfig with the -3
        df = df.merge(quick_tuned_df,
                      on=conf_class.TABLE_COLUMNS[:-3],
                      suffixes=('', ' (quick tuned)'))
        df.rename(columns={'TFlops': 'Quick Tuned MLIR TFlops'}, inplace=True)

    df[f"MLIR/{external_name}"] = df['MLIR TFlops'] / df[external_tflops_col]
    if tuned_df is not None:
        df[f"Tuned/{external_name}"] = df['Tuned MLIR TFlops'] / df[external_tflops_col]
        df["Tuned/Untuned"] = df['Tuned MLIR TFlops'] / df['MLIR TFlops']
    if quick_tuned_df is not None:
        df[f"Quick Tuned/{external_name}"] = df['Quick Tuned MLIR TFlops'] / df[external_tflops_col]
        df["Quick Tuned/Untuned"] = df['Quick Tuned MLIR TFlops'] / df['MLIR TFlops']
    if tuned_df is not None and quick_tuned_df is not None:
        df["Quick Tuned/Tuned"] = df['Quick Tuned MLIR TFlops'] / df['Tuned MLIR TFlops']
    chip = GFX_CHIP_RE.search(arch).group(0)
    if conf_class is HipBLASLtGemmConfig:
        report_file = reportUtils.PERF_REPORT_FILE['hipBLASLt']
    elif conf_class is CKGemmConfig:
        report_file = reportUtils.PERF_REPORT_FILE['CK']
    else:
        report_file = reportUtils.PERF_REPORT_FILE['MIOpen']
    df.fillna(np.nan, inplace=True)
    df.to_csv(chip + '_' + report_file, index=False)


def get_solver_name(test_vector, arch, num_cu, num_chiplets):
    config = ConvConfiguration.from_command_line(test_vector.split(sep=' '), arch, num_cu,
                                                 num_chiplets)
    if config.direction == 'fwd':
        solver_name = 'ConvMlirIgemmFwd'
    elif config.direction == 'bwd':
        solver_name = 'ConvMlirIgemmBwd'
    else:
        solver_name = 'ConvMlirIgemmWrW'
    if config.chip in ['gfx908', 'gfx90a', 'gfx942', 'gfx950']:
        solver_name += 'Xdlops'
    return solver_name


RUNNABLE_TEST_RE = re.compile(r"//\s*RUN\s*:(.*)")
ROCMLIRGEN_RE = re.compile(r"rocmlir-gen.*?-fut\s*(\w+)")


def find_run_command(filename):
    rocmlir_cmd = None
    fut_name = None
    with open(filename, 'r') as f:
        for line in f:
            has_run = RUNNABLE_TEST_RE.search(line)
            has_rocmlir_gen = ROCMLIRGEN_RE.search(line)
            if has_run:
                command = has_run.group(1)
                if not rocmlir_cmd:
                    parts = command.split('|')  # Split the command using the "|" separator
                    if 'rocmlir-driver' in parts[0] or 'rocmlir-opt' in parts[0]:
                        rocmlir_cmd = parts[0].strip()  # Find rocmlir-driver command
                    elif 'rocmlir-driver' in parts[1] or 'rocmlir-opt' in parts[1]:
                        rocmlir_cmd = parts[1].strip()

                if has_rocmlir_gen and not fut_name:
                    fut_name = has_rocmlir_gen.group(1)

                if 'runner' in line:  # Stop processing lines after finding a runner
                    return rocmlir_cmd, fut_name

    # Not found a "RUN" command or a runner
    print("WARNING: cannot find valid RUN command in ", filename)
    return None, None


# Extract test_vector and test function name from the test file
def get_fusion_test_info(filename, paths: Paths, target_chip: Optional[str] = None):
    chip = target_chip if target_chip is not None else get_chip()
    test_entry = {}
    rocmlir_cmd, fut_name = find_run_command(filename)
    if not rocmlir_cmd:
        return test_entry
    # rocmlir-gen -fut test -arch gfx90a --clone-harness
    rocmlirgen_cmd = [
        paths.mlir_paths.rocmlir_gen_path, '-fut', fut_name, '-arch', chip, '--clone-harness',
        filename
    ]
    p0 = subprocess.Popen(rocmlirgen_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if "-migraphx-to-tosa" in rocmlir_cmd:
        rocmliropt_cmd = [paths.mlir_paths.rocmlir_opt_path, '-migraphx-to-tosa']
        rocmlir_driver_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'highlevel', '-kernel-pipeline',
            'highlevel', '-targets', chip
        ]
        # rocmlir-opt -migraphx-to-tosa ../mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-1.mlir
        p1 = subprocess.Popen(rocmliropt_cmd,
                              stdin=p0.stdout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
        # pipe to rocmlir-driver -host-pipeline highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlir_driver_cmd,
                              stdin=p1.stdout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
        p1.stdout.close()
    elif "migraphx" in rocmlir_cmd:
        rocmlir_migraphx_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-kernel-pipeline', 'migraphx,highlevel'
        ]
        rocmlir_driver_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'migraphx,highlevel',
            '-targets', chip
        ]
        # rocmlir-driver -kernel-pipeline migraphx ../mlir/test/fusion/resnet50-e2e/mixr-resnet-fusion-case-1.mlir
        p1 = subprocess.Popen(rocmlir_migraphx_cmd,
                              stdin=p0.stdout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
        # pipe to rocmlir-driver -host-pipeline highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlir_driver_cmd,
                              stdin=p1.stdout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
        p1.stdout.close()
    else:
        rocmlir_driver_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'highlevel', '-kernel-pipeline',
            'highlevel', '-targets', chip
        ]
        # rocmlir-driver -host-pipeline highlevel -targets gfx90a
        p2 = subprocess.Popen(rocmlir_driver_cmd,
                              stdin=p0.stdout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)

    # pipe to rocmlir_gen --emit-tuning-key
    tuning_key = subprocess.Popen([paths.mlir_paths.rocmlir_gen_path, '--emit-tuning-key', '-'],
                                  stdin=p2.stdout,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
    p2.stdout.close()
    output, _ = tuning_key.communicate()
    result = output.decode('utf-8').strip().split('\t')
    test_entry = {'filename': filename, 'testVector': result[3], 'futName': fut_name}
    return test_entry


def run_fusion_kernel(filename, rocmlir_gen_args, paths: Paths):
    arch = get_arch()
    chip = get_chip()
    if os.path.exists(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME)):
        os.remove(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME))

    rocmlir_cmd, fut_name = find_run_command(filename)

    # rocmlir-gen -fut test -arch gfx90a --clone-harness
    rocmlirgen_cmd = [
        paths.mlir_paths.rocmlir_gen_path, '-fut', fut_name, '-arch', chip, '--clone-harness',
        filename
    ]
    commands = [rocmlirgen_cmd]
    if "-migraphx-to-tosa" in rocmlir_cmd:
        rocmliropt_cmd = [paths.mlir_paths.rocmlir_opt_path, '-migraphx-to-tosa', filename]
        commands.append(rocmliropt_cmd)
        rocmlir_driver_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'highlevel', '-kernel-pipeline',
            'highlevel', '-targets', chip
        ]
        commands.append(rocmlir_driver_cmd)
    elif "migraphx" in rocmlir_cmd:
        rocmlir_migraphx_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-kernel-pipeline', 'migraphx,highlevel'
        ]
        commands.append(rocmlir_migraphx_cmd)
        rocmlir_driver_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'migraphx,highlevel',
            '-targets', chip
        ]
        commands.append(rocmlir_driver_cmd)
    else:
        rocmlir_driver_cmd = [
            paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'highlevel', '-kernel-pipeline',
            'highlevel', '-targets', chip
        ]
        commands.append(rocmlir_driver_cmd)

    rocmlir_gen_cmd = [paths.mlir_paths.rocmlir_gen_path] + rocmlir_gen_args
    commands.append(rocmlir_gen_cmd)
    kernel_pipeline_cmd = [
        paths.mlir_paths.rocmlir_driver_path, '-host-pipeline', 'backend', '-kernel-pipeline',
        'full'
    ]
    commands.append(kernel_pipeline_cmd)
    profiler_cmd = [ROCPROF] + get_metric_args_for_rocprof(chip) + [
        '--kernel-trace', '--stats', '-f', 'csv', '-o', BENCHMARKING_RESULT_FILE_NAME
    ] + ['--', paths.mlir_paths.rocm_run_path]
    commands.append(profiler_cmd)
    outs, noerr = run_pipeline(commands)
    nanoseconds = np.nan
    if noerr:
        nanoseconds = get_nanoseconds(get_profiler_output_path(arch, BENCHMARKING_STATS_FILE_NAME))

    return nanoseconds


# Generate fusion vs. gemm/conv performance results
def benchmark_fusion_kernels(test_dir,
                             paths: Paths,
                             arch,
                             num_cu,
                             num_chiplets,
                             tuning_db: MaybeTuningDb,
                             use_rocprof=False,
                             flush_last_level_cache=False):
    all_tests = []  # filename, test_vector, fut_name
    perf_results = {}  # associate test_vector to config and performances
    chip = GFX_CHIP_RE.search(arch).group(0)

    # Prepare test cases
    for filename in glob.glob(test_dir + '/*.mlir'):
        test_entry = get_fusion_test_info(filename, paths)
        if test_entry:
            all_tests.append(test_entry)

    if tuning_db:
        # Force all split-K factors to 1, to avoid trouble because fusion
        # and split-K aren't compatible.  Crude parser mirroring the CSV
        # layout serialized by GemmParamsAttr::getPerfConfigStr (see
        # RockAttrDefs.td); SPLITK_IDX must match the splitKFactor field
        # position.
        for (arch, config), perfconfig in tuning_db.items():
            split_perf = perfconfig.split(',')
            if int(split_perf[SPLITK_IDX]) > 1:
                split_perf[SPLITK_IDX] = '1'
                tuning_db[arch, config] = ','.join(split_perf)

    # Profile each test case
    for test in all_tests:
        filename = test['filename']
        test_vector = test['testVector']
        fut_name = test['futName']

        print("Profiling:", filename)
        # Sanity check
        if not test_vector:
            print("\tCannot find a test vector")
            continue
        if not fut_name:
            print("\tCannot find rocmlir-gen with -fut")
            continue

        commandline = test_vector.split(sep=' ')
        if commandline[0].startswith('conv'):
            op = 'conv'
            config = ConvConfiguration.from_command_line(commandline, arch, num_cu, num_chiplets)
        else:
            op = 'gemm'
            config = GemmConfiguration.from_command_line(commandline, arch, num_cu, num_chiplets)

        # Find the best perf_config
        best_perf = ""
        if tuning_db:
            config_str = config.to_command_line()
            if (arch, config_str) in tuning_db:
                best_perf = tuning_db[arch, config_str]
                config.set_perfconfig(best_perf)
            else:  # Tuning DB present but doesn't contain config, add a NaN entry
                if test_vector not in perf_results:
                    one_entry = config.table_entry(np.nan)
                    one_entry['MLIR TFlops'] = np.nan
                    one_entry['Fusion/MLIR'] = np.nan
                    one_entry['FileName'] = filename
                    perf_results[test_vector] = one_entry
                continue

        # Run fusion test
        rocmlir_gen_args = [
            '-ph', '-fut=' + fut_name + '_wrapper', '--perf_config=' + best_perf, '-'
        ]
        nanoseconds = run_fusion_kernel(filename, rocmlir_gen_args, paths)
        one_entry = config.table_entry(nanoseconds)
        # Keep the best performance
        if test_vector in perf_results and one_entry['TFlops'] <= perf_results[test_vector][
                'TFlops']:
            continue

        # Run gemm or conv op with the same configuration
        nanoseconds = run_config_with_mlir(config, paths, arch, '', use_rocprof,
                                           flush_last_level_cache)
        one_entry['MLIR TFlops'] = config.compute_tflops(nanoseconds)
        one_entry['Fusion/MLIR'] = one_entry['TFlops'] / one_entry['MLIR TFlops']
        one_entry['FileName'] = filename
        perf_results[test_vector] = one_entry

    df = pd.DataFrame(perf_results.values())
    df.fillna(np.nan, inplace=True)
    df.rename(columns={'TFlops': 'Fusion TFlops'}, inplace=True)
    df.to_csv(chip + '_' + op + '_' + reportUtils.PERF_REPORT_FUSION_FILE, index=False)


# Tune MIOpen with MLIR kernels
def tune_mlir_kernels(configs, arch, num_cu, num_chiplets):
    solver_names = {
        test_vector: get_solver_name(test_vector, arch, num_cu, num_chiplets)
        for test_vector in configs
    }

    envs = os.environ.copy()
    envs['MIOPEN_FIND_ENFORCE'] = '4'
    envs['MIOPEN_DRIVER_USE_GPU_REFERENCE'] = '1'
    for test_vector in configs:
        envs['MIOPEN_DEBUG_FIND_ONLY_SOLVER'] = solver_names[test_vector]
        commandline = test_vector.split(sep=' ')
        config = ConvConfiguration.from_command_line(commandline, arch, num_cu, num_chiplets)
        if config.input_layout == 'nchw':
            miopen_driver_cmd = [MIOPENDRIVER, *commandline, '-V', '0']
            print(' '.join(miopen_driver_cmd))
            p1 = subprocess.Popen(miopen_driver_cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  env=envs)
            # get output.
            try:
                _, errs = p1.communicate(timeout=300)
                if len(errs) > 0 and p1.returncode != 0:
                    raise OSError(errs.decode('utf-8'))
            except subprocess.TimeoutExpired:
                p1.kill()
                print("MIOpen tuning timed out")
                _, errs = p1.communicate()


def parse_data_types(data_types):
    if not data_types:
        return DATA_TYPES_GEMM, OUTPUT_DATA_TYPES_MAP
    datatypes = []
    out_map = {}
    for dpair in data_types:
        dt = dpair.split('_')
        datatypes.append(dt[0])
        out_map[dt[0]] = dt[0]
        if len(dt) == 2:
            out_map[dt[0]] = dt[1]
        elif dt[0] == 'i8':
            out_map[dt[0]] = 'i32'
        elif dt[0] == 'fp8':
            out_map[dt[0]] = 'f32'
    return datatypes, out_map


def get_num_cu(chip):
    for props in iter_device_props():
        agent = props.gcnArchName.decode('utf-8')
        if chip in agent:
            return int(props.multiProcessorCount)
    raise RuntimeError(f"Cannot find number of CUs for {chip}")


def found_external_tool(paths: Paths,
                        optype: Operation,
                        gemm_library: Optional[GEMMLibrary] = None):
    if optype == Operation.GEMM:
        if not paths.mlir_paths:
            return False
        if gemm_library == GEMMLibrary.CK and not paths.mlir_paths.ck_gemm_benchmark_driver_path:
            return False
        if gemm_library == GEMMLibrary.HIPBLASLT and not paths.mlir_paths.hipblaslt_benchmark_driver_path:
            return False
    return True


# Main function.
def main(args=None):
    """
    usage examples:

    python3 perfRunner.py
    python3 perfRunner.py --batch_all -o=output_file.csv
    python3 perfRunner.py --batch_all -o=output_file.csv -t=tuning_db.tsv
    python3 perfRunner.py -b
    # Uses results from tuning db when running MLIR benchmarks
    python3 perfRunner.py -b -t=tuning_db.tsv
    python3 perfRunner.py --batch_external
    python3 perfRunner.py --operation gemm --external # hipBLASLt tests
    python3 perfRunner.py -- conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
    python3 perfRunner.py --external -- conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1
    python3 perfRunner.py --operation gemm [--external] -- -t f32 -transA true -transB true -g 1 -m 1024 -k 769 -n 512
    """
    if args is None:
        args = sys.argv[1:]

    arch = get_arch()
    chip = get_chip()
    num_cu = get_num_cu(chip)
    num_chiplets = amd_arch_db.infer_num_chiplets(chip, num_cu)

    root_dir = str(
        subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
    default_conv_configs = root_dir + '/mlir/utils/jenkins/performance/configs/tier1-conv-configs'

    parser = argparse.ArgumentParser(
        prog="rocMLIR performance test runner",
        description="A test runner script for MIOpen and MLIR-based kernel generator",
        allow_abbrev=False,
    )

    parser.add_argument("--op",
                        "--operation",
                        choices=['conv', 'gemm', 'fusion', 'attention', 'gemm_gemm', 'conv_gemm'],
                        default='conv',
                        help="Operation to benchmark")

    mutex_arg_group = parser.add_mutually_exclusive_group()
    mutex_arg_group.add_argument("--tuning", action="store_true", help="Only tune the MLIR kernels")
    mutex_arg_group.add_argument("-b",
                                 "--batch_mlir",
                                 action="store_true",
                                 help="CSV batch benchmarking mode with MLIR")
    mutex_arg_group.add_argument("--batch_external",
                                 action="store_true",
                                 help="CSV batch benchmarking mode with external reference")
    mutex_arg_group.add_argument(
        "--batch_all",
        action="store_true",
        help="CSV batch benchmarking with MLIR and external reference (defalut on no args)")
    mutex_arg_group.add_argument("--external",
                                 action="store_true",
                                 help="benchmark a single config externally")

    parser.add_argument("-c",
                        "--configs_file",
                        type=str,
                        default=default_conv_configs,
                        help="File of configurations to test")

    parser.add_argument("-o",
                        type=str,
                        default=chip + '_' + date.today().strftime("perf.%m%d%y"),
                        help="Output file name",
                        dest="filename")
    parser.add_argument("-t",
                        "--tuning_db",
                        type=str,
                        default=argparse.SUPPRESS,
                        help="Tuning database filename")
    parser.add_argument("-qt",
                        "--quick_tuning_db",
                        type=str,
                        default=argparse.SUPPRESS,
                        help="Quick tuning database filename")

    parser.add_argument("--test_dir",
                        type=str,
                        default="../mlir/test/fusion/resnet50-e2e",
                        help="The directory of tests")
    parser.add_argument(
        "--mlir-build-dir",
        type=str,
        default=find_mlir_build_dir(),
        help="The build directory of MLIR based kernel generator",
    )
    parser.add_argument("config",
                        type=str,
                        nargs='*',
                        help="The specific config to test, if you want to test one")

    parser.add_argument("--rocmlir_gen_flags",
                        type=str,
                        default=argparse.SUPPRESS,
                        help="rocmlir-gen flags to toggle each feature")

    parser.add_argument("--external-gemm-library",
                        type=str,
                        default="hipBLASLt",
                        help="(hipBLASLt | CK) external library to run GEMM routines")

    parser.add_argument(
        '--data-type',
        nargs='+',
        choices=["f32", "f16", "i8", "i8_i32", "i8_i8", "fp8", "fp8_fp8", "fp8_f32"],
        default=["f32", "f16", "i8"],
        help='Force a set of datatypes')

    parser.add_argument(
        '--scale-type',
        nargs='+',
        choices=["f32", "f8E8M0FNU"],
        default=None,
        help=
        'Force a set of scale types for scaled GEMM (only applicable when config includes -scaledGemm)'
    )

    parser.add_argument(
        '--use-rocprof',
        action="store_true",
        help="Use rocprof instead of rocmlir-tuning-driver to collect performance data")

    parser.add_argument(
        "--flush-last-level-cache",
        action='store_true',
        default=False,
        help=
        "Size the cache-flush buffer to the architecture's last-level cache (e.g. AMD Infinity Cache) instead of the per-XCD L2 cache size reported by the HIP runtime. Defaults to the L2 cache size."
    )

    parsed_args = parser.parse_args(args)

    rocmlir_gen_flags = ''
    if 'rocmlir_gen_flags' in parsed_args:
        rocmlir_gen_flags = parsed_args.rocmlir_gen_flags

    tuning_db = None
    quick_tuning_db = None
    if 'tuning_db' in parsed_args:
        tuning_db = read_tuning_db(parsed_args.tuning_db)

    if 'quick_tuning_db' in parsed_args:
        quick_tuning_db = read_tuning_db(parsed_args.quick_tuning_db)

    # Impose default behavior when no args have been passed
    if len(args) == 0:
        parsed_args.batch_all = True

    conf_class = PerfConfiguration
    optype = Operation.from_name(parsed_args.op)
    if optype == Operation.CONV:
        conf_class = ConvConfiguration
        external_lib = None
    elif optype == Operation.GEMM:
        external_lib = GEMMLibrary.from_name(parsed_args.external_gemm_library)
        if external_lib == GEMMLibrary.CK:
            conf_class = CKGemmConfig
        elif external_lib == GEMMLibrary.HIPBLASLT:
            conf_class = HipBLASLtGemmConfig
    elif optype == Operation.ATTENTION:
        conf_class = AttentionConfiguration
        external_lib = None
    elif optype == Operation.GEMM_GEMM:
        conf_class = GemmGemmConfiguration
        external_lib = None
    elif optype == Operation.CONV_GEMM:
        conf_class = ConvGemmConfiguration
        external_lib = None

    configs_path = None if parsed_args.config else parsed_args.configs_file
    paths = create_paths(configs_path, parsed_args.mlir_build_dir)
    configs = None
    if optype == Operation.CONV:
        configs = get_conv_configurations(paths.configuration_file_path)
    elif optype == Operation.GEMM:
        datatypes, output_type_map = parse_data_types(parsed_args.data_type)
        scale_types = parsed_args.scale_type if parsed_args.scale_type else None
        configs = get_gemm_configurations(paths.configuration_file_path, datatypes, output_type_map,
                                          scale_types)
    elif optype == Operation.ATTENTION:
        configs = get_attn_configurations(paths.configuration_file_path)
    elif optype == Operation.GEMM_GEMM:
        configs = get_gemm_gemm_configurations(paths.configuration_file_path)
    elif optype == Operation.CONV_GEMM:
        configs = get_conv_gemm_configurations(paths.configuration_file_path)

    if parsed_args.external or parsed_args.batch_external or parsed_args.batch_all:
        if not found_external_tool(paths, optype, external_lib):
            raise RuntimeError(
                "External benchmark reference (MIOpen or hipBLASLt driver) needed but not found")

    if parsed_args.batch_mlir or parsed_args.batch_all:
        if not paths.mlir_paths:
            raise RuntimeError("MLIR build dir was not provided/found")

    # If no arguments are passed, then benchmark with MLIR and MIOpen
    if parsed_args.batch_all:
        # batch benchmark with MLIR and MIOpen.
        generate_performance_results(configs, conf_class, paths, arch, num_cu, num_chiplets,
                                     tuning_db, quick_tuning_db, rocmlir_gen_flags,
                                     parsed_args.use_rocprof, parsed_args.flush_last_level_cache)
    elif parsed_args.tuning:
        tune_mlir_kernels(configs, arch, num_cu, num_chiplets)
    elif optype == Operation.FUSION:
        if not parsed_args.mlir_build_dir:
            raise RuntimeError("MLIR build dir was not provided/found")
        else:
            benchmark_fusion_kernels(parsed_args.test_dir, paths, arch, num_cu, num_chiplets,
                                     tuning_db, parsed_args.use_rocprof,
                                     parsed_args.flush_last_level_cache)
    else:
        if parsed_args.batch_mlir:
            df = pd.DataFrame(
                benchmark_mlir(test_vector.split(sep=' '), conf_class, paths, arch, num_cu,
                               num_chiplets, tuning_db, rocmlir_gen_flags, parsed_args.use_rocprof,
                               parsed_args.flush_last_level_cache) for test_vector in configs)
        elif parsed_args.batch_external:
            df = pd.DataFrame(
                conf_class.benchmark_external(test_vector.split(
                    sep=' '), paths, arch, num_cu, num_chiplets) for test_vector in configs)
        elif parsed_args.external:
            df = pd.DataFrame([
                conf_class.benchmark_external(parsed_args.config, paths, arch, num_cu, num_chiplets)
            ])
        else:
            # Will only reach here with more than 1 unspecified arguments
            # These are arguments are directly passed through to benchmark_mlir
            if not parsed_args.mlir_build_dir:
                raise RuntimeError("MLIR build dir was not provided/found")
            else:
                if parsed_args.config:
                    df = pd.DataFrame([
                        benchmark_mlir(parsed_args.config, conf_class, paths, arch, num_cu,
                                       num_chiplets, tuning_db, rocmlir_gen_flags,
                                       parsed_args.use_rocprof, parsed_args.flush_last_level_cache)
                    ])
                else:
                    df = pd.DataFrame([
                        benchmark_mlir(config.split(), conf_class, paths, arch, num_cu,
                                       num_chiplets, tuning_db, rocmlir_gen_flags,
                                       parsed_args.use_rocprof, parsed_args.flush_last_level_cache)
                        for config in configs
                    ])
        df.to_csv(parsed_args.filename)
        with pd.option_context('display.precision', reportUtils.ROUND_DIGITS):
            print(df)  # for interactive consumption


if __name__ == '__main__':
    sys.exit(main())
