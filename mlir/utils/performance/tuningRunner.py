#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Automated performance tuning for rocMLIR generated kernels.

This script tunes MLIR kernels by running them with different performance configurations and selecting the best one based on execution time.

Usage examples:
    # Tune GEMM configs from a file
    python3 tuningRunner.py --op gemm -c configs/tier1-gemm-configs -o tuning_db.tsv

    # Tune a single GEMM config
    python3 tuningRunner.py --op gemm --config "-g 3 -m 1024 -k 769 -n 512 -t f32 -transA 0 -transB 0"

    # Quick-tune CONV configs from a file
    python3 tuningRunner.py --op conv -c configs/tier1-conv-configs --tuning-space quick

    # Tune GEMM configs with the adaptive search, on a short budget
    python3 tuningRunner.py --op gemm -c configs/tier1-gemm-configs --tuning-space lfbo --lfbo-effort quick

    # Use a subset of available GPUs
    python3 tuningRunner.py --op gemm -c configs/tier1-gemm-configs --gpus 2 3

    # Tune fusion ops from E2E test directory
    python3 tuningRunner.py --op fusion --test-dir ../mlir/test/fusion/resnet50-e2e

    # Pipe configs from stdin
    cat configs/tier1-gemm-configs | python3 tuningRunner.py --op gemm -c - -o tuning_db.tsv
"""

import argparse
import functools
import glob
import json
import hashlib
import logging
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import perfRunner
from perfCommonUtils import CORRECT_RESULT_RE, Operation
from perfRunner import (
    AttentionConfiguration,
    ConvConfiguration,
    ConvGemmConfiguration,
    GemmConfiguration,
    GemmGemmConfiguration,
    Paths,
    PerfConfiguration,
    SLEEP_US,
    TUNE_REP_MS,
    TUNE_WARMUP_MS,
    auto_precision_flags_att,
    canonicalize_config,
)
from tuningArgumentUtils import add_common_tuning_arguments

# Hard dependency, copied next to the scripts by ci-performance-scripts.
import amd_arch_db

# =============================================================================
# Constants
# =============================================================================

# rocmlir-gen wraps the GPU kernel in a loop when --kernel-repeats > 1. Split-K
# GEMM uses atomic_add on the output buffer; each repeat accumulates another
# full result (e.g. 10 repeats → ~10× vs reference). Verification must use 1.
VERIFY_REPEATS = 1

# Default wall-clock budget for a single verification pipeline (rocmlir-gen
# through the profiler), overridable with --verify-timeout. Generous because CPU
# verification of a large config is single-threaded and can take minutes.
DEFAULT_VERIFY_TIMEOUT_SECONDS = 600

# Compile timeouts are recoverable per-config outcomes inside
# rocmlir-tuning-driver: the driver kills that rocmlir-driver child process,
# emits N/A for that perf config, and continues tuning. A GPU run timeout is
# different: an in-process kernel may have hung and left the HIP context
# untrustworthy, so the driver exits the whole process with this distinct code.
# Must stay in sync with rock::kExitGpuTimeout in
# mlir/include/mlir/Dialect/Rock/utility/compileUtils.h.
GPU_TIMEOUT_EXIT_CODE = 3

# Status tokens that rocmlir-tuning-driver can return.
# - "N/A" means the config never ran;
# - "Discarded" means it ran but it was not inside the topK list.
NOT_APPLICABLE_STATUS = "N/A"
DISCARDED_STATUS = "Discarded"
UNMEASURED_STATUSES = frozenset({NOT_APPLICABLE_STATUS, DISCARDED_STATUS})

OUTPUT_HEADER_COLUMNS = [
    'arch', 'numCUs', 'numChiplets', 'testVector', 'perfConfig', 'TFlops', 'tuningSpace',
    'commitId', 'timestamp', 'durationSec'
]

# Coarse-budget presets selected by --two-stage from the GPU count. Tuning
# benchmarks configs in parallel across the visible GPUs, so a busier (>2 GPU)
# node has noisier per-config measurements that a slightly larger shortlist +
# more coarse iterations rank more robustly -- and, being compile-bound there,
# that costs ~nothing. On few GPUs the run is GPU-bound and measurements are
# clean, so a leaner budget is both cheaper and just as accurate.
LEAN_PRESET = {"topk": 10, "rep_iters": 200, "min_rep_iters": 32}
ROBUST_PRESET = {"topk": 16, "rep_iters": 500, "min_rep_iters": 64}

# =============================================================================
# Logging Setup
# =============================================================================

# ANSI color codes
_LOG_COLORS = {
    logging.DEBUG: '\033[36m',  # Cyan
    logging.INFO: '\033[34m',  # Blue
    logging.WARNING: '\033[33m',  # Yellow
    logging.ERROR: '\033[91m',  # Red
    logging.CRITICAL: '\033[91m',  # Red
}
_COLOR_RESET = '\033[0m'


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that uses tqdm.write() to avoid corrupting progress bars."""

    def __init__(self, use_color: bool = False):
        super().__init__()
        self.use_color = use_color

    def emit(self, record):
        try:
            msg = record.getMessage()
            levelname = record.levelname

            if self.use_color:
                color = _LOG_COLORS.get(record.levelno, '')
                prefix = f"{color}{levelname}{_COLOR_RESET}: "
            else:
                prefix = f"{levelname}: "

            indent = ' ' * 4
            lines = msg.splitlines()
            if len(lines) == 1:
                formatted = prefix + lines[0]
            else:
                formatted = prefix + lines[0] + '\n' + '\n'.join(
                    indent + line for line in lines[1:])

            tqdm.write(formatted, file=sys.stderr)
        except Exception:
            self.handleError(record)


class GpuLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that prefixes messages with GPU ID."""

    def process(self, msg, kwargs):
        gpu_id = self.extra.get('gpu_id')
        if gpu_id is not None:
            return f"[GPU {gpu_id}] {msg}", kwargs
        return msg, kwargs


def setup_logger(quiet: bool = False, verbose: bool = False) -> None:
    """Configure and return a logger for tuningRunner."""
    if quiet and verbose:
        raise ValueError("quiet and verbose are mutually exclusive")

    if quiet:
        logger.setLevel(logging.ERROR)
    elif verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.handlers.clear()
    logger.addHandler(TqdmLoggingHandler(use_color=sys.stderr.isatty()))


def get_gpu_logger(gpu_id: int) -> logging.LoggerAdapter:
    """Get a logger adapter for a specific GPU."""
    return GpuLoggerAdapter(logger, {'gpu_id': gpu_id})


# Module-level logger
logger: logging.Logger = logging.getLogger("tuningRunner")

# =============================================================================
# Configuration & Results
# =============================================================================


@dataclass(frozen=True)
class Options:
    """Configuration options for the tuning process."""
    debug: bool
    debug_quick_tune_data: bool
    tuning_space_kind: str
    lfbo_effort: str
    quiet: bool
    verbose: bool
    chip: str
    arch: str
    num_cu: int
    num_chiplets: int
    rocmlir_gen_flags: str
    verify_winning_config: bool
    verify_all_perfconfigs: bool
    output: str
    abort_on_error: bool
    retune: bool
    retry_states: frozenset
    gpu_ids: List[int]
    num_cpus: Optional[int]
    wait_for_compiles: bool
    flush_last_level_cache: bool
    timeout: Optional[int]
    verify_timeout: int
    perf_config_timeout: int
    gpu_run_timeout: int
    rep_ms: int
    warmup_ms: int
    two_stage_topk: int
    coarse_rep_iters: int
    coarse_warmup_iters: int
    coarse_warmup_floor_ms: int
    coarse_rel_sem_target: float
    coarse_chunk_iters: int
    coarse_min_rep_iters: int
    compile_only_dir: Optional[str] = None
    benchmark_artifacts_dir: Optional[str] = None
    allow_commit_mismatch: bool = False


@dataclass
class TuningResult:
    """Result of tuning a single configuration."""
    test_vector: str
    success: bool
    timed_out: bool = False
    gpu_timed_out: bool = False
    gpu_id: int = -1
    duration_seconds: float = 0.0
    timestamp: Optional[str] = None
    winning_config: Optional[str] = None
    max_tflops: Optional[float] = None
    entries: List[Dict] = field(default_factory=list)
    verify_tflops: Optional[float] = None


# =============================================================================
# Exceptions
# =============================================================================


class TuningError(Exception):
    """Raised when tuning or verification fails."""
    pass


# =============================================================================
# System Topology Discovery
# =============================================================================


@dataclass(frozen=True)
class Gpu:
    """Information about a GPU."""
    gpu_id: int
    sku: str
    numa_node: int


@dataclass(frozen=True)
class GpuTopology:
    """System GPU topology with NUMA mappings."""
    gpus: Dict[int, Gpu]  # GPU ID -> Gpu

    def get_numa_node(self, gpu_id: int) -> int:
        """Get NUMA node for a GPU."""
        return self.gpus[gpu_id].numa_node

    def select(self, gpu_ids: Optional[List[int]]) -> List[int]:
        """Resolve a user GPU selection, where None means every GPU on the system.

        Raises ValueError if an ID is not present or the selection mixes models.
        """
        if gpu_ids is None:
            gpu_ids = sorted(self.gpus)

        unknown = [gpu_id for gpu_id in gpu_ids if gpu_id not in self.gpus]
        if unknown:
            raise ValueError(f"no such GPU(s): {unknown} (available: {sorted(self.gpus)})")

        skus = {self.gpus[gpu_id].sku for gpu_id in gpu_ids}
        if len(skus) > 1:
            details = ", ".join(f"GPU {g}: {self.gpus[g].sku}" for g in gpu_ids)
            raise ValueError(f"mixed GPU models not supported. Found: {details}")

        return gpu_ids

    @staticmethod
    def discover() -> 'GpuTopology':
        """Query GPU topology using rocm-smi.

        rocm-smi reports physical device IDs regardless of environment variables (e.g., ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES).
        """
        rocm_smi = f"{perfRunner.ROCM_PATH}/bin/rocm-smi"
        # rocm-smi can take ~20s to enumerate large multi-GPU systems, so allow
        # a generous timeout to avoid spurious TimeoutExpired failures.
        output = subprocess.check_output(
            [rocm_smi, "--showproductname", "--showtoponuma", "--json"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=60)
        data = json.loads(output)

        gpus = {}
        for key, value in data.items():
            if key.startswith("card"):
                gpu_id = int(key.replace("card", ""))

                sku = value["Card SKU"]

                numa_node_str = value.get("(Topology) Numa Node")
                numa_node = int(numa_node_str) if numa_node_str is not None else 0

                gpus[gpu_id] = Gpu(gpu_id=gpu_id, sku=sku, numa_node=numa_node)

        if not gpus:
            raise RuntimeError("rocm-smi returned no GPU cards")

        return GpuTopology(gpus=gpus)


@functools.lru_cache(maxsize=1)
def get_gpu_topology() -> GpuTopology:
    """Discover the GPU topology, once per process.

    Discovery shells out to rocm-smi, which is slow on large systems and fails
    outright on a GPU-less host, so it is deferred until a phase that actually
    needs a GPU asks for it. --compile-only never does.
    """
    return GpuTopology.discover()


@dataclass(frozen=True)
class NumaTopology:
    """System NUMA topology with CPU mappings."""
    numa_to_cpus: Dict[int, List[int]]  # NUMA node -> list of CPU IDs

    def get_cpus_for_numa_node(self, numa_node: int) -> List[int]:
        """Get CPUs belonging to a NUMA node."""
        return self.numa_to_cpus[numa_node]

    @staticmethod
    def discover() -> 'NumaTopology':
        """Discover NUMA topology for CPUs.

        Returns a topology where all CPUs are on node 0 if discovery fails or system is non-NUMA.
        """
        numa_to_cpus: Dict[int, List[int]] = {}
        numa_base = "/sys/devices/system/node"

        if os.path.exists(numa_base):
            for entry in os.listdir(numa_base):
                if entry.startswith("node") and entry[4:].isdigit():
                    node_id = int(entry[4:])
                    cpulist_path = os.path.join(numa_base, entry, "cpulist")
                    with open(cpulist_path, 'r') as f:
                        numa_to_cpus[node_id] = NumaTopology._parse_cpu_list(f.read())

        # Fallback: single node with all CPUs
        if not numa_to_cpus:
            numa_to_cpus[0] = list(range(os.cpu_count() or 1))

        return NumaTopology(numa_to_cpus=numa_to_cpus)

    @staticmethod
    def _parse_cpu_list(cpu_list_str: str) -> List[int]:
        """Parse CPU list string like '0-55,112-167' into list of CPU IDs."""
        cpus = []
        for part in cpu_list_str.strip().split(','):
            if '-' in part:
                start, end = part.split('-', 1)
                cpus.extend(range(int(start), int(end) + 1))
            else:
                cpus.append(int(part))
        return cpus


# =============================================================================
# State Management
# =============================================================================


class ConfigState(Enum):
    """Possible states for a tuning configuration in the state file.

    State transitions:
        PENDING (implicit) -> RUNNING: Config starts tuning
        RUNNING -> SUCCEEDED (implicit): Tuning completes successfully (removed from state, written to output)
        RUNNING -> FAILED: Tuning completes with error
        RUNNING -> TIMED_OUT: Tuning exceeded timeout
        RUNNING -> GPU_TIMED_OUT: A perf-config's GPU run hung (driver run-timeout tripped)
        RUNNING -> INTERRUPTED: User interrupted (Ctrl+C) during tuning
        RUNNING -> CRASHED: Detected on next startup (stale RUNNING state)
        <state> -> PENDING: User requests retry with --retry <state>

    Note: PENDING and SUCCEEDED are implicit states:
        - PENDING: not in state file AND not in output file
        - SUCCEEDED: in output file (not tracked in state file)
    """
    RUNNING = "running"  # Currently being tuned
    FAILED = "failed"  # Tuning completed with error
    TIMED_OUT = "timed_out"  # Tuning exceeded timeout
    GPU_TIMED_OUT = "gpu_timed_out"  # A perf-config's GPU run hung (driver run-timeout)
    INTERRUPTED = "interrupted"  # User interrupted during tuning (Ctrl+C)
    CRASHED = "crashed"  # Process crashed while tuning (detected on startup)


# States representing unsuccessful tuning outcomes that are skipped by default
UNSUCCESSFUL_STATES = frozenset(
    {ConfigState.FAILED, ConfigState.TIMED_OUT, ConfigState.GPU_TIMED_OUT, ConfigState.CRASHED})


@dataclass
class TuningState:
    """State tracking for configs within a single context."""
    configs: Dict[str, ConfigState] = field(default_factory=dict)
    _pre_running_states: Dict[str, ConfigState] = field(default_factory=dict)

    def set_running(self, test_vector: str) -> None:
        if test_vector in self.configs:
            self._pre_running_states[test_vector] = self.configs[test_vector]
        self.configs[test_vector] = ConfigState.RUNNING

    def set_failed(self, test_vector: str) -> None:
        self.configs[test_vector] = ConfigState.FAILED
        self._pre_running_states.pop(test_vector, None)

    def set_timed_out(self, test_vector: str) -> None:
        self.configs[test_vector] = ConfigState.TIMED_OUT
        self._pre_running_states.pop(test_vector, None)

    def set_gpu_timed_out(self, test_vector: str) -> None:
        self.configs[test_vector] = ConfigState.GPU_TIMED_OUT
        self._pre_running_states.pop(test_vector, None)

    def set_interrupted(self, test_vector: str) -> None:
        self.configs[test_vector] = ConfigState.INTERRUPTED
        self._pre_running_states.pop(test_vector, None)

    def remove(self, test_vector: str) -> None:
        self.configs.pop(test_vector, None)
        self._pre_running_states.pop(test_vector, None)

    def should_skip(self, test_vector: str, retry_states: frozenset = frozenset()) -> bool:
        state = self.configs.get(test_vector)
        return state in UNSUCCESSFUL_STATES and state not in retry_states

    def is_empty(self) -> bool:
        return not self.configs

    def failed_count(self) -> int:
        return sum(1 for s in self.configs.values() if s == ConfigState.FAILED)

    def timed_out_count(self) -> int:
        return sum(1 for s in self.configs.values() if s == ConfigState.TIMED_OUT)

    def gpu_timed_out_count(self) -> int:
        return sum(1 for s in self.configs.values() if s == ConfigState.GPU_TIMED_OUT)

    def crashed_count(self) -> int:
        return sum(1 for s in self.configs.values() if s == ConfigState.CRASHED)

    def promote_running_to_interrupted(self) -> int:
        count = 0
        for tv in self.configs:
            if self.configs[tv] == ConfigState.RUNNING:
                prev_state = self._pre_running_states.pop(tv, None)
                self.configs[tv] = prev_state or ConfigState.INTERRUPTED
                count += 1
        return count


class TuningStateFile:
    """Manages multi-context tuning state in a JSON file.

    File format:
    {
        "contexts": {
            "<chip>/<num_cu>/<num_chiplets>/<tuning_space>": {
                "test_vector_1": "failed",
                "test_vector_2": "crashed"
            }
        }
    }

    If filepath is None, all operations are no-ops.
    """

    def __init__(self, filepath: Optional[str], chip: str, arch: str, num_cu: int,
                 num_chiplets: int, tuning_space: str, conf_class: type):
        self.filepath = filepath
        self.context_key = f"{chip}/{num_cu}/{num_chiplets}/{tuning_space}"
        self._arch = arch
        self._num_cu = num_cu
        self._num_chiplets = num_chiplets
        self._conf_class = conf_class
        self._lock = threading.Lock()
        self._all_contexts: Dict[str, Dict[str, str]] = {}  # context_key -> {tv -> state_str}
        self._state = TuningState()

        self._load()
        self._save_locked()  # Persist any state transitions from load

    def _load(self) -> None:
        """Load state from file.

        For the active context only:
        - INTERRUPTED configs are removed (will be retried)
        - RUNNING configs become CRASHED (stale = crash)
        - Entries that don't parse are kept verbatim so they survive a save/load round-trip
        """
        if not self.filepath or not os.path.exists(self.filepath):
            return

        with open(self.filepath, 'r') as f:
            data = json.load(f)
        self._all_contexts = data['contexts']

        # Process configs for active context with state transitions
        if self.context_key in self._all_contexts:
            for tv, state_str in self._all_contexts[self.context_key].items():
                try:
                    state = ConfigState(state_str)
                except ValueError:
                    logger.warning(f"Unknown state '{state_str}' for config '{tv}' in state file")
                    continue

                if state == ConfigState.INTERRUPTED:
                    continue  # Remove - will retry
                if state == ConfigState.RUNNING:
                    state = ConfigState.CRASHED  # Stale running = crashed

                # Canonicalize so a legacy / non-canonical key still matches
                # the canonicalized configs we tune. Keep the raw key on
                # failure so it survives a save/load round-trip.
                try:
                    canonical_tv = canonicalize_test_vector(tv, self._conf_class, self._arch,
                                                            self._num_cu, self._num_chiplets)
                except ValueError as e:
                    logger.debug(f"Failed to canonicalize config in state file: {e}")
                    canonical_tv = tv  # Keep the raw key so it survives a save/load round-trip

                self._state.configs[canonical_tv] = state

    @property
    def state(self) -> TuningState:
        return self._state

    def _save_locked(self) -> None:
        if not self.filepath:
            return

        # Update active context in all_contexts
        if not self._state.is_empty():
            self._all_contexts[self.context_key] = {
                tv: s.value for tv, s in self._state.configs.items()
            }
        else:
            self._all_contexts.pop(self.context_key, None)

        # Remove empty contexts
        self._all_contexts = {k: v for k, v in self._all_contexts.items() if v}

        # Delete file if nothing left, otherwise save
        if not self._all_contexts:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
            return

        temp_path = self.filepath + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump({'contexts': self._all_contexts}, f, indent=2)
        os.replace(temp_path, self.filepath)

    def set_running(self, test_vector: str) -> None:
        with self._lock:
            self._state.set_running(test_vector)
            self._save_locked()

    def set_failed(self, test_vector: str) -> None:
        with self._lock:
            self._state.set_failed(test_vector)
            self._save_locked()

    def set_timed_out(self, test_vector: str) -> None:
        with self._lock:
            self._state.set_timed_out(test_vector)
            self._save_locked()

    def set_gpu_timed_out(self, test_vector: str) -> None:
        with self._lock:
            self._state.set_gpu_timed_out(test_vector)
            self._save_locked()

    def set_succeeded(self, test_vector: str) -> None:
        with self._lock:
            self._state.remove(test_vector)
            self._save_locked()

    def finalize_interrupted(self) -> None:
        """Mark RUNNING configs as INTERRUPTED on clean shutdown."""
        with self._lock:
            count = self._state.promote_running_to_interrupted()
            if count > 0:
                logger.info(f"Marked {count} running config(s) as interrupted")
            self._save_locked()


def get_state_filepath(output_filepath: str) -> Optional[str]:
    """Get the state file path for a given output file."""
    if output_filepath == '-':
        return None
    return f"{output_filepath}.state"


# =============================================================================
# Tuning Infrastructure
# =============================================================================


@dataclass(frozen=True)
class TunedConfigsCache:
    """Cache for previously tuned configurations loaded from output file."""
    _results: Dict[str, TuningResult] = field(default_factory=dict)

    def contains(self, test_vector: str) -> bool:
        """Check if a test vector has already been tuned."""
        return test_vector in self._results

    def get(self, test_vector: str) -> Optional[TuningResult]:
        """Get cached result for a test vector."""
        return self._results.get(test_vector)

    def get_all_results(self) -> List[TuningResult]:
        """Get all cached tuning results."""
        return list(self._results.values())

    def count(self) -> int:
        """Return number of cached configurations."""
        return len(self._results)

    @classmethod
    def from_output_file(cls, options: Options, conf_class: type) -> 'TunedConfigsCache':
        """Load previously tuned configurations from an output TSV file.

        Format (new): # arch\tnumCUs\tnumChiplets\ttestVector\tperfConfig\tTFlops\ttuningSpace\tcommitId\ttimestamp\tdurationSec
        Format (old): # arch\tnumCUs\tnumChiplets\ttestVector\tperfConfig (tuning_space)\t[TFlops]

        Only loads entries matching current arch, num_cu, num_chiplets, and tuning space.
        """
        if options.output == '-' or not os.path.exists(options.output):
            return cls()

        results: Dict[str, TuningResult] = {}

        current_commit = get_git_commit_hash()
        warned_commits: set = set()

        header_tuning_space: Optional[str] = None
        column_indices: Dict[str, int] = {}

        with open(options.output, mode='r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Check for header line
                if cls._is_header_line(line):
                    column_indices = cls._parse_header_line(line)
                    # Extract tuning space from header for old format (perfConfig (tuning_space))
                    header_tuning_space = cls._extract_tuning_space_from_header(line)
                    continue

                # Skip comment lines
                if line.startswith('#'):
                    continue

                # Skip if we haven't seen a header yet
                if not column_indices:
                    continue

                result = cls._parse_data_line(line.split('\t'), column_indices, options, conf_class,
                                              header_tuning_space, current_commit, warned_commits)
                if not result:
                    logger.debug(f"Skipping invalid output file line: {line}")
                    continue

                results[result.test_vector] = result

        return cls(_results=results)

    @staticmethod
    def _is_header_line(line: str) -> bool:
        """Check if line is a column header."""
        header_prefix = f"# {OUTPUT_HEADER_COLUMNS[0]}\t"
        return line.startswith(header_prefix)

    @staticmethod
    def _extract_tuning_space_from_header(line: str) -> Optional[str]:
        """Extract tuning space from old format header like 'perfConfig (quick)' or 'TFlops (quick)'."""
        match = re.search(r'\((\w+)\)', line)
        return match.group(1) if match else None

    @staticmethod
    def _parse_header_line(line: str) -> Dict[str, int]:
        """Parse column header and return name -> index mapping."""
        # Strip leading '# ' if present
        header_text = line[2:] if line.startswith('# ') else line

        indices = {}
        for i, col in enumerate(header_text.split('\t')):
            if not col:
                continue
            # Extract base column name (handles 'perfConfig (tuning_space)')
            col_name = col.split()[0]
            indices[col_name] = i

        return indices

    @staticmethod
    def _parse_data_line(fields: List[str], column_indices: Dict[str, int], options: Options,
                         conf_class: type, header_tuning_space: Optional[str], current_commit: str,
                         warned_commits: set) -> Optional[TuningResult]:
        """Parse a data line and return TuningResult if valid.

        A line is valid if:
        - arch matches current system (chip or arch for backwards compatibility)
        - numCUs and numChiplets match current system
        - tuning space matches (from column or header)
        - testVector is present, parseable, and belongs to the expected operation
        - perfConfig is present and not 'None'
        """

        def get_field(name: str) -> Optional[str]:
            idx = column_indices.get(name)
            if idx is not None and idx < len(fields) and fields[idx]:
                return fields[idx]
            return None

        # Check arch match (chip or arch)
        file_arch = get_field('arch')
        if file_arch != options.chip and file_arch != options.arch:
            return None

        # Check numCUs match
        file_num_cu = get_field('numCUs')
        if file_num_cu and file_num_cu != str(options.num_cu):
            return None

        # Check numChiplets match
        file_num_chiplets = get_field('numChiplets')
        if file_num_chiplets and file_num_chiplets != str(options.num_chiplets):
            return None

        # Check tuning space match (new format has column, old format used header)
        file_tuning_space = get_field('tuningSpace') or header_tuning_space
        if file_tuning_space != options.tuning_space_kind:
            return None

        test_vector = get_field('testVector')
        if not test_vector:
            return None

        # Canonicalize so a legacy / non-canonical testVector still matches the
        # canonicalized configs we tune (and perfRunner's lookup key). Skip rows
        # we can't parse rather than poisoning the cache with a bad key.
        try:
            test_vector = canonicalize_test_vector(test_vector, conf_class, options.arch,
                                                   options.num_cu, options.num_chiplets)
        except ValueError:
            return None

        perf_config = get_field('perfConfig')
        if not perf_config or perf_config == 'None':
            return None

        # TFlops (optional)
        max_tflops = None
        tflops_str = get_field('TFlops')
        if tflops_str:
            try:
                tflops_val = float(tflops_str)
                if np.isfinite(tflops_val):
                    max_tflops = tflops_val
            except ValueError:
                pass

        # Duration (optional)
        duration_seconds = 0.0
        duration_str = get_field('durationSec')
        if duration_str:
            try:
                duration_seconds = float(duration_str)
            except ValueError:
                pass

        # Timestamp (optional)
        timestamp = get_field('timestamp')

        # Warn if commit differs (avoid spamming for same commit)
        file_commit = get_field('commitId')
        if file_commit and file_commit != current_commit and file_commit not in warned_commits:
            logger.warning(
                f"Loading tuned configs from different commit (file: {file_commit[:8]}, current: {current_commit[:8]})"
            )
            warned_commits.add(file_commit)

        return TuningResult(test_vector=test_vector,
                            success=True,
                            gpu_id=-1,
                            duration_seconds=duration_seconds,
                            timestamp=timestamp,
                            winning_config=perf_config,
                            max_tflops=max_tflops)


@dataclass
class ETATracker:
    """Track completion times for accurate ETA estimation using median of successful configs."""
    total_configs: int
    num_workers: int
    success_times: List[float] = field(default_factory=list)
    ok_count: int = 0
    fail_count: int = 0
    _processed: int = field(default=0, init=False)

    def record(self, result: TuningResult) -> None:
        self._processed += 1
        if result.success:
            self.ok_count += 1
            self.success_times.append(result.duration_seconds)
        else:
            self.fail_count += 1

    def _format_rate(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s/cfg"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m/cfg"
        else:
            return f"{seconds / 3600:.1f}h/cfg"

    def _format_eta(self, seconds: float) -> str:
        if seconds == 0:
            return "0s"
        elif seconds < 60:
            return "<1m"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}m"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days}d{hours}h"

    def get_postfix_str(self) -> str:
        remaining = self.total_configs - self._processed

        rate = "n/a"
        eta = "n/a"
        if len(self.success_times) >= 3:
            median = statistics.median(self.success_times)
            eta_seconds = (remaining / self.num_workers) * median if self.num_workers > 0 else 0
            rate = self._format_rate(median)
            eta = self._format_eta(eta_seconds)

        return f"ok={self.ok_count}, fail={self.fail_count}, rate={rate}, eta={eta}"


@dataclass
class TuningContext:
    """Encapsulates all state and configuration needed for tuning operations."""
    configs: List[str]
    conf_class: type
    paths: Paths
    options: Options
    gpu_topology: Optional[GpuTopology]
    numa_topology: NumaTopology

    _threads_per_gpu: Dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Compute optimal thread allocation after initialization."""
        self._threads_per_gpu = self._compute_thread_allocation()

    def _compute_thread_allocation(self) -> Dict[int, int]:
        """Determine how many compile threads each GPU should use based on NUMA topology."""
        # GPU-less phases size their own thread pool from --num-cpus directly.
        if not self.options.gpu_ids:
            return {}

        # Group GPUs by their NUMA node
        gpus_by_node: Dict[int, List[int]] = {}
        for gpu_id in self.options.gpu_ids:
            node = self.gpu_topology.get_numa_node(gpu_id)
            gpus_by_node.setdefault(node, []).append(gpu_id)

        # Allocate CPUs from each node proportionally to GPUs on that node
        allocation: Dict[int, int] = {}
        for node, gpus_on_node in gpus_by_node.items():
            cpus_on_node = len(self.numa_topology.get_cpus_for_numa_node(node))
            threads_each = max(1, cpus_on_node // len(gpus_on_node))
            for gpu_id in gpus_on_node:
                allocation[gpu_id] = threads_each

        # Apply user-specified CPU limit if provided
        if self.options.num_cpus is not None:
            total_allocated = sum(allocation.values())
            if self.options.num_cpus < total_allocated:
                scale_factor = self.options.num_cpus / total_allocated
                for gpu_id in allocation:
                    allocation[gpu_id] = max(1, int(allocation[gpu_id] * scale_factor))
            else:
                logger.info(
                    f"--num-cpus={self.options.num_cpus} exceeds optimal {total_allocated}, using optimal allocation"
                )

        return allocation

    def get_compile_threads(self, gpu_id: int) -> int:
        """Get the number of compile threads allocated to a GPU."""
        return self._threads_per_gpu[gpu_id]

    def print_gpu_summary(self, num_workers: Optional[int] = None) -> None:
        """Print summary of GPU allocation."""
        num_active = num_workers or len(self.options.gpu_ids)
        lines = [f"Using {num_active} GPU(s)"]
        for gpu_id in self.options.gpu_ids[:num_active]:
            node = self.gpu_topology.get_numa_node(gpu_id)
            threads = self._threads_per_gpu[gpu_id]
            lines.append(f"GPU {gpu_id}: NUMA node {node}, {threads} compile threads")
        logger.info("\n".join(lines))


class NumaNodeLock:
    """Reader-preferring reader-writer lock.

    Shared holders may run concurrently; an exclusive holder excludes all shared holders and any
    other exclusive holder. A new shared holder is admitted as long as no exclusive holder is
    currently active, even if an exclusive holder is waiting (writers may starve under sustained
    reader contention).
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._shared_count = 0
        self._writer_active = False

    def acquire_shared(self):
        with self._cond:
            while self._writer_active:
                self._cond.wait()
            self._shared_count += 1

    def release_shared(self):
        """Release a shared hold. No-op if no shared hold is currently active."""
        with self._cond:
            if self._shared_count == 0:
                return
            self._shared_count -= 1
            if self._shared_count == 0:
                self._cond.notify_all()

    def acquire_exclusive(self):
        with self._cond:
            while self._writer_active or self._shared_count > 0:
                self._cond.wait()
            self._writer_active = True

    def release_exclusive(self):
        """Release the exclusive hold. No-op if no exclusive hold is currently active."""
        with self._cond:
            if not self._writer_active:
                return
            self._writer_active = False
            self._cond.notify_all()


class GpuWorkerPool:
    """Manages assignment of GPUs to worker threads with NUMA-aware CPU affinity."""

    def __init__(self, ctx: TuningContext):
        self._ctx = ctx
        self._assignment_lock = threading.Lock()
        self._unassigned_gpus = deque(ctx.options.gpu_ids)
        self._worker_state = threading.local()
        self._numa_locks: Dict[int, NumaNodeLock] = {}
        for numa_node in ctx.numa_topology.numa_to_cpus:
            self._numa_locks[numa_node] = NumaNodeLock()

    @property
    def worker_count(self) -> int:
        """Number of parallel workers (one per GPU)."""
        return len(self._ctx.options.gpu_ids)

    def get_numa_lock(self, gpu_id: int) -> NumaNodeLock:
        numa_node = self._ctx.gpu_topology.get_numa_node(gpu_id)
        return self._numa_locks[numa_node]

    def acquire_gpu_for_thread(self) -> int:
        """Assign a GPU to the calling thread if not already assigned.

        Also pins the thread to CPUs on the GPU's NUMA node for better memory locality.
        """
        if hasattr(self._worker_state, 'assigned_gpu'):
            return self._worker_state.assigned_gpu

        with self._assignment_lock:
            if not self._unassigned_gpus:
                raise RuntimeError("No GPUs available - more workers than GPUs")
            self._worker_state.assigned_gpu = self._unassigned_gpus.popleft()

        self._apply_numa_affinity(self._worker_state.assigned_gpu)
        return self._worker_state.assigned_gpu

    def _apply_numa_affinity(self, gpu_id: int) -> None:
        """Pin current thread to CPUs on the same NUMA node as the GPU."""
        node = self._ctx.gpu_topology.get_numa_node(gpu_id)
        cpu_list = self._ctx.numa_topology.get_cpus_for_numa_node(node)

        os.sched_setaffinity(0, set(cpu_list))

        self._set_memory_policy(node)

    def _set_memory_policy(self, numa_node: int) -> None:
        """Set memory allocation policy to prefer the specified NUMA node."""
        try:
            import ctypes
            libnuma = ctypes.CDLL("libnuma.so.1", mode=ctypes.RTLD_GLOBAL)

            # MPOL_PREFERRED = 1 (prefer allocations on this node, fall back to others)
            # MPOL_BIND = 2 (strict, fail if node unavailable)
            mpol_preferred = 1

            # Create a nodemask with just our node
            nodemask = 1 << numa_node

            # int set_mempolicy(int mode, const unsigned long *nodemask, unsigned long maxnode)
            libnuma.set_mempolicy(mpol_preferred,
                                  ctypes.byref(ctypes.c_ulong(nodemask)),
                                  maxnode=64)
        except (OSError, AttributeError):
            logger.debug("libnuma not available, skipping memory policy setup")


# =============================================================================
# Output Writers
# =============================================================================


class OutputFileWriter:
    """Context manager for writing tuning results to TSV file."""

    HEADER = "# " + "\t".join(OUTPUT_HEADER_COLUMNS)

    def __init__(self, filepath: str, options: Options):
        self.filepath = filepath
        self.options = options
        self.file = None
        self._header_written = False

    def __enter__(self):
        if self.filepath == '-':
            self.file = sys.stdout
        else:
            self.file = open(self.filepath, 'a')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file and self.file != sys.stdout:
            self.file.close()

    def _write_header(self):
        print(self.HEADER, file=self.file)
        self.file.flush()
        self._header_written = True

    def write_result(self, result: TuningResult):
        if not result.success:
            raise ValueError("write_result called with unsuccessful result")
        if not result.winning_config:
            raise ValueError("write_result called without winning_config")
        if result.max_tflops is None:
            raise ValueError("write_result called without max_tflops")
        if not result.timestamp:
            raise ValueError("write_result called without timestamp")
        if result.duration_seconds <= 0.0:
            raise ValueError("write_result called with invalid duration_seconds")

        if not self._header_written:
            self._write_header()

        fields = [
            self.options.arch,
            str(self.options.num_cu),
            str(self.options.num_chiplets), result.test_vector, result.winning_config,
            str(result.max_tflops), self.options.tuning_space_kind,
            get_git_commit_hash(), result.timestamp, f"{result.duration_seconds:.1f}"
        ]

        print("\t".join(fields), file=self.file)
        self.file.flush()


class DebugFileWriter:
    """Context manager for writing debug entries to TSV file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self._header_written = False
        self._existing_columns: Optional[List[str]] = None

    def __enter__(self):
        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            with open(self.filepath, 'r') as f:
                first_line = f.readline().rstrip('\n')
            if first_line:
                self._existing_columns = first_line.split('\t')
                self._header_written = True
        self.file = open(self.filepath, 'a')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()

    def write_result(self, result: TuningResult):
        if not result.success:
            raise ValueError("write_result called with unsuccessful result")
        if not result.entries:
            raise ValueError("write_result called without entries")

        df = pd.DataFrame(result.entries)
        new_columns = list(df.columns)
        if self._existing_columns is not None and new_columns != self._existing_columns:
            raise ValueError(
                f"Debug file '{self.filepath}' has a schema that does not match the current "
                f"tuning run. Each op writes a different debug schema; please use a per-op "
                f"output path (e.g. '<base>.<op>.tsv') or remove the existing file.\n"
                f"  existing columns: {self._existing_columns}\n"
                f"  new columns:      {new_columns}")

        df.to_csv(self.file, sep='\t', header=not self._header_written, index=False)
        self.file.flush()
        self._header_written = True


# =============================================================================
# Utilities
# =============================================================================

# Signals that indicate user/system requested termination (should not be logged as failures)
TERMINATION_SIGNALS = frozenset({
    signal.SIGINT,  # Ctrl+C
    signal.SIGTERM,  # Graceful termination request
    signal.SIGHUP,  # Terminal hangup
    signal.SIGQUIT,  # Quit from keyboard
})


def raise_if_terminated(returncode: int) -> None:
    """Raise KeyboardInterrupt if returncode indicates termination."""
    if -returncode in TERMINATION_SIGNALS:
        raise KeyboardInterrupt()


class TuningArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with custom validation for tuning arguments."""

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)

        op_type = Operation.from_name(parsed.op)

        if op_type == Operation.FUSION and not parsed.test_dir:
            self.error("argument --op=fusion: requires --test-dir to be specified")

        if parsed.test_dir and op_type != Operation.FUSION:
            self.error("argument --test-dir: only allowed with --op=fusion")

        if parsed.debug_quick_tune_data and (parsed.two_stage or (parsed.two_stage_topk or 0) > 0):
            self.error(
                "argument --debug-quick-tune-data: not allowed with --two-stage/--two-stage-topk")

        # The tuning driver takes these as unsigned options, so a negative value would
        # wrap into a huge timeout instead of being rejected.
        if parsed.gpu_run_timeout < 0:
            self.error("argument --gpu-run-timeout: must be non-negative")
        if parsed.perf_config_timeout < 0:
            self.error("argument --perf-config-timeout: must be non-negative")
        if parsed.verify_timeout < 0:
            self.error("argument --verify-timeout: must be non-negative")

        # The coarse warmup is capped at what --warmup affords, so a larger
        # floor would be silently inert. The driver rejects this too.
        if (parsed.two_stage or (parsed.two_stage_topk or 0) > 0) \
                and parsed.coarse_warmup_floor_ms > parsed.warmup:
            self.error(f"argument --coarse-warmup-floor-ms: must not exceed --warmup "
                       f"({parsed.coarse_warmup_floor_ms} > {parsed.warmup})")

        target_overrides = [
            ("--target-arch", parsed.target_arch),
            ("--target-num-cu", parsed.target_num_cu),
            ("--target-num-chiplets", parsed.target_num_chiplets),
        ]

        # An override describes a machine other than this one. Outside the
        # cross-compile phases nothing checks it against the live GPU: the tuning
        # driver would benchmark a wrong-grid kernel and the result TSV would be
        # keyed on the override, so refuse rather than quietly ignore it.
        if not parsed.compile_only and not parsed.benchmark_artifacts:
            supplied = [name for name, value in target_overrides if value is not None]
            if supplied:
                self.error("argument " + ", ".join(supplied) +
                           ": only allowed with --compile-only or --benchmark-artifacts")

        # The compile phase runs on GPU-less hosts, so it never asks for the topology.
        if parsed.compile_only:
            if parsed.gpus:
                self.error("argument --gpus: not allowed with --compile-only, "
                           "which performs no GPU runs")
            # An adaptive search picks each batch from the previous batch's
            # timings, which a GPU-less compile phase can never produce.
            if parsed.tuning_space == "lfbo":
                self.error("argument --tuning-space=lfbo: not allowed with --compile-only, "
                           "which performs no GPU runs and so cannot measure the batch "
                           "that decides what to compile next")
            # HIP/rocminfo discovery and the getMinNumCU fallback are unavailable
            # here, so an unset override would bake a wrong grid into the bundle.
            missing = [name for name, value in target_overrides if value is None]
            if missing:
                self.error("argument --compile-only: requires " + ", ".join(missing) +
                           " (the compile host is GPU-less; target identity cannot be "
                           "auto-discovered)")
            parsed.gpus = []
            return parsed

        if parsed.benchmark_artifacts:
            unsupported = []
            if parsed.retune:
                unsupported.append("--retune")
            if parsed.retry:
                unsupported.append("--retry")
            if parsed.status:
                unsupported.append("--status")
            if unsupported:
                self.error("argument --benchmark-artifacts: does not support " +
                           ", ".join(unsupported))

        requested_gpus = parsed.gpus
        try:
            parsed.gpus = get_gpu_topology().select(requested_gpus)
        except ValueError as e:
            self.error(f"argument --gpus: {e}")

        if parsed.benchmark_artifacts and requested_gpus is None and parsed.gpus:
            parsed.gpus = [min(parsed.gpus)]

        if parsed.benchmark_artifacts and len(parsed.gpus) != 1:
            self.error("argument --benchmark-artifacts: requires exactly one GPU; "
                       "select it with --gpus GPU_ID")

        return parsed


class UniqueValuesAction(argparse.Action):
    """Argparse action that ensures no duplicate values."""

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) != len(set(values)):
            duplicates = [v for v in values if values.count(v) > 1]
            parser.error(
                f"argument {option_string}: duplicate values not allowed: {set(duplicates)}")
        setattr(namespace, self.dest, values)


@functools.lru_cache(maxsize=1)
def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(['git', '-C', script_dir, 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        logger.warning(f"Failed to get git commit hash for {script_dir}: {e}")
        return "unknown"


def set_isolated_gpu_env(env: Dict[str, str], gpu_id: int) -> None:
    """Modify environment to isolate subprocess to one physical GPU.

    Sets ROCR_VISIBLE_DEVICES at the HSA/ROCr level, providing complete isolation for all higher layers including HIP.
    """
    env["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("HIP_VISIBLE_DEVICES", None)  # Remove HIP_VISIBLE_DEVICES to avoid conflicts


def make_isolated_gpu_env(gpu_id: int) -> Dict[str, str]:
    """Create environment that isolates subprocess to one physical GPU."""
    env = os.environ.copy()
    set_isolated_gpu_env(env, gpu_id)
    return env


def format_error(context: str,
                 command: Optional[str] = None,
                 stdout: Optional[str] = None,
                 stderr: Optional[str] = None,
                 exit_code: Optional[int] = None,
                 gpu_id: Optional[int] = None,
                 max_lines: int = 10) -> str:
    """Format an error message with optional details."""

    def truncate(text: Optional[str]) -> Optional[str]:
        if not text or not text.strip():
            return None
        lines = text.strip().splitlines()
        if len(lines) <= max_lines:
            return text.strip()
        half = max_lines // 2
        return '\n'.join(lines[:half] + [f'... ({len(lines) - max_lines} lines omitted) ...'] +
                         lines[-half:])

    parts = [context]

    if exit_code is not None:
        parts.append(f"Exit code: {exit_code}")

    if command:
        if gpu_id is not None:
            parts.append(f"Reproduce: ROCR_VISIBLE_DEVICES={gpu_id} {command}")
        else:
            parts.append(f"Reproduce: {command}")

    truncated_stdout = truncate(stdout)
    if truncated_stdout:
        parts.append("STDOUT:\n" + truncated_stdout)

    truncated_stderr = truncate(stderr)
    if truncated_stderr:
        parts.append("STDERR:\n" + truncated_stderr)

    return '\n'.join(parts)


# =============================================================================
# Core Tuning Logic
# =============================================================================

# `auto_precision_flags_att` lives in perfRunner so the parameter sweeps
# (attentionSweeps.py) can share the same per-config heuristic. See the
# helper's docstring for the rationale behind --pv-f64.


def verify_perfconfig(perfconfig: str, config: PerfConfiguration, paths: Paths, options: Options,
                      gpu_id: int, numa_lock: NumaNodeLock) -> float:
    """Verify a performance config by running with profiling.

    Returns the execution time in nanoseconds, or raises TuningError on failure.

    Verification compares the GPU output against a CPU reference. That reference is
    single-threaded with a large working set and saturates the NUMA node's memory bandwidth if
    compile threads run alongside it, so an exclusive lock is taken on the GPU's NUMA node for the
    duration of the run. Callers must release any shared hold on this node's lock before invoking
    this function to avoid deadlocking themselves against the exclusive acquire.
    """
    gpu_logger = get_gpu_logger(gpu_id)

    config.set_perfconfig(perfconfig)

    command_line_options = config.generate_mlir_driver_commandline(options.rocmlir_gen_flags,
                                                                   kernel_repeats=VERIFY_REPEATS)
    precision_flags = auto_precision_flags_att(config)
    rocmlir_gen_command = [
        paths.mlir_paths.rocmlir_gen_path,
        '-print-verify-results=summary',
        '-pv',
        *precision_flags,
    ] + command_line_options.split()

    host_pipeline_command = [paths.mlir_paths.rocmlir_driver_path, '--host-pipeline=highlevel']
    rocmlir_driver_command = [paths.mlir_paths.rocmlir_driver_path, '-c']
    profiler_command = [perfRunner.ROCPROF] + perfRunner.get_metric_args_for_rocprof(
        options.arch) + [
            '--kernel-trace', '--stats', '--output-format=csv', '-o',
            perfRunner.BENCHMARKING_RESULT_FILE_NAME, '--', paths.mlir_paths.rocm_run_path
        ]

    verification_commands = [
        rocmlir_gen_command, host_pipeline_command, rocmlir_driver_command, profiler_command
    ]
    verification_pipeline = " | ".join(' '.join(command) for command in verification_commands)
    gpu_logger.debug(f"Verifying perfconfig '{perfconfig}'\nCommand: {verification_pipeline}")

    with tempfile.TemporaryDirectory() as tmpdir:
        env = make_isolated_gpu_env(gpu_id)
        try:
            numa_lock.acquire_exclusive()
            try:
                rc, outs, errs = _run_pipeline(verification_commands,
                                               env=env,
                                               cwd=tmpdir,
                                               timeout=options.verify_timeout)
            except subprocess.TimeoutExpired:
                raise TuningError(
                    format_error(
                        f"Verification timed out after {options.verify_timeout}s for perfconfig '{perfconfig}'",
                        command=verification_pipeline,
                        gpu_id=gpu_id))

            raise_if_terminated(rc)
            if rc != 0 or not CORRECT_RESULT_RE.search(outs):
                raise TuningError(
                    format_error(f"Verification failed for perfconfig '{perfconfig}'",
                                 command=verification_pipeline,
                                 stdout=outs,
                                 stderr=errs,
                                 exit_code=rc,
                                 gpu_id=gpu_id))

            stats_file = os.path.join(
                tmpdir,
                perfRunner.get_profiler_output_path(options.arch,
                                                    perfRunner.BENCHMARKING_STATS_FILE_NAME))
            nano_seconds = perfRunner.get_nanoseconds(stats_file)
        finally:
            numa_lock.release_exclusive()

    return nano_seconds


def find_best_perfconfig(
        tuning_output_lines: List[str], config: PerfConfiguration, paths: Paths, options: Options,
        gpu_id: int, numa_lock: NumaNodeLock) -> Tuple[Optional[str], Optional[float], List[Dict]]:
    """Parse tuning driver output and find the best performing perfconfig.

    Returns the winning config, its TFLOPS, and all entries.

    `numa_lock` is forwarded to `verify_perfconfig` when `--verify-all-perfconfigs` is enabled so
    that verification can take an exclusive hold on the NUMA node. The caller must not be holding
    any shared hold on this lock when invoking this function.
    """
    gpu_logger = get_gpu_logger(gpu_id)

    max_tflops: Optional[float] = None
    winning_config: Optional[str] = None
    entries = []

    for line in tuning_output_lines:
        result = line.strip()
        if not result:
            continue

        parts = result.split('\t')
        if len(parts) < 2:
            gpu_logger.debug(f"Skipping malformed tuning output line: '{result}'")
            continue

        perfconfig = parts[0]
        time = parts[-1]
        try:
            if time in UNMEASURED_STATUSES:
                nano_seconds = np.nan
                measurements = None
            else:
                nano_seconds = float(time)
                measurements = json.loads(parts[1]) if len(parts) == 3 else None
        except (ValueError, json.JSONDecodeError):
            gpu_logger.debug(f"Skipping malformed tuning output line: '{result}'")
            continue

        config.set_perfconfig(perfconfig)
        entry = config.table_entry(nano_seconds)
        if options.debug:
            entry["MeasurementsMs"] = measurements
            entry["Status"] = time if np.isnan(nano_seconds) else "Measured"
        entries.append(entry)

        # Verify Discarded configs too: gating this on the timing
        # instead would silently skip verification of non topK configs.
        if options.verify_all_perfconfigs and time != NOT_APPLICABLE_STATUS:
            verify_ns = verify_perfconfig(perfconfig, config, paths, options, gpu_id, numa_lock)
            if np.isnan(verify_ns):
                raise TuningError(f"Verification returned NaN for perfconfig '{perfconfig}'")

        these_tflops = entry['TFlops']
        if not np.isnan(these_tflops) and (max_tflops is None or these_tflops > max_tflops):
            max_tflops = these_tflops
            winning_config = perfconfig

    return winning_config, max_tflops, entries


def tune_config(test_vector: str, conf_class: type, paths: Paths, options: Options, gpu_id: int,
                num_compile_threads: int, numa_lock: NumaNodeLock) -> TuningResult:
    """Tune a single configuration and return the result."""
    gpu_logger = get_gpu_logger(gpu_id)

    tuning_driver_args = [
        f"--tuning-space={options.tuning_space_kind}",
        f"--rep={options.rep_ms}",
        f"--warmup={options.warmup_ms}",
        "--use-median",
        f"--sleep-us={SLEEP_US}",
        f"--show-all-measurements={options.debug}",
        f"--num-compile-threads={num_compile_threads}",
        f"--perf-config-timeout={options.perf_config_timeout}",
        f"--gpu-run-timeout={options.gpu_run_timeout}",
        f"--two-stage-topk={options.two_stage_topk}",
    ]
    if options.two_stage_topk > 0:
        tuning_driver_args += [
            f"--coarse-rep-iters={options.coarse_rep_iters}",
            f"--coarse-warmup-iters={options.coarse_warmup_iters}",
            f"--coarse-warmup-floor-ms={options.coarse_warmup_floor_ms}",
            f"--coarse-rel-sem-target={options.coarse_rel_sem_target}",
            f"--coarse-chunk-iters={options.coarse_chunk_iters}",
            f"--coarse-min-rep-iters={options.coarse_min_rep_iters}",
        ]
    if options.wait_for_compiles:
        tuning_driver_args.append("--wait-for-compiles")
    if options.flush_last_level_cache:
        tuning_driver_args.append("--flush-last-level-cache")
    # A budget only means something to the search that decides how much of its
    # space to benchmark; passing it to any other space earns a warning.
    if options.tuning_space_kind == "lfbo":
        tuning_driver_args.append(f"--lfbo-effort={options.lfbo_effort}")

    # An adaptive search is the only one whose progress is worth recording: the
    # fixed spaces hand out everything in one batch. Problems tune concurrently
    # in a driver process each, so each gets a file of its own, named the way
    # the artifact directories are.
    trace_path = lfbo_trace_path(test_vector, options)
    if trace_path:
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        tuning_driver_args.append(f"--lfbo-trace={trace_path}")

    env = make_isolated_gpu_env(gpu_id)

    config: Optional[PerfConfiguration] = None
    tuning_output: Optional[str] = None
    try:
        # Hold shared during the tuning pipeline so other workers' verification (exclusive) waits
        # until our tuning driver finishes. We release before any verification on this worker so
        # that the exclusive acquire inside verify_perfconfig does not self-deadlock.
        numa_lock.acquire_shared()
        try:
            rocmlir_gen_command = [paths.mlir_paths.rocmlir_gen_path]
            tuning_driver_command = [paths.mlir_paths.rocmlir_tuning_driver_path
                                    ] + tuning_driver_args

            if not test_vector.endswith(".mlir"):
                command_line = test_vector.split()
                config = conf_class.from_command_line(command_line, options.arch, options.num_cu,
                                                      options.num_chiplets)
                command_line_options = config.generate_mlir_driver_commandline(
                    options.rocmlir_gen_flags, kernel_repeats=None)
                # Note, we don't need the -ph, this goes to the tuning driver.
                # Because we don't set -ph, kernel_repeats is set to None.
                # This is because the kernel-repeats flag is only supported with host harness or CPU validation.
                rocmlir_gen_command += command_line_options.split()
                tuning_commands = [rocmlir_gen_command, tuning_driver_command]
            else:
                rocmlir_gen_command += ['--emit-tuning-key', test_vector]
                rc, output, err = _run_pipeline([rocmlir_gen_command], env=env)
                raise_if_terminated(rc)
                if rc != 0:
                    gpu_logger.error(
                        format_error("Failed to generate tuning key",
                                     command=' '.join(rocmlir_gen_command),
                                     stderr=err,
                                     exit_code=rc,
                                     gpu_id=gpu_id))
                    return TuningResult(test_vector=test_vector, success=False, gpu_id=gpu_id)
                result = output.strip().split('\t')
                command_line = result[2].split()
                config = conf_class.from_command_line(command_line, options.arch, options.num_cu,
                                                      options.num_chiplets)
                tuning_driver_command += [test_vector]
                tuning_commands = [tuning_driver_command]

            tuning_pipeline = " | ".join(' '.join(cmd) for cmd in tuning_commands)
            gpu_logger.debug(f"Tuning '{test_vector}'\nCommand: {tuning_pipeline}")

            try:
                rc, tuning_output, tuning_errors = _run_pipeline(tuning_commands,
                                                                 env=env,
                                                                 timeout=options.timeout)
            except subprocess.TimeoutExpired:
                gpu_logger.error(
                    format_error(f"Tuning timed out after {options.timeout}s",
                                 command=tuning_pipeline,
                                 gpu_id=gpu_id))
                return TuningResult(test_vector=test_vector,
                                    success=False,
                                    timed_out=True,
                                    gpu_id=gpu_id)

            raise_if_terminated(rc)

            if rc == GPU_TIMEOUT_EXIT_CODE:
                # A perf-config's GPU run hung and the driver's run-timeout logic
                # tore the process down (see rock::kExitGpuTimeout). Treat the
                # whole test vector as gpu-timed-out and advance to the next
                # problem config.
                gpu_logger.error(
                    format_error(
                        f"GPU run hung (exceeded --gpu-run-timeout={options.gpu_run_timeout}s)",
                        command=tuning_pipeline,
                        stderr=tuning_errors,
                        exit_code=rc,
                        gpu_id=gpu_id))
                return TuningResult(test_vector=test_vector,
                                    success=False,
                                    gpu_timed_out=True,
                                    gpu_id=gpu_id)

            if rc != 0:
                gpu_logger.error(
                    format_error("Tuning pipeline failed",
                                 command=tuning_pipeline,
                                 stdout=tuning_output,
                                 stderr=tuning_errors,
                                 exit_code=rc,
                                 gpu_id=gpu_id))
                return TuningResult(test_vector=test_vector, success=False, gpu_id=gpu_id)

            # Log any stderr output from tuning driver because it may contain warnings
            if options.verbose and tuning_errors.strip():
                gpu_logger.warning(f"rocmlir-tuning-driver stderr:\n{tuning_errors}")
        finally:
            numa_lock.release_shared()

        winning_config, max_tflops, entries = find_best_perfconfig(tuning_output.splitlines(),
                                                                   config, paths, options, gpu_id,
                                                                   numa_lock)
    except TuningError as e:
        gpu_logger.error(str(e))
        return TuningResult(test_vector=test_vector, success=False, gpu_id=gpu_id)

    if winning_config is None:
        gpu_logger.error("No valid perf config found")
        return TuningResult(test_vector=test_vector, success=False, gpu_id=gpu_id)

    verify_tflops = None
    if options.verify_winning_config or options.verify_all_perfconfigs:
        try:
            verify_ns = verify_perfconfig(winning_config, config, paths, options, gpu_id, numa_lock)
        except TuningError as e:
            gpu_logger.error(str(e))
            return TuningResult(test_vector=test_vector, success=False, gpu_id=gpu_id)

        if np.isnan(verify_ns):
            gpu_logger.error(f"Verification returned NaN for winning perfconfig '{winning_config}'")
            return TuningResult(test_vector=test_vector, success=False, gpu_id=gpu_id)

        verify_tflops = config.compute_tflops(verify_ns)

    return TuningResult(test_vector=test_vector,
                        success=True,
                        gpu_id=gpu_id,
                        winning_config=winning_config,
                        max_tflops=max_tflops,
                        entries=entries,
                        verify_tflops=verify_tflops)


def lfbo_trace_path(test_vector: str, options: Options) -> Optional[str]:
    """Where the search should record this problem's iterations, if anywhere.

    Nothing is traced unless --debug asked for detail, the search is the
    adaptive one, and there is an output file to hang the trace off of. See
    analysis/plotLFBOTrace.py for what reads it.
    """
    if not options.debug or options.tuning_space_kind != "lfbo":
        return None
    if options.output == '-':
        return None
    return os.path.join(f"{options.output}.lfbo", f"{_problem_hash(test_vector, options)}.jsonl")


def _problem_hash(test_vector: str, options: Options) -> str:
    """Stable directory key for a problem.

    Includes every input that affects the compiled artifact so resume cannot
    reuse a bundle produced with different compile settings. Test vectors
    contain filesystem-unsafe characters, hence the hash.
    """
    key = "|".join([
        options.arch,
        str(options.num_cu),
        str(options.num_chiplets), options.tuning_space_kind, options.rocmlir_gen_flags,
        test_vector.strip()
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _run_pipeline(commands: List[List[str]],
                  env: Optional[Dict[str, str]] = None,
                  timeout: Optional[int] = None,
                  cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Run a shell-style pipeline and return (returncode, stdout, stderr) as
    decoded strings.

    Thin adapter over perfRunner.run_command_pipeline (the single shared pipeline
    implementation); see that function for the spawning/teardown/deadlock
    semantics. Raises subprocess.TimeoutExpired on timeout.
    """
    rc, out, err = perfRunner.run_command_pipeline(commands, env=env, cwd=cwd, timeout=timeout)
    return rc, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def _load_index_json(root: str) -> Optional[Dict]:
    """Load index.json, returning None when it does not exist."""
    index_path = os.path.join(root, "index.json")
    if not os.path.exists(index_path):
        return None
    try:
        with open(index_path) as f:
            index = json.load(f)
    except (json.JSONDecodeError, OSError) as error:
        raise TuningError(f"cannot read artifact index {index_path}: {error}") from error
    if not isinstance(index, dict):
        raise TuningError(f"artifact index {index_path} must contain a JSON object")
    return index


def _validate_artifact_options(root: str, options: Options) -> None:
    """Require benchmark options to match the artifact bundle metadata."""
    index_path = os.path.join(root, "index.json")
    index = _load_index_json(root)
    if index is None:
        raise TuningError(f"cannot validate artifact options: {index_path} does not exist")

    fields = [
        ("arch", options.arch, "--target-arch"),
        ("numCUs", options.num_cu, "--target-num-cu"),
        ("numChiplets", options.num_chiplets, "--target-num-chiplets"),
        ("tuningSpace", options.tuning_space_kind, "--tuning-space"),
    ]
    missing = [field for field, _local_value, _option in fields if field not in index]
    if missing:
        raise TuningError(f"artifact index {index_path} is missing required "
                          f"field{'s' if len(missing) != 1 else ''}: {', '.join(missing)}")

    differences = []
    for f, local_value, option in fields:
        artifact_value = index[f]
        if artifact_value != local_value:
            differences.append(
                f"artifacts compiled for {f} {artifact_value!r} but this host resolved "
                f"{local_value!r}; pass {option}={artifact_value}")
    if differences:
        raise TuningError("\n".join(differences))


def _write_index_json(root: str, index: Dict) -> None:
    """Atomically replace index.json without exposing a partial write."""
    index_path = os.path.join(root, "index.json")
    temp_path = f"{index_path}.tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(index, f, indent=2)
            f.write("\n")
        os.replace(temp_path, index_path)
    except OSError as error:
        raise TuningError(f"cannot write artifact index {index_path}: {error}") from error
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


def _merge_index_json(root: str, options: Options, problem_hash: str, test_vector: str,
                      run_id: str) -> None:
    """Atomically update index.json, preserving entries for other problems."""
    index = _load_index_json(root) or {}
    index["arch"] = options.arch
    index["numCUs"] = options.num_cu
    index["numChiplets"] = options.num_chiplets
    index["tuningSpace"] = options.tuning_space_kind
    index["lastRunId"] = run_id
    problems = index.setdefault("problems", {})
    if not isinstance(problems, dict):
        raise TuningError(f"artifact index {os.path.join(root, 'index.json')} has invalid problems")
    problems[problem_hash] = {
        "testVector": test_vector,
        "commitId": get_git_commit_hash(),
        "runId": run_id,
    }
    _write_index_json(root, index)


def _remove_problem_from_index(root: str, problem_hash: str) -> None:
    """Remove one problem's metadata from index.json if present."""
    index = _load_index_json(root)
    if index is None:
        return

    problems = index.get("problems")
    if not isinstance(problems, dict):
        raise TuningError(f"artifact index {os.path.join(root, 'index.json')} has invalid problems")
    if problem_hash not in problems:
        return

    del problems[problem_hash]
    _write_index_json(root, index)


def _discard_problem(root: str, problem_dir: str, problem_hash: str) -> None:
    """
    Drop every trace of a problem whose compile did not finish.
    """
    for path in (problem_dir, f"{problem_dir}.tmp", f"{problem_dir}.old"):
        shutil.rmtree(path, ignore_errors=True)
    _remove_problem_from_index(root, problem_hash)


def _read_problem_commit(root: str, problem_hash: str) -> Optional[str]:
    """Compile-time commit recorded for a problem in index.json, or None if
    index.json is missing or has no entry for the problem."""
    index = _load_index_json(root)
    if index is None:
        return None
    problems = index.get("problems", {})
    if not isinstance(problems, dict):
        raise TuningError(f"artifact index {os.path.join(root, 'index.json')} has invalid problems")
    problem = problems.get(problem_hash, {})
    if not isinstance(problem, dict):
        raise TuningError(f"artifact index entry for problem {problem_hash} is invalid")
    return problem.get("commitId")


def _check_artifact_commit(root: str, problem_hash: str, options: Options) -> bool:
    """Compare the artifact's compile-time commit (index.json) to this checkout.

    Returns True if the benchmark may proceed (commits match, or mismatch was
    explicitly allowed). The compile and benchmark hosts build the tools
    separately, so a commit drift can mean an incompatible bundle format or
    grid logic; fail closed unless --allow-commit-mismatch.
    """
    artifact_commit = _read_problem_commit(root, problem_hash)
    if artifact_commit is None:
        logger.error(f"cannot read commit for problem {problem_hash} from "
                     f"{os.path.join(root, 'index.json')}")
        return options.allow_commit_mismatch

    current_commit = get_git_commit_hash()
    if artifact_commit == "unknown" or current_commit == "unknown":
        msg = (f"cannot validate build commit for problem {problem_hash}: "
               f"artifacts compiled at '{artifact_commit}' but benchmarking from "
               f"'{current_commit}'")
        if options.allow_commit_mismatch:
            logger.warning(f"{msg} (continuing due to --allow-commit-mismatch)")
            return True
        logger.error(f"{msg}; pass --allow-commit-mismatch to override")
        return False

    if artifact_commit == current_commit:
        return True

    msg = (f"build-commit mismatch for problem {problem_hash}: artifacts compiled at "
           f"'{artifact_commit}' but benchmarking from '{current_commit}'")
    if options.allow_commit_mismatch:
        logger.warning(f"{msg} (continuing due to --allow-commit-mismatch)")
        return True
    logger.error(f"{msg}; pass --allow-commit-mismatch to override")
    return False


def run_compile_only(ctx: TuningContext) -> bool:
    """--compile-only: produce the kernel bundle for each problem.

    For each problem: compile every perf config into a HSACO bundle and record
    the problem information in index.json. Verification (if requested) happens
    later on the benchmark host, so no golden reference is produced here.
    """
    options = ctx.options
    paths = ctx.paths.mlir_paths
    root = options.compile_only_dir
    problems_root = os.path.join(root, "problems")
    os.makedirs(problems_root, exist_ok=True)
    run_id = uuid.uuid4().hex

    if options.verify_winning_config or options.verify_all_perfconfigs:
        logger.warning("--compile-only performs no GPU runs; verify flags are ignored "
                       "(verification happens on the benchmark host)")

    all_ok = True
    for test_vector in ctx.configs:
        if test_vector.endswith(".mlir"):
            logger.error(f"--compile-only does not support .mlir test vectors: {test_vector}")
            all_ok = False
            if options.abort_on_error:
                return False
            continue

        problem_hash = _problem_hash(test_vector, options)
        problem_dir = os.path.join(problems_root, problem_hash)

        # Resume by default: a problem is already done when index.json records
        # the current build commit and the bundle it describes is still on
        # disk.
        artifact_commit = _read_problem_commit(root, problem_hash)
        current_commit = get_git_commit_hash()
        if not options.retune and current_commit != "unknown" and \
                artifact_commit == current_commit:
            if os.path.isfile(os.path.join(problem_dir, "manifest.json.z")):
                logger.info(f"Skipping already-compiled problem {problem_hash} "
                            "(use --retune to recompile)")
                _merge_index_json(root, options, problem_hash, test_vector, run_id)
                continue
            logger.warning(f"Recompiling problem {problem_hash}: index.json claims it was "
                           f"compiled at {current_commit} but {problem_dir} has no bundle")

        os.makedirs(problem_dir, exist_ok=True)

        command_line = test_vector.split(sep=" ")
        config = ctx.conf_class.from_command_line(command_line, options.arch, options.num_cu,
                                                  options.num_chiplets)
        gen_opts = config.generate_mlir_driver_commandline(options.rocmlir_gen_flags,
                                                           kernel_repeats=None).split()

        # 1) Kernel bundle. rocmlir-gen emits the raw kernel module (no -ph); the
        #    tuning driver reads arch/num_cu/num_chiplets off the module.
        gen_cmd = [paths.rocmlir_gen_path] + gen_opts
        td_cmd = [
            paths.rocmlir_tuning_driver_path,
            f"--compile-only={problem_dir}",
            f"--tuning-space={options.tuning_space_kind}",
            f"--perf-config-timeout={options.perf_config_timeout}",
        ]
        if options.num_cpus:
            td_cmd.append(f"--num-compile-threads={options.num_cpus}")
        pipeline = " | ".join(" ".join(cmd) for cmd in [gen_cmd, td_cmd])
        try:
            rc, _out, err = _run_pipeline([gen_cmd, td_cmd],
                                          env=os.environ.copy(),
                                          timeout=options.timeout)
        except subprocess.TimeoutExpired:
            logger.error(
                format_error(
                    f"compile-only kernel bundle timed out after {options.timeout}s "
                    f"for problem {problem_hash}",
                    command=pipeline))
            _discard_problem(root, problem_dir, problem_hash)
            all_ok = False
            if options.abort_on_error:
                return False
            continue
        except KeyboardInterrupt:
            _discard_problem(root, problem_dir, problem_hash)
            raise

        if rc != 0:
            _discard_problem(root, problem_dir, problem_hash)
            raise_if_terminated(rc)
            logger.error(f"compile-only kernel bundle failed for problem {problem_hash} "
                         f"(exit {rc}):\n{err}")
            all_ok = False
            if options.abort_on_error:
                return False
            continue

        # 2) Orchestration metadata (records this problem's compile commit).
        _merge_index_json(root, options, problem_hash, test_vector, run_id)
        logger.info(f"Compiled problem {problem_hash}")

    return all_ok


def run_benchmark_artifacts(ctx: TuningContext) -> bool:
    """--benchmark-artifacts: time shipped configs on the GPU, verify, write TSV."""
    options = ctx.options
    paths = ctx.paths
    root = options.benchmark_artifacts_dir
    gpu_id = options.gpu_ids[0]
    # This mode benchmarks one problem at a time on a single GPU, so nothing
    # else contends for the node; the lock only satisfies the verifier's
    # exclusive-hold contract.
    numa_lock = NumaNodeLock()

    # These fields are inputs to _problem_hash. Check them once up front so
    # target-option mismatches do not masquerade as missing problem directories.
    _validate_artifact_options(root, options)

    # This mode intentionally starts a fresh benchmark run. Resume/state flags
    # are rejected by TuningArgumentParser rather than being silently ignored.
    if options.output != '-':
        for output_path in (options.output, f"{options.output}.debug"):
            try:
                os.remove(output_path)
            except FileNotFoundError:
                pass

    timing_args = [
        f"--rep={options.rep_ms}",
        f"--warmup={options.warmup_ms}",
        "--use-median",
        f"--sleep-us={SLEEP_US}",
        f"--show-all-measurements={options.debug}",
        f"--gpu-run-timeout={options.gpu_run_timeout}",
    ]
    if options.flush_last_level_cache:
        timing_args.append("--flush-last-level-cache")

    debug_requested = options.debug or options.debug_quick_tune_data
    debug_enabled = debug_requested and options.output != '-'
    if debug_requested and not debug_enabled:
        logger.warning("Debug output disabled when writing to stdout")

    debug_cm = (DebugFileWriter(f"{options.output}.debug") if debug_enabled else nullcontext())
    all_ok = True
    with OutputFileWriter(options.output, options) as results_writer, \
            debug_cm as debug_writer:
        for test_vector in ctx.configs:
            if test_vector.endswith(".mlir"):
                logger.error(f"--benchmark-artifacts does not support .mlir test vectors: "
                             f"{test_vector}")
                all_ok = False
                if options.abort_on_error:
                    return False
                continue

            problem_hash = _problem_hash(test_vector, options)
            problem_dir = os.path.join(root, "problems", problem_hash)
            if not os.path.isdir(problem_dir):
                logger.error(f"no artifacts for problem {problem_hash} in {problem_dir}")
                all_ok = False
                if options.abort_on_error:
                    return False
                continue

            # Build-commit guardrail (orchestrator-owned): the artifacts were
            # compiled by tools built from index.json's per-problem commit; refuse
            # to benchmark them with a binary built from a different commit, since
            # the bundle format and grid logic can drift across commits.
            if not _check_artifact_commit(root, problem_hash, options):
                all_ok = False
                if options.abort_on_error:
                    return False
                continue

            command_line = test_vector.split(sep=" ")
            config = ctx.conf_class.from_command_line(command_line, options.arch, options.num_cu,
                                                      options.num_chiplets)

            # Benchmark. The C++ tool runs the target-identity guardrail
            # internally and emits perfConfig\t<ns|N/A>.
            td_cmd = [
                paths.mlir_paths.rocmlir_tuning_driver_path, f"--benchmark-artifacts={problem_dir}"
            ] + timing_args
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            start_time = time.time()
            try:
                rc, out, err = _run_pipeline([td_cmd],
                                             env=make_isolated_gpu_env(gpu_id),
                                             timeout=options.timeout)
            except subprocess.TimeoutExpired:
                logger.error(
                    format_error(
                        f"benchmark timed out after {options.timeout}s for problem "
                        f"{problem_hash}",
                        command=" ".join(td_cmd),
                        gpu_id=gpu_id))
                all_ok = False
                if options.abort_on_error:
                    return False
                continue
            duration = time.time() - start_time
            raise_if_terminated(rc)
            if rc == GPU_TIMEOUT_EXIT_CODE:
                logger.error(
                    format_error(
                        f"GPU run hung for problem {problem_hash} "
                        f"(exceeded --gpu-run-timeout={options.gpu_run_timeout}s)",
                        command=" ".join(td_cmd),
                        stderr=err,
                        exit_code=rc,
                        gpu_id=gpu_id))
                all_ok = False
                if options.abort_on_error:
                    return False
                continue
            if rc != 0:
                logger.error(f"benchmark failed for problem {problem_hash} (exit {rc}):\n{err}")
                all_ok = False
                if options.abort_on_error:
                    return False
                continue

            # Winner selection. Suppress find_best_perfconfig's in-process verify
            # via a copy (Options is frozen); verification is run separately
            # below when requested.
            winning_config, max_tflops, entries = find_best_perfconfig(
                out.splitlines(), config, paths, replace(options, verify_all_perfconfigs=False),
                gpu_id, numa_lock)

            if winning_config is None:
                logger.error(f"no valid perf config for problem {problem_hash}")
                all_ok = False
                if options.abort_on_error:
                    return False
                continue

            # Verification (opt-in). The CPU reference is recomputed on this host
            # via the standard verifier path; only the winning config is checked
            # unless --verify-all-perfconfigs is set.
            if options.verify_winning_config or options.verify_all_perfconfigs:
                to_verify = [winning_config]
                if options.verify_all_perfconfigs:
                    to_verify = _successful_perfconfigs(out.splitlines())
                try:
                    for pc in to_verify:
                        verify_ns = verify_perfconfig(pc, config, paths, options, gpu_id, numa_lock)
                        if np.isnan(verify_ns):
                            raise TuningError(f"Verification returned NaN for perfconfig '{pc}'")
                except TuningError as e:
                    logger.error(f"verification failed for problem {problem_hash}: {e}")
                    all_ok = False
                    if options.abort_on_error:
                        return False
                    continue

            result = TuningResult(test_vector=test_vector,
                                  success=True,
                                  winning_config=winning_config,
                                  max_tflops=max_tflops,
                                  entries=entries,
                                  timestamp=timestamp,
                                  duration_seconds=max(duration, 0.1))
            results_writer.write_result(result)
            if debug_writer:
                debug_writer.write_result(result)
            logger.info(f"Benchmarked problem {problem_hash}: winner '{winning_config}' "
                        f"({max_tflops:.1f} TFlops)")

    return all_ok


def _successful_perfconfigs(tuning_output_lines: List[str]) -> List[str]:
    """Perf configs from a benchmark stream that produced a real time."""
    result = []
    for line in tuning_output_lines:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        try:
            float(parts[-1])
        except ValueError:
            continue
        result.append(parts[0])
    return result


def tune_configs(ctx: TuningContext, status_only: bool) -> bool:
    """Tune multiple configurations in parallel across available GPUs."""
    # Load tuned configs from output file (unless --retune)
    if ctx.options.retune:
        cache = TunedConfigsCache()
    else:
        cache = TunedConfigsCache.from_output_file(ctx.options, ctx.conf_class)

    # Load state file
    state_file = TuningStateFile(get_state_filepath(ctx.options.output), ctx.options.chip,
                                 ctx.options.arch, ctx.options.num_cu, ctx.options.num_chiplets,
                                 ctx.options.tuning_space_kind, ctx.conf_class)
    state = state_file.state

    if cache.count() > 0:
        logger.info(f"Found {cache.count()} tuned config(s) in {ctx.options.output}")
    if state.crashed_count() > 0:
        logger.warning(f"Found {state.crashed_count()} crashed config(s) in state file")
    if state.timed_out_count() > 0:
        logger.warning(f"Found {state.timed_out_count()} timed out config(s) in state file")
    if state.gpu_timed_out_count() > 0:
        logger.warning(f"Found {state.gpu_timed_out_count()} gpu-timed-out config(s) in state file")
    if state.failed_count() > 0:
        logger.warning(f"Found {state.failed_count()} failed config(s) in state file")

    pending_configs = ctx.configs

    # Filter out already-tuned configs (unless --retune)
    skipped_successful = 0
    if not ctx.options.retune:
        pending_configs = [c for c in pending_configs if not cache.contains(c)]
        skipped_successful = len(ctx.configs) - len(pending_configs)

    # Filter out unsuccessful configs (unless --retry or --retune)
    skipped_unsuccessful = 0
    if not ctx.options.retune:
        before_filter = len(pending_configs)
        pending_configs = [
            c for c in pending_configs if not state.should_skip(c, ctx.options.retry_states)
        ]
        skipped_unsuccessful = before_filter - len(pending_configs)

    total_skipped = skipped_successful + skipped_unsuccessful

    if skipped_successful > 0:
        logger.info(
            f"Skipping {skipped_successful} already tuned config(s) - use '--retune' to retune")
    if skipped_unsuccessful > 0:
        logger.info(
            f"Skipping {skipped_unsuccessful} unsuccessful config(s) - use '--retry <state>' to retry"
        )

    if status_only:
        logger.info(f"{len(pending_configs)}/{len(ctx.configs)} config(s) pending tuning")
        return True

    if not pending_configs:
        logger.info("No configurations to tune")
        return True

    pool = GpuWorkerPool(ctx)
    num_workers = min(pool.worker_count, len(pending_configs))
    ctx.print_gpu_summary(num_workers=num_workers)

    # Prepare ETA tracker with historical data
    initial_times = [
        r.duration_seconds for r in cache.get_all_results() if r.duration_seconds > 0.0
    ]
    eta_tracker = ETATracker(total_configs=len(pending_configs),
                             num_workers=num_workers,
                             success_times=initial_times,
                             ok_count=skipped_successful,
                             fail_count=skipped_unsuccessful)

    has_errors = False

    debug_requested = ctx.options.debug or ctx.options.debug_quick_tune_data
    debug_enabled = debug_requested and ctx.options.output != '-'
    if debug_requested and not debug_enabled:
        logger.warning("Debug output disabled when writing to stdout")

    debug_cm = (DebugFileWriter(f"{ctx.options.output}.debug") if debug_enabled else nullcontext())
    with OutputFileWriter(ctx.options.output, ctx.options) as results_writer, \
            debug_cm as debug_writer:

        executor = None
        progress_bar = None
        try:  # No context manager for executor because we need to shutdown with wait=False
            progress_bar = tqdm(
                total=len(ctx.configs),
                initial=total_skipped,
                disable=ctx.options.quiet or ctx.options.output == '-' or not sys.stderr.isatty(),
                file=sys.stderr,
                desc=f"Tuning {ctx.conf_class.__name__} ({ctx.options.tuning_space_kind})",
                unit="config",
                leave=False,
                bar_format=
                '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [t={elapsed}{postfix}]')
            progress_bar.set_postfix_str(eta_tracker.get_postfix_str())

            def execute_tuning_task(test_vector: str) -> TuningResult:
                gpu_id = pool.acquire_gpu_for_thread()
                numa_lock = pool.get_numa_lock(gpu_id)

                state_file.set_running(test_vector)

                timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                start_time = time.time()
                compile_threads = ctx.get_compile_threads(gpu_id)
                result = tune_config(test_vector, ctx.conf_class, ctx.paths, ctx.options, gpu_id,
                                     compile_threads, numa_lock)
                result.duration_seconds = time.time() - start_time
                result.timestamp = timestamp

                if result.success:
                    state_file.set_succeeded(result.test_vector)
                elif result.timed_out:
                    state_file.set_timed_out(result.test_vector)
                elif result.gpu_timed_out:
                    state_file.set_gpu_timed_out(result.test_vector)
                else:
                    state_file.set_failed(result.test_vector)

                return result

            executor = ThreadPoolExecutor(max_workers=num_workers)
            pending_futures = {
                executor.submit(execute_tuning_task, test_vector): test_vector
                for test_vector in pending_configs
            }

            for completed_future in as_completed(pending_futures):
                result = completed_future.result()

                if result.success:
                    results_writer.write_result(result)
                    if debug_writer:
                        debug_writer.write_result(result)
                    # Per-config success line on stderr (consumed by lit smoke
                    # tests under mlir/test/perf-scripts/runtime/). Emitted
                    # regardless of --quiet, matching the pre-rewrite behaviour.
                    if result.verify_tflops is not None:
                        print(
                            f"Tuned and verified : {result.test_vector} : {result.winning_config} "
                            f"with {result.max_tflops} TFlops and {result.verify_tflops} on verification",
                            file=sys.stderr,
                            flush=True)
                    else:
                        print(
                            f"Tuned : {result.test_vector} : {result.winning_config} "
                            f"with {result.max_tflops} TFlops",
                            file=sys.stderr,
                            flush=True)
                else:
                    has_errors = True
                    logger.error(
                        f"Tuning unsuccessful for '{result.test_vector}' on GPU {result.gpu_id}")

                eta_tracker.record(result)
                progress_bar.update(1)
                progress_bar.set_postfix_str(eta_tracker.get_postfix_str())

                if has_errors and ctx.options.abort_on_error:
                    return False

        except KeyboardInterrupt:
            logger.info("Tuning interrupted by user")
            raise
        finally:
            if executor:
                executor.shutdown(wait=False, cancel_futures=True)
            if progress_bar:
                progress_bar.close()

            state_file.finalize_interrupted()

    if has_errors:
        logger.error("Encountered errors during tuning")
    else:
        logger.info("Tuning completed successfully")

    return not has_errors


# =============================================================================
# Configuration Loading
# =============================================================================


def resolve_paths(op_type: Operation, parsed_args: argparse.Namespace) -> Paths:
    """Resolve paths based on operation type and arguments."""
    if op_type == Operation.FUSION:
        configs_path = "./fusion_config_file"
    elif parsed_args.config:
        configs_path = None
    else:
        configs_path = parsed_args.configs_file
    return perfRunner.create_paths(configs_path, parsed_args.mlir_build_dir)


def extract_fusion_configs(test_dir: str,
                           paths: Paths,
                           target_chip: Optional[str] = None) -> Operation:
    """Extract tuning configurations from fusion E2E test files.

    Writes extracted configs to paths.configuration_file_path and returns the detected operation type.
    """
    all_configs = []
    op_type = Operation.FUSION

    for filename in glob.glob(test_dir + '/*mlir'):
        logger.info(f"Extracting fusion configs from: {filename}")
        test_entry = perfRunner.get_fusion_test_info(filename, paths, target_chip=target_chip)
        if not test_entry:
            continue

        test_vector = test_entry['testVector']
        if not test_vector:
            continue

        if test_vector in all_configs:
            logger.debug("Duplicate entry skipped")
            continue

        command_line = test_vector.split()
        if command_line[0].startswith('conv'):
            if op_type == Operation.FUSION:
                op_type = Operation.CONV
            elif op_type != Operation.CONV:
                logger.warning(f"Mixed operation types, skipping: {test_vector}")
                continue
        else:
            if op_type == Operation.FUSION:
                op_type = Operation.GEMM
            elif op_type != Operation.GEMM:
                logger.warning(f"Mixed operation types, skipping: {test_vector}")
                continue

        all_configs.append(test_vector)

    with open(paths.configuration_file_path, 'w') as outfile:
        for item in all_configs:
            outfile.write("%s\n" % item)

    return op_type


def get_config_class(op_type: Operation) -> type:
    """Get the configuration class for an operation type."""
    config_classes = {
        Operation.CONV: ConvConfiguration,
        Operation.GEMM: GemmConfiguration,
        Operation.ATTENTION: AttentionConfiguration,
        Operation.GEMM_GEMM: GemmGemmConfiguration,
        Operation.CONV_GEMM: ConvGemmConfiguration,
    }

    if op_type not in config_classes:
        raise ValueError(f"No config class for operation: {str(op_type)}")
    return config_classes[op_type]


def load_configs_from_stdin() -> str:
    """Read configs from stdin and return path to a temporary file."""
    content = sys.stdin.read()
    fd, path = tempfile.mkstemp(suffix='.txt', prefix='tuning_configs_')
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path


def load_configs(op_type: Operation,
                 parsed_args: argparse.Namespace,
                 paths: Paths,
                 arch: str,
                 num_cu: int,
                 num_chiplets: int,
                 target_chip: Optional[str] = None) -> List[str]:
    """Load configurations based on operation type and arguments."""
    if parsed_args.config:
        return [parsed_args.config]

    loaders = {
        Operation.CONV:
            lambda: perfRunner.get_conv_configurations(
                paths.configuration_file_path, arch, num_cu, num_chiplets, target_chip=target_chip),
        Operation.GEMM:
            lambda: perfRunner.get_gemm_configurations(paths.configuration_file_path,
                                                       arch,
                                                       num_cu,
                                                       num_chiplets,
                                                       *perfRunner.parse_data_types(parsed_args.
                                                                                    data_type),
                                                       parsed_args.scale_type,
                                                       target_chip=target_chip),
        Operation.ATTENTION:
            lambda: perfRunner.get_attn_configurations(paths.configuration_file_path, arch, num_cu,
                                                       num_chiplets),
        Operation.GEMM_GEMM:
            lambda: perfRunner.get_gemm_gemm_configurations(paths.configuration_file_path, arch,
                                                            num_cu, num_chiplets),
        Operation.CONV_GEMM:
            lambda: perfRunner.get_conv_gemm_configurations(paths.configuration_file_path, arch,
                                                            num_cu, num_chiplets),
    }

    if op_type not in loaders:
        raise ValueError(f"No config loader for operation: {str(op_type)}")

    return loaders[op_type]()


def canonicalize_test_vector(tv: str, conf_class: type, arch: str, num_cu: int,
                             num_chiplets: int) -> str:
    """Canonicalize a single test vector under ``conf_class``.

    perfRunner resolves tuning-DB entries by ``config.to_command_line()``, so the
    persisted ``testVector`` column, the dedup/state-file keys, and that lookup
    key must all be the same canonical string. ``.mlir`` test vectors are file
    paths (handled via ``--emit-tuning-key``) and pass through unchanged.

    Raises ``ValueError`` if ``conf_class`` cannot parse ``tv``.
    """
    if tv.endswith(".mlir"):
        return tv
    return canonicalize_config(tv, conf_class, arch, num_cu, num_chiplets)


def canonicalize_configs(configs: List[str], conf_class: type, arch: str, num_cu: int,
                         num_chiplets: int) -> List[str]:
    """Best-effort canonicalization of a list of input configs.

    Unparseable vectors are kept verbatim (with a warning) so they surface the
    same error later during tuning instead of being silently dropped here.
    """
    canonicalized = []
    for tv in configs:
        try:
            canonicalized.append(
                canonicalize_test_vector(tv, conf_class, arch, num_cu, num_chiplets))
        except ValueError as e:
            logger.warning(f"Could not canonicalize test vector '{tv}': {e}")
            canonicalized.append(tv)
    return canonicalized


# =============================================================================
# Entry Point
# =============================================================================


def parse_arguments(args=None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = TuningArgumentParser(
        prog="rocmlirTriton tuning runner",
        description="Automated performance tuning for rocMLIR generated kernels",
        allow_abbrev=False)
    add_common_tuning_arguments(parser)

    # Only the adaptive search chooses how much of its space to benchmark, so it
    # is the only one this can make cheaper.
    parser.add_argument("--lfbo-effort",
                        default="full",
                        choices=["quick", "full"],
                        help="How much benchmarking --tuning-space=lfbo may spend looking for a "
                        "fast config. 'quick' finishes sooner, at the risk of settling for a "
                        "slower config than the space holds. Ignored by the other spaces.")

    config_group = parser.add_mutually_exclusive_group(required=True)

    config_group.add_argument(
        "-c",
        "--configs-file",
        "--configs_file",  # for backward compatibility
        type=str,
        metavar='FILE',
        help="Path to file containing list of configurations to tune. Use '-' for stdin.")

    config_group.add_argument(
        "--config",
        type=str,
        metavar='CONFIG',
        help="Specific config to tune. Can be a config string or path to an .mlir file.")

    config_group.add_argument(
        "--test-dir",
        "--test_dir",  # for backward compatibility
        type=str,
        metavar='DIR',
        help=
        "Directory containing fusion E2E tests to extract configs from. Only used when --op=fusion."
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="tuning_results_local.tsv",
        metavar='FILE',
        help=
        "Output file path for tuning results in TSV format. Results will be appended if file exists. Use '-' for stdout."
    )

    parser.add_argument(
        "--mlir-build-dir",
        type=str,
        default=perfRunner.find_mlir_build_dir(),
        metavar='DIR',
        help=
        "Path to rocMLIR build directory containing rocmlir-gen, rocmlir-driver, rocmlir-tuning-driver, and other build artifacts",
    )

    # Cross-compilation: split tuning into a CPU-only compile phase and a
    # GPU-only benchmark phase, exchanging a compressed artifact bundle.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--compile-only",
        default=None,
        metavar='DIR',
        help="Cross-compile phase (no GPU): for each problem compile every perf "
        "config into <DIR>/problems/<hash>/, plus <DIR>/index.json. Requires "
        "--target-arch/--target-num-cu/--target-num-chiplets and a build configured "
        "with -DLLVM_ENABLE_ZSTD=FORCE_ON.")
    mode_group.add_argument(
        "--benchmark-artifacts",
        default=None,
        metavar='DIR',
        help="Benchmark phase (GPU): time the configs compiled into <DIR> by a "
        "prior --compile-only run, optionally verify (see --verify-winning-config), "
        "and write a fresh TSV, replacing any existing output. Does not support "
        "--retune, --retry, or --status. Uses the lowest detected GPU by default; "
        "--gpus must select exactly one GPU when specified. Requires a build configured with "
        "-DLLVM_ENABLE_ZSTD=FORCE_ON.")
    logging_group = parser.add_mutually_exclusive_group()

    logging_group.add_argument("-q",
                               "--quiet",
                               action='store_true',
                               default=False,
                               help="Suppress non-error output")

    logging_group.add_argument("-v",
                               "--verbose",
                               action='store_true',
                               default=False,
                               help="Enable verbose output, including commands being executed")

    parser.add_argument(
        "--tflops",
        action='store_true',
        default=False,
        help="[Deprecated, TFlops is always included] Include achieved TFLOPS in the output")

    parser.add_argument("--abort-on-error",
                        action='store_true',
                        default=False,
                        help="Abort tuning upon first error encounter")

    parser.add_argument(
        "--retune",
        action='store_true',
        default=False,
        help="Force retuning of all configs, ignoring existing results in the output file")

    parser.add_argument("--retry",
                        nargs='+',
                        choices=["failed", "timed_out", "gpu_timed_out", "crashed"],
                        default=[],
                        metavar='STATE',
                        help="Retry configs in specified states")

    parser.add_argument("--verify-timeout",
                        type=int,
                        default=DEFAULT_VERIFY_TIMEOUT_SECONDS,
                        metavar='SECONDS',
                        help="Timeout in seconds for each verification run "
                        f"(default: {DEFAULT_VERIFY_TIMEOUT_SECONDS})")

    parser.add_argument("--gpus",
                        type=int,
                        nargs='+',
                        action=UniqueValuesAction,
                        default=None,
                        metavar='GPU_ID',
                        help="GPUs to use for tuning (default: all GPUs on the system). "
                        "Not applicable to --compile-only.")

    parser.add_argument(
        "--wait-for-compiles",
        action='store_true',
        default=False,
        help=
        "Wait for all compilation tasks to complete before starting tuning. Useful for systems with shared CPU/GPU memory (e.g., APUs)."
    )

    parser.add_argument(
        "--gpu-run-timeout",
        type=int,
        default=0,
        metavar='SECONDS',
        help="Per-perf-config GPU-run timeout in seconds (default: 0 = no timeout). "
        "When > 0, the tuning driver bounds the time spent benchmarking "
        "each config; a run that exceeds this budget is presumed hung. Because a "
        "hung GPU kernel cannot be interrupted in-process, the whole test vector is "
        "marked 'gpu_timed_out' and tuning advances to the next problem config "
        "(retry with '--retry gpu_timed_out').")

    parser.add_argument("--rep",
                        type=int,
                        default=TUNE_REP_MS,
                        metavar='MS',
                        help=f"Per-config measurement time budget in milliseconds "
                        f"(default: {TUNE_REP_MS}).")

    parser.add_argument("--warmup",
                        type=int,
                        default=TUNE_WARMUP_MS,
                        metavar='MS',
                        help=f"Per-config warmup time budget in milliseconds "
                        f"(default: {TUNE_WARMUP_MS}).")

    parser.add_argument(
        "--two-stage",
        action='store_true',
        default=False,
        help="Enable two-stage tuning with top-K and coarse budgets chosen automatically "
        "for this machine."
        " The "
        "preset is selected from the GPU count: >2 GPUs uses a larger "
        "shortlist and more coarse iterations (more robust to the extra "
        "measurement noise of a busy multi-GPU node, still cheap since tuning "
        "is compile-bound); <=2 GPUs uses a cheaper budget."
        " Any of --two-stage-topk / "
        "--coarse-* passed explicitly overrides the preset.")

    parser.add_argument("--two-stage-topk",
                        type=int,
                        default=None,
                        metavar='K',
                        help="Enable two-stage tuning, allowing to select the top-K (overrides the "
                        "--two-stage preset). "
                        "Every applicable config is first benchmarked at "
                        "the cheap coarse budget (by default an adaptive, iteration-based "
                        "measurement that stops once each config's estimate is precise enough to "
                        "rank it, see --coarse-rel-sem-target; capped by --coarse-rep-iters) to "
                        "shortlist the K fastest, then re-benchmarked at the precise budget to "
                        "pick the winner. Unset by default.")

    parser.add_argument("--coarse-rep-iters",
                        type=int,
                        default=None,
                        metavar='N',
                        help="Coarse-pass measurement iteration count, used only in two-stage "
                        "mode (overrides the --two-stage preset; default when unset: 200). When "
                        "adaptive stopping is enabled "
                        "(--coarse-rel-sem-target > 0, the default) this is the maximum/cap; a "
                        "config stops earlier once its estimate is precise enough. An "
                        "iteration-based budget (rather than a millisecond budget) keeps the "
                        "coarse ranking's quality independent of GPU speed, so a value validated "
                        "on one machine transfers to others.")

    parser.add_argument(
        "--coarse-rel-sem-target",
        type=float,
        default=0.005,
        metavar='FRAC',
        help="Coarse-pass adaptive stopping target, used only when --two-stage-topk "
        "> 0 (default: 0.005 = 0.50%%). A config's coarse measurement stops once "
        "the relative standard error of the mean (s / (sqrt(N) * mean)) drops "
        "below this fraction, so quiet configs stop early and noisy ones measure "
        "longer (up to --coarse-rep-iters), automatically and independent of the "
        "GPU. Set to 0 to measure a fixed --coarse-rep-iters iterations instead.")

    parser.add_argument("--coarse-chunk-iters",
                        type=int,
                        default=32,
                        metavar='N',
                        help="Coarse-pass measurement chunk size (default: 32), used only when "
                        "--two-stage-topk > 0 and adaptive stopping is on. The relative SEM is "
                        "recomputed after each chunk of this many iterations (AdaTune-style "
                        "micro-batching).")

    parser.add_argument("--coarse-min-rep-iters",
                        type=int,
                        default=None,
                        metavar='N',
                        help="Coarse-pass minimum measured iterations before adaptive stopping "
                        "may fire, used only in two-stage mode (overrides the --two-stage preset; "
                        "default when unset: 32). Floors the sample count so the variance/SEM "
                        "estimate is trustworthy. Clamped to at most --coarse-rep-iters.")

    parser.add_argument(
        "--coarse-warmup-iters",
        type=int,
        default=50,
        metavar='N',
        help="Coarse-pass warmup iteration count, used only when --two-stage-topk > 0 "
        "(default: 50). Actual warmup is max(this, --coarse-warmup-floor-ms), capped at "
        "the warmup --warmup affords for the kernel being measured so the coarse pass "
        "never warms up longer than the precise pass it feeds.")

    parser.add_argument(
        "--coarse-warmup-floor-ms",
        type=int,
        default=5,
        metavar='MS',
        help="Minimum coarse-pass warmup time in milliseconds, used only when "
        "--two-stage-topk > 0 (default: 5). Floor layered under --coarse-warmup-iters "
        "so DVFS/clock ramp still completes on fast GPUs. Must not exceed --warmup, "
        "which caps the coarse warmup.")

    parser.add_argument("-s",
                        "--status",
                        action='store_true',
                        default=False,
                        help="Only show tuning status without performing any tuning")

    return parser.parse_args(args)


def main(args=None):
    # Capture these before set_isolated_gpu_env overwrites them
    user_rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
    user_hip_visible = os.environ.get("HIP_VISIBLE_DEVICES")

    parsed_args = parse_arguments(args)

    setup_logger(quiet=parsed_args.quiet, verbose=parsed_args.verbose)

    op_type = Operation.from_name(parsed_args.op)
    compile_only_dir = parsed_args.compile_only
    benchmark_artifacts_dir = parsed_args.benchmark_artifacts

    if not compile_only_dir:
        # We call into perfRunner which also queries GPU info using HIP and rocminfo.
        # To ensure consistency, we isolate the process to the first selected GPU.
        gpu_id = parsed_args.gpus[0] if parsed_args.gpus else min(get_gpu_topology().gpus)
        set_isolated_gpu_env(os.environ, gpu_id)

        if user_rocr_visible or user_hip_visible:
            vars_set = []
            if user_rocr_visible:
                vars_set.append(f"ROCR_VISIBLE_DEVICES={user_rocr_visible}")
            if user_hip_visible:
                vars_set.append(f"HIP_VISIBLE_DEVICES={user_hip_visible}")
            logger.warning(f"Ignoring {' and '.join(vars_set)}. This script manages GPU "
                           "visibility internally. Use '--gpus' to select specific GPUs.")

    # Target identity. TuningArgumentParser has already restricted the overrides
    # to the cross-compile phases and required all three under --compile-only, so
    # anything still unset here falls back to live discovery. Resolve it before
    # loading configs so GPU-less compile-only expansion can avoid HIP queries.
    if parsed_args.target_arch is not None:
        arch = parsed_args.target_arch
        chip_match = perfRunner.GFX_CHIP_RE.search(arch)
        if not chip_match:
            logger.error(f"--target-arch '{arch}' does not contain a gfx chip name")
            return 1
        chip = chip_match.group(0)
    else:
        arch = perfRunner.get_arch()
        chip = perfRunner.get_chip()
    num_cu = (parsed_args.target_num_cu
              if parsed_args.target_num_cu is not None else perfRunner.get_num_cu(chip))
    num_chiplets = (parsed_args.target_num_chiplets if parsed_args.target_num_chiplets is not None
                    else amd_arch_db.infer_num_chiplets(chip, num_cu))

    # Handle stdin for configs file
    stdin_temp_file = None
    if parsed_args.configs_file == '-':
        stdin_temp_file = load_configs_from_stdin()
        parsed_args.configs_file = stdin_temp_file

    try:
        paths = resolve_paths(op_type, parsed_args)
        if not paths.mlir_paths:
            logger.error("rocMLIR build dir was not provided/found")
            return 1

        if op_type == Operation.FUSION:
            op_type = extract_fusion_configs(parsed_args.test_dir, paths, target_chip=chip)

        configs = load_configs(op_type,
                               parsed_args,
                               paths,
                               arch,
                               num_cu,
                               num_chiplets,
                               target_chip=chip)
    finally:
        if stdin_temp_file:
            os.unlink(stdin_temp_file)

    conf_class = get_config_class(op_type)
    # Canonicalize configs so the persisted DB key, the dedup/state-file keys,
    # and the perfRunner ``to_command_line()`` lookup key all match.
    configs = canonicalize_configs(configs, conf_class, arch, num_cu, num_chiplets)

    # --- Resolve the two-stage coarse budget ------------------------------
    # --two-stage is the friendly toggle: the user opts in and we pick a coarse
    # budget preset from the GPU count (see LEAN_PRESET / ROBUST_PRESET).
    # Anything passed explicitly wins.
    preset = None
    if parsed_args.two_stage:
        preset = ROBUST_PRESET if len(parsed_args.gpus) > 2 else LEAN_PRESET

    def _resolve_two_stage(explicit, preset_key, hard_default):
        if explicit is not None:
            return explicit  # user passed it explicitly -> wins
        if preset is not None:
            return preset[preset_key]  # --two-stage on -> machine-picked preset
        return hard_default  # neither -> single-pass default

    resolved_two_stage_topk = _resolve_two_stage(parsed_args.two_stage_topk, "topk", 0)
    resolved_coarse_rep_iters = _resolve_two_stage(parsed_args.coarse_rep_iters, "rep_iters", 200)
    resolved_coarse_min_rep_iters = _resolve_two_stage(parsed_args.coarse_min_rep_iters,
                                                       "min_rep_iters", 32)
    if resolved_two_stage_topk > 0:
        logger.info(
            f"Two-stage tuning: top-{resolved_two_stage_topk} shortlist, coarse "
            f"budget up to {resolved_coarse_rep_iters} iters (min "
            f"{resolved_coarse_min_rep_iters}), "
            f"{'robust' if preset is ROBUST_PRESET else 'lean' if preset is LEAN_PRESET else 'custom'} "
            f"preset for {len(parsed_args.gpus)} GPU(s).")

    options = Options(chip=chip,
                      arch=arch,
                      num_cu=num_cu,
                      num_chiplets=num_chiplets,
                      debug=parsed_args.debug,
                      debug_quick_tune_data=parsed_args.debug_quick_tune_data,
                      quiet=parsed_args.quiet,
                      verbose=parsed_args.verbose,
                      tuning_space_kind=parsed_args.tuning_space,
                      lfbo_effort=parsed_args.lfbo_effort,
                      rocmlir_gen_flags=parsed_args.rocmlir_gen_flags,
                      verify_winning_config=parsed_args.verify_winning_config,
                      verify_all_perfconfigs=parsed_args.verify_all_perfconfigs,
                      output=parsed_args.output,
                      abort_on_error=parsed_args.abort_on_error,
                      retune=parsed_args.retune,
                      retry_states=frozenset(ConfigState(s) for s in parsed_args.retry),
                      gpu_ids=parsed_args.gpus,
                      num_cpus=parsed_args.num_cpus,
                      wait_for_compiles=parsed_args.wait_for_compiles,
                      flush_last_level_cache=parsed_args.flush_last_level_cache,
                      timeout=parsed_args.timeout,
                      verify_timeout=parsed_args.verify_timeout,
                      perf_config_timeout=parsed_args.perf_config_timeout,
                      gpu_run_timeout=parsed_args.gpu_run_timeout,
                      rep_ms=parsed_args.rep,
                      warmup_ms=parsed_args.warmup,
                      two_stage_topk=resolved_two_stage_topk,
                      coarse_rep_iters=resolved_coarse_rep_iters,
                      coarse_warmup_iters=parsed_args.coarse_warmup_iters,
                      coarse_warmup_floor_ms=parsed_args.coarse_warmup_floor_ms,
                      coarse_rel_sem_target=parsed_args.coarse_rel_sem_target,
                      coarse_chunk_iters=parsed_args.coarse_chunk_iters,
                      coarse_min_rep_iters=resolved_coarse_min_rep_iters,
                      compile_only_dir=compile_only_dir,
                      benchmark_artifacts_dir=benchmark_artifacts_dir,
                      allow_commit_mismatch=parsed_args.allow_commit_mismatch)

    ctx = TuningContext(configs=configs,
                        conf_class=get_config_class(op_type),
                        paths=paths,
                        options=options,
                        gpu_topology=get_gpu_topology() if parsed_args.gpus else None,
                        numa_topology=NumaTopology.discover())

    try:
        if compile_only_dir:
            succeeded = run_compile_only(ctx)
        elif benchmark_artifacts_dir:
            succeeded = run_benchmark_artifacts(ctx)
        else:
            succeeded = tune_configs(ctx, status_only=parsed_args.status)
    except KeyboardInterrupt:
        return 128 + signal.SIGINT
    except TuningError as error:
        logger.error(str(error))
        return 1

    return 0 if succeeded else 1


if __name__ == '__main__':
    sys.exit(main())
