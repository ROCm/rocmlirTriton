#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Sweep random (problem-shape, perf-config) combinations through the rocMLIR
pipeline (rocmlir-gen | rocmlir-driver -c | mlir-runner) and classify each as
PASS / NOT_APPLICABLE / FAIL. Run from the build directory.

Requires Python 3.9 or newer (uses ``asyncio.to_thread``).

Usage:
    $ ninja check-rocmlir-build-only ci-performance-scripts
    $ python3 bin/parameterSweeps.py {conv|gemm} [--samples N] [--seed S]
                                                 [--jobs J] [--debug]"""

from __future__ import annotations

import argparse
import asyncio
import enum
import itertools
import os
import random
import sys

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, List, Sequence, Optional, Tuple, TypeVar

import perfRunner
from perfCommonUtils import PERF_CONFIG_FIELD_NAMES
from perfRunner import (ConvConfiguration, Paths, get_arch, get_num_cu)

# Hard dependency, copied next to the scripts by ci-performance-scripts.
import amd_arch_db


@dataclass(frozen=True)
class Options:
    """Class for keeping option state for the parameter sweep script."""
    debug: bool
    quiet: bool
    arch: str
    concurrent_tests: int
    num_cu: int
    num_chiplets: int
    log_failures: bool
    test_timeout_sec: int
    max_timeout_rate: Optional[float]


async def _kill_process(proc: asyncio.subprocess.Process):
    if proc.returncode is None:
        proc.kill()
        try:
            await proc.wait()
        except ProcessLookupError:
            pass


async def _communicate_with_timeout(proc: asyncio.subprocess.Process,
                                    timeout_sec: int,
                                    input_data: Optional[bytes] = None):
    if timeout_sec and timeout_sec > 0:
        return await asyncio.wait_for(proc.communicate(input=input_data), timeout=timeout_sec)
    return await proc.communicate(input=input_data)


def _decode_cmd_output(data: bytes) -> str:
    """Decode subprocess stdout/stderr without failing the sweep on invalid UTF-8."""
    return data.decode('utf-8', errors='replace')


class PerfConfig:
    """Serialized perf-config for the Triton-backed rocMLIR pipeline.

    The canonical format is a ``<kind>:key=value,...`` list. Only the tunable
    fields are emitted here; the trailing knob fields are omitted and default to
    -1 on parse (see ``GemmParamsAttr::get`` in RockDialect.cpp). Two ``kind``
    values exist (see RockAttrDefs.td):

    * ``gemm`` / GemmParamsAttr (used for gemm and conv kernels), with 11
      fields:
        mPerBlock, nPerBlock, kPerBlock, kpack, numCTAs, numWaves,
        matrixInstrNonkdim, splitKFactor, numStages, wavesPerEU, gridGroupSize

    * ``attn`` / GemmGemmParamsAttr (used for attention / gemm-gemm kernels),
      with 12 fields:
        mPerBlockG0, nPerBlockG0, nPerBlockG1, kPerBlock, kpack, numCTAs,
        numWaves, matrixInstrNonkdim, splitKFactor, numStages, wavesPerEU,
        gridGroupSize
      ``nPerBlockG1`` is the second-gemm N/head tile; ``0`` means untiled.
    """

    _KEYS = PERF_CONFIG_FIELD_NAMES
    _NUM_FIELDS = {kind: len(names) for kind, names in PERF_CONFIG_FIELD_NAMES.items()}

    def __init__(self, config: Sequence[int], kind: str = 'gemm'):
        if kind not in self._NUM_FIELDS:
            raise ValueError(f"Invalid PerfConfig kind: {kind!r}")
        expected = self._NUM_FIELDS[kind]
        if len(config) != expected:
            raise ValueError(f"PerfConfig(kind={kind!r}) expects {expected} fields, "
                             f"got {len(config)}: {config!r}")
        self._config = tuple(config)
        self._kind = kind

    def __str__(self):
        body = ','.join(f'{k}={v}' for k, v in zip(self._KEYS[self._kind], self._config))
        return f'{self._kind}:{body}'


def multiline_repr(obj, num_fields=4):
    """ Returns a multi-line string representation of the given object,
    inserting a newline after every defined number of comma-separated
    fields in its repr(). Useful for making long configuration
    representations more readable in logs or debug output."""
    s = repr(obj).replace('\n', ' ')  # Flatten to one line
    lines = []
    field = ''
    fields = []
    in_quotes = False
    # Bracket depth so commas inside list/tuple/dict literals (e.g.
    # ``last_valid_kv_index=[0, 100, 88]``) don't split a field.
    bracket_depth = 0
    perf_config_str = None

    i = 0
    while i < len(s):
        # Detect start of perf_config to prevent it from being split
        if s.startswith('perf_config=', i):
            perf_config_str = s[i:]
            break
        c = s[i]
        if c == "'":
            in_quotes = not in_quotes
            field += c
        elif not in_quotes and c in '[({':
            bracket_depth += 1
            field += c
        elif not in_quotes and c in '])}':
            bracket_depth -= 1
            field += c
        elif c == ',' and not in_quotes and bracket_depth == 0:
            fields.append(field.strip() + ',')
            field = ''
        else:
            field += c
        i += 1
    if field:
        fields.append(field.strip())
    for j in range(0, len(fields), num_fields):
        prefix = '\t' if j > 0 else ''
        group = fields[j:j + num_fields]
        if j + num_fields >= len(fields) and group and group[-1].endswith(','):
            group[-1] = group[-1][:-1]
        lines.append(f"{prefix}{' '.join(group)}")
    if perf_config_str:
        lines.append('\t' + perf_config_str.strip())

    return '\n'.join(lines)


class TestResult(enum.Enum):
    PASS = 1
    # Matches the `rock.not_applicable` module attribute that rock passes set
    # when they cleanly reject a (kernel x perf-config x hw) combination.
    NOT_APPLICABLE = 2
    FAIL = 3
    TIMEOUT = 4


def _log_path(config, prefix: str) -> str:
    """Per-kind log file used by ``--log-failures``, named
    ``{prefix}_{config.SWEEP_KIND}_configs.txt``.

    Callers pass ``prefix="failing"`` for FAILs and ``prefix="timed_out"`` for
    TIMEOUTs; keeping the two streams in separate files means tolerated
    timeouts never land in the failing-configs logs that downstream tooling
    treats as bugs."""
    return f"{prefix}_{config.SWEEP_KIND}_configs.txt"


def _needs_host_highlevel(config) -> bool:
    """Configs whose -pv verifier emits ``tosa.*`` ops and therefore needs the
    rocmlir-driver ``--host-pipeline=highlevel`` pre-stage before the kernel
    pipeline's bufferizer."""
    return isinstance(config, (perfRunner.AttentionConfiguration, perfRunner.GemmGemmConfiguration))


def _build_rocmlir_gen_opts(config) -> List[str]:
    """Full rocmlir-gen argv for ``config``, including ``-pv`` and any
    per-kind flag tweaks. Used by both ``test_config`` (to actually run) and
    ``_repro_command`` (to print the failure-summary repro line) so the two
    cannot drift."""
    # generate_mlir_driver_commandline is the single source of truth for the
    # driver argv, so it already includes every optional flag a config needs.
    # Building on top of it keeps the perf-run and tuning paths in sync without
    # per-flag special-casing here.
    opts = config.generate_mlir_driver_commandline('', kernel_repeats=None).split()
    opts.append('-pv')
    # Per-config precision-aware rocmlir-gen flags (e.g. --pv-f64)
    # attached by callers such as attentionSweeps.to_attn_test to combat
    # CPU reference drift at long seq_len for f32/bf16 attention.
    extra_flags = getattr(config, "extra_rocmlir_gen_flags", None)
    if extra_flags:
        opts.extend(extra_flags)
    return opts


def _repro_command(config) -> str:
    """Command line that reproduces the given config under rocmlir-gen."""
    return ' '.join(_build_rocmlir_gen_opts(config))


def _print_failure(config,
                   cmd: Sequence[str],
                   reason: str,
                   output: str = "",
                   errors: str = "") -> None:
    """Single-source-of-truth FAIL printer (always to stderr).

    ``cmd`` is the rocmlir-gen argv list; we render it as a space-joined
    shell command so the user can copy-paste it directly."""
    msg = [f"FAIL: {reason}", f"Config = {config!r}", f"Command line = {' '.join(cmd)}"]
    if output:
        msg.append(f"Output = {output}")
    if errors:
        msg.append(f"Errors = {errors}")
    print("\n".join(msg), file=sys.stderr)


def _print_timeout(config, cmd: Sequence[str], reason: str, debug: bool) -> None:
    """Single-source-of-truth TIMEOUT printer (to stderr, only when ``debug``).

    Distinct from ``_print_failure`` so timeouts, which are tolerated up to
    ``Options.max_timeout_rate``, are never mistaken for FAILs in logs or by
    downstream log scrapers grepping for ``^FAIL``."""
    if not debug:
        return

    print("\n".join([
        f"TIMEOUT: {reason}",
        f"Config = {config!r}",
        f"Command line = {' '.join(cmd)}",
    ]),
          file=sys.stderr)


def _positive_int(s: str) -> int:
    """argparse type for `--samples`: must parse to an int > 0."""
    n = int(s)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {s!r}")
    return n


async def test_config(config, options: Options, paths: Paths) -> TestResult:
    """Runs the given configuration through rocmlir-gen | (optional
    rocmlir-driver --host-pipeline=highlevel) | rocmlir-driver -c |
    mlir-runner and returns ``PASS`` (correct), ``NOT_APPLICABLE`` (rejected
    upstream as inapplicable), or ``FAIL`` (lowering bug, runner crash, or
    incorrect numerical result).

    The optional ``--host-pipeline=highlevel`` pre-stage is needed for
    attention because rocmlir-gen ``-pv`` emits a CPU verifier as a chain of
    ``tosa.*`` ops; OneShotBufferize in the kernel pipeline can't bufferize
    those, so they must be lowered (tosa->linalg etc.) first. Conv/gemm
    verifiers use plain memref/loops and don't need it."""
    rocmlir_gen_opts = _build_rocmlir_gen_opts(config)
    needs_host_highlevel = _needs_host_highlevel(config)

    # rocmlir-driver classifies its failures via the `rock.not_applicable`
    # marker on the module: exit code 2 means the lowering pipeline cleanly
    # rejected the (kernel x perf-config x hw) combination as structurally
    # inapplicable (e.g. LDS-too-big in ResolveKernelLaunchParams) and exit
    # code 1 (or any other non-zero) means a real lowering bug. We mirror
    # that classification below: 2 -> NOT_APPLICABLE, anything else
    # non-zero -> FAIL.
    timeout = options.test_timeout_sec
    active: List[asyncio.subprocess.Process] = []

    async def _kill_active():
        for proc in reversed(active):
            await _kill_process(proc)

    try:
        generator = await asyncio.create_subprocess_exec(paths.mlir_paths.rocmlir_gen_path,
                                                         *rocmlir_gen_opts,
                                                         stdout=asyncio.subprocess.PIPE,
                                                         stderr=asyncio.subprocess.PIPE,
                                                         stdin=asyncio.subprocess.DEVNULL)
        active.append(generator)
        try:
            lowering_in, gen_errs = await _communicate_with_timeout(generator, timeout)
        except asyncio.TimeoutError:
            await _kill_process(generator)
            _print_timeout(config, rocmlir_gen_opts, f"Timeout in rocmlir-gen stage ({timeout}s)",
                           options.debug)
            return TestResult.TIMEOUT

        if generator.returncode != 0:
            gen_err_text = _decode_cmd_output(gen_errs)
            _print_failure(
                config,
                rocmlir_gen_opts,
                f"rocmlir-gen failed (exit {generator.returncode})",
                errors=gen_err_text,
            )
            return TestResult.FAIL

        if needs_host_highlevel:
            host_lowering = await asyncio.create_subprocess_exec(
                paths.mlir_paths.rocmlir_driver_path,
                '--host-pipeline=highlevel',
                '-',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            active.append(host_lowering)
            try:
                lowering_in, host_errs = await _communicate_with_timeout(host_lowering,
                                                                         timeout,
                                                                         input_data=lowering_in)
            except asyncio.TimeoutError:
                await _kill_process(host_lowering)
                _print_timeout(config, rocmlir_gen_opts,
                               f"Timeout in --host-pipeline=highlevel stage ({timeout}s)",
                               options.debug)
                return TestResult.TIMEOUT
            if host_lowering.returncode == 2:
                if options.debug:
                    print("\n".join([
                        "Host lowering rejected config as not applicable",
                        f"Command line = {' '.join(rocmlir_gen_opts)}",
                        f"Errors = {_decode_cmd_output(host_errs)}",
                    ]))
                return TestResult.NOT_APPLICABLE
            if host_lowering.returncode != 0:
                _print_failure(config,
                               rocmlir_gen_opts,
                               f"Host lowering failed (exit {host_lowering.returncode})",
                               errors=_decode_cmd_output(host_errs))
                return TestResult.FAIL

        # Pipe is created here (right before its only use) so early-return paths
        # above don't have to remember to close it.
        runner_from_lowering, lowering_to_runner = os.pipe()
        lowering = await asyncio.create_subprocess_exec(paths.mlir_paths.rocmlir_driver_path,
                                                        '-c',
                                                        '-',
                                                        stdin=asyncio.subprocess.PIPE,
                                                        stdout=lowering_to_runner,
                                                        stderr=asyncio.subprocess.PIPE)
        os.close(lowering_to_runner)
        active.append(lowering)

        runner = await asyncio.create_subprocess_exec(paths.mlir_paths.rocm_run_path,
                                                      stdin=runner_from_lowering,
                                                      stdout=asyncio.subprocess.PIPE,
                                                      stderr=asyncio.subprocess.PIPE)
        os.close(runner_from_lowering)
        active.append(runner)

        try:
            _, lowering_errs = await _communicate_with_timeout(lowering,
                                                               timeout,
                                                               input_data=lowering_in)
        except asyncio.TimeoutError:
            await _kill_process(lowering)
            await _kill_process(runner)
            _print_timeout(config, rocmlir_gen_opts,
                           f"Timeout in rocmlir-driver stage ({timeout}s)", options.debug)
            return TestResult.TIMEOUT
        try:
            runner_out, runner_errs = await _communicate_with_timeout(runner, timeout)
        except asyncio.TimeoutError:
            await _kill_process(runner)
            _print_timeout(config, rocmlir_gen_opts, f"Timeout in mlir-runner stage ({timeout}s)",
                           options.debug)
            return TestResult.TIMEOUT
        runner_out = _decode_cmd_output(runner_out)

        # Exit code 2 from rocmlir-driver = `rock.not_applicable` marker was set,
        # i.e. a rock pass cleanly rejected the config. Any other non-zero exit
        # is a real lowering bug (this includes SIGSEGV / -11 in the driver,
        # which is itself a bug worth surfacing).
        if lowering.returncode == 2:
            if options.debug:
                print("\n".join([
                    "Lowering rejected config as not applicable",
                    f"Command line = {' '.join(rocmlir_gen_opts)}",
                    f"Errors = {_decode_cmd_output(lowering_errs)}",
                ]))
            return TestResult.NOT_APPLICABLE

        if lowering.returncode != 0:
            _print_failure(config,
                           rocmlir_gen_opts,
                           f"Lowering failed (exit {lowering.returncode})",
                           errors=_decode_cmd_output(lowering_errs))
            return TestResult.FAIL

        if runner.returncode != 0:
            _print_failure(config,
                           rocmlir_gen_opts,
                           f"Runner failed (exit {runner.returncode})",
                           output=runner_out,
                           errors=_decode_cmd_output(runner_errs))
            return TestResult.FAIL

        output_lines = [line.strip() for line in runner_out.splitlines() if len(line.strip()) > 0]
        expected_output = "[1 1 1]"
        # ``all([])`` is True in Python; empty stdout must not count as PASS.
        if not output_lines:
            _print_failure(config,
                           rocmlir_gen_opts,
                           "Runner produced no verifier output",
                           output=runner_out,
                           errors=_decode_cmd_output(runner_errs))
            return TestResult.FAIL
        if not all(line == expected_output for line in output_lines):
            _print_failure(config,
                           rocmlir_gen_opts,
                           "Runner returned incorrect result",
                           output=runner_out,
                           errors=_decode_cmd_output(runner_errs))
            return TestResult.FAIL
        return TestResult.PASS
    except asyncio.CancelledError:
        await _kill_active()
        raise


IterType = TypeVar('IterType')


def grouper(iterable: Iterable[IterType], n: int):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


async def drop_good_config(config: perfRunner.PerfConfiguration, options: Options,
                           paths: Paths) -> Tuple[TestResult, perfRunner.PerfConfiguration]:
    """Run the given config and return ``(result, config)``. When
    ``--log-failures`` is set, FAILs are appended to the per-kind failure log
    and TIMEOUTs to the (separate) per-kind timeout log."""
    result = await test_config(config, options, paths)
    if not options.quiet:
        # Single print() so concurrent jobs don't interleave the separator
        # and the result line.
        print("-" * 100 + f"\n{result.name}: {multiline_repr(config)}")
    if options.log_failures and result in (TestResult.FAIL, TestResult.TIMEOUT):
        # Push blocking I/O off the asyncio loop. Concurrent writes to the
        # same path are still safe because POSIX `O_APPEND` makes each
        # `write()` atomic up to PIPE_BUF. Timeouts go to their own log so
        # tolerated compile-time blowups never pollute the failing-configs
        # files that downstream tooling treats as bugs.
        prefix = "failing" if result == TestResult.FAIL else "timed_out"
        await asyncio.to_thread(_append_failure, _log_path(config, prefix), config)
    return (result, config)


def _append_failure(log_path: str, config) -> None:
    """Append one config (failing or timed-out) to ``log_path``.

    Each entry is a ``# ``-prefixed multiline config repr followed by the
    rocmlir-gen argv (no binary prefix). Strip the comment lines and prepend
    your rocmlir-gen path to rerun, e.g.::

        grep -v '^#' log_path | grep -v '^$' \\
            | xargs -L1 bin/rocmlir-gen
    """
    config_lines = multiline_repr(config).splitlines()
    block = "\n".join(f"# {line}" for line in config_lines)
    with open(log_path, "a") as f:
        f.write(f"{block}\n{_repro_command(config)}\n\n")


async def sweep_parameters(
    param_iter: Iterable[IterType], to_config: Callable[[IterType, Options],
                                                        perfRunner.PerfConfiguration],
    options: Options, paths: Paths
) -> Tuple[int, int, List[perfRunner.PerfConfiguration], List[perfRunner.PerfConfiguration]]:
    failing_configs: List[perfRunner.PerfConfiguration] = []
    timed_out_configs: List[perfRunner.PerfConfiguration] = []
    passed = 0
    not_applicable = 0
    configs = (to_config(p, options) for p in param_iter)
    for chunk in grouper((drop_good_config(c, options, paths) for c in configs),
                         options.concurrent_tests):
        tasks = [asyncio.create_task(coro) for coro in chunk]
        try:
            configs_results = await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for result, config in configs_results:
            if result == TestResult.PASS:
                passed += 1
            elif result == TestResult.NOT_APPLICABLE:
                not_applicable += 1
            elif result == TestResult.TIMEOUT:
                timed_out_configs.append(config)
            else:
                failing_configs.append(config)

    return (passed, not_applicable, timed_out_configs, failing_configs)


# Sweep spaces. We deliberately go wider than the production tuning space in
# mlir/lib/Dialect/Rock/Tuning/RockTuningImpl.cpp so this script can find bugs
# in combinations the heuristic would never pick. The driver rejects combos
# that violate the kernel's applicability constraints (tile-too-large for
# LDS, etc.) — those are reported as NOT_APPLICABLE, not FAIL. A 0 in
# matrixInstrNonkdim / waves_per_eu / grid_group_size means "let the
# heuristic pick".
# The non-power-of-two m/n_per_block entries (48, 80, 96, 160, 192) are
# deliberately included to exercise the rock-decompose-nonpow2-tiles pass,
# which splits a blockwise GEMM tile with non-pow2 M and/or N into a grid of
# power-of-two sub-tiles (e.g. 80 -> 64 + 16, 96 -> 64 + 32).
# The non-power-of-two k_per_block entries (48, 80, 96, 112, 160, 192) are
# likewise included to exercise the non-pow2 k_per_block.
PERF_CONFIG_OPTIONS = {
    'm_per_block': [16, 32, 48, 64, 80, 96, 128, 160, 192, 256],
    'n_per_block': [16, 32, 48, 64, 80, 96, 128, 160, 192, 256],
    'k_per_block': [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 512],
    # `kpack` is sampled via _kpack_choices(arch); see below. `kpack != 1` is
    # deprecated on gfx950 and gfx1250 (and newer); older archs still take
    # {1, 2}.
    # numCTAs is hardcoded to 1 in RockTuningImpl::createGemmTuningRangeBF
    # (TODO(roctriton): numCTAs for gfx1250).
    'num_ctas': [1],
    # The C++ "Quick" range is {2, 4, 8}; Exhaustive sweeps powers of 2 up to
    # maxHardwareWorkgroupSize / waveSize. We pick a slightly wider range.
    'num_waves': [1, 2, 4, 8, 16],
    # MFMA: {16, 32}. WMMA: {0}. Union exercises both paths.
    'matrix_instr_nonkdim': [0, 16, 32],
    # The driver heuristic only emits {1} or {1, 3, 4}; we sweep more to
    # exercise the splitK lowering path.
    'split_k_factor': [1, 2, 3, 4],
    'num_stages': [1, 2, 3],
    # The C++ tuner pins these at 0 ("use heuristic"); the commented-out code
    # in getRangeGemm shows the intended sweep range. 0 is kept so the
    # heuristic path is also exercised.
    'waves_per_eu': [0, 1, 2, 4, 8],
    'grid_group_size': [0, 1, 2, 4, 8],
    # attn-only second-gemm N/head tile (the attn nPerBlockG1 field). These
    # are the non-zero (tiled) options; the untiled case (0) is sampled
    # separately (see sample_perf_config). All powers of two so the chunk count
    # (gemm1N / nPerBlockG1, after gemm1N is padded to a power of two in
    # AttnToGridwise.cpp) stays a power of two -- the invariant
    # GridwiseAttnToBlockwise.cpp's tree-concat relies on.
    'n_per_block_g1': [16, 32, 64, 128, 256],
}

# Conv problem-shape sweep. Sizes are a mix of common CNN shapes (e.g. 224, 56,
# 28) and small/odd ones to hit padding/edge paths.
CONV_SHAPE_OPTIONS = {
    'op': ['fwd', 'bwd', 'wrw'],
    'layout': ['NCHW', 'NHWC'],
    'dtype': ['f32', 'f16', 'bf16', 'i8', 'fp8'],
    'n': [1, 2, 4, 8, 16],
    'c': [1, 3, 8, 16, 32, 64, 128, 256, 512],
    'k': [1, 8, 16, 32, 64, 128, 256, 512],
    'hi': [4, 8, 14, 16, 28, 32, 56, 64, 112, 224],
    'wi': [4, 8, 14, 16, 28, 32, 56, 64, 112, 224],
    'y': [1, 2, 3, 5, 7],
    'x': [1, 2, 3, 5, 7],
    'conv_stride': [1, 2, 3],
    'padding': [0, 1, 2, 3],
    'dilation': [1, 2],
    'group': [1, 2, 4],
}

# GEMM problem-shape sweep.
MAX_GEMM_DIM = 512
GEMM_SHAPE_OPTIONS = {
    'dtype': ['f32', 'f16', 'bf16', 'i8', 'fp8'],
    'g': [1, 2],
    'trans_a': [False, True],
    'trans_b': [False, True],
    'trans_o': [False, True],
}


def _arch_id(arch: str) -> Optional[int]:
    """Return the integer encoding of ``arch`` (parsed as hex, so 'gfx950'
    -> 0x950, 'gfx90a' -> 0x90a, 'gfx1250' -> 0x1250) for ordered
    comparisons. Returns ``None`` if no ``gfxNNN`` token is found or the
    digits don't parse as hex.

    Reuses ``perfRunner.GFX_CHIP_RE`` (the project's chip-token primitive)
    so it transparently accepts every arch-string form already in use:
    bare 'gfx950', HIP gcnArchName 'gfx950:sramecc+:xnack-' (from
    ``get_arch()``), LLVM-triple 'amdgcn-amd-amdhsa:gfx950[:...]', and the
    rocminfo double-dash 'amdgcn-amd-amdhsa--gfx950:...' form parsed by
    Jenkinsfile."""
    if not arch:
        return None
    m = perfRunner.GFX_CHIP_RE.search(arch)
    if not m:
        return None
    try:
        return int(m.group(0)[len('gfx'):], 16)
    except ValueError:
        return None


def _kpack_choices(arch: str) -> List[int]:
    """Valid ``kpack`` values for the perf-config sweep, by arch.

    Sourced from ``rock::getMaxKpack`` via the AmdArchDB pybind module."""
    return list(range(1, amd_arch_db.get_max_kpack(arch) + 1))


# Dtypes whose Triton fp_to_fp lowering expands into many LLVM ops on AMD
# targets that lack a packed hardware conversion.
_AMPLIFIED_DTYPES = frozenset({'fp8', 'fp8_fp8', 'bf8'})


# Returns the amplifier value for the given data type and arch.
# - Amplifier=10 if fp8 and RDNA3 or RDNA4
# - Amplifier=0 otherwise
def _dtype_amplifier(dtype: str, arch: str) -> int:
    """Multiplier on the number of elements held per thread for dtypes whose
    Triton fp_to_fp lowering expands into many LLVM ops.

    On RDNA3 (gfx11xx) and RDNA4 (gfx12xx) the fp8->f16 path lowers to ~25
    scalar LLVM ops per element because there is no packed hardware
    conversion intrinsic available. CDNA3 keeps the conversion packed via
    v_cvt_pk_f32_fp8, and CDNA4 (gfx950) doesn't need it at all because
    tt.dot_scaled accepts fp8 operands natively, so they aren't amplified.
    Any other AMD target (gfx9 pre-940, gfx10, gfx13+, future families)
    defaults to alpha=1 until measured."""
    if dtype not in _AMPLIFIED_DTYPES:
        return 1
    n = _arch_id(arch)
    if n is None:
        return 1
    is_rdna3_or_4 = 0x1100 <= n < 0x1250  # gfx11xx, gfx12xx
    return 10 if is_rdna3_or_4 else 1


# Needed because IR explosion issue in Triton
# https://github.com/ROCm/triton/issues/940
# Remove this compile cost cap heuristic if the issue is fixed.
#
# AMDGPU codegen, in particular the PostRA machine instruction scheduler,
# scales with the instruction count *per basic block*, not in the total
# across the kernel. A GEMM kernel has two basic blocks that matter for cost,
# each measured in number of elements held per thread:
#
#   1. The K-loop body. Each iteration loads a tile slice of A and B into
#      registers, contributing
#          num_elements_kloop_body = (MPB + NPB) * KPB / (threads * kpack)
#      elements per thread, where ``threads = numWaves * waveSize``.
#
#   2. The C-tile epilogue. The accumulator is held per-thread at
#          num_elements_c_epilogue = (MPB * NPB) / threads
#      elements per thread, and is stored back in one shot.
#
# The scheduler's cost is roughly the *larger* of these, not their sum:
# halving KPB shortens the K-loop body but leaves the C-epilogue unchanged,
# and vice versa. So we cap on
#     max(num_elements_kloop_body, num_elements_c_epilogue) * alpha
# where ``alpha`` is a dtype amplifier: fp8 on RDNA lowers tt.fp_to_fp into
# ~25 scalar LLVM ops per element (no packed hw conversion), inflating the
# K-loop body specifically; on CDNA3/4 the conversion is packed/native and
# alpha == 1.


def _compile_cost_score(perf: Sequence[int], dtype: str, arch: str) -> float:
    """Estimate AMDGPU codegen cost for this (perf-config, dtype, arch).

    Returns a unit-less score; higher means the LLVM backend is expected
    to take longer to compile the kernel. The score is
        max(num_elements_kloop_body, num_elements_c_epilogue) * alpha
    where each term counts elements held per thread in the larger of the
    two cost-dominating basic blocks (the K-loop body and the C-epilogue).
    The PostRA scheduler bottlenecks on whichever basic block is larger,
    so the per-block max is a better proxy than their sum."""
    mpb, npb, kpb, kpack, _, num_waves, *_ = perf
    threads = max(1, num_waves * amd_arch_db.get_wave_size(arch))
    num_elements_kloop_body = (mpb + npb) * kpb / (threads * max(1, kpack))
    num_elements_c_epilogue = (mpb * npb) / threads
    largest_num_elements = max(num_elements_kloop_body, num_elements_c_epilogue)
    return largest_num_elements * _dtype_amplifier(dtype, arch)


def _compile_cost_budget(arch: str) -> int:
    """Per-arch cap on the compile cost score (see :func:`_compile_cost_score`).

    RDNA build times start to go wild above 8000.
    CDNA archs process the same workload faster (wider waves, native fp8 paths)
    we give them a bit more budget."""
    if _arch_family(arch) == 'rdna':
        # RDNA1-4: more expensive LLVM processing (post-RA scheduler).
        return 8000
    # Everything else (gfx9, gfx1250, gfx13+, future): looser cap until measured.
    return 12000


def _arch_family(arch: str) -> str:
    """
    - ``'rdna'`` spans the whole RDNA line: RDNA1 (gfx101x), RDNA2 (gfx103x),
      RDNA3 (gfx11xx), and RDNA4 (gfx12xx < gfx1250).
    - ``'cdna'`` is the catch-all non-RDNA bucket: the whole CDNA line
      (MI100 gfx908, MI200/MI250 gfx90a, MI300 gfx942, gfx950) plus gfx1250,
      gfx13+, GCN5_1, and unknown targets until they're measured.
    """
    rdna = (amd_arch_db.ISAFamily.RDNA1, amd_arch_db.ISAFamily.RDNA2, amd_arch_db.ISAFamily.RDNA3,
            amd_arch_db.ISAFamily.RDNA4)
    return 'rdna' if amd_arch_db.get_isa_family(arch) in rdna else 'cdna'


def _timeout_rate(arch: str, kind: str) -> float:
    """Tolerated timeout rate for ``(arch, kind)``: the fraction of applicable
    configs (PASS + FAIL + TIMEOUT, i.e. every config the pipeline actually
    committed to compiling, excluding the ones rejected upstream as
    NOT_APPLICABLE) that may time out before the sweep is failed.
    """
    # Each rate is the smallest 0.5%-granularity value that still covers the
    # worst timeout rate observed in weekly 1000-sample sweeps for that
    # (op, family)
    rates = {
        'conv': {
            'cdna': 0.015,
            'rdna': 0.020
        },
        'gemm': {
            'cdna': 0.005,
            'rdna': 0.010
        },
        'gemm_gemm': {
            'cdna': 0.005,
            'rdna': 0.010
        },
        'attn': {
            'cdna': 0.035,
            'rdna': 0.045
        },
    }
    return rates[kind][_arch_family(arch)]


def _perf_within_budget(perf: Sequence[int], dtype: str, arch: str) -> bool:
    """Whether this (perf-config, dtype, arch) tuple's compile cost score
    is within the per-arch budget."""
    return _compile_cost_score(perf, dtype, arch) <= _compile_cost_budget(arch)


# Hard cap on resampling in the random-cases generators; defensive in case
# the budget is misconfigured for an arch and rejects nearly everything.
_MAX_PERF_CONFIG_RETRIES = 256


def _sampled_perf_within_budget(rng: random.Random, arch: str, dtype: str,
                                split_k_choices: Sequence[int]) -> Tuple[int, ...]:
    """Like :func:`sample_perf_config` but rejects perf-configs whose
    effective per-thread state exceeds the arch budget. We loop with a
    generous retry cap rather than enumerating the valid subspace because the
    rejection rate is small (~few %) under PERF_CONFIG_OPTIONS for the archs
    we care about, and we don't want to silently bias the sample distribution
    by filtering an enumerated list."""
    for _ in range(_MAX_PERF_CONFIG_RETRIES):
        perf = sample_perf_config(rng, arch, split_k_choices)
        if _perf_within_budget(perf, dtype, arch):
            return perf
    raise RuntimeError(f"sample_perf_config exceeded {_MAX_PERF_CONFIG_RETRIES} retries for "
                       f"arch={arch!r} dtype={dtype!r}; PERF_CONFIG_OPTIONS may have no "
                       "config inside the effective-state budget.")


def _is_pow2(n: int) -> bool:
    """True iff ``n`` is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def sample_perf_config(rng: random.Random,
                       arch: str,
                       split_k_choices: Sequence[int],
                       pow2_only: bool = False,
                       is_attention: bool = False) -> Tuple[int, ...]:
    """Returns one random perf-config tuple: 11 fields for gemm/conv, or 12
    fields for attention / gemm+gemm, with the ``nPerBlockG1`` second-gemm
    N/head tile spliced in as the 3rd field.

    ``arch`` selects the valid ``kpack`` set (see :func:`_kpack_choices`).
    ``split_k_choices`` is the list of permissible ``splitKFactor`` values
    for this caller — typically :func:`_split_k_choices(dtype)` for conv/gemm
    and ``[1]`` for attention (whose K-split is exposed via the separate
    ``-split_kv`` kernel arg, not via the perf-config splitK).

    ``pow2_only`` restricts ``mPerBlock``/``nPerBlock``/``kPerBlock`` to
    powers of two. PERF_CONFIG_OPTIONS now includes non-pow2 m/n tiles (to
    exercise the rock-decompose-nonpow2-tiles pass) and non-pow2 k tiles (to
    exercise the non-pow2 K peeling in rock-gridwise-gemm-to-blockwise), both
    gemm/conv only; callers whose pipeline doesn't run those passes
    (attention / gemm+gemm, which use GemmGemmParamsAttr) pass
    ``pow2_only=True`` to stay on the pow2 grid.

    ``is_attention`` adds the attn-only ``nPerBlockG1`` field: untiled (``0``)
    half the time, otherwise a power-of-two tile from
    ``PERF_CONFIG_OPTIONS['n_per_block_g1']``."""
    opts = PERF_CONFIG_OPTIONS
    m_choices = opts['m_per_block']
    n_choices = opts['n_per_block']
    k_choices = opts['k_per_block']
    if pow2_only:
        m_choices = [v for v in m_choices if _is_pow2(v)]
        n_choices = [v for v in n_choices if _is_pow2(v)]
    if pow2_only or not amd_arch_db.supports_non_pow2_k_per_block(arch):
        k_choices = [v for v in k_choices if _is_pow2(v)]
    return (
        rng.choice(m_choices),
        rng.choice(n_choices),
        # attn inserts nPerBlockG1 here as the 3rd field; gemm omits it.
        # Untiled (0) half the time, otherwise a power-of-two tile.
        *([0 if rng.choice([True, False]) else rng.choice(opts['n_per_block_g1'])]
          if is_attention else []),
        rng.choice(k_choices),
        rng.choice(_kpack_choices(arch)),
        rng.choice(opts['num_ctas']),
        rng.choice(opts['num_waves']),
        rng.choice(opts['matrix_instr_nonkdim']),
        rng.choice(split_k_choices),
        rng.choice(opts['num_stages']),
        rng.choice(opts['waves_per_eu']),
        rng.choice(opts['grid_group_size']),
    )


# Hard cap on per-axis resampling in ``_sample_conv_axis``. The valid-shape
# rejection rate is well below 50% on CONV_SHAPE_OPTIONS, so 100 attempts is
# astronomically more than enough; we only enforce a cap so a future bad
# edit to CONV_SHAPE_OPTIONS that makes the axis infeasible fails loudly
# instead of hanging.
_MAX_CONV_AXIS_RETRIES = 100


def _conv_out_dim(in_dim: int, pad_l: int, pad_r: int, filt: int, dilation: int,
                  stride: int) -> int:
    """Output spatial dim for one axis. Mirrors
    ``ConvConfiguration.ho``/``wo``."""
    return (in_dim + pad_l + pad_r - (filt - 1) * dilation - 1) // stride + 1


def _sample_conv_axis(rng: random.Random, in_dim_choices, filt_choices, opts):
    """Sample (in_dim, filt, stride, dilation, pad_l, pad_r) for one spatial
    axis such that the corresponding output dim is >= 1. Sampling each
    axis independently (vs. jointly with the rest of the shape) keeps the
    rejection-resample localized: we never throw away a valid
    dtype/op/layout/n/c/k/g pick because the *other* axis happened to
    underflow. ``rocmlir-gen`` rejects out_dim<=0 outright, so we filter
    those here rather than letting them surface as FAILs."""
    for _ in range(_MAX_CONV_AXIS_RETRIES):
        in_dim = rng.choice(in_dim_choices)
        filt = rng.choice(filt_choices)
        stride = rng.choice(opts['conv_stride'])
        dilation = rng.choice(opts['dilation'])
        pad_l = rng.choice(opts['padding'])
        pad_r = rng.choice(opts['padding'])
        if _conv_out_dim(in_dim, pad_l, pad_r, filt, dilation, stride) >= 1:
            return in_dim, filt, stride, dilation, pad_l, pad_r
    raise RuntimeError("_sample_conv_axis exceeded retry cap; CONV_SHAPE_OPTIONS likely has "
                       "no feasible (in_dim, filt, stride, dilation, padding) combination.")


# 8-bit conv input dtypes are forward-only in rocmlir-gen / ConvGenerator
# (see ConvGenerator.cpp::parseConvConfig: any non-fwd direction with an
# i8/fp8/bf8 input is rejected outright). ``fp8_fp8`` is the mixed-precision
# input variant.
_FWD_ONLY_CONV_DTYPES = frozenset({'i8', 'fp8', 'fp8_fp8', 'bf8'})


def _sample_conv_shape(rng: random.Random):
    opts = CONV_SHAPE_OPTIONS
    op = rng.choice(opts['op'])
    valid_dtypes = (opts['dtype'] if op == 'fwd' else
                    [dt for dt in opts['dtype'] if dt not in _FWD_ONLY_CONV_DTYPES])
    # Grouped convolution requires both the input and the output channel
    # counts to be divisible by the group size (each group owns ``c/g``
    # input channels and produces ``k/g`` output channels). Sample ``g``
    # first and restrict ``c``/``k`` to multiples of it.
    g = rng.choice(opts['group'])
    valid_c = [c for c in opts['c'] if c % g == 0]
    valid_k = [k for k in opts['k'] if k % g == 0]
    hi, y, sh, dh, phl, phr = _sample_conv_axis(rng, opts['hi'], opts['y'], opts)
    wi, x, sw, dw, pwl, pwr = _sample_conv_axis(rng, opts['wi'], opts['x'], opts)
    return (
        op,
        rng.choice(opts['layout']),
        rng.choice(valid_dtypes),
        rng.choice(opts['n']),
        rng.choice(valid_c),
        rng.choice(valid_k),
        hi,
        wi,
        y,
        x,
        sh,
        sw,
        phl,
        phr,
        pwl,
        pwr,
        dh,
        dw,
        g,
    )


def _sample_gemm_shape(rng: random.Random):
    opts = GEMM_SHAPE_OPTIONS
    return (
        rng.choice(opts['dtype']),
        rng.choice(opts['g']),
        rng.randint(1, MAX_GEMM_DIM),  # m
        rng.randint(1, MAX_GEMM_DIM),  # k
        rng.randint(1, MAX_GEMM_DIM),  # n
        rng.choice(opts['trans_a']),
        rng.choice(opts['trans_b']),
        rng.choice(opts['trans_o']),
    )


def default_seed() -> int:
    """ISO week number, so weekly CI is reproducible across script runs."""
    return datetime.now(timezone.utc).isocalendar()[1]


# Accumulator/output dtype used to decide whether splitK is legal in the
# sweep. Any floating-point input lowers to an f32 accumulator on AMD GPUs;
# i8 is the odd one out (i32 accumulator). This is intentionally simpler
# than ``perfRunner.OUTPUT_DATA_TYPES_MAP`` — for the splitK gate we only
# care about the f32 / not-f32 boundary, not the exact user-visible output.
_OUTPUT_DTYPE_FOR_SPLITK = {
    'f32': 'f32',
    'f16': 'f32',
    'bf16': 'f32',
    'fp8': 'f32',
    'i8': 'i32',
}


def _split_k_choices(input_dtype: str) -> List[int]:
    """Permissible ``splitKFactor`` values for ``input_dtype``. ``splitKFactor
    > 1`` is only legal when the accumulator is f32, so non-f32 outputs
    (i.e. i8) are restricted to ``[1]``; everything else gets the full
    ``PERF_CONFIG_OPTIONS['split_k_factor']`` set."""
    if _OUTPUT_DTYPE_FOR_SPLITK[input_dtype] == 'f32':
        return PERF_CONFIG_OPTIONS['split_k_factor']
    return [1]


def random_conv_cases(num_samples: int, arch: str, seed: Optional[int] = None):
    """Yields ``num_samples`` random ``(conv_shape, perf_config)`` tuples.

    Perf-configs are filtered through :func:`_sampled_perf_within_budget` so
    we never feed the pipeline a (tile, dtype, arch) combination known to
    drive TritonToHsacoPass into the multi-minute regime."""
    rng = random.Random(seed if seed is not None else default_seed())
    for _ in range(num_samples):
        shape = _sample_conv_shape(rng)
        # shape[2] is the input dtype (op, layout, dtype, n, c, k, ...).
        dtype = shape[2]
        yield (shape, _sampled_perf_within_budget(rng, arch, dtype, _split_k_choices(dtype)))


def random_gemm_cases(num_samples: int, arch: str, seed: Optional[int] = None):
    """Yields ``num_samples`` random ``(gemm_shape, perf_config)`` tuples.

    Perf-configs are filtered through :func:`_sampled_perf_within_budget` so
    we never feed the pipeline a (tile, dtype, arch) combination known to
    drive TritonToHsacoPass into the multi-minute regime."""
    rng = random.Random(seed if seed is not None else default_seed())
    for _ in range(num_samples):
        shape = _sample_gemm_shape(rng)
        # shape[0] is the input dtype (dtype, g, m, k, n, trans_a, trans_b, trans_o).
        dtype = shape[0]
        yield (shape, _sampled_perf_within_budget(rng, arch, dtype, _split_k_choices(dtype)))


def to_conv_test(params, options: Options) -> ConvConfiguration:
    shape, perf = params
    op, layout, dtype, n, c, k, hi, wi, y, x, sh, sw, phl, phr, pwl, pwr, dh, dw, g = shape
    # ConvConfiguration applies FILTER/INPUT/OUTPUT_LAYOUT_MAP internally, so
    # we pass the same NCHW/NHWC-style uppercase string for all three layouts.
    return ConvConfiguration(dtype,
                             op,
                             layout,
                             layout,
                             layout,
                             n,
                             c,
                             hi,
                             wi,
                             k,
                             y,
                             x,
                             sh,
                             sw,
                             phl,
                             phr,
                             pwl,
                             pwr,
                             dh,
                             dw,
                             g,
                             options.arch,
                             options.num_cu,
                             options.num_chiplets,
                             perf_config=str(PerfConfig(perf, kind='gemm')))


def to_gemm_test(params, options: Options) -> perfRunner.GemmConfiguration:
    shape, perf = params
    dtype, g, m, k, n, trans_a, trans_b, trans_o = shape
    out_dtype = perfRunner.OUTPUT_DATA_TYPES_MAP.get(dtype, dtype)
    return perfRunner.GemmConfiguration(dtype=dtype,
                                        out_dtype=out_dtype,
                                        g=g,
                                        m=m,
                                        k=k,
                                        n=n,
                                        trans_a=trans_a,
                                        trans_b=trans_b,
                                        trans_o=trans_o,
                                        arch=options.arch,
                                        num_cu=options.num_cu,
                                        num_chiplets=options.num_chiplets,
                                        perf_config=str(PerfConfig(perf, kind='gemm')))


async def run_config(param_iter: Iterable[IterType],
                     to_config: Callable[[IterType, Options], perfRunner.PerfConfiguration],
                     options: Options, paths: Paths, *, samples: int) -> bool:
    n_passes, n_not_applicable, timeouts, failures = \
        await sweep_parameters(param_iter, to_config, options, paths)
    if len(failures) != 0:
        print("*** Summary of failures ***")
        for c in failures:
            print(_repro_command(c))

    n_timeouts = len(timeouts)
    n_failures = len(failures)
    timeouts_over_budget = False
    budget = 0
    if n_timeouts != 0:
        # All configs in a single run_config call come from one to_config, so
        # they share a kind; timeouts[0] is a safe representative for the
        # per-operation rate lookup.
        rate = (options.max_timeout_rate if options.max_timeout_rate is not None else _timeout_rate(
            options.arch, timeouts[0].SWEEP_KIND))
        if rate < 0:
            # Negative rate disables the check entirely.
            budget_str = "unlimited"
        else:
            # Denominator is the configs the pipeline actually committed to
            # compiling (everything except NOT_APPLICABLE).
            n_applicable = n_passes + n_failures + n_timeouts
            budget = round(rate * n_applicable)
            timeouts_over_budget = n_timeouts > budget
            budget_str = f"{budget} = {rate:.1%} of {n_applicable} applicable"
        verdict = "OVER BUDGET" if timeouts_over_budget else "within budget"
        print(f"*** Summary of timeouts ({n_timeouts}, budget {budget_str}: {verdict}) ***")
        for c in timeouts:
            print(_repro_command(c))

    print(f"Passed: {n_passes}, Not applicable: {n_not_applicable}, "
          f"Timed out: {n_timeouts}, Failed: {n_failures}")

    if timeouts_over_budget:
        print(
            f"Sweep recorded {n_timeouts} timeouts, exceeding the budget of "
            f"{budget} for arch {options.arch!r}. Raise --max-timeout-rate (or "
            "pass a negative value to disable the check) if this is expected, "
            "or investigate the configs above for a compile-time regression.",
            file=sys.stderr,
        )
        return False

    # Fail the run if we intended to validate kernels but recorded no PASS and
    # no FAIL. This happens when every sample
    # was NOT_APPLICABLE and/or timed out (within budget) for this arch.
    if samples > 0 and n_passes == 0 and n_failures == 0:
        reason = ("every sample was NOT_APPLICABLE or timed out"
                  if n_timeouts else "the sample space is entirely NOT_APPLICABLE")
        print(
            f"Sweep did not record any PASS results (samples > 0, failures == 0, "
            f"timeouts == {n_timeouts}). Check arch, build, or whether {reason} "
            "for this target.",
            file=sys.stderr,
        )
        return False
    return n_failures == 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the CLI flags shared by parameterSweeps and attentionSweeps.

    Sister scripts (e.g. attentionSweeps) call this so users see the same
    flag names and defaults across all sweep front-ends. Add a flag here only
    if every sweep script needs it; per-sweep options stay local."""
    # Failures (FAIL) are always printed in full; --debug additionally prints
    # the (much noisier) details of NOT_APPLICABLE configs and the per-stage
    # detail of (tolerated) TIMEOUTs as they happen. Timeouts are always
    # recorded by the end-of-run summary regardless of --debug.
    parser.add_argument('--debug',
                        '-d',
                        action='store_true',
                        help='Also print details for NOT_APPLICABLE configs and '
                        'inline per-stage detail for TIMEOUTs as they happen '
                        '(failures are always printed; the timeout summary is '
                        'printed regardless)')
    parser.add_argument('--quiet',
                        '-q',
                        action='store_true',
                        help="Don't print a per-config PASS/NOT_APPLICABLE/FAIL line")
    parser.add_argument('--log-failures',
                        '-L',
                        action='store_true',
                        help='Save failing configs to a .txt file')
    parser.add_argument('--jobs',
                        '-j',
                        type=_positive_int,
                        default=max(1,
                                    len(os.sched_getaffinity(0)) // 2),
                        help='Number of concurrent test tasks, must be > 0 '
                        '(default %(default)s)')
    parser.add_argument('--test-timeout-sec',
                        type=int,
                        default=120,
                        help='Per-stage timeout in seconds, applied independently '
                        'to rocmlir-gen, rocmlir-driver, and mlir-runner '
                        '(0 disables the timeout). Default %(default)s.')
    parser.add_argument('--max-timeout-rate',
                        type=float,
                        default=None,
                        help='Fraction of applicable configs (PASS + FAIL + '
                        'TIMEOUT) that may time out before the sweep is failed. '
                        'Timeouts up to this rate are reported '
                        'but not counted as failures (they are compile-time '
                        'blowups, not correctness bugs). Default: a per-arch, '
                        'per-operation rate (see _timeout_rate). Pass a negative '
                        'value to tolerate any number of timeouts.')
    parser.add_argument('--samples',
                        type=_positive_int,
                        default=1000,
                        help='Number of random samples to test, must be > 0 '
                        '(default %(default)s)')
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='RNG seed for sampling. Defaults to the current ISO '
                        'week so weekly CI is reproducible.')
    parser.add_argument(
        "--mlir-build-dir",
        type=str,
        default=None,
        help="The build directory of MLIR based kernel generator",
    )
    # Offline cap-validation flag. --dry-run prints sampled (shape, perf)
    # pairs and whether the per-thread state cap (see _perf_within_budget)
    # would ACCEPT or REJECT each one, without running rocmlir-gen,
    # rocmlir-driver, or mlir-runner.
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Sample configs and print whether the cap would '
                        'accept or reject each. No subprocesses are spawned.')


def build_options_and_paths(args: argparse.Namespace) -> Tuple[Options, Paths]:
    """Materialize the ``Options`` and ``Paths`` shared by every sweep script
    from a Namespace produced by a parser populated via ``add_common_args``."""
    mlir_build_dir = args.mlir_build_dir or perfRunner.find_mlir_build_dir()

    arch = get_arch()
    chip = perfRunner.get_chip()
    num_cu = get_num_cu(chip)
    options = Options(debug=args.debug,
                      quiet=args.quiet,
                      log_failures=args.log_failures,
                      arch=arch,
                      concurrent_tests=args.jobs,
                      num_cu=num_cu,
                      num_chiplets=amd_arch_db.infer_num_chiplets(chip, num_cu),
                      test_timeout_sec=args.test_timeout_sec,
                      max_timeout_rate=args.max_timeout_rate)
    paths = perfRunner.create_paths(None, mlir_build_dir)
    return options, paths


def _dry_run(kind: str, num_samples: int, arch: str, seed: Optional[int]) -> bool:
    """Print sampled (shape, perf) pairs together with the cap's verdict.

    Does NOT spawn rocmlir-gen / rocmlir-driver / mlir-runner. The point is to
    verify the per-thread state cap (_perf_within_budget) cheaply.

    The RNG sequence here matches a *real* run only until the first rejection
    (in a real run, _sampled_perf_within_budget resamples on reject and so
    consumes extra RNG state). That's fine for the purpose of finding a seed
    whose first N draws contain a rejection."""
    rng = random.Random(seed if seed is not None else default_seed())
    accept = 0
    reject = 0
    print(f"# dry-run: arch={arch}, kind={kind}, samples={num_samples}, "
          f"seed={seed if seed is not None else default_seed()}")
    print(f"# budget = {_compile_cost_budget(arch)} on "
          f"max(num_elements_kloop_body, num_elements_c_epilogue) * alpha")
    for i in range(num_samples):
        if kind == 'gemm':
            shape = _sample_gemm_shape(rng)
            dtype = shape[0]
        else:  # 'conv'
            shape = _sample_conv_shape(rng)
            dtype = shape[2]
        perf = sample_perf_config(rng, arch, _split_k_choices(dtype))
        score = _compile_cost_score(perf, dtype, arch)
        verdict = "ACCEPT" if _perf_within_budget(perf, dtype, arch) else "REJECT"
        if verdict == "ACCEPT":
            accept += 1
        else:
            reject += 1
        print(f"[{i:4d}] {verdict} score={score:8.1f} dtype={dtype!s:5} "
              f"perf={perf} shape={shape}")
    print(f"# total: accept={accept} reject={reject}")
    return reject > 0


def main() -> bool:
    parser = argparse.ArgumentParser(
        description='Sweep parameter values to check correctness of MLIR')
    parser.add_argument('config',
                        choices=['conv', 'gemm'],
                        help="Kind of kernel to sweep: 'conv' or 'gemm'. Both "
                        "modes randomly sample problem shape and perf-config "
                        "on every iteration.")
    add_common_args(parser)
    args = parser.parse_args()

    if args.dry_run:
        arch = get_arch()
        # Returns True iff at least one config was rejected by the cap; that
        # makes `--dry-run` a usable success indicator for "found a seed that
        # exercises the filter".
        return _dry_run(args.config, args.samples, arch, args.seed)

    options, paths = build_options_and_paths(args)

    if args.config == 'conv':
        param_iter = random_conv_cases(args.samples, options.arch, seed=args.seed)
        return asyncio.run(
            run_config(param_iter, to_conv_test, options, paths, samples=args.samples))
    if args.config == 'gemm':
        param_iter = random_gemm_cases(args.samples, options.arch, seed=args.seed)
        return asyncio.run(
            run_config(param_iter, to_gemm_test, options, paths, samples=args.samples))
    raise ValueError(f"Unknown config {args.config!r} (expected 'conv' or 'gemm')")


if __name__ == '__main__':
    ret = main()
    sys.exit(int(not ret))
