#!/usr/bin/env python3
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Brute-force autotuner for the CPU verifier's matmul tile sizes.

For a given GEMM shape (and any host CPU), sweep a Cartesian product of
`(mFuse, nFuse, kTile)` candidates, run the standard CPU verifier pipeline
once per candidate, and report the triple with the lowest median
`CPU validation time`.

The autotuner drives the existing rocmlir binaries through subprocess
calls; no compilation is needed per candidate. Tile-size selection is
overridden via the env vars `ROCMLIR_CPU_TILE_{M,N,K}` (read by
`mlir/lib/Conversion/CPU/Transforms/CpuTileLUT.cpp`).

Typical usage from the repo root:

    python3 mlir/utils/performance/cpuTileAutotuner.py \\
        --build-dir build --arch gfx942 \\
        --m 1000 --n 405 --k 1024 --trials 3

Output is plain text: a sorted table of (median ms, mFuse, nFuse, kTile)
followed by the recommended triple. Nothing is persisted to disk; if a
triple looks good, paste it into `pickSapphireRapids` (or the appropriate
per-CPU picker) in `CpuTileLUT.cpp`.

The script intentionally has no third-party dependencies: stdlib only.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Default search space: powers of two from 8 to 512 inclusive, for each of
# M, N, K. That's 7 values per dim = 7^3 = 343 candidates, which is
# manageable on a fast box but still big enough to need a timeout safety
# net. Callers can shrink it via --tile-values.
DEFAULT_TILE_VALUES = (8, 16, 32, 64, 128, 256, 512)

# Pattern used by `--cpu-timers` in
# `mlir/lib/ExecutionEngine/conv-validation-wrappers.cpp`:
#   printf("CPU validation time: %.3f ms\n", elapsed);
_CPU_TIME_RE = re.compile(r"CPU validation time:\s+([0-9]+(?:\.[0-9]+)?)\s+ms")


@dataclass(frozen=True)
class Triple:
    """A `(mFuse, nFuse, kTile)` candidate."""

    m: int
    n: int
    k: int

    def env(self) -> dict[str, str]:
        return {
            "ROCMLIR_CPU_TILE_M": str(self.m),
            "ROCMLIR_CPU_TILE_N": str(self.n),
            "ROCMLIR_CPU_TILE_K": str(self.k),
        }


@dataclass
class TrialResult:
    """Outcome of a single (triple, trial) measurement."""

    triple: Triple
    ms: Optional[float]  # None if compile/run failed or timed out.
    note: str = ""


def build_command(args: argparse.Namespace) -> list[list[str]]:
    """Build the three-stage pipeline command as a list of argv lists.

    Each inner list corresponds to one process in a pipeline:
        rocmlir-gen | rocmlir-driver -c | rocm-run
    The autotuner re-runs this whole pipeline once per trial. Per-trial
    JIT cost dominates for small shapes, but `--cpu-timers` only measures
    CPU verifier time, so JIT cost doesn't bias the comparison between
    triples.
    """
    build_bin = Path(args.build_dir) / "bin"
    gen_argv = [
        str(build_bin / "rocmlir-gen"),
        "--cpu-timers",
        "--arch",
        args.arch,
        "--operation",
        "gemm",
        "-t",
        args.dtype,
        "-out_datatype",
        args.out_dtype,
        f"-transA={'true' if args.transa else 'false'}",
        f"-transB={'true' if args.transb else 'false'}",
        "-g",
        str(args.g),
        "-m",
        str(args.m),
        "-n",
        str(args.n),
        "-k",
        str(args.k),
        "-pv",
    ]
    if args.perf_config:
        gen_argv += [f"--perf_config={args.perf_config}"]
    driver_argv = [str(build_bin / "rocmlir-driver"), "-c"]
    runner_argv = [str(build_bin / "rocm-run")]
    return [gen_argv, driver_argv, runner_argv]


def run_pipeline(
    cmds: list[list[str]],
    env_overrides: dict[str, str],
    timeout_s: float,
) -> tuple[Optional[str], str]:
    """Run a 3-stage pipeline with `env_overrides` merged into os.environ.

    Returns `(stdout_or_None, diagnostic)`. `stdout` is None on any
    failure (non-zero exit anywhere, timeout, runner crash). `diagnostic`
    is a short string suitable for the per-trial `note`.
    """
    env = os.environ.copy()
    env.update(env_overrides)
    try:
        # subprocess.Popen pipeline. We capture stdout of the final stage
        # and merge stderr into stdout so error messages from any stage
        # are visible. We do NOT pipe stderr to subprocess.PIPE separately
        # for each stage -- too fiddly for what is essentially a
        # diagnostic on failure.
        procs: list[subprocess.Popen] = []
        prev_stdout = None
        for i, argv in enumerate(cmds):
            stdin = prev_stdout
            stdout = subprocess.PIPE
            stderr = subprocess.STDOUT if i == len(cmds) - 1 else subprocess.DEVNULL
            p = subprocess.Popen(
                argv,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                env=env,
            )
            if prev_stdout is not None:
                # Allow the previous stage to receive SIGPIPE if this
                # stage exits early. See `man 7 pipe`.
                prev_stdout.close()
            prev_stdout = p.stdout
            procs.append(p)

        try:
            out_bytes, _ = procs[-1].communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            for p in procs:
                p.kill()
            for p in procs:
                p.wait()
            return None, "timeout"

        # Wait for all upstream stages so their return codes are settled.
        for p in procs[:-1]:
            p.wait(timeout=5)

        if any(p.returncode != 0 for p in procs):
            codes = ",".join(str(p.returncode) for p in procs)
            return None, f"nonzero exit ({codes})"

        return out_bytes.decode(errors="replace"), "ok"
    except FileNotFoundError as e:
        return None, f"missing binary: {e.filename}"


def parse_cpu_time_ms(stdout: str) -> Optional[float]:
    """Extract the `CPU validation time: X ms` value from runner output.

    Returns None when the line isn't present (e.g. the verifier was
    skipped because the function has unsupported argument types).
    """
    m = _CPU_TIME_RE.search(stdout)
    if not m:
        return None
    return float(m.group(1))


def trial(
    cmds: list[list[str]],
    triple: Triple,
    timeout_s: float,
) -> TrialResult:
    """Run a single (triple, trial) pair and parse the timing."""
    stdout, note = run_pipeline(cmds, triple.env(), timeout_s)
    if stdout is None:
        return TrialResult(triple=triple, ms=None, note=note)
    ms = parse_cpu_time_ms(stdout)
    if ms is None:
        return TrialResult(
            triple=triple, ms=None, note="no `CPU validation time` line"
        )
    return TrialResult(triple=triple, ms=ms, note=note)


def autotune(args: argparse.Namespace) -> int:
    """Drive the sweep. Returns a process exit code."""
    tile_values = tuple(args.tile_values)
    cmds = build_command(args)

    print(f"# Build dir : {args.build_dir}")
    print(f"# Arch      : {args.arch}")
    print(
        f"# Shape     : M={args.m} N={args.n} K={args.k} "
        f"dtype={args.dtype}->{args.out_dtype} "
        f"transA={args.transa} transB={args.transb} g={args.g}"
    )
    print(f"# Tile vals : {tile_values}")
    candidates = [Triple(*t) for t in itertools.product(tile_values, repeat=3)]
    print(f"# Trials    : {args.trials} per candidate")
    print(f"# Candidates: {len(candidates)}")
    print(f"# Timeout   : {args.timeout_s}s per trial")
    print()

    # Aggregate per-triple medians.
    medians: list[tuple[float, Triple, list[Optional[float]], str]] = []
    start = time.monotonic()
    for idx, triple in enumerate(candidates, 1):
        per_trial: list[Optional[float]] = []
        notes: list[str] = []
        for _ in range(args.trials):
            r = trial(cmds, triple, args.timeout_s)
            per_trial.append(r.ms)
            if r.ms is None:
                notes.append(r.note)
        valid = [v for v in per_trial if v is not None]
        if not valid:
            note = notes[0] if notes else "no valid trial"
            medians.append((float("inf"), triple, per_trial, note))
        else:
            medians.append((statistics.median(valid), triple, per_trial, ""))
        if args.verbose:
            elapsed = time.monotonic() - start
            est_total = elapsed / idx * len(candidates)
            print(
                f"  [{idx:>4}/{len(candidates)}] "
                f"M={triple.m:>4} N={triple.n:>4} K={triple.k:>4}  "
                f"median={medians[-1][0]:>9.3f} ms  "
                f"trials={per_trial}  "
                f"({elapsed:.1f}s elapsed, ~{est_total:.0f}s total)",
                flush=True,
            )

    medians.sort(key=lambda x: x[0])
    print()
    print("# Results (sorted by median CPU validation time):")
    print(
        f"# {'rank':>4}  {'mFuse':>5}  {'nFuse':>5}  {'kTile':>5}  "
        f"{'median ms':>10}  notes"
    )
    head = args.show_top if args.show_top > 0 else len(medians)
    for rank, (median_ms, triple, _per_trial, note) in enumerate(
        medians[:head], 1
    ):
        median_str = (
            f"{median_ms:>10.3f}" if median_ms != float("inf") else f"{'N/A':>10}"
        )
        print(
            f"  {rank:>4}  {triple.m:>5}  {triple.n:>5}  {triple.k:>5}  "
            f"{median_str}  {note}"
        )

    best = medians[0]
    if best[0] == float("inf"):
        print("\n# No candidate produced a measurable timing. "
              "Check `--build-dir`, `--arch`, and that `-pv` works manually.")
        return 1

    print()
    print(
        f"# Best: mFuse={best[1].m} nFuse={best[1].n} kTile={best[1].k} "
        f"(median {best[0]:.3f} ms)"
    )
    print(
        f"# Paste into pick<CPU> in CpuTileLUT.cpp:"
        f"\n#   return {{/*mFuse=*/{best[1].m}, "
        f"/*nFuse=*/{best[1].n}, /*kTile=*/{best[1].k}}};"
    )
    return 0


def parse_tile_values(s: str) -> list[int]:
    """Parse `--tile-values 8,16,32,64` into [8, 16, 32, 64]."""
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v <= 0:
            raise argparse.ArgumentTypeError(
                f"tile values must be positive, got {v}"
            )
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("--tile-values must be non-empty")
    return out


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--build-dir", default="build",
                   help="Path to the rocmlirTriton build directory.")
    p.add_argument("--arch", required=True,
                   help="GPU arch passed to rocmlir-gen, e.g. gfx942.")
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("-g", type=int, default=1)
    p.add_argument("--dtype", default="f32",
                   help="rocmlir-gen `-t` value (default: f32).")
    p.add_argument("--out-dtype", default="f32",
                   help="rocmlir-gen `-out_datatype` value (default: f32).")
    p.add_argument("--transa", action="store_true")
    p.add_argument("--transb", action="store_true")
    p.add_argument("--perf-config", default="",
                   help="Optional `--perf_config=...` value for rocmlir-gen.")
    p.add_argument("--tile-values", type=parse_tile_values,
                   default=list(DEFAULT_TILE_VALUES),
                   help="Comma-separated list of values to sweep per dim. "
                        "Default: 8,16,32,64,128,256,512.")
    p.add_argument("--trials", type=int, default=3,
                   help="Number of measurements per candidate (default: 3).")
    p.add_argument("--timeout-s", type=float, default=120.0,
                   help="Per-trial timeout in seconds (default: 120).")
    p.add_argument("--show-top", type=int, default=20,
                   help="How many top entries to print (0 = all).")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-candidate progress.")
    args = p.parse_args(argv)
    return autotune(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
