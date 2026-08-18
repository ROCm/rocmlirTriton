#!/usr/bin/env python3
"""Generate LdsBlacklistPerfconfigs.inc: GEMM tile shapes that overflow LDS.

For each (arch, dtype), this enumerates the *exhaustive* tuning space of a large
GEMM (large so tile sizes are never capped by the problem dims), lowers each
perf config only as far as the ``triton`` stage -- where ``ttg.shared`` is
computed -- and then runs the standalone ``resolve-kernel-launch-params`` pass.
If that pass reports the config's shared-memory usage exceeds the arch LDS
limit, the config's LDS-relevant projection is recorded.

Only a handful of perf-config fields actually change ``ttg.shared``. Measured
empirically (vary one field, hold the rest, watch ttg.shared):

    mPerBlock, nPerBlock, kPerBlock, numWaves, matrixInstrNonkdim, numStages

change it; kpack, numCTAs, splitKFactor, wavesPerEU, gridGroupSize and the
knob fields do NOT. So we key the blacklist on just those six fields (see
PROJECTION_NAMES / GemmLdsKey in LdsBlacklist.h -- the field order must match
the C++ struct). This makes the table small and, crucially, lets the C++
consumer also drop configs that share the projection but differ in an
LDS-irrelevant field (e.g. every splitK/kpack variant), which the big-K sweep
never even emits.

Per-block LDS is not strictly independent of the problem's M/N/K: it grows with
the K-loop trip count (K / kPerBlock) because ``numStages`` software pipelining
multi-buffers the LDS operands, but it is monotonic non-decreasing in that trip
count and saturates for large K. Sweeping at a large K (DEFAULT_DIMS) therefore
captures the saturated, worst-case footprint, which makes the blacklist safe and
effectively shape-independent: it never blacklists a config that fits at large K
(so it never prunes one that could overflow at runtime for any K), and the only
over-pruning -- at trip count 1, K <= kPerBlock -- is harmless because there the
pipeliner caps the effective stage count to 1, so the config produces the same
kernel as its (never-blacklisted) numStages=1 variant. See the LdsBlacklist.h
header comment for the full argument and the C++ consumption in
RockTuningImpl.cpp.

Data types are keyed the same way the C++ consumer canonicalizes them
(``ParamLookupTable::getDataTypeString``): all 16-bit floats collapse to ``f16``
(so ``f16`` also covers ``bf16``), all 8-bit floats to ``fp8``, all 4-bit floats
to ``f4``. We therefore enumerate one representative ``-t`` value per key.

The arch/dtype matrix and GEMM dims are fixed internally (DEFAULT_ARCHES /
DEFAULT_DTYPES / DEFAULT_DIMS); edit those constants to change the sweep.

Usage:
    # Regenerate the in-tree .inc for the full arch/dtype matrix:
    python3 generateLDSBlacklist.py

    # Drift detection (nightly): re-check that projections already in the .inc
    # still overflow with the current compiler; exit non-zero otherwise. Use
    # --samples N (seeded by the ISO week via --seed) to check only a random
    # subset and keep runtime bounded:
    python3 generateLDSBlacklist.py --verify --samples 500
"""

import argparse
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import perfRunner
from perfCommonUtils import PERF_CONFIG_FIELD_NAMES, parse_perfconfig, serialize_perfconfig
from tqdm import tqdm

# Tells LdsBlacklist::lookupGemm (see LdsBlacklist.cpp) to return an empty set so
# `--emit-tuning-space=exhaustive` yields the *unfiltered* space. Without this
# the generator would only ever see configs the current .inc hasn't already
# blacklisted, making regeneration non-idempotent (it would shrink to nothing).
BLACKLIST_BYPASS_ENV = "ROCMLIR_DISABLE_LDS_BLACKLIST"

# Diagnostic emitted by ResolveKernelLaunchParams when ttg.shared > LDS limit.
# Keep in sync with mlir/lib/Dialect/Rock/Transforms/ResolveKernelLaunchParams.cpp.
LDS_OVERFLOW_MARKER = "exceeds LDS limit"

# The perf-config fields that affect ttg.shared, in the order of GemmLdsKey in
# LdsBlacklist.h and the projection built in RockTuningImpl.cpp. Perf configs are
# keyed by field name, so only this order (which the .inc tuples follow) matters.
PROJECTION_NAMES = ("mPerBlock", "nPerBlock", "kPerBlock", "numWaves", "matrixInstrNonkdim",
                    "numStages")

Projection = Tuple[int, ...]

# Canonical dtype key (matching ParamLookupTable::getDataTypeString) -> the
# rocmlir-gen ``-t`` value used to (re)produce it. We only pick one -t per key
# because LDS usage depends on element bit-width, which is identical within a
# key (e.g. bf16 and f16 both key as "f16"). We don't pass -out_datatype:
# rocmlir-gen derives a valid output type from -t.
DTYPE_GEN = {
    "f32": "f32",
    "f16": "f16",  # also covers bf16 (same 16-bit LDS footprint)
    "i8": "i8",
    "fp8": "fp8_fp8",
    "f4": "f4E2M1FN",
}

# f4 is scaled-only on current targets and often enumerates to nothing; keep it
# out of the default sweep but allow it via --data-type.
DEFAULT_DTYPES = ["f32", "f16", "i8", "fp8"]

# One representative per supported ISA family (see getArch/ISAFamily in
# AmdArchDb.cpp). We key per family -- not just per distinct LDS size -- because
# ttg.shared is a function of the matmul lowering (MFMA vs WMMA vs non-accel),
# which differs between families that happen to share a 64 KB budget, so one
# family's overflow set is not a safe substitute for another's. Unlisted archs
# fall back at lookup time to the same-family listed arch whose gfx id is
# numerically closest and whose LDS size is identical (e.g. gfx1031 -> gfx1030,
# gfx1101 -> gfx1100).
#
# When a Triton/LLVM bump adds a new ISA family, add a representative chip here
# (a new chip in an existing family is covered by the fallback above and needs
# no entry). See docs/bump_triton_version.md
DEFAULT_ARCHES = [
    "gfx906",  # GCN5_1
    "gfx908",  # CDNA1 (MI100)
    "gfx90a",  # CDNA2 (MI200)
    "gfx942",  # CDNA3 (MI300)
    "gfx950",  # CDNA4
    "gfx1010",  # RDNA1
    "gfx1030",  # RDNA2
    "gfx1100",  # RDNA3
    "gfx1150",  # RDNA3 (APU)
    "gfx1170",  # GFX1170 (own ISA family, not RDNA3/RDNA4)
    "gfx1200",  # RDNA4
    "gfx1250",  # GFX1250
]

# [g, m, n, k], chosen large for two independent reasons, both needed for the
# blacklist derived from this single GEMM to be a valid worst case for every
# GEMM shape:
#   1. M/N/K must exceed the largest per-block tile so the problem shape never
#      clips the tile ladders (computeDPerBlock caps M/N tiles at
#      MAX_MN_PER_BLOCK=256, capKPerBlockByK caps K tiles at MAX_K_PER_BLOCK=512,
#      both in RockTuningImpl.cpp). 32768 sits ~64-128x above those caps.
#   2. K must be large enough that the numStages software pipeline saturates:
#      per-block LDS grows with the K-loop trip count (K / kPerBlock) and only
#      reaches its worst case once trip count >= numStages. At K=32768 the trip
#      count is >= 64 for every kPerBlock (<=512), far past saturation, so we
#      always record the maximal LDS footprint (see the LdsBlacklist.h header).
# The extra headroom over the tile caps also keeps even the densest 1-byte (i8)
# tiles well beyond any AMD LDS budget (<=320 KB on gfx1250), so no dtype's
# widest tile can accidentally fit.
DEFAULT_DIMS = [1, 32768, 32768, 32768]


def gen_args(paths: perfRunner.Paths, arch: str, gen_dtype: str, dims: List[int]) -> List[str]:
    g, m, n, k = dims
    return [
        paths.mlir_paths.rocmlir_gen_path, "-operation", "gemm", "--arch", arch, "-t", gen_dtype,
        "-g",
        str(g), "-m",
        str(m), "-n",
        str(n), "-k",
        str(k)
    ]


def emit_exhaustive_space(paths: perfRunner.Paths, arch: str, gen_dtype: str,
                          dims: List[int]) -> List[str]:
    """Return the exhaustive list of perf-config strings for a large GEMM."""
    cmd = gen_args(paths, arch, gen_dtype, dims) + ["--emit-tuning-space=exhaustive"]
    env = {**os.environ, BLACKLIST_BYPASS_ENV: "1"}
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    # Fail fast: a non-zero exit is a real toolchain/config error, which is
    # indistinguishable from an empty space once we drop it. Swallowing it here
    # would let regeneration commit an empty/partial blacklist. A genuinely
    # unsupported arch/dtype exits 0 with no configs and is skipped downstream.
    if proc.returncode != 0:
        raise RuntimeError(f"rocmlir-gen failed (exit {proc.returncode}) for "
                           f"{arch}/{gen_dtype}:\n  cmd: {' '.join(cmd)}\n"
                           f"{proc.stderr.strip()}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def project(perf_config: str) -> Optional[Projection]:
    """Extract the LDS-relevant field tuple from a perf-config string.

    Fields are looked up by name in the canonical ``gemm:key=value,...`` form,
    so this is insensitive to which other tunable or knob fields the string
    carries."""
    try:
        _, params = parse_perfconfig(perf_config)
        return tuple(params[name] for name in PROJECTION_NAMES)
    except (KeyError, ValueError):
        return None


def synth_config(proj: Projection) -> str:
    """Reconstruct a representative perf-config string from a projection.

    The LDS-irrelevant tunable fields are pinned to the constants the GEMM
    tuning space always uses (kpack=1, numCTAs=1, splitKFactor=1, wavesPerEU=0,
    gridGroupSize=0) and the knob fields are omitted, so the parser fills them
    with the kKnobDefault sentinel. None of these affect ttg.shared (see
    PROJECTION_*), so the reconstructed config overflows LDS iff the original
    projection does.

    Field order comes from PERF_CONFIG_FIELD_NAMES so it is defined in exactly
    one place."""
    values = {
        **dict(zip(PROJECTION_NAMES, proj)),
        "kpack": 1,
        "numCTAs": 1,
        "splitKFactor": 1,
        "wavesPerEU": 0,
        "gridGroupSize": 0,
    }
    ordered_values = {name: values[name] for name in PERF_CONFIG_FIELD_NAMES["gemm"]}
    return serialize_perfconfig("gemm", ordered_values)


def default_seed() -> int:
    """ISO week number, so nightly sampling is reproducible within a week
    (mirrors parameterSweeps.py::default_seed)."""
    return datetime.now(timezone.utc).isocalendar()[1]


# Verdicts from lowering one config as far as the LDS check.
LDS_OVERFLOW = "overflow"  # ttg.shared exceeded the arch LDS limit
LDS_FITS = "fits"  # lowered cleanly and stayed within budget
LDS_INCONCLUSIVE = "inconclusive"  # config rejected / lowering errored out


def classify_lds(paths: perfRunner.Paths, arch: str, gen_dtype: str, perf_config: str,
                 dims: List[int]) -> str:
    """Lower one config through the triton stage and run the LDS check.

    Returns LDS_OVERFLOW when the overflow diagnostic is seen, LDS_FITS when
    every stage succeeds without it, and LDS_INCONCLUSIVE when any stage errors
    (e.g. rocmlir-gen rejects the reconstructed config). Distinguishing the last
    case matters for --verify: a rejected config must NOT be misread as "no
    longer overflows" (that would be spurious drift).
    """
    gen = subprocess.Popen(gen_args(paths, arch, gen_dtype, dims) +
                           [f"--perf_config={perf_config}"],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)
    driver = subprocess.Popen(
        [paths.mlir_paths.rocmlir_driver_path, "--kernel-pipeline=gpu,triton", "--arch", arch],
        stdin=gen.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)
    gen.stdout.close()
    opt = subprocess.Popen([paths.mlir_paths.rocmlir_opt_path, "-resolve-kernel-launch-params"],
                           stdin=driver.stdout,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE)
    driver.stdout.close()

    _, err = opt.communicate()
    gen.wait()
    driver.wait()
    if LDS_OVERFLOW_MARKER in err.decode("utf-8", errors="replace"):
        return LDS_OVERFLOW
    if gen.returncode != 0 or driver.returncode != 0 or opt.returncode != 0:
        return LDS_INCONCLUSIVE
    return LDS_FITS


def config_overflows_lds(paths: perfRunner.Paths, arch: str, gen_dtype: str, perf_config: str,
                         dims: List[int]) -> bool:
    """True only when the config provably overflows LDS. Used by generation,
    where any non-overflow outcome (fits, or an unrelated bail) must be treated
    as "don't blacklist" so shape-dependent verdicts never poison the table."""
    return classify_lds(paths, arch, gen_dtype, perf_config, dims) == LDS_OVERFLOW


def overflowing_projections(paths: perfRunner.Paths, arch: str, gen_dtype: str, dims: List[int],
                            jobs: int) -> Tuple[Dict[Projection, str], List[Projection]]:
    """Return ({projection: representative config}, [projections that overflow]).

    Dedups the exhaustive space by projection (LDS is a function of the
    projection only, so one representative per projection suffices) and tests
    each representative in parallel.
    """
    configs = emit_exhaustive_space(paths, arch, gen_dtype, dims)
    reps: Dict[Projection, str] = {}
    for pc in configs:
        p = project(pc)
        if p is not None:
            reps.setdefault(p, pc)
    if not reps:
        return {}, []

    projs = list(reps.keys())
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        results = ex.map(lambda p: config_overflows_lds(paths, arch, gen_dtype, reps[p], dims),
                         projs)
        verdicts = list(
            tqdm(results, total=len(projs), desc=f"{arch}/{gen_dtype}", unit="perf_config"))
    overflowing = [p for p, bad in zip(projs, verdicts) if bad]
    return reps, overflowing


# =============================================================================
# .inc emission (mirrors QuickTuningPerfconfigs.inc structure)
# =============================================================================


def ident(dtype: str, arch: str) -> str:
    """C++ identifier stem for an (arch, dtype), e.g. gemmF16Gfx942."""
    d = "".join(c for c in dtype.title() if c.isalnum())
    a = arch.capitalize()
    return f"gemm{d}{a}"


REL_INC = "mlir/include/mlir/Dialect/Rock/Tuning/LdsBlacklistPerfconfigs.inc"


def default_output_path() -> Path:
    # Resolve the in-tree .inc whether we're the source script or the build/bin
    # copy: walk up from both this file and the cwd looking for the repo root
    # (the dir that contains REL_INC's parent), then fall back to git toplevel.
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in (start, *start.parents):
            if (parent / REL_INC).parent.is_dir():
                return parent / REL_INC
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
        return Path(top) / REL_INC
    except Exception:
        return Path(REL_INC)


def generator_rel_path() -> str:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() or (parent / "mlir").is_dir():
            try:
                return str(here.relative_to(parent))
            except ValueError:
                pass
    return here.name


def emit_inc(results: Dict[Tuple[str, str], List[Projection]], path: Path) -> None:
    """Write the full .inc from {(arch, dtype_key): [projection,...]}.

    Regenerated wholesale each run (unlike quickTuningGen's incremental patching)
    because this tool always sweeps the complete arch/dtype matrix it's given.
    """
    field_comment = "// GemmLdsKey fields: " + ", ".join(PROJECTION_NAMES)
    lines: List[str] = [
        f"// Generated by: {generator_rel_path()}", "", field_comment, "", "// clang-format off", ""
    ]

    # DEFINITIONS
    lines.append("#ifdef GemmLdsBlacklist_DEFINITIONS_GEN")
    lines.append("")
    for (arch, dtype), projs in results.items():
        stem = ident(dtype, arch)
        lines.append(f"// BEGIN_gemm_{dtype}_{arch}_LDS_BLACKLIST")
        lines.append(f"const GemmLdsKey LdsBlacklistGemm::{stem}[] = {{")
        for i, p in enumerate(projs):
            comma = "," if i < len(projs) - 1 else ""
            lines.append(f'    {{{", ".join(str(x) for x in p)}}}{comma}')
        lines.append("};")
        lines.append(f"// END_gemm_{dtype}_{arch}_LDS_BLACKLIST")
        lines.append("")
    lines.append("#endif  // GemmLdsBlacklist_DEFINITIONS_GEN")
    lines.append("")

    # DECLARATIONS
    lines.append("#ifdef GemmLdsBlacklist_DECLARATIONS_GEN")
    lines.append("")
    for (arch, dtype), projs in results.items():
        stem = ident(dtype, arch)
        count = f"n{stem[0].upper()}{stem[1:]}"
        lines.append(f"static constexpr size_t {count} = {len(projs)};")
        lines.append(f"static const GemmLdsKey {stem}[{count}];")
    lines.append("")
    lines.append("#endif  // GemmLdsBlacklist_DECLARATIONS_GEN")
    lines.append("")

    # LOOKUP TABLE
    lines.append("#ifdef GemmLdsBlacklist_LOOKUP_TABLE_GEN")
    lines.append("")
    for (arch, dtype), projs in results.items():
        stem = ident(dtype, arch)
        count = f"n{stem[0].upper()}{stem[1:]}"
        key = f"{arch}_gemm_{dtype}"
        lines.append(f'{{"{key}", {{LdsBlacklistGemm::{stem}, LdsBlacklistGemm::{count}}}}},')
    lines.append("")
    lines.append("#endif  // GemmLdsBlacklist_LOOKUP_TABLE_GEN")
    lines.append("")

    path.write_text("\n".join(lines))


# Marker delimiting each per-(arch,dtype) array in the DEFINITIONS section.
_BEGIN_RE = re.compile(r"//\s*BEGIN_gemm_(?P<dtype>\w+)_(?P<arch>gfx\w+)_LDS_BLACKLIST")
_KEY_RE = re.compile(r"\{\s*([\d,\s]+?)\s*\}")


def parse_inc(path: Path) -> Dict[Tuple[str, str], List[Projection]]:
    """Parse an existing .inc back into {(arch, dtype_key): [projection,...]}.

    Reads the DEFINITIONS blocks (delimited by BEGIN_/END_ markers) so it works
    regardless of which macro is defined at compile time.
    """
    results: Dict[Tuple[str, str], List[Projection]] = {}
    if not path.exists():
        return results
    cur_key = None
    for line in path.read_text().splitlines():
        m = _BEGIN_RE.search(line)
        if m:
            cur_key = (m.group("arch"), m.group("dtype"))
            results[cur_key] = []
            continue
        if cur_key is None:
            continue
        if "END_gemm_" in line:
            cur_key = None
            continue
        km = _KEY_RE.search(line)
        if km:
            nums = tuple(int(x) for x in km.group(1).split(","))
            if len(nums) == len(PROJECTION_NAMES):
                results[cur_key].append(nums)
    return {k: v for k, v in results.items() if v}


# =============================================================================
# Modes
# =============================================================================


def run_generate(paths: perfRunner.Paths, arches: List[str], dtypes: List[str], dims: List[int],
                 jobs: int, out_path: Path) -> int:
    results: Dict[Tuple[str, str], List[Projection]] = {}
    total = 0
    for arch in arches:
        for dtype in dtypes:
            gen_dtype = DTYPE_GEN[dtype]
            reps, overflowing = overflowing_projections(paths, arch, gen_dtype, dims, jobs)
            if not reps:
                print(f"== {arch}/{dtype}: no configs (unsupported?), skipping", file=sys.stderr)
                continue
            print(
                f"== {arch}/{dtype}: {len(overflowing)}/{len(reps)} distinct tile shapes "
                f"blacklisted",
                file=sys.stderr)
            if overflowing:
                results[(arch, dtype)] = sorted(overflowing)
                total += len(overflowing)

    emit_inc(results, out_path)
    print(
        f"Wrote {total} blacklisted tile shapes across {len(results)} (arch,dtype) key(s) to "
        f"{out_path}",
        file=sys.stderr)
    return 0


def run_verify(paths: perfRunner.Paths,
               dims: List[int],
               jobs: int,
               inc_path: Path,
               samples: Optional[int] = None,
               seed: Optional[int] = None) -> int:
    """Re-check blacklisted projections still overflow LDS. Non-zero on drift.

    Each entry is verified independently by reconstructing a representative
    perf-config (synth_config) and re-lowering it, so the cost is one compile
    per checked entry -- no need to re-enumerate the whole space. With
    ``samples`` set, only that many randomly chosen entries are checked (seeded
    by ``seed``, defaulting to the ISO week) to keep nightly runtime bounded.
    """
    existing = parse_inc(inc_path)
    if not existing:
        # An empty blacklist is a failure, not a pass: it almost always means the
        # .inc was lost, truncated, or never generated. Passing here would let the
        # drift check silently green-light a table that prunes nothing, defeating
        # its purpose. Regenerate with generateLDSBlacklist.py to recover.
        print(
            f"ERROR: no blacklist entries found in {inc_path}; the blacklist is empty. "
            f"Regenerate with:\n    python3 {generator_rel_path()}",
            file=sys.stderr)
        return 1

    flat: List[Tuple[str, str, Projection]] = [(arch, dtype, p)
                                               for (arch, dtype), projs in existing.items()
                                               if DTYPE_GEN.get(dtype) is not None for p in projs]
    if not flat:
        # Entries exist but none are verifiable (all under unknown dtype keys):
        # also a failure, since nothing can actually be re-checked.
        print(f"ERROR: no verifiable blacklist entries in {inc_path} (unknown dtype keys?).",
              file=sys.stderr)
        return 1

    total = len(flat)
    if samples is not None and samples < total:
        used_seed = seed if seed is not None else default_seed()
        flat = random.Random(used_seed).sample(flat, samples)
        print(f"Sampling {samples}/{total} blacklist entries (seed={used_seed}).", file=sys.stderr)

    def check(item: Tuple[str, str, Projection]):
        arch, dtype, p = item
        return item, classify_lds(paths, arch, DTYPE_GEN[dtype], synth_config(p), dims)

    stale: Dict[Tuple[str, str], List[Projection]] = {}
    inconclusive = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        results = ex.map(check, flat)
        for (arch, dtype, p), verdict in tqdm(results,
                                              total=len(flat),
                                              desc="verify",
                                              unit="perf_config"):
            # Only a clean lowering that no longer overflows is real drift; a
            # rejected/errored reconstruction is inconclusive, not drift.
            if verdict == LDS_FITS:
                stale.setdefault((arch, dtype), []).append(p)
            elif verdict == LDS_INCONCLUSIVE:
                inconclusive += 1

    checked = len(flat)
    if inconclusive:
        print(
            f"Note: {inconclusive}/{checked} sampled entries were inconclusive "
            f"(config rejected/errored) and treated as still-valid.",
            file=sys.stderr)
    if stale:
        print(
            f"\nDRIFT DETECTED: {sum(len(v) for v in stale.values())} of {checked} checked "
            f"tile shape(s) no longer overflow LDS. Regenerate with:\n"
            f"    python3 {generator_rel_path()}",
            file=sys.stderr)
        for (arch, dtype), projs in stale.items():
            for p in projs:
                print(f"    {arch}/{dtype}: {p}", file=sys.stderr)
        return 1

    print(f"\nOK: all {checked} checked tile shape(s) still overflow LDS.", file=sys.stderr)
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate/verify LdsBlacklistPerfconfigs.inc (GEMM tiles that overflow LDS).")
    parser.add_argument("-o",
                        "--output",
                        default=str(default_output_path()),
                        metavar="FILE",
                        help="Output/input .inc path (default: the in-tree include location)")
    parser.add_argument("--verify",
                        action="store_true",
                        help="Drift check: re-verify the existing .inc instead of regenerating")
    parser.add_argument("--samples",
                        type=int,
                        default=None,
                        metavar="N",
                        help="With --verify, check only N randomly sampled blacklist entries "
                        "instead of all (keeps nightly runtime bounded; default: all)")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        metavar="S",
                        help="RNG seed for --samples. Defaults to the current ISO week so a "
                        "given week's nightly run is reproducible (see parameterSweeps.py)")
    parser.add_argument("-j",
                        "--jobs",
                        type=int,
                        default=os.cpu_count() or 8,
                        metavar="N",
                        help="Parallel compile pipelines (default: CPU count)")
    parser.add_argument("--mlir-build-dir",
                        default=perfRunner.find_mlir_build_dir(),
                        metavar="DIR",
                        help="rocmlirTriton build dir containing bin/rocmlir-gen etc.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    paths = perfRunner.create_paths(None, args.mlir_build_dir)
    if not paths.mlir_paths:
        print(
            "ERROR: rocmlirTriton build dir was not provided/found; "
            "specify it with --mlir-build-dir DIR",
            file=sys.stderr)
        return 1

    if args.verify:
        return run_verify(paths, DEFAULT_DIMS, args.jobs, Path(args.output), args.samples,
                          args.seed)
    return run_generate(paths, DEFAULT_ARCHES, DEFAULT_DTYPES, DEFAULT_DIMS, args.jobs,
                        Path(args.output))


if __name__ == "__main__":
    sys.exit(main())
