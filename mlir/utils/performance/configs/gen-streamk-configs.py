#!/usr/bin/env python3
"""Derive the stream-K candidate subsets of the tier1 gemm/conv config lists.

These lists target the regime the current StreamKDecompose pass can actually
handle: an *over*-filled tile grid with a ragged tail. The pass builds a
persistent grid of P = streamKMultiple * num_cu workgroups out of rectangular
gridwise_gemm waves plus a split-K remainder, which structurally forces
P <= gridFull. So if the data-parallel grid underfills the machine
(gridFull < num_cu) the pass cannot build a P >= num_cu decomposition and bails,
and only plain split-K can fill the CUs. (Underutilized / GEMV shapes therefore
fall to split-K, not stream-K -- see the gridFull < num_cu skip in
computeOptimalStreamKMultiples.)

A config is kept when *all* of the following hold. The grid is modelled at the
*per-dimension* efficient tile the exhaustive tuner actually picks -- the next
power of two that fits the dim, capped at ``MAX_TILE`` (128):

    tile(d) = min(MAX_TILE, next_pow2(d))
    grid    = gemmG * ceil(gemmM / tile(gemmM)) * ceil(gemmN / tile(gemmN))

Modelling at the *smallest* tile instead would over-count tiles and wrongly
predict an over-filled grid: the tuner selects a large efficient tile for
compute-bound work, which shrinks the grid, and if that underfills the CUs it
uses split-K (not stream-K) to fill them.

0. Fills a tile: ``gemmM >= MIN_TILE`` and ``gemmN >= MIN_TILE``. A GEMV / skinny
   shape (dim < one tile) is a memory-bound kernel that falls to split-K --
   stream-K can't help a shape that never fills a tile.

1. Over-filled grid: ``grid >= num_cu`` (a P >= num_cu wave decomposition exists).

2. Ragged tail: the grid does not divide evenly across the CUs, so the trailing
   wave is partially empty -- exactly the waste stream-K reclaims:

       imbalance = ceil(grid / num_cu) * num_cu / grid  >=  1.20

3. Splittable K: the remainder slab re-splits K, so ``gemmK >= 2 * MIN_K_PER_SPLIT``.

4. Compute-bound: arithmetic intensity
   ``2*M*N*K / (dtype_bytes*(M*K + K*N + M*N)) >= MIN_ARITH_INTENSITY``. Stream-K
   repacks the compute tail but does nothing for a bandwidth-bound kernel (it even
   adds atomic read-modify-write traffic), so memory-bound shapes fall to split-K.

Per-arch knobs (num_cu, atomic-add dtype support, and hence the eligible dtype /
conv op tokens) are in ``ARCHS`` below; one gemm + one conv list is emitted per
arch.

Backward-data conv (-F 2) is mapped with the aggregate implicit-gemm size and
does NOT model the per-gemmId stride decomposition, so its imbalance is a lower
bound (conservative: may under-count a few bwd candidates).

Re-run to regenerate:
    python3 gen-streamk-configs.py
"""

import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# Smallest efficient output tile the tuner will realistically pick (MFMA/WMMA
# f32 tiles bottom out around 32; the observed tuned perf_configs use 32/64).
# Used only for the GEMV / "fills a tile" gate.
MIN_TILE = 32
# Largest output tile the tuner picks (observed tuned perf_configs top out at
# 128x128). The grid is modelled at the *per-dimension* efficient tile
# min(MAX_TILE, next_pow2(dim)): exhaustive tuning selects the tile that best fits
# each dim for compute efficiency, so a small dim uses a small tile (few tiles)
# and only genuinely large dims yield many tiles. Modelling the grid at the tiny
# MIN_TILE instead would wildly over-count tiles and wrongly predict an
# over-filled grid for shapes the tuner actually runs at a large tile (where the
# grid underfills and split-K -- not stream-K -- fills the CUs).
MAX_TILE = 128
IMBALANCE_THRESHOLD = 1.20
# Splittability gate: each K-split workgroup must reduce over at least this many
# K elements (a few kPerBlock iterations) to be worth its launch + atomic write.
MIN_K_PER_SPLIT = 128
# Arithmetic-intensity gate (FLOP per byte). Stream-K only pays off on
# compute-bound GEMMs: it repacks the compute tail but does nothing for a
# bandwidth-bound kernel (it even adds atomic read-modify-write traffic). A shape
# below this intensity is memory-bound and falls to split-K, so drop it. The knee
# is dtype-dependent (higher-peak matrix dtypes have a higher roofline knee); 60
# is a conservative f32 threshold that also excludes the clearly memory-bound
# low-K shapes for the matrix dtypes.
MIN_ARITH_INTENSITY = 60
# Bytes per element, used for the arithmetic-intensity estimate.
DTYPE_BYTES = {'f32': 4, 'f16': 2, 'bf16': 2, 'int8': 1}


class ArchSpec:
    def __init__(self, name, label, num_cu, gemm_dtypes, conv_tokens):
        self.name = name              # file-name suffix
        self.label = label            # header description
        self.num_cu = num_cu          # real device CU count (getNumCUValue)
        self.gemm_dtypes = gemm_dtypes  # eligible `-t` values
        self.conv_tokens = conv_tokens  # eligible conv op tokens -> dtype


# Fast atomic_add support (isFastAtomicAddSupported, AmdArchDb.cpp):
#   RDNA3 (gfx1100): f32 only.
#   CDNA4 (gfx950) : f32, f16, bf16.
ARCHS = [
    ArchSpec(
        name='navi3x',
        label='Navi3x (RDNA3, gfx1100)',
        num_cu=70,
        gemm_dtypes={'f32'},
        conv_tokens={'conv': 'f32'},
    ),
    ArchSpec(
        name='mi350',
        label='MI350 (CDNA4, gfx950)',
        num_cu=256,
        gemm_dtypes={'f32', 'f16'},
        conv_tokens={'conv': 'f32', 'convfp16': 'f16'},
    ),
]


def _pow2_ceil(x: int) -> int:
    """Smallest power of two >= x (x >= 1)."""
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _tile(dim: int) -> int:
    """Efficient tile the tuner would pick for an output dim: the next power of
    two that fits it, capped at MAX_TILE (the tuner won't use a tile larger than
    128, nor one much bigger than the dim)."""
    return min(MAX_TILE, _pow2_ceil(dim))


def _grid(gemm_g: int, gemm_m: int, gemm_n: int) -> int:
    """Data-parallel grid at the per-dimension efficient tile the tuner picks."""
    return (gemm_g * math.ceil(gemm_m / _tile(gemm_m))
            * math.ceil(gemm_n / _tile(gemm_n)))


def imbalance(gemm_g: int, gemm_m: int, gemm_n: int, num_cu: int) -> float:
    """Data-parallel work imbalance, mirroring computeWorkImbalance() with
    splitK=1 at the per-dimension efficient tile the tuner picks."""
    grid = _grid(gemm_g, gemm_m, gemm_n)
    if grid <= 0:
        return 1.0
    max_wg_per_cu = math.ceil(grid / num_cu)
    return (max_wg_per_cu * num_cu) / grid


def arith_intensity(gemm_m: int, gemm_n: int, gemm_k: int,
                    dtype_bytes: int) -> float:
    """GEMM arithmetic intensity in FLOP/byte (ignores G, which cancels):
        2*M*N*K / (dtype_bytes * (M*K + K*N + M*N))."""
    flops = 2 * gemm_m * gemm_n * gemm_k
    byts = dtype_bytes * (gemm_m * gemm_k + gemm_k * gemm_n + gemm_m * gemm_n)
    return flops / byts if byts else 0.0


def is_candidate(gemm_g: int, gemm_m: int, gemm_n: int, gemm_k: int,
                 num_cu: int, dtype_bytes: int) -> bool:
    if gemm_m <= 0 or gemm_n <= 0 or gemm_k <= 0:
        return False
    # Gate 0: reject GEMV / skinny shapes. If gemmM or gemmN is below one output
    # tile, ceil(dim / MIN_TILE) == 1 counts a *full* tile while <1/MIN_TILE of it
    # holds real data -- the grid looks over-filled but the kernel is a
    # memory-bound GEMV, so it lands on split-K with poor efficiency (stream-K
    # can't rescue a shape that never fills a tile). Require at least one full
    # tile in each output dim.
    if gemm_m < MIN_TILE or gemm_n < MIN_TILE:
        return False
    grid = _grid(gemm_g, gemm_m, gemm_n)
    # Gate 1: the grid must *over*-fill the machine. StreamKDecompose forces the
    # persistent grid P <= gridFull, so if gridFull < num_cu it cannot build a
    # P >= num_cu wave decomposition and bails (only plain split-K helps there).
    if grid < num_cu:
        return False
    # Gate 2: ragged tail -- the over-filled grid does not divide evenly across
    # CUs, so the last wave is partially empty (this is what stream-K reclaims).
    if imbalance(gemm_g, gemm_m, gemm_n, num_cu) < IMBALANCE_THRESHOLD:
        return False
    # Gate 3: the remainder slab re-splits K, so K must be splittable at all.
    if gemm_k < 2 * MIN_K_PER_SPLIT:
        return False
    # Gate 4: compute-bound only. Stream-K repacks the compute tail but does
    # nothing for a bandwidth-bound kernel (and adds atomic traffic), so a
    # memory-bound shape falls to split-K -- drop it.
    return arith_intensity(gemm_m, gemm_n, gemm_k, dtype_bytes) >= \
        MIN_ARITH_INTENSITY


# ---------------------------------------------------------------------------
# GEMM
# ---------------------------------------------------------------------------
def _int_flag(line: str, flag: str):
    m = re.search(rf"(?<!\S){re.escape(flag)}\s+(-?\d+)(?!\S)", line)
    return int(m.group(1)) if m else None


def _str_flag(line: str, flag: str):
    m = re.search(rf"(?<!\S){re.escape(flag)}\s+(\S+)", line)
    return m.group(1) if m else None


def gemm_candidates(lines, arch: ArchSpec):
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        dtype = _str_flag(line, '-t')
        if dtype not in arch.gemm_dtypes:
            continue
        g = _int_flag(line, '-g') or 1
        m = _int_flag(line, '-m')
        n = _int_flag(line, '-n')
        k = _int_flag(line, '-k')
        if m is None or n is None or k is None:
            continue
        if is_candidate(g, m, n, k, arch.num_cu, DTYPE_BYTES[dtype]):
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# CONV  (MIOpen driver syntax: `conv[fp16|int8] -F <dir> ... `)
# ---------------------------------------------------------------------------
def conv_gemm_size(line: str, arch: ArchSpec):
    """Return (gemmG, gemmM, gemmN, gemmK) for a conv line, or None to skip.

    Only op tokens whose dtype has fast atomic_add on this arch are kept.
    Forward and weight-grad use the exact implicit-gemm mapping; backward-data
    uses the aggregate size (see module docstring)."""
    tok = line.split()[0]
    if tok not in arch.conv_tokens:
        return None
    direction = {1: 'fwd', 2: 'bwd', 4: 'wrw'}.get(_int_flag(line, '-F'), 'fwd')
    n = _int_flag(line, '-n')
    c = _int_flag(line, '-c')
    hi = _int_flag(line, '-H')
    wi = _int_flag(line, '-W')
    k = _int_flag(line, '-k')
    y = _int_flag(line, '-y')
    x = _int_flag(line, '-x')
    p = _int_flag(line, '-p') or 0
    q = _int_flag(line, '-q') or 0
    u = _int_flag(line, '-u') or 1
    v = _int_flag(line, '-v') or 1
    dil_h = _int_flag(line, '-l') or 1
    dil_w = _int_flag(line, '-j') or 1
    g = _int_flag(line, '-g') or 1
    if None in (n, c, hi, wi, k, y, x):
        return None

    # Output spatial dims (mirrors ConvConfiguration.ho/wo in perfRunner.py).
    ho = math.floor((hi + 2 * p - (y - 1) * dil_h - 1) / u) + 1
    wo = math.floor((wi + 2 * q - (x - 1) * dil_w - 1) / v) + 1

    if direction == 'fwd':
        # gemmK = im2col reduction = (Cin/g) * filterY * filterX.
        return (g, k // g, n * ho * wo, (c // g) * y * x)
    if direction == 'wrw':
        # gemmK = batch * output spatial (the reduction axis of weight-grad).
        return (g, k // g, (c // g) * y * x, n * ho * wo)
    # bwd (data): aggregate implicit-gemm size; gemmK = (Cout/g) * filterY*X.
    return (g, c // g, n * hi * wi, (k // g) * y * x)


def conv_candidates(lines, arch: ArchSpec):
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        size = conv_gemm_size(line, arch)
        if size is None:
            continue
        dtype = arch.conv_tokens[line.split()[0]]
        if is_candidate(*size, arch.num_cu, DTYPE_BYTES[dtype]):
            out.append(line)
    return out


def header(src: str, arch: ArchSpec) -> str:
    return f"""\
# Stream-K candidate subset of {src} for {arch.label}.
#
# Generated by gen-streamk-configs.py. Modelling the grid at the per-dimension
# efficient tile the tuner picks, tile(d) = min({MAX_TILE}, next_pow2(d)),
#   grid = gemmG * ceil(gemmM / tile(gemmM)) * ceil(gemmN / tile(gemmN))   (num_cu = {arch.num_cu})
# a config is kept only when it (0) fills at least one output tile in each dim
# (gemmM >= {MIN_TILE} and gemmN >= {MIN_TILE}, else it is a memory-bound GEMV that
# falls to split-K), (1) over-fills the machine (grid >= num_cu, so
# StreamKDecompose can build a P >= num_cu wave decomposition rather than bailing),
# (2) has a ragged tail -- imbalance = ceil(grid/num_cu)*num_cu/grid >=
# {IMBALANCE_THRESHOLD:.2f} -- (3) has splittable K (gemmK >= {2 * MIN_K_PER_SPLIT}), and (4) is
# compute-bound -- arith. intensity 2MNK/(bytes*(MK+KN+MN)) >= {MIN_ARITH_INTENSITY}.
# Underutilized / GEMV / memory-bound shapes are excluded: the pass can't
# decompose them (or gains nothing) and they fall to plain split-K instead.
# Eligible dtypes are those with fast
# atomic_add on this arch: {', '.join(sorted(arch.gemm_dtypes))}.
# See gen-streamk-configs.py for the full rationale.
"""


def write_list(path, src, arch, entries):
    with open(path, 'w') as f:
        f.write(header(src, arch))
        f.write('\n')
        if entries:
            f.write('\n'.join(entries) + '\n')


def main():
    gemm_src = os.path.join(HERE, 'tier1-gemm-configs')
    conv_src = os.path.join(HERE, 'tier1-conv-configs')

    with open(gemm_src) as f:
        gemm_lines = f.readlines()
    with open(conv_src) as f:
        conv_lines = f.readlines()

    for arch in ARCHS:
        gemm_out = gemm_candidates(gemm_lines, arch)
        conv_out = conv_candidates(conv_lines, arch)

        gemm_dst = os.path.join(HERE, f'tier1-gemm-streamk-{arch.name}-configs')
        conv_dst = os.path.join(HERE, f'tier1-conv-streamk-{arch.name}-configs')
        write_list(gemm_dst, 'tier1-gemm-configs', arch, gemm_out)
        write_list(conv_dst, 'tier1-conv-configs', arch, conv_out)

        n_gemm = sum(1 for l in gemm_lines if l.strip() and not l.startswith('#')
                     and _str_flag(l, '-t') in arch.gemm_dtypes)
        n_conv = sum(1 for l in conv_lines if l.strip() and not l.startswith('#')
                     and l.split()[0] in arch.conv_tokens)
        print(f"[{arch.name}] gemm: {len(gemm_out)} / {n_gemm} eligible kept; "
              f"conv: {len(conv_out)} / {n_conv} eligible kept")


if __name__ == '__main__':
    main()
