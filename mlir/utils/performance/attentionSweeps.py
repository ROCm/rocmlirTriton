#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Sweep random (problem-shape, perf-config) combinations for the
``GemmGemmParamsAttr`` perf-config family (``attn:``) — i.e. attention
and gemm+elementwise+gemm — through the rocMLIR pipeline and classify each
as PASS / NOT_APPLICABLE / FAIL. Run from the build directory.

Companion script to ``parameterSweeps.py``: this one targets attention and
gemm+gemm kernels, and reuses the shared ``Options``, ``PerfConfig``,
perf-config sampler, and the ``run_config`` orchestrator from there.

Requires Python 3.9 or newer (uses ``asyncio.to_thread``).

Usage:
    $ ninja check-rocmlir-build-only ci-performance-scripts
    $ python3 bin/attentionSweeps.py {attention|gemm_gemm} [--samples N] [--seed S]
                                                            [--jobs J] [--debug]

The config-name spelling matches ``rocmlir-gen --operation`` so the
positional argument here is the same string you would pass to the binary."""

from __future__ import annotations

import argparse
import asyncio
import random
import sys

from typing import Sequence, Optional, Tuple

import perfRunner
from perfRunner import auto_precision_flags_att
from parameterSweeps import (
    Options,
    PerfConfig,
    _MAX_PERF_CONFIG_RETRIES,
    _split_k_choices,
    add_common_args,
    build_options_and_paths,
    default_seed,
    run_config,
    sample_perf_config,
)

# Hard dependency, copied next to the scripts by ci-performance-scripts.
import amd_arch_db

# Per-thread VGPR-budget guard for the attention / gemm-gemm sweep.
# Complements ``parameterSweeps._compile_cost_score`` (which models post-RA
# *scheduler* cost from basic-block instruction count) by modelling
# *register-allocator* cost. The rocmlir-driver path stamps
# ``amdgpu-waves-per-eu="N,N"`` on the kernel function for every
# ``wavesPerEU > 0`` (see ``setKernelAttributes`` in
# ``mlir/lib/Translation/TritonToHsaco/TritonToHsaco.cpp``), so the AMDGPU
# backend caps each thread's VGPRs at ``vgprs_per_eu / N``. When the
# C-tile accumulator
#     accum_vgprs_per_thread = (mPerBlock * nPerBlock) / threads
# (one VGPR per element since every supported dtype accumulates into
# f32 / i32) overruns that cap, the allocator can't avoid spilling and
# spends multi-minute time searching the live-range graph -- so we reject.
#
# ``wavesPerEU == 0`` is the "let the backend pick" case: no
# ``amdgpu-waves-per-eu`` attribute is emitted, so the LLVM AMDGPU backend
# has no occupancy limit and the upper bound comes from the hardware --
# at most one wave per EU getting the full SIMD VGPR file. Tiles wider
# than ``vgprs_per_eu`` still trigger a ~10s-to-multi-minute ISel /
# post-RA scheduling blow-up even without spills, so we treat ``wpe == 0``
# as ``wpe == 1`` for the budget check.
#
# The per-EU VGPR-file size comes from ``rock::getVGPRsPerEU`` via the
# AmdArchDB pybind module (``amd_arch_db.get_vgprs_per_eu``), so the budget
# tracks LLVM's per-arch VGPR coverage instead of a hand-coded heuristic.


def _waves_per_eu_register_budget_ok(perf_config: Sequence[int], arch: str) -> bool:
    """``True`` iff the C-tile accumulator fits the wpe-implied VGPR budget:
    ``(mPerBlock * nPerBlock) / threads <= vgprs_per_eu / effective_wpe``,
    where ``threads = numWaves * waveSize`` and ``effective_wpe`` is
    ``wavesPerEU`` itself, or ``1`` when ``wavesPerEU == 0``."""
    # 12-field attn layout (nPerBlockG1 is field 2), so numWaves/wavesPerEU
    # sit one slot later than in the 11-field gemm layout.
    mpb, npb = perf_config[0], perf_config[1]
    num_waves = perf_config[6]
    waves_per_eu = perf_config[10]
    threads = max(1, num_waves * amd_arch_db.get_wave_size(arch))
    effective_waves_per_eu = waves_per_eu if waves_per_eu > 0 else 1
    # Integer cross-multiplication is exact; avoids any FP rounding at the
    # boundary. Equivalent to ``(mpb * npb) / threads <= vgprs_per_eu / wpe``.
    return (mpb * npb) * effective_waves_per_eu <= amd_arch_db.get_vgprs_per_eu(arch) * threads


def _sampled_perf_within_vgpr_budget(rng: random.Random,
                                     arch: str,
                                     split_k_choices: Sequence[int],
                                     pow2_only: bool = False) -> Tuple[int, ...]:
    """Like :func:`sample_perf_config` but rejects perf-configs whose
    C-tile accumulator overruns the VGPR budget implied by ``wavesPerEU``
    (see :func:`_waves_per_eu_register_budget_ok`). ``pow2_only`` is forwarded
    to ``sample_perf_config`` for callers whose pipeline does not decompose
    non-pow2 tiles. We loop with a
    generous retry cap rather than enumerating the in-budget subspace
    because the rejection rate is small (<10% on the archs we care about)
    and we don't want to silently bias the sample distribution by
    filtering an enumerated list -- mirrors the rationale in
    :func:`parameterSweeps._sampled_perf_within_budget`.

    We deliberately do NOT also apply ``parameterSweeps._perf_within_budget``:
    its basic-block instruction-count cap is calibrated for conv / gemm
    and doesn't catch the regalloc thrash this predicate guards against."""
    for _ in range(_MAX_PERF_CONFIG_RETRIES):
        perf_config = sample_perf_config(rng,
                                         arch,
                                         split_k_choices,
                                         pow2_only=pow2_only,
                                         is_attention=True)
        if _waves_per_eu_register_budget_ok(perf_config, arch):
            return perf_config
    raise RuntimeError(
        f"_sampled_perf_within_vgpr_budget exceeded {_MAX_PERF_CONFIG_RETRIES} retries "
        f"for arch={arch!r}; PERF_CONFIG_OPTIONS may have no config inside the "
        "VGPR budget implied by the sampled wavesPerEU values.")


# Cap on the per-config CPU validation cost: max(seq_len_q, seq_len_k) * g.
# Every shape produced by ``_sample_attn_shape`` already respects this bound
# by construction, so no post-hoc filtering or top-up loop is needed.
MAX_TOKENS = 4096
MAX_FIRST_MATMUL = 512 * 512
MAX_HEAD_DIM = 256
SPLIT_KV_OPTIONS = [1, 2, 4, 8, 16, 32]

# Attention problem-shape sweep. Dtype list is sourced from perfRunner so the
# sweep automatically picks up dtype additions/removals on the runner side.
ATTN_SHAPE_OPTIONS = {
    'dtype': perfRunner.DATA_TYPES_ATTENTION,
    # Powers of two up to 128 cover the typical num_heads_q/num_heads_kv range.
    'num_heads_pow2': [1, 2, 3, 4, 5, 6, 7],
}


def _sample_num_heads(rng: random.Random) -> tuple[int, int]:
    """Pick (num_heads_q, num_heads_kv) for attention.

    Half the time returns the no-GQA case ``(1, 1)``. The other half picks
    a real GQA pair such that ``q > kv and q % kv == 0`` (strict ``>`` so we
    actually exercise the GQA path; the equal-heads case is already covered
    by the no-GQA branch). Both ``q`` and ``kv`` are powers of two."""
    if rng.choice([True, False]):
        return (1, 1)
    while True:
        num_heads_q = 2**rng.choice(ATTN_SHAPE_OPTIONS['num_heads_pow2'])
        num_heads_kv = 2**rng.choice(ATTN_SHAPE_OPTIONS['num_heads_pow2'])
        if num_heads_q > num_heads_kv and num_heads_q % num_heads_kv == 0:
            return (num_heads_q, num_heads_kv)


def _sample_attn_shape(rng: random.Random, n_per_block: int):
    """Returns one random attention problem-shape tuple. By construction,
    ``max(seq_len_q, seq_len_k) * g <= MAX_TOKENS``,
    ``seq_len_q * seq_len_k <= MAX_FIRST_MATMUL``, and -- when ``causal`` and
    ``split_kv > 1`` are picked -- ``seq_len_q <= n_per_block``, so callers
    don't need to re-filter."""
    g = rng.randint(1, 8)
    # Per-group sequence-length budget (so total tokens stay under MAX_TOKENS).
    max_valid_seqlen = max(1, MAX_TOKENS // g)

    # Hoisted out of the trailing return tuple so we can apply the
    # causal+split_kv+seq_len_q rocmlir-gen limitation when we pick seq_len_q
    # below. ``split_kv`` only makes sense with ``return_lse`` (the LSE output
    # is what makes the partial reductions reassemblable across the K split).
    return_lse = rng.choice([True, False])
    split_kv = rng.choice(SPLIT_KV_OPTIONS) if return_lse else 1

    use_kvcache = rng.choice([True, False])
    seqlen_k = rng.randint(1, max_valid_seqlen)

    # Causal masking combined with KV-cache decode (seq_len_q == 1) doesn't
    # make sense: the single query attends only to cached past keys anyway.
    causal = False if use_kvcache else rng.choice([True, False])

    # KV-cache mode: a single query token per call. Each group's inclusive
    # `last_valid_kv_index` P makes keys [0, P] valid. Otherwise cap seq_len_q so:
    #   1. The first matmul (Q @ K^T, shape seq_len_q x seq_len_k) stays under
    #      MAX_FIRST_MATMUL elements -- the dominant cost on prefill shapes.
    #   2. When causal masking is combined with split-KV, seq_len_q stays at
    #      most n_per_block, matching rocmlir-gen.cpp's check
    #      "Causal masking + split-KV is not supported with sequenceLengthQ
    #      > nPerBlock (rocmlir-gen limitation)" (around the splitKV>1 host
    #      harness path). KV-cache mode (seqlen_q=1) is always within bound
    #      since n_per_block >= 16 in PERF_CONFIG_OPTIONS.
    if use_kvcache:
        seqlen_q = 1
    else:
        max_seqlen_q = max(1, min(max_valid_seqlen, MAX_FIRST_MATMUL // seqlen_k))
        if causal and split_kv > 1:
            max_seqlen_q = min(max_seqlen_q, n_per_block)
        seqlen_q = rng.randint(1, max_seqlen_q)
    # P is an inclusive index, not a count. In particular, P == 0 means one
    # valid cached key. Sample the complete legal range [0, seqlen_k - 1].
    last_valid_kv_index = ([rng.randint(0, seqlen_k - 1) for _ in range(g)]
                           if use_kvcache else None)

    # A sliding look-back L is strictly positive and bounded by seqlen_k - 1.
    # Sweeps use explicit last-valid indices, while reconstructed tuning keys
    # may use rocmlir-gen's full-cache default. None is the non-sliding spelling;
    # keep that path and skip sliding when seqlen_k == 1 because no valid L exists.
    sliding_window_look_back = (rng.randint(1, seqlen_k - 1) if use_kvcache and seqlen_k > 1 and
                                rng.choice([True, False]) else None)

    num_heads_q, num_heads_kv = _sample_num_heads(rng)

    return (
        rng.choice(ATTN_SHAPE_OPTIONS['dtype']),
        g,
        seqlen_q,
        seqlen_k,
        num_heads_q,
        num_heads_kv,
        rng.randint(1, MAX_HEAD_DIM),  # head_dim_qk
        rng.randint(1, MAX_HEAD_DIM),  # head_dim_v
        rng.choice([True, False]),  # with_attn_scale
        rng.choice([True, False]),  # with_attn_bias
        rng.choice([True, False]),  # trans_q
        rng.choice([True, False]),  # trans_k
        rng.choice([True, False]),  # trans_v
        rng.choice([True, False]),  # trans_o
        causal,
        return_lse,
        split_kv,
        last_valid_kv_index,
        sliding_window_look_back,
    )


def random_attn_cases(num_samples: int, arch: str, seed: Optional[int] = None):
    """Yields ``num_samples`` random ``(attn_shape, perf_config)`` tuples.

    The perf-config's ``splitKFactor`` is pinned to 1: attention exposes its
    K-dim split via the dedicated ``-split_kv`` kernel arg (sampled inside
    ``_sample_attn_shape``), not via the perf-config splitK.

    Perf-config is sampled first (and filtered through
    :func:`_sampled_perf_within_vgpr_budget`) so we can feed ``nPerBlock``
    -- field 1 in the 11-tuple -- into ``_sample_attn_shape`` to enforce
    the rocmlir-gen ``causal + split_kv`` limitation on ``seq_len_q``."""
    rng = random.Random(seed if seed is not None else default_seed())
    for _ in range(num_samples):
        # Attention uses GemmGemmParamsAttr and isn't lowered through
        # rock-decompose-nonpow2-tiles, so keep the m/n tile on the pow2 grid.
        perf = _sampled_perf_within_vgpr_budget(rng, arch, [1], pow2_only=True)
        n_per_block = perf[1]
        yield (_sample_attn_shape(rng, n_per_block=n_per_block), perf)


# Gemm+gemm problem-shape sweep.
MAX_GEMM_GEMM_DIM = 128
GEMM_GEMM_SHAPE_OPTIONS = {
    'dtype': perfRunner.DATA_TYPES_GEMM_GEMM,
    'g': [1, 2],
    'trans': [False, True],
}


def _sample_gemm_gemm_shape(rng: random.Random):
    opts = GEMM_GEMM_SHAPE_OPTIONS
    return (
        rng.choice(opts['dtype']),
        rng.choice(opts['g']),
        rng.randint(1, MAX_GEMM_GEMM_DIM),  # m
        rng.randint(1, MAX_GEMM_GEMM_DIM),  # k
        rng.randint(1, MAX_GEMM_GEMM_DIM),  # n
        rng.randint(1, MAX_GEMM_GEMM_DIM),  # o
        rng.choice(opts['trans']),  # trans_a
        rng.choice(opts['trans']),  # trans_b
        rng.choice(opts['trans']),  # trans_c
        rng.choice(opts['trans']),  # trans_o
    )


def random_gemm_gemm_cases(num_samples: int, arch: str, seed: Optional[int] = None):
    """Yields ``num_samples`` random ``(gemm_gemm_shape, perf_config)`` tuples.

    Unlike attention, ``splitKFactor`` is left free: for gemm+gemm the
    second gemm's K-dim split is a real degree of freedom rather than a
    duplicate of a separate kernel arg. Perf-configs are filtered through
    :func:`_sampled_perf_within_vgpr_budget`."""
    rng = random.Random(seed if seed is not None else default_seed())
    for _ in range(num_samples):
        shape = _sample_gemm_gemm_shape(rng)
        # shape[0] is the input dtype (dtype, g, m, k, n, o, trans_a, ...).
        # gemm+gemm also uses GemmGemmParamsAttr (no non-pow2 tile decompose),
        # so keep the m/n tile on the pow2 grid.
        dtype = shape[0]
        yield (shape,
               _sampled_perf_within_vgpr_budget(rng, arch, _split_k_choices(dtype), pow2_only=True))


def to_gemm_gemm_test(params, options: Options) -> perfRunner.GemmGemmConfiguration:
    shape, perf = params
    dtype, g, m, k, n, o, ta, tb, tc, to = shape
    # ``kind='attn'`` is intentional: gemm+gemm and attention share the same
    # GemmGemmParamsAttr perf-config family (serialized as ``attn:...``);
    # see the ``PerfConfig`` docstring and RockAttrDefs.td.
    return perfRunner.GemmGemmConfiguration(
        dtype=dtype,
        g=g,
        m=m,
        k=k,
        n=n,
        o=o,
        trans_a=ta,
        trans_b=tb,
        trans_c=tc,
        trans_o=to,
        arch=options.arch,
        num_cu=options.num_cu,
        num_chiplets=options.num_chiplets,
        perf_config=str(PerfConfig(perf, kind='attn')),
    )


def to_attn_test(params, options: Options) -> perfRunner.AttentionConfiguration:
    shape, perf = params
    (dtype, g, slq, slk, nhq, nhkv, hdqk, hdv, scale, bias, tq, tk, tv, to, causal, return_lse,
     split_kv, last_valid_kv_index, sliding_window_look_back) = shape
    attn_config = perfRunner.AttentionConfiguration(
        dtype=dtype,
        g=g,
        seq_len_q=slq,
        seq_len_k=slk,
        num_heads_q=nhq,
        num_heads_kv=nhkv,
        head_dim_qk=hdqk,
        head_dim_v=hdv,
        with_attn_scale=scale,
        with_attn_bias=bias,
        trans_q=tq,
        trans_k=tk,
        trans_v=tv,
        trans_o=to,
        causal=causal,
        return_lse=return_lse,
        split_kv=split_kv,
        arch=options.arch,
        num_cu=options.num_cu,
        num_chiplets=options.num_chiplets,
        perf_config=str(PerfConfig(perf, kind='attn')),
        last_valid_kv_index=last_valid_kv_index,
        sliding_window_look_back=sliding_window_look_back,
    )
    # Precision-aware rocmlir-gen flags (e.g. --pv-f64) picked up per-config
    # in parameterSweeps._build_rocmlir_gen_opts to combat CPU reference drift
    # at long seq_len for f32/bf16 attention.
    attn_config.extra_rocmlir_gen_flags = auto_precision_flags_att(attn_config)
    return attn_config


def main() -> bool:
    parser = argparse.ArgumentParser(
        description='Sweep parameter values for attention / gemm+gemm to detect bugs')
    parser.add_argument('config',
                        choices=['attention', 'gemm_gemm'],
                        help="Kind of kernel to sweep: 'attention' or 'gemm_gemm'. "
                        "The spelling matches rocmlir-gen --operation. Both share "
                        "the GemmGemmParamsAttr (attn:) perf-config family but "
                        "have different problem-shape spaces.")
    add_common_args(parser)
    args = parser.parse_args()

    options, paths = build_options_and_paths(args)

    if args.config == 'attention':
        param_iter = random_attn_cases(args.samples, options.arch, seed=args.seed)
        return asyncio.run(
            run_config(param_iter, to_attn_test, options, paths, samples=args.samples))
    if args.config == 'gemm_gemm':
        param_iter = random_gemm_gemm_cases(args.samples, options.arch, seed=args.seed)
        return asyncio.run(
            run_config(param_iter, to_gemm_gemm_test, options, paths, samples=args.samples))
    raise ValueError(f"Unknown config {args.config!r} (expected 'attention' or 'gemm_gemm')")


if __name__ == '__main__':
    ret = main()
    sys.exit(int(not ret))
