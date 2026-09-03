# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Describe the problem and the chip it is being tuned for.

Rewritten rather than ported. Helion's `helion/autotuner/llm/workload.py`
inspects Python source and torch tensors to work out what a kernel is doing;
here the compiler already knows, and C++ has put it in the request. So the
counterpart of upstream's `describe_kernel` is a renderer, and the counterpart
of `_gpu_hardware_lines` is longer, because a Triton config as Helion spells it
has a handful of hardware-sensitive knobs and a Rock perf config has
seventeen.

What does carry over is the idea behind `_matmul_hints` and
`_attention_reduction_hints`: read the shapes, and say out loud which way they
point. The hints below are keyed on M/N/K against CU count, chiplet count and
LDS size, where upstream keys on SM count.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List

Hardware = Dict[str, Any]
Problem = Dict[str, Any]

# What the kernel actually computes, and where its GEMM dimensions came from.
#
# Keyed on the `kernelType` C++ sends, which is the name rock's `KernelType`
# enum gives itself (RockAttrDefs.td, via the generated getNameForKernelType):
# the single-GEMM ops (RockGemmWrapperInterface) report `Gemm`, `Conv` or
# `ConvBwdData`, and the fused ops (RockGemmGemmWrapperInterface) report
# `Attention`, `GemmElementwiseGemm` or `ConvElementwiseGemm`.
#
# The distinction is worth spelling out because everything the model is shown
# downstream is in GEMM terms, and the same M/K/N mean different things
# depending on how they arose: a convolution's N is a batch times a spatial
# extent and is therefore huge, its K is a filter footprint and is therefore
# not a power of two, and attention's four dimensions are two sequence lengths
# and two head dimensions. A model told only "M=50176" tiles it like a GEMM.
_KERNEL_NOTES = {
    "Gemm": "A batched matrix multiply: C[G,M,N] = A[G,M,K] x B[G,K,N]. The "
            "dimensions are the user's own, so no dimension is systematically larger "
            "than another.",
    "Conv": "A forward convolution, lowered to an implicit GEMM, so the dimensions "
            "below are derived rather than given: M is the output channels per group, "
            "N is the batch size times the output spatial extent, and K is the input "
            "channels times the filter footprint. Expect N >> M, and expect K to be a "
            "multiple of the filter footprint rather than a power of two, which is "
            "what makes a kPerBlock that divides K worth more here than elsewhere.",
    "ConvBwdData": "The data gradient of a convolution, lowered to an implicit GEMM: M is "
                   "the input channels per group, N is the batch size times the input "
                   "spatial extent, and K is the output channels times the filter footprint. "
                   "As with the forward pass, N >> M and K follows the filter footprint.",
    "Attention": "Fused attention: gemm0 is Q x K^T and gemm1 is softmax(gemm0) x V, with "
                 "the softmax fused between them. So M is the number of query rows "
                 "(seq_len_q), K is the query/key head dimension (head_dim_qk), N is the "
                 "number of key/value rows (seq_len_k), and O is the value head dimension "
                 "(head_dim_v). The head dimensions are small and fixed by the model "
                 "architecture; the sequence lengths are what grow.",
    "GemmElementwiseGemm":
        "Two chained matrix multiplies with an elementwise operation fused "
        "between them: gemm0 is (M x K) x (K x N), the result passes through the "
        "elementwise op, and gemm1 contracts that against O.",
    "ConvElementwiseGemm":
        "A convolution chained into a second matrix multiply, with an elementwise "
        "operation fused between them. The first GEMM is the convolution's "
        "implicit GEMM, so its M is the output channels per group, its N is the "
        "batch times the output spatial extent and its K is the input channels "
        "times the filter footprint; gemm1 then contracts that N against O.",
}

# Both GEMMs are tiled by one perf config, which is the fact a model has to
# have to read the G0/G1 parameters at all. Shared by every fused kernel.
_FUSED_TILING_NOTE = ("One perf config tiles both GEMMs: mPerBlockG0 and nPerBlockG0 tile "
                      "gemm0, and nPerBlockG1 tiles gemm1's O, where 0 means untiled -- the "
                      "whole of O at once.")


def _note(text: str) -> str:
    """Wrap a paragraph into the indented block the prompt sections use."""
    return textwrap.fill(text, width=76, initial_indent="  ", subsequent_indent="  ")


def _si(value: int) -> str:
    """A byte count as somebody would say it."""
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.0f} MiB"
    if value >= 1024:
        return f"{value / 1024:.0f} KiB"
    return f"{value} B"


def describe_problem(problem: Problem) -> str:
    """The GEMM, convolution or attention shape the kernel computes."""
    kernel = problem.get("kernelType", "unknown")
    size = problem.get("gemmSize", {})
    lines = [f"  Kernel type: {kernel}"]

    dims = f"G={size.get('g')} M={size.get('m')} K={size.get('k')} N={size.get('n')}"
    fused = "o" in size
    if fused:
        dims += f" O={size.get('o')}"
    lines.append(f"  Shape: {dims}")

    if kernel in _KERNEL_NOTES:
        lines.append(_note(_KERNEL_NOTES[kernel]))
    if fused:
        lines.append(_note(_FUSED_TILING_NOTE))

    types = f"  Element types: A={problem.get('aType')} B={problem.get('bType')}"
    # For a single GEMM C is the result; for a fused one it is the second
    # GEMM's other operand (attention's V) and the result is reported apart.
    if problem.get("cType"):
        types += f" C={problem['cType']}"
    if problem.get("outType"):
        types += f" out={problem['outType']}"
    lines.append(types)

    if problem.get("quantBlockSize") is not None:
        lines.append(f"  Block-scaled GEMM: {problem['quantBlockSize']} elements per scale, "
                     f"scale types A={problem.get('aScaleType')} B={problem.get('bScaleType')}. "
                     "kPerBlock must be a multiple of the scale block size.")

    lines.extend(_describe_layout(problem))
    lines.extend(_describe_convolution(problem))
    lines.extend(_describe_fusion(problem))
    lines.extend(_describe_masking(problem))
    return "\n".join(lines)


def _describe_layout(problem: Problem) -> List[str]:
    """Which operands are stored transposed, and what that costs.

    A fact worth stating plainly: the same M/N/K with A stored K-major wants a
    different kPerBlock and a different answer on useAsyncCopy than one stored
    M-major, because it decides whether a tile's loads are contiguous.
    """
    named = [
        ("A", "transposedA"),
        ("B", "transposedB"),
        ("C", "transposedC"),
        ("out", "transposedOut"),
    ]
    stated = [(name, problem[key]) for name, key in named if key in problem]
    if not stated:
        return []

    transposed = [name for name, yes in stated if yes]
    lines = [
        "  Operand layout: " + (", ".join(
            f"{name} transposed" for name in transposed) if transposed else "none transposed") +
        f" (of {', '.join(name for name, _ in stated)})."
    ]
    if transposed:
        lines.append(
            _note("A transposed operand has its reduction (K) dimension contiguous "
                  "instead of strided, so loads along K coalesce and loads across "
                  "the free dimension do not. That flips which kPerBlock and kpack "
                  "read the memory efficiently, and it is what decides whether "
                  "useAsyncCopy can issue a direct-to-LDS load at all."))
    for key, extra in (("transposedAScale", "A"), ("transposedBScale", "B")):
        if problem.get(key):
            lines.append(f"  The {extra} scale tensor is transposed too.")
    return lines


def _describe_convolution(problem: Problem) -> List[str]:
    """The layout and window of the convolution underneath the implicit GEMM.

    The dimensions above are the implicit GEMM's; these are how the data
    actually sits in memory, which is what a tile of that GEMM has to gather.
    """
    lines: List[str] = []
    for label, key in (
        ("Filter layout", "filterLayout"),
        ("Input layout", "inputLayout"),
        ("Output layout", "outputLayout"),
    ):
        if problem.get(key):
            lines.append(f"  {label}: {', '.join(problem[key])}")
    window = [
        f"{label}={problem[key]}" for label, key in (("strides", "strides"), ("dilations",
                                                                              "dilations"),
                                                     ("padding", "padding")) if problem.get(key)
    ]
    if window:
        lines.append("  Convolution window: " + ", ".join(window))
    if lines:
        lines.append(
            _note("The GEMM dimensions above are this convolution's implicit-GEMM "
                  "image. A stride or dilation above 1 means consecutive GEMM "
                  "columns read non-consecutive input, so the gather is strided "
                  "however the tile is shaped."))
    alignment = problem.get("kPerBlockAlignment") or 1
    if alignment > 1:
        lines.append(
            _note(f"K here is a merge of the input channels and the filter's "
                  f"spatial extents, in that order, so only the channel part of it "
                  f"advances freely: a kPerBlock that is not a multiple of "
                  f"{alignment} lands mid-filter and moves the input window with it. "
                  f"So multiples of {alignment} are the tiles this layout is asking "
                  f"for."))
    return lines


def _describe_fusion(problem: Problem) -> List[str]:
    """What else the kernel does besides the GEMMs, and so what it has to hold.

    Both kinds of fusion cost registers that a tile's accumulator would
    otherwise have, which is why they belong in a description of the problem
    rather than in advice about it.
    """
    lines: List[str] = []
    if problem.get("hasPreSecondGemmFusion"):
        inputs = problem.get("numElemwiseInputs") or 0
        lines.append(f"  An elementwise operation is fused between the two GEMMs, "
                     f"reading {inputs} extra tensor(s) -- for attention this is the "
                     "usual scale and/or bias applied before the softmax.")
        lines.append(
            _note("Its operands are live across the first GEMM's accumulator, so "
                  "the register budget a tile is spent out of is tighter here than "
                  "the unfused shape would suggest."))
    if problem.get("hasFusedReduction"):
        lines.append(
            _note("The kernel also reduces this GEMM's output. That reduction "
                  "already writes with atomics, so splitKFactor above 1 adds a "
                  "second reduction on top of one that is being paid for anyway."))
    return lines


def _describe_masking(problem: Problem) -> List[str]:
    """Attention's head counts, masking and optional outputs."""
    if problem.get("numHeadsQ") is None:
        return []

    heads_q, heads_kv = problem["numHeadsQ"], problem.get("numHeadsKV")
    lines = [f"  Attention heads: {heads_q} query, {heads_kv} key/value"]
    if heads_kv and heads_q > heads_kv:
        lines.append(
            _note(f"Grouped-query attention: {heads_q // heads_kv} query heads share "
                  "each key/value head, so a K/V tile brought into LDS is reused by "
                  "that many query heads. Tiling M generously amortises the load."))

    if problem.get("causal"):
        lines.append(
            _note("Causal masking: a query only attends to keys at or before its own "
                  "position, so tiles above the diagonal are skipped entirely and "
                  "tiles on it do half their work. The loop over N is therefore "
                  "shorter for early query blocks than for late ones, which makes "
                  "the work per workgroup uneven."))
    if problem.get("slidingWindowLookBack") is not None:
        lines.append(f"  Sliding-window attention with a look-back of "
                     f"{problem['slidingWindowLookBack']}, so each query attends to at "
                     "most that many keys plus itself, however long the sequence is.")
    if problem.get("hasLastValidKVIndex"):
        lines.append("  The valid K/V extent is a run-time value, so the loop over N is "
                     "bounded by something the compiler cannot see.")
    if problem.get("hasPrefixOffset"):
        lines.append("  The causal mask is offset by a run-time prefix length.")

    split_kv = problem.get("splitKV") or 1
    if split_kv > 1:
        lines.append(
            _note(f"Flash decoding with splitKV={split_kv}: the grid is multiplied by "
                  f"{split_kv}, each workgroup covering a slice of the K/V sequence, "
                  "and the partial results are combined afterwards. So the machine "
                  "fills at a smaller mPerBlockG0 than the shape alone suggests."))
    if problem.get("hasLse"):
        lines.append("  The log-sum-exp output is written as well, which is another "
                     "tile of results each workgroup keeps and stores.")
    if problem.get("softmaxType"):
        lines.append(f"  The softmax accumulates in {problem['softmaxType']}.")
    return lines


def describe_hardware(hardware: Hardware) -> str:
    """The chip, named, and the budgets a config is spent out of."""
    line = "CDNA" if hardware.get("isCDNA") else "RDNA" if hardware.get("isRDNA") else "GCN"
    lines = [
        f"  Chip: {hardware.get('chip')} ({line}), full target {hardware.get('arch')}",
        f"  Compute units: {hardware.get('numCUs')} across "
        f"{hardware.get('numChiplets')} chiplet(s)",
        f"  Matrix instructions: {hardware.get('accelKind')}, "
        f"wave size {hardware.get('waveSize')}",
        f"  LDS per workgroup: {_si(hardware.get('ldsSize', 0))}",
        f"  VGPRs per EU: {hardware.get('vgprsPerEU')}, "
        f"up to {hardware.get('maxWavesPerEU')} waves per EU",
        f"  Last-level cache: {_si(hardware.get('lastLevelCacheSize', 0))}",
    ]

    # What the ladders can only imply. `GemmParamAxes` already pins a knob to
    # its default when the arch cannot act on it, so a model reading the space
    # alone would see a one-value ladder and not know why. Saying why is what
    # lets it reason instead of pattern-match.
    capability: List[str] = [f"kpack up to {hardware.get('maxKpack')}"]
    if hardware.get("maxNumCTAs", 1) > 1:
        capability.append(f"multi-CTA launch up to {hardware['maxNumCTAs']} CTAs")
    else:
        capability.append("no multi-CTA launch (numCTAs is fixed at 1)")
    capability.append(
        "asynchronous (direct-to-LDS) global loads available" if hardware.get("supportsAsyncCopy")
        else "no direct-to-LDS load width, so useAsyncCopy cannot change the kernel")
    if hardware.get("supportsTDM"):
        capability.append("tensor descriptor memory (TDM) available")
    capability.append("kPerBlock need not be a power of two" if hardware.
                      get("supportsNonPow2KPerBlock") else "kPerBlock must be a power of two")
    if hardware.get("supportsScaledGemm"):
        capability.append("scaled (MXFP) matrix instructions available")
    lines.append("  Capabilities: " + "; ".join(capability) + ".")

    # Kept out of the list above, which is otherwise all things the chip can or
    # cannot do. This one is a claim about which of two kernels wins, and the
    # model has no way to tell the difference unless it is told.
    if hardware.get("preferBf16x3ForF32Dot"):
        lines.append(
            _note("One heuristic, rather than a property of the chip: this family "
                  "was measured to come out ahead overall with an f32 dot "
                  "decomposed into three bf16 products, and that is what "
                  "useBf16x3ForF32 = -1 resolves to here. The measurement was an "
                  "average over many shapes, so it is a starting point and not a "
                  "fact about this one. Where the operands are f32, and so the knob "
                  "reaches the dot at all, propose 0 as well as 1 and let the "
                  "timings settle it."))
    return "\n".join(lines)


def compute_workload_hints(problem: Problem, hardware: Hardware) -> List[str]:
    """Say which way the shapes point, in the spirit of Helion's `_matmul_hints`.

    Every hint is a reading of the numbers already given above, not new
    information. They are here because a model given only the numbers tends to
    propose a generic config, and a model told "this is skinny" reaches for
    splitKFactor.

    Split by kernel shape, because the two perf configs tile differently
    enough that one set of readings would be wrong for one of them: a plain
    GEMM parallelises over M and N, while a fused kernel parallelises over G
    and M and loops over N inside the kernel.
    """
    size = problem.get("gemmSize", {})
    g = size.get("g") or 0
    m, n, k = size.get("m") or 0, size.get("n") or 0, size.get("k") or 0
    o = size.get("o")
    num_cus = hardware.get("numCUs") or 0

    if not (m and n and k and num_cus):
        return []
    if o is not None:
        return _fused_hints(g, m, n, k, o, num_cus)
    return _gemm_hints(m, n, k, num_cus, hardware, problem.get("kPerBlockAlignment") or 1)


def _gemm_hints(m: int, n: int, k: int, num_cus: int, hardware: Hardware,
                k_alignment: int) -> List[str]:
    """Readings for a single GEMM, whether written as one or lowered to one."""
    hints: List[str] = []

    # How many workgroups a mid-sized tile would launch, which is what decides
    # whether the machine is even full. Helion compares its tile count against
    # the SM count for the same reason.
    for tile in (64, 128, 256):
        tiles = ((m + tile - 1) // tile) * ((n + tile - 1) // tile)
        if tiles >= num_cus:
            hints.append(f"A {tile}x{tile} output tile gives {tiles} workgroups, which fills "
                         f"all {num_cus} CUs, so tiles of that size or smaller keep the "
                         "machine busy.")
            break
    else:
        tiles = ((m + 63) // 64) * ((n + 63) // 64)
        hints.append(f"Even a 64x64 output tile gives only {tiles} workgroups against "
                     f"{num_cus} CUs, so this problem cannot fill the machine by tiling "
                     "M and N alone. splitKFactor above 1 is the way to get more "
                     "parallelism out of it, at the cost of a reduction across the "
                     "partial results.")

    if k >= 8 * max(m, n):
        hints.append(f"K ({k}) dwarfs both M ({m}) and N ({n}): this is a skinny GEMM, and "
                     "the useful families here are a deep kPerBlock and splitKFactor "
                     "above 1.")
    elif m >= 8 * n or n >= 8 * m:
        hints.append(f"M ({m}) and N ({n}) are very different sizes, so a square tile "
                     "wastes work on the short dimension. Tile the long dimension "
                     "generously and the short one tightly.")

    if hardware.get("numChiplets", 1) > 1:
        hints.append(f"This chip has {hardware['numChiplets']} chiplets, each with its own "
                     "path to the last-level cache, so gridGroupSize matters more here "
                     "than on a single-chiplet part: grouping M-tiles keeps a chiplet's "
                     "accesses together.")

    if k <= 64:
        hints.append(f"K is only {k}, so kPerBlock cannot usefully exceed it and deep "
                     "pipelining (large numStages) has little to pipeline.")

    # The one hint that is not a reading of the numbers above: the reasoning
    # rests on gemmK being a merged (channel, filter) dimension, which is a
    # fact about the conv underneath rather than about K, so the compiler hands
    # the factor over ready-made (see `kPerBlockAlignmentFactor`).
    if k_alignment > 1:
        divides = [
            tile for tile in range(k_alignment,
                                   min(k, 512) + 1, k_alignment) if k % tile == 0
        ]
        hints.append(f"This convolution's K is its input channels times a filter "
                     f"footprint of {k_alignment}, laid out channels-first, so a "
                     f"kPerBlock that is a multiple of {k_alignment} advances K without "
                     "moving the input window and keeps the validity mask out of the K "
                     f"loop. Such tiles are in kPerBlock's list and look unusual -- "
                     f"neither powers of two nor multiples of 16 -- but they are the "
                     f"ones to reach for here" +
                     (f", the best of them also dividing K ({k}) exactly: "
                      f"{', '.join(str(tile) for tile in divides[-4:])}." if divides else "."))
    return hints


def _fused_hints(g: int, m: int, n: int, k: int, o: int, num_cus: int) -> List[str]:
    """Readings for a fused kernel, where N is looped rather than tiled.

    Deliberately says nothing about gridGroupSize: attention lays its grid out
    with `makeGxNGridLayout`, which takes no group size, so the field reaches
    nothing and the axes pin it at 0 (see Rock_GemmGemmParamsAttr in
    RockAttrDefs.td). Advising a knob that cannot do anything spends a round.
    """
    hints: List[str] = []

    # The grid is G by the M tiles: one workgroup per (batch*head, block of
    # query rows). N is the loop extent inside the kernel, so unlike a plain
    # GEMM it buys no parallelism.
    for tile in (32, 64, 128, 256):
        groups = g * ((m + tile - 1) // tile)
        if groups >= num_cus:
            hints.append(f"The grid is G x (M / mPerBlockG0), so mPerBlockG0={tile} gives "
                         f"{groups} workgroups against {num_cus} CUs, which fills the "
                         "machine. N buys no parallelism here: it is the loop the kernel "
                         "runs inside each workgroup.")
            break
    else:
        groups = g * ((m + 31) // 32)
        hints.append(f"Even mPerBlockG0=32 gives only {groups} workgroups against "
                     f"{num_cus} CUs, because the grid is only G x (M / mPerBlockG0) and "
                     "N is looped inside the kernel. No tiling fills this machine, so "
                     "prefer configs that make each workgroup efficient over configs "
                     "that make more of them; splitKFactor, where the Configuration "
                     "Space offers a value above 1, is the one knob that adds "
                     "workgroups here, and it splits that loop over N.")

    hints.append(f"O is {o}, so nPerBlockG1=0 (untiled, the whole of O in one workgroup) "
                 "is the usual choice; tiling O only pays once O is large enough that a "
                 "whole row of it crowds out LDS or registers." if o <=
                 128 else f"O is {o}, which is large enough that nPerBlockG1=0 (the whole of O at "
                 "once) may not fit; tiling O is worth trying here.")

    if n >= 8 * m:
        hints.append(f"N ({n}) dwarfs M ({m}): most of the time goes into the loop over "
                     "N, so nPerBlockG0 and numStages are what to move -- they set how "
                     "much of N each iteration covers and how deeply those iterations "
                     "are pipelined.")

    # Not a reading of the shapes but of the pipeline the two dots go through,
    # which is why it is here rather than in the hints a plain GEMM gets:
    # `ChainedDotSchedule` and `transformChainedDotSchedule` both test
    # numStages == 4 exactly and return otherwise.
    hints.append("Two chained dots go through a schedule written for numStages of "
                 "exactly 4: at that depth the pipeliner double-buffers each dot's "
                 "loads into separate memory and compute clusters, and the pingpong "
                 "scheduler rearranges them. At 1, 2, 3 or above 4 both are skipped "
                 "and the loop keeps the ordinary schedule. The sweeps behind the seed "
                 "configs never went past 3, so 4 is untried rather than rejected here "
                 "and is worth one config of its own.")

    if k <= 64:
        hints.append(f"K is only {k}, which is normal for this kernel shape: it is a head "
                     "dimension, not a reduction length, so kPerBlock cannot usefully "
                     "exceed it. This is not the dimension splitKFactor splits: that "
                     f"one is N ({n}), which the two GEMMs share as the first's N and "
                     "the second's K.")
    return hints
