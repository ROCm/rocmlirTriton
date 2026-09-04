# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Build the prompts for the LLM-guided tuning search.

Ported from Helion's `helion/autotuner/llm/prompting.py`, but this is the one
file in the package where the port is scaffolding only. What steers a model is
general; what to steer it towards is not.

Ported as-is, because it is about talking to a model rather than about Triton:
the section helpers, the section ordering of `build_initial_prompt` and
`build_refinement_prompt`, the output contract (minified JSON, one
`{"configs":[...]}` object, sparse configs, no Python literals, omit rather
than guess), `RETURN_JSON_ONLY`, the sparse-config field-count guidance, the
"cover three families, roughly 40/40/20 safe/balanced/aggressive" split, and
the failure-heavy versus default branching of the refinement step.

Rewritten, because every word of it was Triton vocabulary: the knob glossary.
Upstream talks about `block_sizes`, `num_warps`, `num_stages`, `pid_type`,
`indexing`, `l2_groupings`, `maxnreg` and the `range_*` toggles. This talks
about the Rock perf config, and has to explain one thing upstream has no
equivalent of at all: the eight tri-state knobs, where -1 is not a missing
answer but the usual one.

Dropped, because there is nothing here for them to describe: upstream's
`_ADVANCED_TOGGLE_FIELDS`, `_HEURISTIC_PURPOSES` and
`build_author_seed_section`. `build_compiler_analysis_section` survives in
spirit as `build_seed_config_section`, repointed at the quick tuning list.

The glossary is prose, curated from the parameter documentation in
RockAttrDefs.td (`kRockGemmParams`) and reviewed like any other source. What a
knob *is* is already machine-readable and the generated Configuration Space
section carries it; what is worth saying about it -- when to reach for it, what
it trades against, which knobs are coupled -- is judgement, and generating that
would either lose it or bury it in a template.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .configs import knob_names, render_space
from .feedback import (
    MAX_CHANGED_FIELDS_PER_CONFIG,
    format_config_for_prompt,
)
from .workload import compute_workload_hints, describe_hardware, describe_problem

RETURN_JSON_ONLY = 'Return minified JSON only: {"configs":[...]}'

FEASIBILITY_RULE = ("Every value you choose must appear in that parameter's list in the "
                    "Configuration Space, but that is necessary and not sufficient: the lists "
                    "hold each parameter's values independently, and the tuning space also "
                    "checks combinations of them against LDS capacity, Triton's per-tensor "
                    "element cap, a compile-cost budget and the register budget behind "
                    "wavesPerEU. A config can sit entirely on the lists and still be refused "
                    "before it is ever compiled. Refused configs are reported back to you "
                    "under Refused Configs, so read them.")

_INITIAL_STRATEGY_BASE_LINES = (
    "First read the problem shape, the hardware, and the configuration space.",
    "Cover 3 config families with a rough mix of about 40% near-default safe, 40% balanced throughput, and 20% aggressive configs, while keeping most candidates ones the space will accept.",
    "If the problem is an unusual shape, stay closer to the default and avoid aggressive coupled changes.",
    ("Keep each config sparse: usually 2-6 changed fields, omit unchanged "
     "defaults, and exceed 6 only when several coupled changes are needed "
     "for a distinct family."),
    "Use the block tiles to define families: include at least 3 materially different tilings rather than tiny perturbations of one tile.",
    "Vary the M, N and K tiles coherently rather than by arbitrary skew.",
    "Do not repeat unchanged defaults.",
    "Avoid configs that max out several aggressive knobs at once (largest tiles plus deepest numStages plus highest splitKFactor) unless there is a reason.",
)

_FAILURE_HEAVY_REFINEMENT_LINES = (
    "Recent rounds had many failures or refusals. Use only the best 1-2 anchors.",
    "At least 80% of configs should be 1-2 field mutations of those anchors.",
    "Back off the aggressive settings first: smaller tiles, lower numStages, splitKFactor back to 1, and leave the use* knobs at -1.",
)

_DEFAULT_REFINEMENT_LINES = (
    "About two thirds of configs should be 1-field mutations of Anchor 1.",
    "Use most of the rest for 1-2 field mutations of Anchor 2.",
    "Reserve at most a small minority for one clearly different family, not random noise.",
)

_SYSTEM_PREAMBLE = textwrap.dedent("""\
    You are an expert GPU kernel autotuner for AMD GPUs. You are tuning a
    rocmlirTriton "perf config": a fixed set of integer parameters that decides
    how a matrix-multiply kernel is tiled, scheduled and lowered.

    rocmlirTriton is an MLIR kernel generator for AMD GPUs: it builds the
    kernel in its own `rock` dialect and lowers that to Triton's TTIR, and on
    through TTGIR and LLIR, so a perf config settles the shape of the Triton
    kernel that comes out.

    Use the provided Configuration Space and Default Configuration as the
    source of truth for allowed parameter names, the values each may take, and
    what an unspecified parameter means. Every parameter is a scalar integer.

    The block tiles (the primary knobs):""")

# The same bullet under the two names the tiles go by, rather than one bullet
# that renames itself halfway through. Which one the kernel uses is in the
# space, so there is no reason to make the model read past the other.
_GEMM_TILE_BULLET = textwrap.dedent("""\
    - mPerBlock, nPerBlock: the M x N output tile one workgroup computes. This
      is the single most consequential choice: it fixes how many workgroups
      launch, how much LDS a stage needs, and how many registers the
      accumulator occupies.""")

_GEMM_GEMM_TILE_BULLET = textwrap.dedent("""\
    - mPerBlockG0, nPerBlockG0: the M x N output tile one workgroup computes
      for the first GEMM. This is the single most consequential choice: it
      fixes how many workgroups launch, how much LDS a stage needs, and how
      many registers the accumulator occupies.
    - nPerBlockG1: the second GEMM's output tile, where 0 means untiled.""")

_KPERBLOCK_BULLET = textwrap.dedent("""\
    - kPerBlock: how much of the contraction dimension one iteration consumes.
      Deeper means fewer, larger LDS loads and better matrix-instruction
      utilization, but more LDS per stage. The A and B tiles together must fit
      in LDS numStages times over, which is what makes large tiles and deep
      pipelining compete for the same budget.""")

_GENERAL_HEURISTICS = textwrap.dedent("""\
    General heuristics:
    - Read the shapes against the CU count before choosing a tile. A tile that
      leaves most of the machine idle cannot be fast however well scheduled.
    - Powers of two are the usual tiles, but they are not the only ones the
      space allows, and its lists are the authority on that rather than this
      habit. The M and N tiles also take every multiple of 16, and the powers
      of two below 16. kPerBlock goes further: where the kernel is a
      convolution whose K index wants an aligned tile, the list carries the
      multiples of that alignment, which are usually neither powers of two nor
      multiples of 16 and are the right answer on that kernel. The Problem
      section says so when it applies.
    - Deep pipelining and large tiles compete for LDS; do not raise both.
    - Prefer changes whose effect is attributable: move the tiles, or
      numStages, or splitKFactor, rather than rewriting every field at once.""")

_OUTPUT_CONTRACT = textwrap.dedent("""\
    Output contract:
    - Return minified JSON on a single line. No markdown, code fences,
      comments, pretty-printing, or trailing commas.
    - Emit exactly one top-level object: {"configs":[...]} and make every
      config unique.
    - Do not use Python syntax or expressions.
    - Only specify parameters you want to change; unspecified = default.
    - Use only parameter names that appear in the configuration space, and only
      values that appear in that parameter's list.
    - Every value is a plain integer. Never a list, a string, or a float.
    - If you are unsure about a parameter, omit it rather than guessing.
    - Use null not None, true/false not True/False.
    - Return ONLY minified JSON: {"configs":[...]}""")


def _section(title: str, body: str) -> str:
    """Render a titled prompt section."""
    return f"## {title}\n{body}"


def _bullet_section(title: str, lines: Sequence[str]) -> str:
    """Render a titled prompt section whose body is a bullet list."""
    return _section(title, "\n".join(f"  - {line}" for line in lines))


def _join_sections(*sections: str) -> str:
    """Join non-empty prompt sections with a blank line."""
    return "\n\n".join(section for section in sections if section)


def _is_tunable(space: Optional[Dict[str, Sequence[int]]], name: str) -> bool:
    """Whether `name` is a parameter this run can actually move.

    A one-value ladder is a parameter the arch or the problem left no room for,
    and `TuningSearch`'s `addKnobAxes` pins exactly those. Prose about one is
    prose the model cannot act on, so the blocks below are keyed on this.

    Says yes when there is no space to consult, so that a caller without one
    gets the whole prompt rather than a silently abridged one.
    """
    if space is None:
        return True
    values = space.get(name)
    if values is None:
        return True
    return len(values) > 1


def _any_tunable(space: Optional[Dict[str, Sequence[int]]], *names: str) -> bool:
    return any(_is_tunable(space, name) for name in names)


def _may_pipeline_one_stage(space: Optional[Dict[str, Sequence[int]]]) -> bool:
    """Whether numStages can be 1, the depth at which the two schedule knobs
    stop reaching anything. Where the axes rule 1 out, the trap they lay is one
    the model cannot fall into and the warning is dead weight."""
    if space is None:
        return True
    return 1 in space.get("numStages", [1])


# One entry per tunable, emitted only where the space gives it a choice. Keyed
# on the name the perf config spells; a gemm+gemm renames the block tiles, so
# those live in the always-on text above rather than here.
_PARAM_BULLETS: Dict[str, str] = {
    "numWaves":
        """- numWaves: waves per workgroup. More waves split the tile more finely, so
  a large tile usually wants more of them and a small tile is starved by
  them.""",
    "matrixInstrNonkdim":
        """- matrixInstrNonkdim: the M/N extent of the matrix instruction, typically
  16 or 32. 16 and 32 are genuinely different families rather than points on a
  scale: 32 amortizes more work per instruction, 16 wastes less on a tile that
  does not divide by 32.""",
    "kpack":
        """- kpack: how many matrix instructions issue from one LDS load. Above 1 it
  reduces LDS traffic; the ceiling is in the hardware section.""",
    "numStages":
        """- numStages: software pipeline depth over the K loop. 1 is safest. 2 to 4
  overlaps loads with math on a streaming loop, at numStages times the LDS for
  the tiles. The sweeps behind the seed configs stopped at 3, so any higher
  value the Configuration Space offers is unmeasured here rather than known to
  be bad.""",
    "splitKFactor":
        """- splitKFactor: splits the contraction across that many workgroups, which
  then reduce their partial results. This is the answer to a problem too small
  to fill the machine by tiling M and N, and it is a distinct family: it buys
  parallelism and pays for it with a reduction. Leave it at 1 unless M x N is
  small relative to the CU count.""",
    "gridGroupSize":
        """- gridGroupSize: how many M-tile blocks are grouped when workgroups are
  mapped onto the grid. Larger groups improve last-level-cache locality across
  the group and cost scheduling flexibility. 0 lets the compiler choose.""",
    "numCTAs":
        "- numCTAs: workgroups per cooperative cluster.",
    "wavesPerEU":
        """- wavesPerEU: a hint to the backend for how many waves to keep resident per
  execution unit, which it honours by limiting registers per wave. 0 means no
  hint. A large tile plus a high wavesPerEU cannot both be satisfied, and the
  space refuses that combination rather than compiling it.""",
}

# Sentences appended to the bullet above where the kernel is a gemm+gemm. Both
# are about the two dots and the schedule they go through, so they are wrong
# rather than merely idle on a single GEMM.
_GEMM_GEMM_PARAM_TAILS: Dict[str, str] = {
    "numStages":
        """Here 4 is worth a proposal of its own: the chained-dot pipeline
  schedule, and the pingpong that rides on it, are written for exactly that
  depth and are skipped at any other.""",
    "splitKFactor":
        """The contraction being split here is the one the two GEMMs share, the
  first GEMM's N and so the second GEMM's K. That dimension is the long one, so
  the knob is a real option on this kernel.""",
}

_KNOB_INTRO = textwrap.dedent("""\
    The use* knobs are tri-state, and this is the part most easily got wrong.
    Each takes -1, 0 or 1:
    - -1 means "let the compiler decide", and applies a per-architecture
      heuristic. It is the default, and it is a reasonable answer. The hardware
      section says which way it goes here.
    - 0 forces the transform off; 1 forces it on.
    Every seed config below spells -1 in all of them, and 0 in wavesPerEU and
    gridGroupSize. Read nothing into that. Those configs are distilled from
    sweeps that pinned exactly those fields and varied only the tiles and the
    schedule, so a column of -1 records what was never tried rather than what
    won. They are the least explored part of this space, not the settled part.
    Do not treat these as booleans and do not set them all. Move one when you
    have a reason to believe the heuristic is wrong for this shape, and leave
    the rest at -1. What each gates:""")

# The knobs the space left room for. Same rule as `_PARAM_BULLETS`: a knob
# pinned to -1 builds the one kernel whatever it is asked for.
_KNOB_BULLETS: Dict[str, str] = {
    "useAsyncCopy":
        "- useAsyncCopy: direct-to-LDS global loads, bypassing registers.",
    # Nothing here about the MFMA layout the pass wants, which is a condition on
    # the target rather than on the config: `addKnobAxes` pins this knob where a
    # single dot's layout would not be one, so a ladder that reached this bullet
    # is a ladder whose kernel can carry the schedule.
    "useBlockPingpong":
        """- useBlockPingpong: the pingpong schedule, which alternates two wave groups
  between loading and computing.""",
    "useInThreadTranspose":
        "- useInThreadTranspose: an in-thread transpose of a loaded tile.",
    "useBufferOps":
        """- useBufferOps: the buffer-ops pass cluster (buffer addressing rather than
  flat pointers).""",
    "useBufferAtomics":
        """- useBufferAtomics: buffer atomics, which require useBufferOps to be on.
  Setting this to 1 with useBufferOps at 0 is refused.""",
    "useReductionLayout":
        """- useReductionLayout: redistributes warps onto the reduction dimension to
  cut register spill. -1 rewrites convolutions only.""",
    "useOptimizeEpilogue":
        "- useOptimizeEpilogue: Triton's epilogue optimization.",
    "useBf16x3ForF32":
        """- useBf16x3ForF32: decomposes an f32 dot into three bf16 dots. Only
  relevant to f32 inputs.""",
}

_DUPLICATE_RULES_INTRO = textwrap.dedent("""\
    Some knob values build a kernel that has already been timed, and spending a
    candidate on one measures nothing. Before proposing an explicit 0 or 1,
    check it against these:""")

_BUFFER_DUPLICATE_RULE = textwrap.dedent("""\
    - On useBufferOps and useBufferAtomics, -1 resolves to on with no
      architecture or shape entering into it. So 1 is the same kernel as -1 on
      both, and 0 is the only value that changes anything.""")

_EPILOGUE_DUPLICATE_RULE = textwrap.dedent("""\
    - useOptimizeEpilogue at -1 weighs the store tail against the depth of the
      K loop, and it only ever declines to bypass registers when the stored
      element is 16 bits wide. On any other output type -1 and 1 agree, and
      again only 0 differs.""")

# One rule per knob rather than one covering both, so that a space with only
# one of them open does not carry the other's name into the prompt.
_PINGPONG_DUPLICATE_RULE = textwrap.dedent("""\
    - The pingpong pass is not run at all when numStages is 1, so at that depth
      useBlockPingpong=1 builds the kernel -1 already built. Raise numStages in
      the same config or leave the knob alone.""")

_ASYNC_COPY_DUPLICATE_RULE = textwrap.dedent("""\
    - An asynchronous load needs at least two pipeline buffers, so at numStages
      of 1 useAsyncCopy=1 builds the kernel -1 already built. Again, move
      numStages in the same config or leave the knob alone.""")

_FEASIBILITY_INTRO = textwrap.dedent("""\
    Arithmetic worth doing before you propose a config. A refused config costs
    a whole candidate and tells you only which check it failed, and the checks
    are these, by the name they are reported under. Write t for the threads in
    a workgroup, numWaves times the wave size from the hardware section:
    - exceedsTritonTensorCap: kPerBlock * max(mPerBlock, nPerBlock) must be at
      most 1048576, Triton's limit of 2^20 elements in one tensor. Deep K tiles
      on a wide block are what run into it.""")

_REGISTER_BUDGET_RULE = textwrap.dedent("""\
    - wavesPerEURegisterBudget: a nonzero wavesPerEU caps each thread at
      (VGPRs per EU) / wavesPerEU registers, so mPerBlock * nPerBlock *
      wavesPerEU must be at most (VGPRs per EU) * t. This is why a large tile
      and a high wavesPerEU cannot be asked for together.""")

# Both budgets rather than the one this chip uses, since which it is follows
# from the family the hardware section already names. Worth a number at all
# because the model can compare its own arithmetic against one, where it cannot
# against "the low thousands" (`compileCostBudget` in TuningSearch.cpp).
_COMPILE_COST_RULE = textwrap.dedent("""\
    - compileCostBudget: a config whose kernel would take the backend minutes
      is passed over. The cost is the larger of (mPerBlock + nPerBlock) *
      kPerBlock / (t * kpack), the K-loop body, and mPerBlock * nPerBlock / t,
      the accumulator, in elements per thread, and it has to stay under 8000 on
      RDNA or 12000 elsewhere. Note which one dominates before moving a field:
      halving kPerBlock does nothing for a config the accumulator dominates.""")

_LDS_RULE = textwrap.dedent("""\
    - ldsBlacklist: the A and B tiles are staged in LDS once per stage, so
      (mPerBlock * kPerBlock * bits(A) + nPerBlock * kPerBlock * bits(B)) / 8 *
      numStages has to fit the LDS per workgroup given in the hardware section.
      Tile shapes known to overrun it are refused up front.
    - notOnAxis: a value that is not in that parameter's list. Reread the
      list.""")

_BUFFER_KNOBS_RULE = "- bufferKnobsDisagree: useBufferAtomics=1 with useBufferOps=0."


def _bullets_for(bullets: Dict[str, str], space: Optional[Dict[str, Sequence[int]]],
                 gemm_gemm: bool) -> List[str]:
    """The bullets among `bullets` this space has a use for, in order."""
    chosen = []
    for name, text in bullets.items():
        if not _is_tunable(space, name):
            continue
        tail = _GEMM_GEMM_PARAM_TAILS.get(name) if gemm_gemm else None
        chosen.append(f"{text}\n  {tail}" if tail else text)
    return chosen


def build_system_prompt(space: Optional[Dict[str, Sequence[int]]] = None) -> str:
    """The global instruction block, cut down to what this run can act on.

    Most of it is fixed, but a good deal describes parameters that a given arch
    or kernel leaves no choice about: attention pins splitKFactor and
    gridGroupSize, WMMA pins matrixInstrNonkdim and kpack, an arch with no
    direct-to-LDS width pins useAsyncCopy, and a dot that is not f32 pins
    useBf16x3ForF32. Explaining one of those spends the model's attention on a
    decision it does not have, and the axes say which they are without this
    having to know why, so `space` is read to drop those blocks. Passing none
    yields the full text.
    """
    gemm_gemm = space is not None and "nPerBlockG1" in space
    tiles = _GEMM_GEMM_TILE_BULLET if gemm_gemm else _GEMM_TILE_BULLET
    blocks = ["\n".join((_SYSTEM_PREAMBLE, tiles, _KPERBLOCK_BULLET))]

    scheduling = _bullets_for(_PARAM_BULLETS, space, gemm_gemm)
    if scheduling:
        blocks.append("Scheduling and layout:\n" + "\n".join(scheduling))

    knobs = _bullets_for(_KNOB_BULLETS, space, gemm_gemm)
    if knobs:
        blocks.append(_KNOB_INTRO + "\n" + "\n".join(knobs))

        duplicates = []
        if _any_tunable(space, "useBufferOps", "useBufferAtomics"):
            duplicates.append(_BUFFER_DUPLICATE_RULE)
        if _is_tunable(space, "useOptimizeEpilogue"):
            duplicates.append(_EPILOGUE_DUPLICATE_RULE)
        if _may_pipeline_one_stage(space):
            if _is_tunable(space, "useBlockPingpong"):
                duplicates.append(_PINGPONG_DUPLICATE_RULE)
            if _is_tunable(space, "useAsyncCopy"):
                duplicates.append(_ASYNC_COPY_DUPLICATE_RULE)
        if duplicates:
            blocks.append(_DUPLICATE_RULES_INTRO + "\n" + "\n".join(duplicates))

    blocks.append(_GENERAL_HEURISTICS)

    feasibility = [_FEASIBILITY_INTRO]
    if _is_tunable(space, "wavesPerEU"):
        feasibility.append(_REGISTER_BUDGET_RULE)
    feasibility.append(_COMPILE_COST_RULE)
    feasibility.append(_LDS_RULE)
    if _any_tunable(space, "useBufferOps", "useBufferAtomics"):
        feasibility.append(_BUFFER_KNOBS_RULE)
    blocks.append("\n".join(feasibility))

    blocks.append(_OUTPUT_CONTRACT)
    return "\n\n".join(blocks)


def _initial_strategy_lines(
    *,
    configs_requested: int,
    space: Dict[str, Sequence[int]],
    hints: Sequence[str],
) -> List[str]:
    """Build the bullet list used for the initial search-strategy section."""
    lines = [
        f"Propose up to {configs_requested} UNIQUE candidate configs. "
        "Fewer is better than invalid JSON.",
        *_INITIAL_STRATEGY_BASE_LINES,
    ]
    if len(space.get("splitKFactor", [1])) > 1:
        lines.append("splitKFactor above 1 is available here. Include at least one "
                     "config using it if the shape hints above suggest the machine "
                     "cannot be filled by tiling M and N.")
    if len(space.get("matrixInstrNonkdim", [])) > 1:
        lines.append("matrixInstrNonkdim can vary, so treat 16 and 32 as two families "
                     "and put some configs on each.")
    if knobs := knob_names(space):
        lines.append("Leave the tri-state knobs (" + ", ".join(knobs) + ") at -1 in most "
                     "configs; use an explicit 0 or 1 in a minority, and only where you "
                     "can say why.")
    lines.append("Spread numStages and the tiles across the batch rather than clustering "
                 "on one choice: a later refinement search will anchor on whatever this "
                 "round measures, so variety here is worth more than a batch of "
                 "near-identical safe configs.")
    lines.append(FEASIBILITY_RULE)
    return lines


def _refinement_strategy_lines(
    *,
    unmeasured_count: int,
    total_count: int,
    rejected_count: int,
) -> List[str]:
    """Build the bullet list used for the refinement-step section."""
    trouble = unmeasured_count + rejected_count
    if total_count > 0 and trouble * 3 >= total_count:
        lines = list(_FAILURE_HEAVY_REFINEMENT_LINES)
    else:
        lines = list(_DEFAULT_REFINEMENT_LINES)
    lines.append("Prefer edits with attributable effects: move the block tiles, "
                 "numWaves, numStages, kpack, matrixInstrNonkdim, splitKFactor or "
                 "gridGroupSize rather than rewriting every field.")
    lines.append("Keep each config sparse: usually 1-4 changed fields, and no more than "
                 f"{MAX_CHANGED_FIELDS_PER_CONFIG} unless absolutely necessary.")
    lines.append(FEASIBILITY_RULE)
    lines.append("If unsure, return fewer valid configs instead of verbose or malformed JSON.")
    return lines


def build_seed_config_section(seed_configs: Sequence[Dict[str, int]]) -> str:
    """Show the compiler's own heuristic configs as an unmeasured prior.

    Helion's `build_compiler_analysis_section`, repointed. Upstream surfaces
    the heuristics its compiler fired and the seed configs they derived,
    described as "structural priors ... treat them as strong starting points".
    rocmlirTriton's analogue is the quick tuning list: a heuristic's best guess
    before anything has been measured. In round 0 these are unmeasured, exactly
    as upstream's are; from round 1 on they reappear with real timings in the
    Results section, so this section only earns its place in the first prompt.

    One asymmetry with upstream is worth spelling out to the model, and the
    body below does. That list is distilled from exhaustive sweeps, and
    `createGemmTuningRangeBF` and `createGemmGemmTuningRangeBF` pin every use*
    knob to `kKnobDefault` and both `wavesPerEU` and `gridGroupSize` to 0 while
    enumerating -- so every entry in the checked-in list agrees on those
    fields, without a single one of them having been measured against its
    alternatives. Left unsaid, a column that never varies reads as a
    consensus, and the knobs are precisely where a search over the axes can
    find something the sweeps could not.
    """
    if not seed_configs:
        return ""
    body = ("rocmlirTriton's tuning heuristic proposes the following configs for "
            "this problem before anything is measured. Treat them as strong "
            "starting points: include configs matching these and nearby mutations "
            "before venturing to unrelated families.\n"
            "They are evidence about the fields the sweeps behind them varied: the "
            "block tiles, kpack, numWaves, matrixInstrNonkdim, splitKFactor and "
            "numStages. Their use* knobs, wavesPerEU and gridGroupSize were held "
            "fixed throughout those sweeps, so on those fields these configs are "
            "unmeasured rather than confirmed.\n" +
            "\n".join(f"  - {format_config_for_prompt(config)}" for config in seed_configs))
    return _section("Heuristic Seed Configs", body)


def _build_problem_context(request: Dict[str, Any]) -> Tuple[str, List[str]]:
    """The context a conversation needs before it can reason about results."""
    problem = request.get("problem", {})
    hardware = request.get("hardware", {})
    space = request.get("space", {})
    default_config = request.get("defaultConfig", {})
    hints = compute_workload_hints(problem, hardware, space)
    default_section = _section(
        "Default Configuration",
        format_config_for_prompt(default_config) +
        "\n  Any parameter you do not mention takes its value from this config.",
    )
    return _join_sections(
        _section("Problem", describe_problem(problem)),
        _section("GPU Hardware", describe_hardware(hardware)),
        _bullet_section("What The Shapes Suggest", hints) if hints else "",
        _section("Configuration Space", render_space(space, default_config)),
        default_section,
    ), hints


def build_initial_prompt(request: Dict[str, Any]) -> str:
    """Build the full initial user prompt, for the round that has no results."""
    space = request.get("space", {})
    context, hints = _build_problem_context(request)
    task_section = ("Propose the first batch of configs. Include both near-default and "
                    f"exploratory candidates. {RETURN_JSON_ONLY}")
    return _join_sections(
        context,
        build_seed_config_section(request.get("seedConfigs", [])),
        _bullet_section(
            "Search Strategy",
            _initial_strategy_lines(
                configs_requested=request.get("configsRequested", 15),
                space=space,
                hints=hints,
            ),
        ),
        _section("Task", task_section),
    )


def build_refinement_prompt(
    request: Dict[str, Any],
    *,
    search_state: str,
    anchor_configs: str,
    results: str,
    top_patterns: str,
    failed_patterns: str,
    rejected_patterns: str,
    unmeasured_count: int,
    total_count: int,
    rejected_count: int,
) -> str:
    """Build the refinement prompt sent after each benchmarking round."""
    configs_requested = request.get("configsRequested", 15)
    task_section = (f"Propose up to {configs_requested} NEW UNIQUE configs around the "
                    "anchors above. Avoid the failed and refused patterns above, and favour "
                    f"targeted edits with attributable effects. {RETURN_JSON_ONLY}")
    # Ordinarily the initial request established this context and a resumed
    # conversation retains it. With --llm-wait-for-seeds, however, round 0
    # already has measurements and enters this refinement path on the first
    # request. Put the context in that first message rather than asking the
    # model to tune anonymous M/N/K timings.
    context = _build_problem_context(request)[0] if request.get("round") == 0 else ""
    return _join_sections(
        context,
        _section("Search State", search_state),
        _section("Anchor Configs", anchor_configs),
        _section("Results (best first)", results),
        _section("Top Config Patterns", top_patterns),
        _section("Failed Config Patterns", failed_patterns),
        _section("Refused Configs", rejected_patterns),
        _bullet_section(
            "Next Step",
            _refinement_strategy_lines(
                unmeasured_count=unmeasured_count,
                total_count=total_count,
                rejected_count=rejected_count,
            ),
        ),
        _section("Task", task_section),
    )
