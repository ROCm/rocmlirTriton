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
from typing import Any, Dict, List, Sequence

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

_SYSTEM_PROMPT = textwrap.dedent("""\
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

    The block tiles (the primary knobs):
    - mPerBlock, nPerBlock: the M x N output tile one workgroup computes. This
      is the single most consequential choice: it fixes how many workgroups
      launch, how much LDS a stage needs, and how many registers the
      accumulator occupies. A gemm+gemm (attention) config spells these
      mPerBlockG0 and nPerBlockG0, and adds nPerBlockG1 for the second GEMM's
      output tile, where 0 means untiled.
    - kPerBlock: how much of the contraction dimension one iteration consumes.
      Deeper means fewer, larger LDS loads and better matrix-instruction
      utilization, but more LDS per stage. The A and B tiles together must fit
      in LDS numStages times over, which is what makes large tiles and deep
      pipelining compete for the same budget.

    Scheduling and layout:
    - numWaves: waves per workgroup. More waves split the tile more finely, so
      a large tile usually wants more of them and a small tile is starved by
      them.
    - matrixInstrNonkdim: the M/N extent of the matrix instruction, typically
      16 or 32. 16 and 32 are genuinely different families rather than points
      on a scale: 32 amortizes more work per instruction, 16 wastes less on a
      tile that does not divide by 32.
    - kpack: how many matrix instructions issue from one LDS load. Above 1 it
      reduces LDS traffic; the ceiling is in the hardware section.
    - numStages: software pipeline depth over the K loop. 1 is safest. 2 to 4
      overlaps loads with math on a streaming loop, at numStages times the LDS
      for the tiles. The sweeps behind the seed configs stopped at 3, so any
      higher value the Configuration Space offers is unmeasured here rather
      than known to be bad. On a gemm+gemm kernel 4 is worth a proposal of its
      own: the chained-dot pipeline schedule, and the pingpong that rides on
      it, are written for exactly that depth and are skipped at any other.
    - splitKFactor: splits the contraction across that many workgroups, which
      then reduce their partial results. This is the answer to a problem too
      small to fill the machine by tiling M and N, and it is a distinct family:
      it buys parallelism and pays for it with a reduction. Leave it at 1
      unless M x N is small relative to the CU count. On a gemm+gemm kernel
      the contraction being split is the one the two GEMMs share, the first
      GEMM's N and so the second GEMM's K, and the factor is the second
      GEMM's; that dimension is the long one, so the knob is a real option
      there. Attention is the exception and must leave this at 1. It splits
      that same dimension by a different route, which is a property of the
      problem rather than a knob: the Problem section reports it as splitKV.
    - gridGroupSize: how many M-tile blocks are grouped when workgroups are
      mapped onto the grid. Larger groups improve last-level-cache locality
      across the group and cost scheduling flexibility. 0 lets the compiler
      choose. Attention lays out its grid differently and takes no group size,
      so the space pins this at 0 there.
    - numCTAs: workgroups per cooperative cluster. Only useful where the
      hardware section says multi-CTA launch is available.
    - wavesPerEU: a hint to the backend for how many waves to keep resident per
      execution unit, which it honours by limiting registers per wave. 0 means
      no hint. A large tile plus a high wavesPerEU cannot both be satisfied,
      and the space refuses that combination rather than compiling it.

    The eight use* knobs are tri-state, and this is the part most easily got
    wrong. Each takes -1, 0 or 1:
    - -1 means "let the compiler decide", and applies a per-architecture
      heuristic. It is the default, and it is a reasonable answer.
    - 0 forces the transform off; 1 forces it on.
    Every seed config below spells -1 in all eight, and 0 in wavesPerEU and
    gridGroupSize. Read nothing into that. Those configs are distilled from
    sweeps that pinned exactly those fields and varied only the tiles and the
    schedule, so a column of -1 records what was never tried rather than what
    won. They are the least explored part of this space, not the settled part.
    Do not treat these as booleans and do not set them all. Move one when you
    have a reason to believe the heuristic is wrong for this shape, and leave
    the rest at -1. What each gates:
    - useAsyncCopy: direct-to-LDS global loads, bypassing registers.
    - useBlockPingpong: the pingpong schedule, which alternates two wave groups
      between loading and computing. Needs MFMA and enough waves to split.
    - useInThreadTranspose: an in-thread transpose of a loaded tile.
    - useBufferOps: the buffer-ops pass cluster (buffer addressing rather than
      flat pointers).
    - useBufferAtomics: buffer atomics, which require useBufferOps to be on.
      Setting this to 1 with useBufferOps at 0 is refused.
    - useReductionLayout: redistributes warps onto the reduction dimension to
      cut register spill. -1 rewrites convolutions only.
    - useOptimizeEpilogue: Triton's epilogue optimization.
    - useBf16x3ForF32: decomposes an f32 dot into three bf16 dots. Only
      relevant to f32 inputs.

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
      numStages, or splitKFactor, rather than rewriting every field at once.

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


def build_system_prompt() -> str:
    """Return the global instruction block shared by every request."""
    return _SYSTEM_PROMPT


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


def build_initial_prompt(request: Dict[str, Any]) -> str:
    """Build the full initial user prompt, for the round that has no results."""
    problem = request.get("problem", {})
    hardware = request.get("hardware", {})
    space = request.get("space", {})
    default_config = request.get("defaultConfig", {})
    hints = compute_workload_hints(problem, hardware)

    default_section = _section(
        "Default Configuration",
        format_config_for_prompt(default_config) +
        "\n  Any parameter you do not mention takes its value from this config.",
    )
    task_section = ("Propose the first batch of configs. Include both near-default and "
                    f"exploratory candidates. {RETURN_JSON_ONLY}")
    return _join_sections(
        _section("Problem", describe_problem(problem)),
        _section("GPU Hardware", describe_hardware(hardware)),
        _bullet_section("What The Shapes Suggest", hints) if hints else "",
        _section("Configuration Space", render_space(space, default_config)),
        default_section,
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
    return _join_sections(
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
