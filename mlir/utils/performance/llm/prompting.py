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

from .configs import knob_names, render_response_aliases, render_space, space_admits
from .feedback import (
    MAX_CHANGED_FIELDS_PER_CONFIG,
    format_config_diff,
    format_config_for_prompt,
)
from .workload import (
    compute_workload_hints,
    summarize_hardware_for_prompt,
    summarize_problem_for_prompt,
)

RETURN_JSON_ONLY = 'Return minified JSON only: {"configs":[...]}'

_INITIAL_STRATEGY_BASE_LINES = (
    "Use about 40% near-default, 40% balanced and 20% aggressive candidates.",
    # The same bound the output contract and the refinement rounds give. It
    # read "2-6 changed fields" until a sweep showed the two disagreeing: a
    # one-field config obeys the contract and breaks this, and the model
    # settled it by padding, restating a field at its own default in getting
    # on for half of every config it proposed.
    f"Keep configs sparse: usually 1-4 changed fields and never more than "
    f"{MAX_CHANGED_FIELDS_PER_CONFIG}, omitting unchanged defaults.",
    "Cover at least 3 coherent M/N/K tile families.",
    "Do not combine the largest tiles, deepest pipeline and highest split.",
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

# The same bullet under the two names the tiles go by, rather than one bullet
# that renames itself halfway through. Which one the kernel uses is in the
# space, so there is no reason to make the model read past the other.
_GEMM_TILE_BULLET = textwrap.dedent("""\
    - mPerBlock, nPerBlock: the M x N output tile one workgroup computes. This
      is the single most consequential choice: it fixes how many workgroups
      launch, how much LDS a stage needs, and how many registers the
      accumulator occupies. It is also the choice worth spending proposals on:
      shape the tile like the GEMM rather than keeping it square, taking
      nPerBlock from N and mPerBlock from M. Which side can afford a generous
      tile is set by the smallest of M, N and K, since that one decides how
      much reuse there is to amortize the loads a larger tile issues: tile the
      long dimensions generously and the short one tightly.""")

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
      pipelining compete for the same budget. That budget is counted in bytes,
      so narrow operands afford depth that f32 does not; take the scale from
      the element width and the LDS size in the hardware section rather than
      from any fixed number.""")


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


def _offers(space: Optional[Dict[str, Sequence[int]]], name: str, value: int) -> bool:
    """Whether the space still lists `value` for `name`.

    Distinct from `_is_tunable`, which asks whether the parameter can move at
    all. This asks about one rung, because `without_no_op_values` drops rungs
    rather than parameters and the prose about a rung has to go with it.
    """
    if space is None:
        return True
    values = space.get(name)
    return values is None or value in values


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
        "- numWaves: waves per workgroup; match it to tile size.",
    "matrixInstrNonkdim":
        "- matrixInstrNonkdim: matrix-instruction M/N width; 16 and 32 are distinct families.",
    "kpack":
        "- kpack: matrix instructions issued per LDS load.",
    "numStages":
        "- numStages: K-loop pipeline depth; more overlap costs more LDS.",
    "splitKFactor":
        "- splitKFactor: splits the contraction across that many workgroups, which then "
        "reduce their partial results; 1 disables the split and any integer above it is "
        "legal. The grid is G x ceil(M/mPerBlock) x ceil(N/nPerBlock) workgroups, so a "
        "split pays where that count falls short of the CU count, by about the factor it "
        "falls short by.",
    "gridGroupSize":
        "- gridGroupSize: M-grid grouping for cache locality; 0 is heuristic.",
    "numCTAs":
        "- numCTAs: workgroups per cooperative cluster.",
    "wavesPerEU":
        "- wavesPerEU: occupancy hint that limits registers; 0 means no hint.",
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

# The knobs the space left room for, in the order they are listed. Same rule as
# `_PARAM_BULLETS`: a knob pinned to -1 builds the one kernel whatever it is
# asked for. Names only, since one bullet describes all of them.
_KNOB_NAMES: Tuple[str, ...] = (
    "useAsyncCopy",
    "useBlockPingpong",
    "useInThreadTranspose",
    "useBufferOps",
    "useBufferAtomics",
    "useReductionLayout",
    "useOptimizeEpilogue",
    "useBf16x3ForF32",
)


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
    """A compact instruction block, gated to parameters this run can move."""
    gemm_gemm = space is not None and "nPerBlockG1" in space
    tiles = _GEMM_GEMM_TILE_BULLET if gemm_gemm else _GEMM_TILE_BULLET
    blocks = [
        textwrap.dedent("""\
            You are tuning a rocmlirTriton integer perf config for an AMD GPU.
            Use only the supplied Configuration Space and defaults. Return 15
            useful, sparse candidates, no two alike, as minified JSON and no
            prose."""),
        tiles,
        _KPERBLOCK_BULLET,
    ]

    scheduling = _bullets_for(_PARAM_BULLETS, space, gemm_gemm)
    if scheduling:
        blocks.append("Scheduling and layout:\n" + "\n".join(scheduling))

    knob_bullets = {
        name: f"- {name}: tri-state -1=compiler heuristic, 0=off, 1=on." for name in _KNOB_NAMES
    }
    knobs = _bullets_for(knob_bullets, space, gemm_gemm)
    if knobs:
        blocks.append("The use* knobs are tri-state, and -1 is not a missing answer but the usual "
                      "one: with the tile held fixed, moving one off -1 has usually been worth "
                      "nothing measurable. Move one when something about this problem gives you a "
                      "reason to expect it to help, and otherwise leave it at -1 and spend the "
                      "proposal on the tile.\n" + "\n".join(knobs))

        # Only where the ladder still holds the value being warned against.
        # `without_no_op_values` takes these out of the space it renders, and a
        # warning about a rung the model was not offered is a rung named twice:
        # naming one is what made the model reach for it in the first place.
        duplicates = []
        if _offers(space, "useBufferOps", 1) or _offers(space, "useBufferAtomics", 1):
            duplicates.append("- useBufferOps/useBufferAtomics: -1 and 1 produce the same "
                              "kernel, so an explicit 1 measures nothing.")
        if _offers(space, "useOptimizeEpilogue", 1):
            duplicates.append("- useOptimizeEpilogue: except for a 16 bits wide output, -1 "
                              "and 1 agree and an explicit 1 measures nothing.")
        if _may_pipeline_one_stage(space):
            if _is_tunable(space, "useBlockPingpong"):
                duplicates.append("- At numStages=1, useBlockPingpong=1 builds the kernel "
                                  "that -1 already selected.")
            if _is_tunable(space, "useAsyncCopy"):
                duplicates.append("- At numStages=1, useAsyncCopy=1 builds the kernel "
                                  "that -1 already selected.")
        if duplicates:
            blocks.append("Avoid duplicate kernels:\n" + "\n".join(duplicates))

    blocks.append(
        textwrap.dedent("""\
        Output contract:
        - Return exactly {"configs":[...]} on one line: no markdown or prose.
        - Use short names from Response Aliases; full names are accepted.
        - Values are scalar integers from the Configuration Space.
        - Omit unchanged/default fields; usually change 1-4 fields, at most 6.
        - Make every config different from the others; if unsure, return fewer
          valid configs."""))
    return "\n\n".join(blocks)


def _initial_strategy_lines(
    *,
    configs_requested: int,
    space: Dict[str, Sequence[int]],
    hints: Sequence[str],
) -> List[str]:
    """Build the bullet list used for the initial search-strategy section."""
    lines = [
        f"Propose up to {configs_requested} candidate configs, no two alike. "
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
    inert = [
        *knob_names(space), *(name for name in ("wavesPerEU", "gridGroupSize", "numStages")
                              if len(space.get(name, ())) > 1)
    ]
    if inert:
        held = "its" if len(inert) == 1 else "their"
        plural = "" if len(inert) == 1 else "s"
        lines.append(f"Keep {', '.join(inert)} at {held} Default Configuration value{plural} "
                     "unless something about this problem argues otherwise: held against a "
                     "fixed tile, moving one off its default has not been worth a measurable "
                     "amount.")
    # The fields whose effect outruns measurement noise, named so the budget the
    # line above frees up has somewhere to go. numWaves and splitKFactor belong
    # here as much as the tiles do, and neither is reachable from a tile family.
    decisive = [
        "the block tiles", "kPerBlock",
        *(name for name in ("numWaves", "splitKFactor") if len(space.get(name, ())) > 1)
    ]
    lines.append("Put the budget into " + ", ".join(decisive[:-1]) + " and " + decisive[-1] +
                 ", which are the fields whose effect is large enough to read off a single "
                 "timing. Spend the aggressive fifth on one of those that the other configs "
                 "miss.")
    lines.append("Spread the tiles: distinct M/N/K families and aspect ratios matter more "
                 "than near-duplicates.")
    return lines


def _refinement_strategy_lines(
    *,
    unmeasured_count: int,
    total_count: int,
    rejected_count: int,
    space: Optional[Dict[str, Sequence[int]]] = None,
) -> List[str]:
    """Build the bullet list used for the refinement-step section."""
    trouble = unmeasured_count + rejected_count
    if total_count > 0 and trouble * 3 >= total_count:
        lines = list(_FAILURE_HEAVY_REFINEMENT_LINES)
    else:
        lines = list(_DEFAULT_REFINEMENT_LINES)
    # Named from the space, because a fixed list is wrong in both directions on
    # a problem whose axes are narrow. It recommended kpack and
    # matrixInstrNonkdim on a convolution that pins both, spending a third of
    # the advice on moves the space refuses. The fields named here are the ones
    # whose effect is separable from measurement noise.
    movable = [
        name for name in ("kPerBlock", "numWaves", "kpack", "matrixInstrNonkdim", "splitKFactor")
        if space is None or len(space.get(name, ())) > 1
    ]
    lines.append("Prefer edits with attributable effects: move the block tiles" +
                 ("".join(f", {name}" for name in movable[:-1]) +
                  f" or {movable[-1]}" if movable else "") + " rather than rewriting every field.")
    # A tri-state flip on an otherwise unchanged anchor is a clean experiment
    # whose answer is already in: matched on the tile, a median +0.00%.
    if knobs := knob_names(space or {}):
        lines.append("A flip of " + ", ".join(knobs) + " that looks like a win in Results is "
                     "more likely to be noise: held against the same tile, none of them has "
                     "been worth a measurable amount. Spend configs on the tiles, kPerBlock, "
                     "numWaves and splitKFactor instead.")
    # Results are ranked by a single timing apiece, and nothing in them says how
    # far apart two configs have to be before the order means anything. Unsaid,
    # the model reads the top of the list as a gradient and mutates the luckiest
    # measurement: the winning config is the best of some 460 draws, and taking
    # the fastest sample of one is already optimistic by a median 5%.
    lines.append("Read gaps under 10% in Results as noise rather than signal: repeat runs of "
                 "one unchanged kernel routinely differ by that much. Follow the families "
                 "that lead by more than 10%, and do not read an ordering among the configs "
                 "bunched at the top.")
    lines.append("Keep each config sparse: usually 1-4 changed fields, and no more than "
                 f"{MAX_CHANGED_FIELDS_PER_CONFIG} unless absolutely necessary.")
    lines.append("If unsure, return fewer valid configs instead of verbose or malformed JSON.")
    return lines


def build_seed_config_section(seed_configs: Sequence[Dict[str, int]],
                              default_config: Optional[Dict[str, int]] = None,
                              space: Optional[Dict[str, Sequence[int]]] = None,
                              bounds: Optional[Dict[str, Any]] = None) -> str:
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
    alternatives.

    A seed naming a value this problem refuses is left out. The list is checked
    in for no particular chip, so its seeds routinely name values the axes do
    not carry -- held up as starting points by a prompt that also calls the
    Configuration Space the authority on legal values, and refused by `accept`
    when a run copies one. Asked of the bounds where there are bounds, since a
    tile off the ladder is exactly what such a seed usually names and exactly
    what the space now admits: dropping those would hide most of the list on
    the small problems, where it is the only evidence there is.

    The seeds are written as diffs against the default, like every other
    config the model is shown. Printed in full they repeat nineteen fields
    thirty-four times to say the five or six that differ: on that same
    convolution, 5845 of the initial prompt's 10234 characters.

    The body also has to say that these configs are spoken for. `buildSeedBatch`
    has handed every one of them to the benchmark by the time this prompt goes
    out and `accept` drops a proposal that repeats one, so asking the model to
    "include configs matching these" spent the round on configs that could not
    land: eight of fifteen proposals were exact repeats of seeds printed here,
    and saying so instead took the duplicates a round from 3.7 to 0.3.
    """
    default_config = default_config or {}
    if space:
        seed_configs = [
            config for config in seed_configs if all(
                space_admits(space, bounds, field, value) for field, value in config.items()
                if field in space)
        ]
    if not seed_configs:
        return ""
    body = (
        "rocmlirTriton's tuning heuristic proposes the following configs for "
        "this problem. They are already being benchmarked while you read this, "
        "so a config matching one of them is dropped rather than measured: "
        "propose mutations of them instead. A config matches a seed only when "
        "it changes the same fields to the same values; one field at a "
        "different value makes it new.\n"
        "Each is written as its difference from the default config above.\n" +
        "\n".join(f"  - {format_config_diff(default_config, config)}" for config in seed_configs))
    return _section("Heuristic Seed Configs", body)


def _build_problem_context(request: Dict[str, Any]) -> Tuple[str, List[str]]:
    """The context a conversation needs before it can reason about results."""
    problem = request.get("problem", {})
    hardware = request.get("hardware", {})
    space = request.get("space", {})
    bounds = request.get("bounds", {})
    default_config = request.get("defaultConfig", {})
    hints = compute_workload_hints(problem, hardware, space)
    default_section = _section(
        "Default Configuration",
        format_config_for_prompt(default_config) +
        "\n  Any parameter you do not mention takes its value from this config.",
    )
    return _join_sections(
        _section("Problem", summarize_problem_for_prompt(problem)),
        _section("GPU Hardware", summarize_hardware_for_prompt(hardware, space)),
        _bullet_section("What The Shapes Suggest", hints) if hints else "",
        _section("Configuration Space", render_space(space, default_config, bounds)),
        _section("Response Aliases", render_response_aliases(space)),
        default_section,
    ), hints


def build_initial_prompt(request: Dict[str, Any]) -> str:
    """Build the full initial user prompt, for the round that has no results."""
    space = request.get("space", {})
    context, hints = _build_problem_context(request)
    # The shapes and the CU count are argued for in the standing instructions
    # and worked out in What The Shapes Suggest, both of which the model reads
    # long before it reads this. What it reads last is a strategy section
    # asking for coverage and spread, and that is what a round 0 reply looked
    # like: fifteen configs justified as "vary this field", with the
    # convolution, its odd filter and the chip's 48 CUs never mentioned. So
    # this points back at those sections rather than restating them.
    task_section = ("Propose the first batch of configs. Include both near-default and "
                    "exploratory candidates, and justify every value you change against "
                    "the Problem, GPU Hardware and What The Shapes Suggest sections rather "
                    f"than by spreading values. {RETURN_JSON_ONLY}")
    return _join_sections(
        context,
        build_seed_config_section(request.get("seedConfigs", []), request.get("defaultConfig", {}),
                                  space, request.get("bounds", {})),
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
    task_section = (f"Propose up to {configs_requested} configs around the anchors above, "
                    "none of them already measured above and no two alike. Avoid the failed "
                    "and refused patterns above, and favour targeted edits with attributable "
                    f"effects. {RETURN_JSON_ONLY}")
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
                space=request.get("space", {}),
            ),
        ),
        _section("Task", task_section),
    )
