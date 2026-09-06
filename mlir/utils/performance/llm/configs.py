# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Read the configs out of a model's reply, and describe the space to it.

A shrunken port of Helion's `helion/autotuner/llm/configs.py`. The parse,
validate and dedupe skeleton of `parse_response_configs` survives; almost all
of upstream's validation does not, because it exists to catch mistakes that
cannot be made here. Upstream has list-valued fields (`BlockIdSequence`,
`ListOf`) whose lengths a model has to get right, and a power-of-two rule on
`num_warps`; every Rock perf-config parameter is a scalar integer, so what is
left to check is "is this an integer?". Whether it is a *legal* integer is the
tuning space's question, and C++ asks it (`TuningParamAxes::isFeasible`).

`render_perf_config` has no upstream counterpart: it is what turns the sparse
config a model answers with into the whole named perf config that is the only
form anything in rocmlirTriton accepts. It lives here, rather than the merge
being done in C++ or the sparseness being left to the perf-config parser's own
handling of missing keys, because "unspecified" has to mean the exemplar the
prompt was written around -- and a partial `gemm:key=value,...` string means
the schema defaults in RockAttrDefs.td instead, which are a serialization
fallback rather than a config anyone would run.

`describe_config_space` is a rewrite rather than a port, for reasons under
`render_space`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .parsing import parse_jsonish

Config = Dict[str, int]

# Ladders longer than this are elided in the middle. Roughly seventeen
# parameters at a few dozen values each is a few hundred integers, which is
# cheap, but `kPerBlock` in a wide space can run to well over a hundred values
# on its own and listing all of them buys nothing.
MAX_LADDER_VALUES = 40

# The tri-state knob sentinel, `rock::kKnobDefault`. Not a boolean: it means
# "the compiler decides", and it is what every config the quick tuning list
# hands out spells, so it is frequently the right answer rather than a
# fallback.
KNOB_DEFAULT = -1

# Short wire names reduce the model's output tokens without changing the
# perf-config schema. Full names remain accepted for compatibility.
RESPONSE_ALIASES = {
    "mPerBlock": "m",
    "nPerBlock": "n",
    "mPerBlockG0": "m0",
    "nPerBlockG0": "n0",
    "nPerBlockG1": "n1",
    "kPerBlock": "k",
    "kpack": "p",
    "numCTAs": "c",
    "numWaves": "w",
    "matrixInstrNonkdim": "i",
    # Neither of these is a single letter, where every other alias is. `s` and
    # `d` came from upstream's `num_stages`, and here they read as split and
    # depth to this side and as stages and something else to the model: it
    # annotated `s` as "more stages" and "stages 6" repeatedly, and once caught
    # itself mid-sentence ("but d3 = stages, okay"). A config meant as a deeper
    # pipeline that arrives as a three-way split of the contraction is not a
    # config anybody proposed, and the two extra characters buy that back.
    "splitKFactor": "sk",
    "numStages": "st",
    "wavesPerEU": "e",
    "gridGroupSize": "g",
    "useAsyncCopy": "ac",
    "useBlockPingpong": "bp",
    "useInThreadTranspose": "it",
    "useBufferOps": "bo",
    "useBufferAtomics": "ba",
    "useReductionLayout": "rl",
    "useOptimizeEpilogue": "oe",
    "useBf16x3ForF32": "bf",
}

# The parameters the system prompt describes whatever the space says about
# them, because they are what a perf config is mostly about. `build_system_prompt`
# gates every other bullet on the parameter having more than one value; these
# it emits unconditionally, so their aliases are worth the handful of
# characters even where the space pins them. The alternative is a model that
# has just read a paragraph about mPerBlock and finds no way to spell it.
ALWAYS_DESCRIBED = ("mPerBlock", "nPerBlock", "mPerBlockG0", "nPerBlockG0", "nPerBlockG1",
                    "kPerBlock")


def alias_config(config: Config) -> Config:
    """Spell known perf-config fields with their short response names."""
    return {RESPONSE_ALIASES.get(name, name): value for name, value in config.items()}


def render_response_aliases(space: Dict[str, Sequence[int]]) -> str:
    """The aliases available for this problem, as a compact legend.

    A parameter the space pins to a single value is left out, for the reason
    `knob_names` gives for leaving out a pinned knob: naming it reads as an
    invitation, and a config that takes the invitation is refused before it is
    compiled. Across one conv sweep every prompt offered aliases for numCTAs,
    kpack, matrixInstrNonkdim, useAsyncCopy and useBlockPingpong while the
    Configuration Space three lines below said each was fixed.
    """
    pairs = [
        f"{alias}={name}" for name, alias in RESPONSE_ALIASES.items()
        if name in space and (len(space[name]) > 1 or name in ALWAYS_DESCRIBED)
    ]
    return "  " + ", ".join(pairs)


def parse_response_configs(response: str, *, space: Dict[str, Sequence[int]]) -> List[Config]:
    """Pull the sparse configs out of a reply, dropping what cannot be one.

    Silence rather than failure on a malformed config: a model that returns
    twelve good configs and one with a typo has had a useful round, and the
    round the search actually loses is the one that raises. A reply with
    *nothing* usable in it comes back as an empty list, which the caller
    reports as an error, since that is a real failure.
    """
    parsed = parse_jsonish(response)
    raw_configs: Any = None
    if isinstance(parsed, dict):
        raw_configs = parsed.get("configs")
    elif isinstance(parsed, list):
        raw_configs = parsed
    if not isinstance(raw_configs, list):
        return []

    configs: List[Config] = []
    seen = set()
    for raw in raw_configs:
        config = _sparse_config(raw, space)
        if config is None:
            continue
        # A model asked for unique configs will still repeat itself; C++ dedupes
        # against every earlier round as well, but there is no point sending it
        # the same config twice in one batch.
        key = tuple(sorted(config.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(config)
    return configs


def _sparse_config(raw: Any, space: Dict[str, Sequence[int]]) -> Config | None:
    """One config as the response gave it, or None if it is not one at all."""
    if not isinstance(raw, dict):
        return None
    config: Config = {}
    names_by_alias = {alias: name for name, alias in RESPONSE_ALIASES.items() if name in space}
    for key, value in raw.items():
        key = names_by_alias.get(key, key)
        if key not in space:
            # A parameter this space does not have. Dropping it rather than the
            # whole config: the rest of what the model said about this config
            # may still be worth trying.
            continue
        # `True` is an int in Python, and a model that writes `true` for a knob
        # means 1, so bools are accepted and narrowed rather than refused.
        if isinstance(value, bool):
            config[key] = int(value)
        elif isinstance(value, int):
            config[key] = value
        elif isinstance(value, float) and value.is_integer():
            config[key] = int(value)
    return config or None


def render_perf_config(exemplar: str, config: Config) -> str | None:
    """Complete one sparse config against `exemplar` and spell it out in full.

    `exemplar` is the request's `defaultPerfConfig`: the config the prompt was
    written around, serialized by the attribute that owns the format. Editing
    that string rather than building one from scratch is what keeps the prefix
    (`gemm:` or `attn:`), the field names and their order on the C++ side --
    nothing here has to know what a perf config looks like beyond it being
    `prefix:key=value,...`.

    Every field is emitted, including the ones the model did not touch, for
    the reason `Rock_PerfConfigSchema` gives for doing the same: defaults can
    change over time, so a config is never left to be reconstructed on parse.

    Returns None if `exemplar` is not a named perf config, which would mean
    this and the search disagree about the request schema.
    """
    prefix, sep, body = exemplar.partition(":")
    if not sep or not prefix or not body:
        return None

    fields = []
    for piece in body.split(","):
        key, sep, value = piece.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            return None
        # A parameter the model did not name keeps the exemplar's value. One
        # it named that the exemplar does not have cannot be spelled at all,
        # and was dropped by `_sparse_config` before it got here.
        fields.append(f"{key}={config.get(key, value)}")
    return f"{prefix}:" + ",".join(fields)


def render_perf_configs(exemplar: str, configs: Sequence[Config]) -> List[str]:
    """Every proposal as a whole perf config, in the order they arrived.

    Deduped again, because two sparse configs that differ only in a field they
    both set to the exemplar's own value are one config once completed.
    """
    rendered = []
    for config in configs:
        perf_config = render_perf_config(exemplar, config)
        if perf_config is not None and perf_config not in rendered:
            rendered.append(perf_config)
    return rendered


def bound_admits(bound: Dict[str, Any], value: int) -> bool:
    """Whether `value` is one this parameter may hold, by C++'s `TileBounds`.

    The same arithmetic as `TileBounds::admits`, kept in step with it because
    both answer the same question and only one of them is the authority: a
    config this side lets through and C++ refuses costs the round a proposal,
    while one this side drops never reaches C++ to be measured.
    """
    minimum = bound.get("min", 1)
    if not isinstance(value, int) or value < minimum:
        return False
    if not bound.get("pow2Only"):
        return True
    return value == 0 or (value & (value - 1)) == 0


def space_admits(space: Dict[str, Sequence[int]], bounds: Optional[Dict[str, Any]], name: str,
                 value: int) -> bool:
    """Whether this problem accepts `value` for `name`, by whichever rule owns it.

    Two rules, because C++ has two: a tile is held to its bounds and everything
    else to its ladder. Reading the bounds off the request rather than keeping a
    list of tile names here, so that the two sides cannot drift over which
    parameters are tiles.
    """
    bound = (bounds or {}).get(name)
    if isinstance(bound, dict):
        return bound_admits(bound, value)
    values = space.get(name)
    return values is None or value in values


def render_bounds(bound: Dict[str, Any]) -> str:
    """Describe what a tile may be, which is a rule rather than a list.

    A tile is legal wherever it is positive, and a power of two besides on the
    kernels that cannot decompose one that isn't; the ladder beside it is what
    the enumerated search walks, capped by the problem's own dimensions (see
    `TileBounds` in RockTuning.h). Rendering the ladder here would be reading
    the wrong one of the two out loud: it stops at 16 on a problem with 16 rows
    while every wider tile still builds, and a model told otherwise spends its
    proposals asking permission it already has.

    No ceiling is named because there is none to name. What rules out a wide
    tile is the LDS it needs or Triton's cap on one tensor's elements, both of
    them facts about the whole config, and both reported back by name when they
    fire.
    """
    minimum = bound.get("min", 1)
    if bound.get("pow2Only"):
        described = f"a power of two, {max(minimum, 1)} or more"
        return described if minimum > 0 else f"0, or {described}"
    return f"any integer {minimum} or more"


def render_ladder(values: Sequence[int], default: int) -> str:
    """Render one parameter's values, marking the one it defaults to.

    Enumerated rather than described. Helion renders a type descriptor --
    `power_of_2(min=64, max=1024, default=128)` -- which works because its axes
    are regular. Ours are not: `numWaves` is the powers of two a workgroup can
    hold, `matrixInstrNonkdim` is two instruction widths and a zero meaning
    "you choose", and `wavesPerEU` counts up one at a time. A min/max/step
    descriptor would misdescribe them, and a model that believed it would spend
    proposals on values that do not exist.

    Which is exactly why the tiles are described instead: theirs is a rule, and
    `render_bounds` states it.
    """
    if not values:
        return "(no values)"
    if len(values) == 1:
        return f"fixed at {values[0]}"

    def mark(value: int) -> str:
        return f"{value}*" if value == default else str(value)

    if len(values) <= MAX_LADDER_VALUES:
        return "[" + ", ".join(mark(value) for value in values) + "]"

    head = MAX_LADDER_VALUES // 2
    tail = MAX_LADDER_VALUES - head - 1
    shown = ([mark(value) for value in values[:head]] +
             [f"... ({len(values) - head - tail} more) ..."] +
             [mark(value) for value in values[-tail:]])
    return "[" + ", ".join(shown) + "]"


def render_space(space: Dict[str, Sequence[int]],
                 default_config: Config,
                 bounds: Optional[Dict[str, Any]] = None) -> str:
    """Describe every parameter, what it may be, and its default.

    Generated, never hand-written: the names and values come from the tuning
    space itself, so a parameter added to or removed from the perf config
    reaches the prompt with no change here. What is worth *saying* about a
    parameter is hand-written prose, in prompting.py.

    A parameter with bounds is described by them and a parameter without by its
    ladder, which is the same split C++ makes when it decides what to refuse.
    """
    lines = []
    for name, values in space.items():
        default = default_config.get(name)
        bound = (bounds or {}).get(name)
        if isinstance(bound, dict):
            line = f"  {name}: {render_bounds(bound)}"
            if default is not None:
                line += f"   (default {default})"
            lines.append(line)
            continue
        line = f"  {name}: {render_ladder(values, default)}"
        if len(values) > 1 and default is not None:
            line += f"   (default {default}, marked *)"
        lines.append(line)
    return "\n".join(lines)


def knob_names(space: Dict[str, Sequence[int]]) -> List[str]:
    """The parameters that accept -1 and something else, which is what makes
    them tri-state here.

    Read off the values C++ sent rather than from a list kept here, so that a
    knob added to the perf config is described correctly without this file
    knowing it exists.

    A knob pinned to -1 alone is left out. Both callers name these to the model
    as knobs it may set, and on one convolution three of the eight named --
    useAsyncCopy, useBf16x3ForF32 and useBlockPingpong -- were pinned by the
    same Configuration Space the prompt calls the authority, so setting one is
    refused before it is compiled.
    """
    return [name for name, values in space.items() if KNOB_DEFAULT in values and len(values) > 1]
