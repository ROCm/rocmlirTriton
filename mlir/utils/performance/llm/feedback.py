# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Format benchmark results into compact feedback for LLM prompts.

Ported nearly unchanged from Helion's `helion/autotuner/llm/feedback.py`.
Everything here works on (config dict, time) pairs, which is a shape both
autotuners share, so only two things differ:

- Times arrive as nanoseconds, since that is what `BenchmarkResult::timeNs`
  carries, and are shown to the model as microseconds rather than upstream's
  milliseconds; `format_time` says why.
- The statuses are `success` / `notApplicable` / `failed`, from
  `BenchmarkResult::Status`, in place of upstream's `error` / `timeout` /
  `peer_compilation_fail`. `notApplicable` has no upstream counterpart and is
  worth keeping distinct: it means the config was refused before it ran, which
  is a different thing to be told than a compile that broke.
"""

from __future__ import annotations

import collections
import json
import math
from typing import Any, Dict, List, Sequence, Tuple

from .configs import alias_config

MAX_RESULTS_IN_PROMPT = 8
MAX_ANCHORS_IN_PROMPT = 2
MAX_CHANGED_FIELDS_PER_CONFIG = 6
MAX_FAILURES_IN_PROMPT = 5
MAX_REJECTIONS_IN_PROMPT = 4

Config = Dict[str, int]
Result = Dict[str, Any]


def format_time(time_ns: float) -> str:
    """Render a kernel time in the unit these kernels are measured in.

    Microseconds, and not the milliseconds Helion prints, because these
    kernels are far quicker than the ones upstream tunes: across
    rocmlirTriton's own checked-in GEMM results, half run in under 30 us and
    the quickest in under 3. At two decimals in milliseconds most of the field
    would print as `0.00 ms`, leaving the model to rank candidates it cannot
    tell apart. Nanoseconds, the unit on the wire, go the other way and spend
    digits on run-to-run noise.
    """
    return f"{time_ns / 1000.0:.2f} us"


def format_config_for_prompt(config: Config) -> str:
    """Serialize a config exactly as it should appear in prompt examples."""
    return json.dumps(alias_config(config), sort_keys=True, separators=(",", ":"))


def config_diff(default_config: Config, config: Config) -> Config:
    """Drop unchanged defaults so feedback focuses on meaningful tunables."""
    diff = {
        key: value
        for key, value in config.items()
        if key not in default_config or value != default_config[key]
    }
    if not diff:
        return {"(default)": True}
    return diff


def format_config_diff(default_config: Config, config: Config) -> str:
    """Format only the changed fields as compact JSON."""
    return json.dumps(
        alias_config(config_diff(default_config, config)),
        sort_keys=True,
        separators=(",", ":"),
    )


def _was_timed(result: Result) -> bool:
    """Whether a result carries a benchmark time that can be compared."""
    time_ns = result.get("timeNs")
    return (result.get("status") == "success" and isinstance(time_ns, (int, float)) and
            math.isfinite(time_ns))


def measured_results(results: Sequence[Result]) -> List[Tuple[Config, float]]:
    """Return successful benchmark results sorted from fastest to slowest."""
    timed = [(result["config"], result["timeNs"]) for result in results if _was_timed(result)]
    return sorted(timed, key=lambda pair: pair[1])


def unmeasured_results(results: Sequence[Result]) -> List[Result]:
    """Return the configs that could not be timed, whatever the reason."""
    return [result for result in results if result.get("status") != "success"]


def format_results_for_llm(
    results: Sequence[Result],
    default_config: Config,
    *,
    limit: int = MAX_RESULTS_IN_PROMPT,
) -> str:
    """Format successful and failed benchmark results into a compact block."""
    if not results:
        return "No results yet."

    ranked = measured_results(results)
    unmeasured = len(unmeasured_results(results))

    lines: List[str] = []
    for index, (config, time_ns) in enumerate(ranked[:limit], start=1):
        lines.append(f"  #{index}: {format_time(time_ns)} - "
                     f"{format_config_diff(default_config, config)}")
    if unmeasured > 0:
        lines.append(f"  ({unmeasured} configs could not be compiled or run)")
    return "\n".join(lines)


def summarize_search_state_for_llm(
    results: Sequence[Result],
    default_config: Config,
) -> str:
    """Summarize the current best result, coverage, and failure counts."""
    ranked = measured_results(results)
    if not ranked:
        return "  No successful configs yet."

    best_config, best_time = ranked[0]
    lines = [
        f"  Best so far: {format_time(best_time)} - "
        f"{format_config_diff(default_config, best_config)}"
    ]
    if len(ranked) >= 2 and ranked[1][1] > 0:
        gap_pct = ((ranked[1][1] - best_time) / ranked[1][1]) * 100
        lines.append(f"  Margin vs runner-up: {gap_pct:.1f}%")
    lines.append(f"  Search coverage: {len(ranked)} measured / {len(results)} total configs")
    unmeasured = len(unmeasured_results(results))
    if unmeasured > 0:
        lines.append(f"  Configs that could not be run: {unmeasured}")
    return "\n".join(lines)


def summarize_failed_configs_for_llm(
    results: Sequence[Result],
    default_config: Config,
) -> str:
    """Show a small sample of failed configs so the next round can avoid them."""
    unmeasured = unmeasured_results(results)
    if not unmeasured:
        return "  Every config so far ran."

    # `buildSeedBatch` pads its batch with configs drawn at random across every
    # axis at once, and those fail often. Listed as failures the same way a
    # proposal is, they read as patterns to avoid: one round told the model
    # that six configs had failed and offered kPerBlock=208 and numWaves=16
    # among the things they had in common, when the only thing they had in
    # common was having been generated at random. A config that moves more
    # fields than any proposal is allowed to cannot be a proposal, so it is
    # counted rather than held up as an example.
    probes = [result for result in unmeasured
              if len(config_diff(default_config, result["config"])) >
              MAX_CHANGED_FIELDS_PER_CONFIG]
    proposals = [result for result in unmeasured if result not in probes]
    if probes:
        lines = [f"  {len(probes)} of the random configs padding the seed batch could not "
                 "be run. Each moves most of its fields at once, so there is no pattern "
                 "in them for you to avoid."]
    else:
        lines = []
    if not proposals:
        return "\n".join(lines) or "  Every config so far ran."

    counts = collections.Counter(result.get("status", "failed") for result in proposals)
    count_summary = ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
    lines.append(f"  Counts: {count_summary}")

    seen = set()
    for result in proposals:
        label = result.get("status", "failed")
        diff_text = format_config_diff(default_config, result["config"])
        key = f"{label}:{diff_text}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  {label}: {diff_text}")
        if len(seen) >= MAX_FAILURES_IN_PROMPT:
            break
    return "\n".join(lines)


def summarize_rejected_configs_for_llm(
    rejected: Sequence[Result],
    default_config: Config,
) -> str:
    """Show which proposals the tuning space refused, and on which check.

    No Helion counterpart. Upstream normalizes an out-of-range config into
    range, so nothing it proposes is ever turned away; here the space checks a
    config against constraints its parameter ranges cannot express -- LDS
    capacity, Triton's per-tensor element cap, the register budget behind
    `wavesPerEU` -- and refuses it before anything is compiled. A model that is
    not told which of its proposals died that way will keep making them.
    """
    if not rejected:
        return "  None: every config proposed so far was accepted."

    counts = collections.Counter(entry.get("reason", "unknown") for entry in rejected)
    lines = [
        "  Counts by check: " +
        ", ".join(f"{reason}={count}" for reason, count in sorted(counts.items()))
    ]
    seen = set()
    for entry in rejected:
        reason = entry.get("reason", "unknown")
        diff_text = format_config_diff(default_config, entry["config"])
        key = f"{reason}:{diff_text}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  {reason}: {diff_text}")
        if len(lines) > MAX_REJECTIONS_IN_PROMPT:
            break
    return "\n".join(lines)


def summarize_anchor_configs_for_llm(
    results: Sequence[Result],
    default_config: Config,
    *,
    limit: int = MAX_ANCHORS_IN_PROMPT,
) -> str:
    """Show the strongest current configs that the next round should refine."""
    ranked = measured_results(results)
    if not ranked:
        return "  No successful configs yet."

    best_time = ranked[0][1]
    lines: List[str] = []
    for index, (config, time_ns) in enumerate(ranked[:limit], start=1):
        if index == 1 or best_time <= 0:
            delta = "best"
        else:
            gap_pct = ((time_ns - best_time) / best_time) * 100
            delta = f"+{gap_pct:.1f}%"
        lines.append(f"  Anchor {index} ({delta}): {format_time(time_ns)} - "
                     f"{format_config_diff(default_config, config)}")
    return "\n".join(lines)


def analyze_top_configs(
    results: Sequence[Result],
    default_config: Config,
) -> str:
    """Highlight which field values repeat across the best configs so far."""
    ranked = measured_results(results)
    if len(ranked) < 3:
        return "Not enough results for analysis yet."

    def changed_fields(config: Config) -> Config:
        diff = config_diff(default_config, config)
        return {key: value for key, value in diff.items() if key != "(default)"}

    top = [changed_fields(config) for config, _ in ranked[:5]]
    all_keys = sorted({key for config in top for key in config})
    if not all_keys:
        return "The best configs are all close to the default config."

    lines: List[str] = []
    for key in all_keys:
        # Every top config votes, including the ones whose sparse diff leaves
        # this field alone: an omitted field is a vote for the default, not an
        # abstention. Counting only the configs that mention it turned two
        # fastest configs at mPerBlock=64 into "mPerBlock: always 128", a claim
        # the Results section directly above it contradicted.
        values = [config.get(key, default_config.get(key)) for config in top]
        if not values:
            continue
        counts = collections.Counter(json.dumps(value) for value in values)
        common = counts.most_common(2)
        if len(common) == 1:
            lines.append(f"  {key}: always {common[0][0]}")
        elif common[0][1] >= 3:
            lines.append(f"  {key}: mostly {common[0][0]} (also {common[1][0]} x{common[1][1]})")
        else:
            summary = ", ".join(f"{value} x{count}" for value, count in common)
            lines.append(f"  {key}: {summary}")
    return "\n".join(lines) or "No clear patterns yet."
