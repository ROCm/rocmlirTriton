# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Filter attention configs whose generated loop bound may change in PR #347."""

import argparse
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from attentionPerfUtils import canonical_config, normalize_option, option_value, parse_bool
from attentionPerfUtils import parse_config, read_config_lines, write_json, write_lines

DEFAULT_CONFIGS = Path(__file__).resolve().parent / "configs" / "tier1-attention-configs"
DEFAULT_DATA_TYPES = ("f16", "f32", "i8", "bf16")


def classify_config(config: str, conservative: bool = True) -> Tuple[Optional[str], List[str]]:
    """Return impact confidence and reasons, or ``None`` when unaffected."""
    options = parse_config(config)
    definite_reasons = []
    potential_reasons = []

    causal = option_value(options, "causal", "false")
    if causal is not None and parse_bool(causal):
        definite_reasons.append("causal")
    if option_value(options, "prefix_offset") is not None:
        definite_reasons.append("prefix-causal")
    if option_value(options, "current_seq_len") is not None:
        definite_reasons.append("kv-cache")

    seq_len_q = option_value(options, "seq_len_q")
    if conservative and seq_len_q == "1" and not definite_reasons:
        potential_reasons.append("decode-shaped SeqLenQ=1 KV-cache proxy")

    if definite_reasons:
        return "definite", definite_reasons
    if potential_reasons:
        return "potential", potential_reasons
    return None, []


def filter_configs(configs: Sequence[Tuple[int, str]],
                   conservative: bool = True) -> Tuple[List[str], List[Dict]]:
    selected = []
    records = []
    for line_number, config in configs:
        confidence, reasons = classify_config(config, conservative)
        if confidence is None:
            continue
        selected.append(config)
        records.append({
            "line": line_number,
            "config": config,
            "confidence": confidence,
            "reasons": reasons,
        })
    return selected, records


def set_config_option(config: str, name: str, value: str) -> str:
    tokens = shlex.split(config)
    normalized_name = normalize_option(name)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        option = token.split("=", 1)[0]
        if normalize_option(option) == normalized_name:
            if "=" in token:
                tokens[index] = f"{option}={value}"
            else:
                tokens[index + 1] = value
            return shlex.join(tokens)
        index += 1 if "=" in token else 2
    tokens.extend([f"-{name}", value])
    return shlex.join(tokens)


def remove_config_option(config: str, name: str) -> str:
    tokens = shlex.split(config)
    normalized_name = normalize_option(name)
    retained = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        option = token.split("=", 1)[0]
        consumed = 1 if "=" in token else 2
        if normalize_option(option) != normalized_name:
            retained.extend(tokens[index:index + consumed])
        index += consumed
    return shlex.join(retained)


def expand_data_types(config: str) -> List[str]:
    options = parse_config(config)
    if "t" in options:
        return [config]
    return [set_config_option(config, "t", data_type) for data_type in DEFAULT_DATA_TYPES]


def create_performance_configs(configs: Sequence[Tuple[int, str]]) -> Tuple[List[str], List[Dict]]:
    """Create causal-only and pure KV-cache variants for performance testing."""
    causal_configs = []
    causal_records = []
    kv_configs = []
    kv_records: Dict[str, Dict] = {}

    for line_number, config in configs:
        for expanded in expand_data_types(config):
            options = parse_config(expanded)
            causal = option_value(options, "causal", "false")
            if causal is not None and parse_bool(causal):
                causal_configs.append(expanded)
                causal_records.append({
                    "kind": "causal",
                    "source_lines": [line_number],
                    "config": expanded,
                })

            group_size = int(options["g"])
            seq_len_k = int(options["seq_len_k"])
            if seq_len_k < 2:
                raise ValueError("KV-cache configs require SeqLenK >= 2")
            current_seq_len = ",".join([str(seq_len_k - 1)] * group_size)
            kv_config = remove_config_option(expanded, "prefix_offset")
            kv_config = set_config_option(kv_config, "causal", "false")
            kv_config = set_config_option(kv_config, "current_seq_len", current_seq_len)
            identity = canonical_config(kv_config)
            if identity in kv_records:
                kv_records[identity]["source_lines"].append(line_number)
                continue
            kv_configs.append(kv_config)
            kv_records[identity] = {
                "kind": "kv-cache",
                "source_lines": [line_number],
                "config": kv_config,
            }

    records = causal_records + list(kv_records.values())
    return causal_configs + kv_configs, records


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mode",
                        choices=("conservative", "exact", "performance"),
                        default="conservative",
                        help=("Conservative mode retains SeqLenQ=1 proxies; performance mode "
                              "creates causal-only and pure KV-cache variants"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_arguments(argv)
    configs = read_config_lines(args.input)
    if args.mode == "performance":
        selected, records = create_performance_configs(configs)
        counts = {
            "causal_count": sum(record["kind"] == "causal" for record in records),
            "kv_cache_count": sum(record["kind"] == "kv-cache" for record in records),
        }
    else:
        selected, records = filter_configs(configs, conservative=args.mode == "conservative")
        counts = {
            "definite_count": sum(record["confidence"] == "definite" for record in records),
            "potential_count": sum(record["confidence"] == "potential" for record in records),
        }
    manifest = args.manifest or Path(f"{args.output}.json")
    write_lines(args.output, selected)
    write_json(
        manifest, {
            "source": str(args.input),
            "mode": args.mode,
            "input_count": len(configs),
            "selected_count": len(selected),
            **counts,
            "selected": records,
        })
    print(f"Selected {len(selected)} of {len(configs)} configs into {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
