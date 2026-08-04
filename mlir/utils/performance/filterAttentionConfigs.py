# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Filter attention configs whose generated loop bound may change in PR #347."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from attentionPerfUtils import option_value, parse_bool, parse_config, read_config_lines, write_json
from attentionPerfUtils import write_lines

DEFAULT_CONFIGS = Path(__file__).resolve().parent / "configs" / "tier1-attention-configs"


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


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mode",
                        choices=("conservative", "exact"),
                        default="conservative",
                        help="Conservative mode also retains SeqLenQ=1 decode-shaped cases")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_arguments(argv)
    configs = read_config_lines(args.input)
    selected, records = filter_configs(configs, conservative=args.mode == "conservative")
    manifest = args.manifest or Path(f"{args.output}.json")
    write_lines(args.output, selected)
    write_json(
        manifest, {
            "source": str(args.input),
            "mode": args.mode,
            "input_count": len(configs),
            "selected_count": len(selected),
            "definite_count": sum(record["confidence"] == "definite" for record in records),
            "potential_count": sum(record["confidence"] == "potential" for record in records),
            "selected": records,
        })
    print(f"Selected {len(selected)} of {len(configs)} configs into {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
