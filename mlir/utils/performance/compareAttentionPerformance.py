# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compare base and candidate attention benchmark CSV files."""

import argparse
import csv
import io
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from attentionPerfUtils import atomic_write_text, canonical_config, parse_config

PARAMETER_COLUMNS = [
    ("DataType", "t", ""),
    ("TransQ", "transq", "false"),
    ("TransK", "transk", "false"),
    ("TransV", "transv", "false"),
    ("TransO", "transo", "false"),
    ("Causal", "causal", "false"),
    ("ReturnLSE", "return_lse", "false"),
    ("SplitKV", "split_kv", "1"),
    ("G", "g", ""),
    ("SeqLenQ", "seq_len_q", ""),
    ("SeqLenK", "seq_len_k", ""),
    ("NumHeadsQ", "num_heads_q", "1"),
    ("NumHeadsKV", "num_heads_kv", "1"),
    ("HeadDimQK", "head_dim_qk", ""),
    ("HeadDimV", "head_dim_v", ""),
    ("WithAttnScale", "with_attn_scale", "false"),
    ("WithAttnBias", "with_attn_bias", "false"),
    ("TransBias", "transbias", "false"),
]
OUTPUT_FIELDS = [
    "Chip", *[column for column, _, _ in PARAMETER_COLUMNS], "Config", "RocmlirGenFlags",
    "Base SHA", "Candidate SHA", "Base PerfConfig", "Candidate PerfConfig", "Base Samples",
    "Candidate Samples", "Base TFlops", "Candidate TFlops", "% Diff"
]


def read_csv_rows(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def aggregate_samples(rows: Sequence[Mapping[str, str]], expected_label: str) -> Dict:
    revisions = {row["SourceSha"] for row in rows}
    if len(revisions) != 1:
        raise ValueError(f"{expected_label} results must contain exactly one source revision")
    grouped: Dict[Tuple[str, str], List[Mapping[str, str]]] = {}
    seen_samples = set()
    for row in rows:
        if row["RunLabel"] != expected_label:
            raise ValueError(f"Expected RunLabel={expected_label}, found {row['RunLabel']}")
        key = (row["Chip"], canonical_config(row["Config"]))
        sample_key = (*key, int(row["Sample"]))
        if sample_key in seen_samples:
            raise ValueError(f"Duplicate sample for {key}: {row['Sample']}")
        seen_samples.add(sample_key)
        grouped.setdefault(key, []).append(row)

    result = {}
    for key, samples in grouped.items():
        shas = {sample["SourceSha"] for sample in samples}
        perf_configs = {sample["PerfConfig"] for sample in samples}
        runtime_flags = {sample["RocmlirGenFlags"] for sample in samples}
        if len(shas) != 1 or len(perf_configs) != 1 or len(runtime_flags) != 1:
            raise ValueError(f"Inconsistent benchmark metadata for {key}")
        measurements = [float(sample["TFlops"]) for sample in samples]
        if any(not math.isfinite(value) or value <= 0 for value in measurements):
            raise ValueError(f"Invalid TFlops measurement for {key}")
        result[key] = {
            "sha": next(iter(shas)),
            "perf_config": next(iter(perf_configs)),
            "config": samples[0]["Config"],
            "runtime_flags": next(iter(runtime_flags)),
            "samples": len(samples),
            "sample_ids": {int(sample["Sample"]) for sample in samples},
            "tflops": statistics.median(measurements),
        }
    return result


def config_columns(config: str) -> Dict[str, str]:
    options = parse_config(config)
    return {column: options.get(option, default) for column, option, default in PARAMETER_COLUMNS}


def compare_results(base_rows: Sequence[Mapping[str, str]], candidate_rows: Sequence[Mapping[str,
                                                                                             str]],
                    expected_samples: int) -> List[Dict]:
    if expected_samples < 1:
        raise ValueError("expected samples must be positive")
    base = aggregate_samples(base_rows, "base")
    candidate = aggregate_samples(candidate_rows, "candidate")
    if set(base) != set(candidate):
        missing_candidate = sorted(set(base) - set(candidate))
        missing_base = sorted(set(candidate) - set(base))
        raise ValueError(
            f"Benchmark keys differ; missing candidate={missing_candidate}, missing base={missing_base}"
        )

    compared = []
    for chip, config_id in sorted(base):
        base_result = base[(chip, config_id)]
        candidate_result = candidate[(chip, config_id)]
        expected_sample_ids = set(range(1, expected_samples + 1))
        if base_result["sample_ids"] != candidate_result["sample_ids"]:
            raise ValueError(f"Sample IDs differ for {(chip, config_id)}")
        if base_result["sample_ids"] != expected_sample_ids:
            raise ValueError(f"Incomplete sample IDs for {(chip, config_id)}")
        if base_result["runtime_flags"] != candidate_result["runtime_flags"]:
            raise ValueError(f"Runtime flags differ for {(chip, config_id)}")
        base_tflops = base_result["tflops"]
        candidate_tflops = candidate_result["tflops"]
        row = {
            "Chip": chip,
            **config_columns(base_result["config"]),
            "Config": base_result["config"],
            "RocmlirGenFlags": base_result["runtime_flags"],
            "Base SHA": base_result["sha"],
            "Candidate SHA": candidate_result["sha"],
            "Base PerfConfig": base_result["perf_config"],
            "Candidate PerfConfig": candidate_result["perf_config"],
            "Base Samples": base_result["samples"],
            "Candidate Samples": candidate_result["samples"],
            "Base TFlops": base_tflops,
            "Candidate TFlops": candidate_tflops,
            "% Diff": 100.0 * (candidate_tflops - base_tflops) / base_tflops,
        }
        compared.append(row)
    return compared


def write_comparison(path: Path, rows: Sequence[Mapping]) -> None:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_arguments(argv)
    rows = compare_results(read_csv_rows(args.base), read_csv_rows(args.candidate),
                           args.expected_samples)
    write_comparison(args.output, rows)
    print(f"Wrote {len(rows)} comparisons to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
