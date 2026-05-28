#!/usr/bin/env python3
"""Group tuned GEMM problem configs by winning kpack value.

Reads a tier1-style TSV (arch, numCUs, testVector, perfConfig, ...) and writes a
report listing each problem config under its winning kpack.

Perf-config kpack field positions (rocMLIR AccelGemmParamsAttr):
  v3: mPerBlock, nPerBlock, kpackPerBlock, mPerWave, nPerWave, kpack, ...
      kpack is field index 5 (0-based).
  v4: mPerBlock, nPerBlock, kpackPerBlock, mPerWave, nPerWave, mnPerXdl, kpack, ...
      kpack is field index 6 (0-based).

Usage:
  python3 reportWinningKpackByProblem.py
  python3 reportWinningKpackByProblem.py -i /path/to/tier1-gemm-configs-exhaustive.tsv
  python3 reportWinningKpackByProblem.py -i input.tsv -o /path/to/out -f my-report.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, List, Optional, TextIO, Tuple

# ---------------------------------------------------------------------------
# User knobs — edit these defaults; CLI flags override them.
# ---------------------------------------------------------------------------
DEFAULT_INPUT_TSV = (
    "/mnt/dcgpuval/mhalilce/tuning-data/gfx942/tier1-gemm-configs-exhaustive.tsv"
)
# None => write next to the input file.
DEFAULT_OUTPUT_DIR: Optional[str] = None
# None => "<input-stem>-by-kpack.txt"
DEFAULT_OUTPUT_FILENAME: Optional[str] = None

V3_KPACK_FIELD_INDEX = 5
V4_KPACK_FIELD_INDEX = 6
PERFCONFIG_RE = re.compile(r"(v[34]:[\d,]+)")


@dataclass(frozen=True)
class TunedRow:
    line_no: int
    arch: str
    test_vector: str
    perf_config: str
    perf_version: str
    kpack: int
    raw_line: str


def extract_perf_config(line: str) -> str:
    match = PERFCONFIG_RE.search(line)
    if not match:
        raise ValueError(f"no v3/v4 perf config in line: {line!r}")
    return match.group(1)


def parse_kpack(perf_config: str) -> Tuple[str, int]:
    if perf_config.startswith("v4:"):
        fields = [int(x) for x in perf_config[3:].split(",")]
        index = V4_KPACK_FIELD_INDEX
        version = "v4"
    elif perf_config.startswith("v3:"):
        fields = [int(x) for x in perf_config[3:].split(",")]
        index = V3_KPACK_FIELD_INDEX
        version = "v3"
    else:
        raise ValueError(f"unsupported perf config prefix: {perf_config!r}")

    if len(fields) <= index:
        raise ValueError(
            f"{version} config has {len(fields)} fields, need at least "
            f"{index + 1}: {perf_config!r}"
        )
    return version, fields[index]


def parse_row(line_no: int, line: str) -> TunedRow:
    parts = line.split("\t")
    if len(parts) < 3:
        raise ValueError(f"expected at least 3 tab columns, got {len(parts)}")

    perf_config = extract_perf_config(line)
    perf_idx = next(i for i, part in enumerate(parts) if part == perf_config)
    arch = parts[0]
    test_vector = parts[perf_idx - 1] if perf_idx > 0 else ""
    perf_version, kpack = parse_kpack(perf_config)

    return TunedRow(
        line_no=line_no,
        arch=arch,
        test_vector=test_vector,
        perf_config=perf_config,
        perf_version=perf_version,
        kpack=kpack,
        raw_line=line,
    )


def read_tuned_rows(path: Path) -> Tuple[List[TunedRow], List[str]]:
    rows: List[TunedRow] = []
    warnings: List[str] = []

    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(parse_row(line_no, line))
            except ValueError as exc:
                warnings.append(f"L{line_no}: {exc}")
    return rows, warnings


def resolve_output_path(
    input_path: Path,
    output_dir: Optional[str],
    output_filename: Optional[str],
) -> Path:
    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_name = output_filename if output_filename else f"{input_path.stem}-by-kpack.txt"
    return out_dir / out_name


def write_report(
    out: TextIO,
    input_path: Path,
    rows: List[TunedRow],
    warnings: List[str],
) -> None:
    by_kpack: DefaultDict[int, List[TunedRow]] = defaultdict(list)
    for row in rows:
        by_kpack[row.kpack].append(row)

    versions = sorted({row.perf_version for row in rows})

    out.write("# Winning kpack grouped by problem config\n")
    out.write(f"# source: {input_path}\n")
    out.write(f"# rows: {len(rows)}\n")
    out.write(f"# perf config versions: {', '.join(versions)}\n")
    out.write(f"# distinct kpack values: {len(by_kpack)}\n")
    if warnings:
        out.write(f"# parse warnings: {len(warnings)}\n")
    out.write("\n")

    for kpack in sorted(by_kpack):
        group = by_kpack[kpack]
        out.write(f"=== kpack={kpack} ({len(group)} problem configs) ===\n")
        for row in sorted(group, key=lambda r: r.line_no):
            out.write(
                f"L{row.line_no}\t{row.perf_version}\t{row.arch}\t"
                f"{row.test_vector}\t{row.perf_config}\n"
            )
        out.write("\n")

    if warnings:
        out.write("=== parse warnings ===\n")
        for msg in warnings:
            out.write(f"{msg}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report problem configs grouped by winning kpack (v3/v4).",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT_TSV,
        help="Input tier1-style TSV (default: DEFAULT_INPUT_TSV in script)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: same directory as input TSV)",
    )
    parser.add_argument(
        "-f",
        "--output-filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help='Output file name (default: "<input-stem>-by-kpack.txt")',
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"error: input TSV not found: {input_path}", file=sys.stderr)
        return 1

    output_path = resolve_output_path(
        input_path,
        args.output_dir,
        args.output_filename,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows, warnings = read_tuned_rows(input_path)
    if not rows:
        print(f"error: no data rows parsed from {input_path}", file=sys.stderr)
        return 1

    with output_path.open("w", encoding="utf-8") as out:
        write_report(out, input_path, rows, warnings)

    print(f"Wrote {output_path} ({len(rows)} rows, {len(warnings)} warnings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
