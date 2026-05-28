#!/usr/bin/env python3
"""Extract tier1-style GEMM problem configs from a kpack report.

Reads a report produced by reportWinningKpackByProblem.py (or the same layout)
and writes one test vector per line, matching mlir/utils/performance/configs/
tier1-gemm-configs.

Input line format (tab-separated):
  L<line>\t<arch>\t<numCUs>\t<testVector>\t<perfConfig>

Usage:
  python3 extractProblemConfigsFromKpackReport.py --arch 942
  python3 extractProblemConfigsFromKpackReport.py --arch 90a --kpack 4 -o 90a-gemm-kpack-4
  python3 extractProblemConfigsFromKpackReport.py -i gemm-kpack-90a.txt
  python3 extractProblemConfigsFromKpackReport.py --arch 942 --split-by-kpack
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable, List, Optional, Set

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kpackProblemConfigUtils import (
    DEFAULT_KPACK_GROUPS,
    kpack_group_path,
    kpack_report_path,
    normalize_arch_token,
)

PERFCONFIG_RE = re.compile(r"(v[34]:[\d,]+)")
SECTION_RE = re.compile(r"^=== kpack=(\d+)")
DATA_LINE_RE = re.compile(r"^L\d+\t")


@dataclass(frozen=True)
class ProblemConfig:
    line_no: int
    kpack: Optional[int]
    test_vector: str


def extract_test_vector(line: str) -> str:
    match = PERFCONFIG_RE.search(line)
    if not match:
        raise ValueError(f"no v3/v4 perf config in line: {line!r}")

    perf_config = match.group(1)
    parts = line.split("\t")
    perf_idx = next(i for i, part in enumerate(parts) if part == perf_config)
    if perf_idx <= 0:
        raise ValueError(f"test vector not found before perf config: {line!r}")

    test_vector = parts[perf_idx - 1].strip()
    if not test_vector.startswith("-t "):
        raise ValueError(f"expected test vector to start with '-t ', got: {test_vector!r}")
    return test_vector


def parse_data_line(line_no: int, line: str, kpack: Optional[int]) -> ProblemConfig:
    line_no_prefix = line.split("\t", 1)[0]
    if not DATA_LINE_RE.match(line):
        raise ValueError(f"expected data line starting with L<number>, got: {line_no_prefix!r}")

    return ProblemConfig(
        line_no=line_no,
        kpack=kpack,
        test_vector=extract_test_vector(line),
    )


def read_problem_configs(
    path: Path,
    kpack_filter: Optional[Set[int]] = None,
) -> tuple[List[ProblemConfig], List[str]]:
    configs: List[ProblemConfig] = []
    warnings: List[str] = []
    current_kpack: Optional[int] = None

    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            section_match = SECTION_RE.match(line)
            if section_match:
                current_kpack = int(section_match.group(1))
                continue
            if line.startswith("==="):
                continue

            if kpack_filter is not None and current_kpack not in kpack_filter:
                continue

            try:
                configs.append(parse_data_line(line_no, line, current_kpack))
            except ValueError as exc:
                warnings.append(f"L{line_no}: {exc}")

    return configs, warnings


def dedupe_preserve_order(configs: Iterable[ProblemConfig]) -> List[ProblemConfig]:
    seen: Set[str] = set()
    unique: List[ProblemConfig] = []
    for config in configs:
        if config.test_vector in seen:
            continue
        seen.add(config.test_vector)
        unique.append(config)
    return unique


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input:
        return Path(args.input).expanduser().resolve()
    return kpack_report_path(args.arch)


def resolve_single_output_path(args: argparse.Namespace, arch_token: str) -> Path | str:
    if args.output:
        return args.output
    if args.stdout:
        return "-"
    if args.kpack and len(args.kpack) == 1:
        return kpack_group_path(arch_token, args.kpack[0])
    return "-"


def write_config_lines(path: Path | str, configs: Sequence[str]) -> None:
    output_text = "\n".join(configs) + ("\n" if configs else "")
    if path == "-":
        sys.stdout.write(output_text)
        return

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")
    print(f"Wrote {output_path} ({len(configs)} configs)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract tier1-style GEMM problem configs from a kpack report.",
    )
    parser.add_argument(
        "--arch",
        default="942",
        help="Arch token for default input/output names (e.g. 942, 90a, 1101)",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="Input kpack report (default: gemm-kpack-<arch>.txt in repo root)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help='Output file (default: stdout, or "<arch>-gemm-kpack-<kpack>" when --kpack has one value)',
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write extracted configs to stdout",
    )
    parser.add_argument(
        "--kpack",
        type=int,
        nargs="+",
        default=None,
        help="Only include configs from these kpack groups",
    )
    parser.add_argument(
        "--split-by-kpack",
        action="store_true",
        help=(
            "Write one '<arch>-gemm-kpack-<N>' file per kpack group found "
            f"(default groups when filtering: {' '.join(map(str, DEFAULT_KPACK_GROUPS))})"
        ),
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate test vectors (default: dedupe, preserving first occurrence)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        arch_token = normalize_arch_token(args.arch)
        input_path = resolve_input_path(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not input_path.is_file():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 1

    kpack_filter = set(args.kpack) if args.kpack else None
    configs, warnings = read_problem_configs(input_path, kpack_filter)
    if not configs:
        print(f"error: no problem configs parsed from {input_path}", file=sys.stderr)
        return 1

    if not args.keep_duplicates:
        configs = dedupe_preserve_order(configs)

    if args.split_by_kpack:
        grouped: DefaultDict[int, List[str]] = defaultdict(list)
        for config in configs:
            if config.kpack is None:
                continue
            grouped[config.kpack].append(config.test_vector)

        target_kpacks = sorted(grouped)
        if not target_kpacks:
            print("error: no kpack sections found to split", file=sys.stderr)
            return 1

        for kpack in target_kpacks:
            output_path = kpack_group_path(arch_token, kpack)
            write_config_lines(output_path, grouped[kpack])
        return 0

    output_lines = [config.test_vector for config in configs]
    output_target = resolve_single_output_path(args, arch_token)
    write_config_lines(output_target, output_lines)

    if warnings:
        for msg in warnings:
            print(msg, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
