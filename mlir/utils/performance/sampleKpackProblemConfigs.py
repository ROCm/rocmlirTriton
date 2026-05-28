#!/usr/bin/env python3
"""Randomly sample GEMM problem configs from kpack-grouped config lists.

Reads tier1-style test vectors (one per line) from files such as
942-gemm-kpack-4 / 942-gemm-kpack-8 / 942-gemm-kpack-16 and writes a
combined list with a fixed number of random draws from each input file.

Usage:
  python3 sampleKpackProblemConfigs.py --arch 942
  python3 sampleKpackProblemConfigs.py --arch 90a -o 90a-gemm-kpack-sample.txt
  python3 sampleKpackProblemConfigs.py --arch 1101 --count 15 --seed 42
  python3 sampleKpackProblemConfigs.py --input 942-gemm-kpack-4 942-gemm-kpack-8
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kpackProblemConfigUtils import (
    DEFAULT_KPACK_GROUPS,
    kpack_group_paths,
    kpack_sample_output_path,
    normalize_arch_token,
)

DEFAULT_ARCH = "942"


def read_configs(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")

    configs: List[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith("-t "):
                raise ValueError(
                    f"{path}:L{line_no}: expected test vector starting with '-t ', "
                    f"got: {line!r}"
                )
            configs.append(line)
    return configs


def sample_configs(configs: Sequence[str], count: int, rng: random.Random) -> List[str]:
    if count <= 0:
        return []
    if count > len(configs):
        raise ValueError(
            f"requested {count} samples but only {len(configs)} configs available"
        )
    return rng.sample(list(configs), count)


def dedupe_preserve_order(configs: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for config in configs:
        if config in seen:
            continue
        seen.add(config)
        unique.append(config)
    return unique


def resolve_input_paths(args: argparse.Namespace) -> tuple[str, list[Path]]:
    arch_token = normalize_arch_token(args.arch)
    if args.input:
        return arch_token, [path.expanduser().resolve() for path in args.input]

    kpacks = args.kpack if args.kpack else list(DEFAULT_KPACK_GROUPS)
    return arch_token, kpack_group_paths(arch_token, kpacks)


def resolve_output_path(args: argparse.Namespace, arch_token: str) -> str | Path:
    if args.output is not None:
        return args.output
    if args.stdout:
        return "-"
    return kpack_sample_output_path(arch_token)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample random GEMM problem configs from kpack-grouped lists.",
    )
    parser.add_argument(
        "--arch",
        default=DEFAULT_ARCH,
        help=(
            "Arch token for default input/output names "
            "(e.g. 942, 90a, 1101, 1201, or gfx942). "
            f"Default: {DEFAULT_ARCH}"
        ),
    )
    parser.add_argument(
        "--kpack",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Kpack groups to sample from when using --arch "
            f"(default: {' '.join(map(str, DEFAULT_KPACK_GROUPS))})"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit input files (overrides --arch/--kpack defaults)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=15,
        help="Number of configs to draw from each input file (default: 15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output file. Default: '<arch>-gemm-kpack-sample.txt' in repo root, "
            "or stdout with --stdout"
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write the combined list to stdout",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate test vectors in the combined list (default: dedupe)",
    )
    parser.add_argument(
        "--section-comments",
        action="store_true",
        help="Include '# from <file>' headers before each sampled group",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rng = random.Random(args.seed)

    try:
        arch_token, input_paths = resolve_input_paths(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sections: List[tuple[Path, List[str]]] = []
    try:
        for input_path in input_paths:
            all_configs = read_configs(input_path)
            picked = sample_configs(all_configs, args.count, rng)
            sections.append((input_path, picked))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sampled_all: List[str] = []
    for _, picked in sections:
        sampled_all.extend(picked)

    combined = sampled_all if args.keep_duplicates else dedupe_preserve_order(sampled_all)

    lines: List[str] = [
        "# Sampled GEMM problem configs",
        f"# arch: {arch_token}",
        f"# draw count per file: {args.count}",
    ]
    if args.seed is not None:
        lines.append(f"# seed: {args.seed}")
    if not args.keep_duplicates and len(combined) < len(sampled_all):
        lines.append(f"# deduped: removed {len(sampled_all) - len(combined)} duplicate(s)")
    lines.append("")

    if args.section_comments:
        for path, picked in sections:
            lines.append(f"# from {path.name} ({len(picked)} configs)")
            lines.extend(picked)
            lines.append("")
    else:
        lines.extend(combined)

    output_text = "\n".join(lines).rstrip() + "\n"
    output_target = resolve_output_path(args, arch_token)

    if output_target == "-":
        sys.stdout.write(output_text)
    else:
        output_path = Path(output_target).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
        print(
            f"Wrote {output_path} "
            f"({len(combined)} configs from {len(sections)} files)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
