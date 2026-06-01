#!/usr/bin/env python3
"""Compare TFlops between a benchmarking tuning DB and a reference tuning DB.

Both files are tier1-style tuning TSVs with a header line:
  # arch  numCUs  numChiplets  testVector  perfConfig  TFlops  tuningSpace ...

For each entry in the benchmarking file, the matching row in the reference
file is found by a configurable key (default: arch, numCUs, numChiplets,
testVector), and the difference in TFlops (bench - ref) plus the percentage
change relative to the reference is reported.

Use --match to relax the key when the two files were tuned on different GPU
partitions (e.g. benchmark on numCUs=152/numChiplets=1 vs reference on
numCUs=304/numChiplets=8): --match arch-vector or --match vector.

When arch strings differ (e.g. gfx1201 vs gfx1200), add --ignore-arch or use
--match vector. When both arch and numCUs/numChiplets differ, use
--match vector or --ignore-arch --ignore-partition.

Usage:
  python3 compareTuningTFlops.py -b bench.tsv -r ref.tsv
  python3 compareTuningTFlops.py -b bench.tsv -r ref.tsv --match arch-vector
  python3 compareTuningTFlops.py -b bench.tsv -r ref.tsv --match vector
  python3 compareTuningTFlops.py -b bench.tsv -r ref.tsv --ignore-arch --ignore-partition
  python3 compareTuningTFlops.py -b bench.tsv -r ref.tsv -o diff.tsv
  python3 compareTuningTFlops.py -b bench.tsv -r ref.tsv --sort pct
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Column indices in the tuning TSV.
ARCH_IDX = 0
NUM_CUS_IDX = 1
NUM_CHIPLETS_IDX = 2
TEST_VECTOR_IDX = 3
PERF_CONFIG_IDX = 4
TFLOPS_IDX = 5
MIN_COLUMNS = 6

Key = Tuple[str, ...]

# Which columns make up the match key for each --match mode.
MATCH_COLUMNS = {
    "full": (ARCH_IDX, NUM_CUS_IDX, NUM_CHIPLETS_IDX, TEST_VECTOR_IDX),
    "arch-vector": (ARCH_IDX, TEST_VECTOR_IDX),
    "vector": (TEST_VECTOR_IDX,),
}


@dataclass
class Row:
    line_no: int
    arch: str
    num_cus: str
    num_chiplets: str
    test_vector: str
    perf_config: str
    tflops: float
    raw_line: str

    def key(self, columns: Tuple[int, ...]) -> Key:
        field_by_idx = {
            ARCH_IDX: self.arch,
            NUM_CUS_IDX: self.num_cus,
            NUM_CHIPLETS_IDX: self.num_chiplets,
            TEST_VECTOR_IDX: self.test_vector,
        }
        return tuple(field_by_idx[idx] for idx in columns)


def resolve_match_columns(
    match: str,
    ignore_arch: bool,
    ignore_partition: bool,
) -> Tuple[int, ...]:
    columns = MATCH_COLUMNS[match]
    if ignore_arch:
        columns = tuple(c for c in columns if c != ARCH_IDX)
    if ignore_partition:
        columns = tuple(
            c for c in columns if c not in (NUM_CUS_IDX, NUM_CHIPLETS_IDX)
        )
    if not columns:
        columns = (TEST_VECTOR_IDX,)
    return columns


def parse_row(line_no: int, line: str) -> Optional[Row]:
    parts = line.split("\t")
    if len(parts) < MIN_COLUMNS:
        raise ValueError(f"expected at least {MIN_COLUMNS} tab columns, got {len(parts)}")

    try:
        tflops = float(parts[TFLOPS_IDX])
    except ValueError as exc:
        raise ValueError(f"invalid TFlops value {parts[TFLOPS_IDX]!r}") from exc

    return Row(
        line_no=line_no,
        arch=parts[ARCH_IDX].strip(),
        num_cus=parts[NUM_CUS_IDX].strip(),
        num_chiplets=parts[NUM_CHIPLETS_IDX].strip(),
        test_vector=parts[TEST_VECTOR_IDX].strip(),
        perf_config=parts[PERF_CONFIG_IDX].strip(),
        tflops=tflops,
        raw_line=line,
    )


def read_rows(path: Path) -> Tuple[List[Row], List[str]]:
    rows: List[Row] = []
    warnings: List[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = parse_row(line_no, line)
            except ValueError as exc:
                warnings.append(f"{path.name}:L{line_no}: {exc}")
                continue
            if row is not None:
                rows.append(row)
    return rows, warnings


def build_reference_index(
    rows: List[Row], columns: Tuple[int, ...]
) -> Tuple[Dict[Key, Row], List[str]]:
    index: Dict[Key, Row] = {}
    warnings: List[str] = []
    for row in rows:
        key = row.key(columns)
        if key in index:
            warnings.append(
                f"reference L{row.line_no}: duplicate key, keeping first occurrence "
                f"(L{index[key].line_no})"
            )
            continue
        index[key] = row
    return index, warnings


@dataclass
class Comparison:
    arch: str
    num_cus: str
    num_chiplets: str
    test_vector: str
    bench_perf_config: str
    ref_perf_config: str
    bench_tflops: float
    ref_tflops: float

    @property
    def diff(self) -> float:
        return self.bench_tflops - self.ref_tflops

    @property
    def pct_change(self) -> Optional[float]:
        if self.ref_tflops == 0.0:
            return None
        return (self.diff / self.ref_tflops) * 100.0


def compare(
    bench_rows: List[Row],
    ref_index: Dict[Key, Row],
    columns: Tuple[int, ...],
) -> Tuple[List[Comparison], List[Row]]:
    comparisons: List[Comparison] = []
    missing: List[Row] = []
    for row in bench_rows:
        ref = ref_index.get(row.key(columns))
        if ref is None:
            missing.append(row)
            continue
        comparisons.append(
            Comparison(
                arch=row.arch,
                num_cus=row.num_cus,
                num_chiplets=row.num_chiplets,
                test_vector=row.test_vector,
                bench_perf_config=row.perf_config,
                ref_perf_config=ref.perf_config,
                bench_tflops=row.tflops,
                ref_tflops=ref.tflops,
            )
        )
    return comparisons, missing


def format_pct(pct: Optional[float]) -> str:
    if pct is None:
        return "n/a"
    return f"{pct:+.2f}%"


def write_report(
    out,
    comparisons: List[Comparison],
    missing: List[Row],
    warnings: List[str],
) -> None:
    header = [
        "arch",
        "numCUs",
        "numChiplets",
        "testVector",
        "refTFlops",
        "benchTFlops",
        "diffTFlops",
        "pctChange",
    ]
    out.write("\t".join(header) + "\n")
    for cmp in comparisons:
        fields = [
            cmp.arch,
            cmp.num_cus,
            cmp.num_chiplets,
            cmp.test_vector,
            f"{cmp.ref_tflops:.6g}",
            f"{cmp.bench_tflops:.6g}",
            f"{cmp.diff:+.6g}",
            format_pct(cmp.pct_change),
        ]
        out.write("\t".join(fields) + "\n")

    if missing:
        out.write(f"\n# {len(missing)} benchmark entries had no reference match:\n")
        for row in missing:
            out.write(
                "# "
                + "\t".join(
                    [row.arch, row.num_cus, row.num_chiplets, row.test_vector]
                )
                + "\n"
            )

    if warnings:
        out.write(f"\n# {len(warnings)} warnings:\n")
        for msg in warnings:
            out.write(f"# {msg}\n")


def classify(cmp: "Comparison", threshold_pct: float) -> str:
    """Bucket a comparison as improved / regressed / unchanged.

    Anything within +/- threshold_pct of the reference counts as unchanged.
    When the reference TFlops is 0 (pct is undefined), fall back to the raw
    diff sign.
    """
    pct = cmp.pct_change
    if pct is None:
        if cmp.diff > 0:
            return "improved"
        if cmp.diff < 0:
            return "regressed"
        return "unchanged"
    if pct > threshold_pct:
        return "improved"
    if pct < -threshold_pct:
        return "regressed"
    return "unchanged"


def print_summary(
    comparisons: List[Comparison], missing: List[Row], threshold_pct: float
) -> None:
    matched = len(comparisons)
    pcts = [c.pct_change for c in comparisons if c.pct_change is not None]
    buckets = [classify(c, threshold_pct) for c in comparisons]
    improved = buckets.count("improved")
    regressed = buckets.count("regressed")
    unchanged = buckets.count("unchanged")

    print(
        f"matched={matched} improved={improved} regressed={regressed} "
        f"unchanged={unchanged} missing={len(missing)} "
        f"(unchanged within +/-{threshold_pct:g}%)",
        file=sys.stderr,
    )
    if pcts:
        avg = sum(pcts) / len(pcts)
        print(
            f"avg pct change={avg:+.2f}% "
            f"(min={min(pcts):+.2f}%, max={max(pcts):+.2f}%)",
            file=sys.stderr,
        )


def sort_comparisons(comparisons: List[Comparison], key: str) -> List[Comparison]:
    if key == "none":
        return comparisons
    if key == "diff":
        return sorted(comparisons, key=lambda c: c.diff)
    if key == "pct":
        return sorted(
            comparisons,
            key=lambda c: (c.pct_change is None, c.pct_change or 0.0),
        )
    return comparisons


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare TFlops between a benchmarking and a reference tuning DB.",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        required=True,
        help="Benchmarking tuning TSV (each entry is looked up in the reference)",
    )
    parser.add_argument(
        "-r",
        "--reference",
        required=True,
        help="Reference tuning TSV (superset of entries)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help='Output TSV (default: stdout). Use "-" for stdout.',
    )
    parser.add_argument(
        "--sort",
        choices=["none", "diff", "pct"],
        default="none",
        help="Sort rows by raw diff or percentage change (default: none)",
    )
    parser.add_argument(
        "--match",
        choices=sorted(MATCH_COLUMNS),
        default="full",
        help=(
            "Key columns used to match rows: "
            "'full' = arch+numCUs+numChiplets+testVector (default), "
            "'arch-vector' = arch+testVector, "
            "'vector' = testVector only. Use a relaxed key when the two files "
            "were tuned on different GPU partitions (numCUs/numChiplets differ)."
        ),
    )
    parser.add_argument(
        "--ignore-arch",
        action="store_true",
        help="Drop arch from the match key (e.g. gfx1201 bench vs gfx1200 ref)",
    )
    parser.add_argument(
        "--ignore-partition",
        action="store_true",
        help="Drop numCUs and numChiplets from the match key",
    )
    parser.add_argument(
        "--unchanged-threshold",
        type=float,
        default=5.0,
        metavar="PCT",
        help=(
            "Percentage band treated as unchanged in the summary: a change "
            "within +/- this many percent counts as unchanged (default: 5.0)"
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    bench_path = Path(args.benchmark).expanduser().resolve()
    ref_path = Path(args.reference).expanduser().resolve()
    for label, path in (("benchmark", bench_path), ("reference", ref_path)):
        if not path.is_file():
            print(f"error: {label} file not found: {path}", file=sys.stderr)
            return 1

    bench_rows, bench_warnings = read_rows(bench_path)
    ref_rows, ref_warnings = read_rows(ref_path)
    if not bench_rows:
        print(f"error: no data rows parsed from {bench_path}", file=sys.stderr)
        return 1

    match_columns = resolve_match_columns(
        args.match, args.ignore_arch, args.ignore_partition
    )
    ref_index, dup_warnings = build_reference_index(ref_rows, match_columns)
    comparisons, missing = compare(bench_rows, ref_index, match_columns)
    comparisons = sort_comparisons(comparisons, args.sort)

    warnings = bench_warnings + ref_warnings + dup_warnings

    if args.output == "-":
        write_report(sys.stdout, comparisons, missing, warnings)
    else:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as out:
            write_report(out, comparisons, missing, warnings)
        print(f"Wrote {output_path}", file=sys.stderr)

    print_summary(comparisons, missing, args.unchanged_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
