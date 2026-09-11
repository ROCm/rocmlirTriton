#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Quick Tuning Generator

Generates the quick-tuning shards under
include/mlir/Dialect/Rock/Tuning/QuickTuningShards/ from tuning data produced
by tuningRunner.py, plus the QuickTuningShards.inc index that lists them.

One shard holds one lookup key: its set cover, and the best split-K and
non-split-K config recorded for each problem the data measured. A shard is
always written whole, so its contents depend only on the input data.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perfCommonUtils import SPLITK_KEY, parse_perfconfig  # noqa: E402

# Column definitions for grouping problems
GEMM_COLUMNS = ['TransA', 'TransB', 'TransO', 'G', 'M', 'K', 'N']
CONV_COLUMNS = [
    'Direction', 'FilterLayout', 'InputLayout', 'OutputLayout', 'N', 'C', 'H', 'W', 'K', 'Y', 'X',
    'DilationH', 'DilationW', 'StrideH', 'StrideW', 'PaddingH', 'PaddingW'
]
ATTENTION_COLUMNS = [
    'TransQ', 'TransK', 'TransV', 'TransO', 'Causal', 'ReturnLSE', 'SplitKV',
    'SlidingWindowLookBack', 'WithAttnScale', 'WithAttnBias', 'TransBias', 'G', 'SeqLenQ',
    'SeqLenK', 'NumHeadsQ', 'NumHeadsKV', 'HeadDimQK', 'HeadDimV'
]
GEMM_GEMM_COLUMNS = ['TransA', 'TransB', 'TransC', 'TransO', 'G', 'M', 'K', 'N', 'O']
CONV_GEMM_COLUMNS = [
    'FilterLayout', 'InputLayout', 'TransC', 'TransO', 'N', 'C', 'H', 'W', 'K', 'Y', 'X',
    'DilationH', 'DilationW', 'StrideH', 'StrideW', 'PaddingH', 'PaddingW', 'O'
]

# Maps the user-facing --op value to its C++ KernelType name
OP_TO_KERNEL_TYPE = {
    'gemm': 'Gemm',
    'conv': 'Conv',
    'attention': 'Attention',
    'gemm_gemm': 'GemmElementwiseGemm',
    'conv_gemm': 'ConvElementwiseGemm',
}

# Operations whose problems may or may not permit split-K depending on the
# fusion they end up in (including gemm+gemm and conv+gemm). Attention is
# excluded: it parallelizes over the KV sequence via the -split_kv kernel
# argument, never via the perf-config's splitKFactor, so a split-K-free
# duplicate would be the same problem twice.
SPLIT_K_AWARE_OPS = frozenset({'gemm', 'conv', 'gemm_gemm', 'conv_gemm'})

# Column carrying the problem hash that rocmlir-gen emits into .debug rows. The
# generator groups by it and never constructs one, so it stays an opaque token:
# a `0x`-prefixed lowercase 64-bit hex string. Rows without it (older data)
# contribute to the set cover only.
PROBLEM_HASH_COLUMN = 'ProblemHash'
PROBLEM_HASH_PATTERN = re.compile(r'^0x[0-9a-f]{1,16}$')

# Legacy positional perf configs spell their fields as `prefix:vN:a,b,c,...`
# instead of `key=value`. splitKFactor sits at index 7 in every version up to
# v5; v6 inserts a field ahead of it, so a positional v6 has to be rejected
# rather than silently misread.
POSITIONAL_SPLITK_INDEX = 7
POSITIONAL_MAX_VERSION = 5
POSITIONAL_VERSION_PATTERN = re.compile(r'^v(\d+)$')

# Where the shards live, both on disk (relative to mlir/) and as C++ sees them.
SHARD_INCLUDE_DIR = "mlir/Dialect/Rock/Tuning/QuickTuningShards"
SHARD_DIR = f"include/{SHARD_INCLUDE_DIR}"

# kQuickTuningNoConfig: the config index a problem carries when no config of
# that kind was measured for it. Mirrors QuickTuningShardDb.h.
NO_CONFIG = 0xFFFF

# kQuickTuningNoProblem: the hash a lookup passes when it has no problem to
# name. Reserved, so that such a lookup cannot be narrowed by a problem that
# happens to hash to zero. Mirrors QuickTuningShardDb.h. rocmlir-gen prints all
# 16 digits, but the column's grammar admits fewer, so match any spelling of it.
NO_PROBLEM_PATTERN = re.compile(r'^0x0+$')

# How much of a shard's problem count may vanish in a rewrite before it looks
# less like new data and more like a run missing part of the .debug set.
PROBLEM_LOSS_RATIO = 0.9

# =============================================================================
# Helper Functions
# =============================================================================


def op_from_kernel(kernel):
    """Reverse-search the --op value for a kernel type via OP_TO_KERNEL_TYPE.

    The match is case-insensitive, so `kernel` may be either a PascalCase KernelType name
    (e.g. 'GemmElementwiseGemm') or its lowercase lookup-key segment (e.g. 'gemmelementwisegemm').
    """
    kernel = kernel.lower()
    for op, kernel_type in OP_TO_KERNEL_TYPE.items():
        if kernel_type.lower() == kernel:
            return op
    raise ValueError(f"Unknown kernel type: {kernel}")


def get_target_columns(op):
    """Get the columns used to identify unique problems for an operation."""
    if op == "gemm":
        return GEMM_COLUMNS
    elif op == "conv":
        return CONV_COLUMNS
    elif op == "attention":
        return ATTENTION_COLUMNS
    elif op == "gemm_gemm":
        return GEMM_GEMM_COLUMNS
    elif op == "conv_gemm":
        return CONV_GEMM_COLUMNS
    else:
        raise ValueError(f"Unknown operation: {op}")


def get_splitk_value(perfconfig):
    """Extract the Split-K value (as a string) from a perfconfig string.

    Handles both the named `prefix:key=value,...` form and the legacy
    positional `prefix:vN:a,b,c,...` form, which measurements taken before the
    named form landed still carry.
    """
    _, _, rest = perfconfig.partition(":")
    version, sep, body = rest.partition(":")
    match = POSITIONAL_VERSION_PATTERN.match(version) if sep else None
    if match:
        if int(match.group(1)) > POSITIONAL_MAX_VERSION:
            raise ValueError(f"Positional perfconfig too new to read positionally: {perfconfig}")
        return body.split(",")[POSITIONAL_SPLITK_INDEX].strip()

    _, params = parse_perfconfig(perfconfig)
    value = params.get(SPLITK_KEY)
    return None if value is None else str(value)


def is_splitk(perfconfig):
    """Whether `perfconfig` asks for a Split-K factor greater than one."""
    return get_splitk_value(perfconfig) not in (None, '1')


# =============================================================================
# Data Loading & Processing
# =============================================================================


def validate_files(files):
    """Validate that all files exist and are .debug files."""
    errors = []
    for f in files:
        if not f.endswith('.debug'):
            errors.append(f"{f} is not a .debug file")
        elif not os.path.isfile(f):
            errors.append(f"{f} not found")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def load_data(files, no_splitk):
    """Load tuning data from files or stdin."""
    if files:
        validate_files(files)

        print(f"Processing {len(files)} file(s):")
        for f in files:
            print(f"  {f}")

        dfs = [pd.read_csv(f, sep='\t', index_col=None, low_memory=False) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        # Read TSV content from stdin
        print("Reading from stdin...")
        df = pd.read_csv(sys.stdin, sep='\t', index_col=None, low_memory=False)

    if 'WithAttnBias' in df.columns and 'TransBias' not in df.columns:
        df['TransBias'] = False

    # Sliding look-back is optional (KV-cache only) and omitted from the key when
    # disabled, so legacy attention rows may lack the column or carry NaN after a
    # mixed-file concat; normalize it to the disabled sentinel -1. Only attention
    # grouping reads SlidingWindowLookBack, so defaulting it is a no-op elsewhere.
    if 'SlidingWindowLookBack' not in df.columns:
        df['SlidingWindowLookBack'] = -1
    else:
        df['SlidingWindowLookBack'] = df['SlidingWindowLookBack'].fillna(-1)

    # Drop rows that are repeated header lines (happens when using --retry=failed in tuningRunner.py).
    before = len(df)
    df = df[df['DataType'] != 'DataType']
    if len(df) < before:
        print(f"Dropped {before - len(df)} repeated header row(s)")

    # Embedded header rows force pandas to infer 'TFlops' as object dtype, which
    # later breaks numeric comparisons. Coerce to numeric and drop rows from
    # failed tuning runs that have no TFlops measurement.
    before = len(df)
    df['TFlops'] = pd.to_numeric(df['TFlops'], errors='coerce')
    df = df.dropna(subset=['TFlops'])
    if len(df) < before:
        print(f"Dropped {before - len(df)} row(s) with missing/invalid TFlops")

    if no_splitk and not df.empty:
        # Filter out configs where Split-K != 1
        before = len(df)
        mask = df['PerfConfig'].apply(lambda x: not is_splitk(x))
        df = df[mask]
        if len(df) < before:
            print(f"Filtered out {before - len(df)} out of {before} Split-K configs")

    return df


def build_coverage(df_typed, target_cols, op, threshold):
    """Map each problem to the perfconfigs performing within ``threshold`` of its best.

    Keys are ``(problem, split_k_allowed)``. A problem whose fusion forbids
    split-K can only run a ``splitKFactor == 1`` config, so covering it well
    needs a second entry whose candidates are restricted to those. Both entries
    come from the same measurements, so the extra accuracy costs no tuning time.

    The restricted entry is dropped when it would duplicate the unrestricted one,
    which happens whenever split-K did not win the problem in the first place.
    """
    coverage = {}
    for name, group in df_typed.groupby(target_cols):
        max_tflops = group['TFlops'].max()
        top = group[group['TFlops'] >= max_tflops * threshold]['PerfConfig'].tolist()
        coverage[name, True] = top

        if op not in SPLIT_K_AWARE_OPS:
            continue

        no_splitk = group[~group['PerfConfig'].apply(is_splitk)]
        if no_splitk.empty:
            print(f"WARNING: no splitKFactor=1 config measured for {name}; the quick list "
                  "cannot cover it when split-K is illegal")
            continue

        cutoff = no_splitk['TFlops'].max() * threshold
        top_no_splitk = no_splitk[no_splitk['TFlops'] >= cutoff]['PerfConfig'].tolist()
        if set(top_no_splitk) != set(top):
            coverage[name, False] = top_no_splitk

    return coverage


def find_perfconfigs(df, op, threshold):
    """Find minimal covering set of perfconfigs using set cover optimization.

    For each problem (unique combination of problem dimensions), we identify
    configs that achieve >= threshold * best_tflops. We then solve a set cover
    problem to find the minimum number of configs that cover all problems.

    The ILP formulation:
        minimize    sum(x[j] for all configs j)
        subject to  sum(coverage[i,j] * x[j]) >= 1  for each problem i
                    x[j] in {0, 1}

    where coverage[i,j] = 1 if config j is among the top performers for problem i.
    """
    target_cols = get_target_columns(op)
    results = {}

    for dtype in sorted(df['DataType'].unique()):
        df_typed = df[df['DataType'] == dtype]

        # Aggregate by keeping only the best TFlops per (problem, config)
        df_typed = df_typed.groupby(target_cols + ['PerfConfig'], as_index=False)['TFlops'].max()

        coverage = build_coverage(df_typed, target_cols, op, threshold)

        problems = sorted(coverage.keys())
        configs = sorted({c for cs in coverage.values() for c in cs})
        config_idx = {c: i for i, c in enumerate(configs)}

        # Build coverage matrix: matrix[i,j] = 1 if config j covers problem i
        n_problems, n_configs = len(problems), len(configs)
        matrix = np.zeros((n_problems, n_configs), dtype=int)
        for i, prob in enumerate(problems):
            for cfg in coverage[prob]:
                matrix[i, config_idx[cfg]] = 1

        # Solve set cover with ILP
        prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", range(n_configs), cat='Binary')

        # Objective: minimize number of selected configs
        prob += pulp.lpSum(x[j] for j in range(n_configs))

        # Constraints: each problem must be covered by at least one config
        for i in range(n_problems):
            prob += pulp.lpSum(matrix[i, j] * x[j] for j in range(n_configs)) >= 1

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if status != pulp.LpStatusOptimal:
            status_name = pulp.LpStatus.get(status, "Unknown")
            raise RuntimeError(f"Set cover failed for {dtype}: {status_name}. "
                               f"This likely indicates corrupted input data or a bug.")

        # Extract selected configs, sorted by how many problems they cover
        selected = [configs[j] for j in range(n_configs) if x[j].varValue == 1]
        counts = {c: sum(matrix[i, config_idx[c]] for i in range(n_problems)) for c in selected}
        results[dtype] = sorted(selected, key=lambda c: counts[c], reverse=True)

    return results


def pick_best(group):
    """The winning PerfConfig of `group`, breaking TFlops ties lexicographically.

    The tie-break is what makes a rerun on the same data reproduce the same
    shard: measurements repeat to the last digit often enough that leaving the
    winner to row order would churn the output.
    """
    if group.empty:
        return None
    ordered = group.sort_values(['TFlops', 'PerfConfig'], ascending=[False, True])
    return ordered['PerfConfig'].iloc[0]


def find_problem_bests(df):
    """Best non-split-K and split-K config per measured problem, per data type.

    Returns ``{dtype: {problem_hash: (best_non_splitk, best_splitk)}}`` with
    either config possibly ``None``. The hash is C++'s to define: it arrives in
    the `ProblemHash` column and is only ever grouped by, never constructed.
    Data predating the column yields no bests at all, which leaves every key it
    covers with the pre-sharding set-cover-only behaviour.
    """
    if PROBLEM_HASH_COLUMN not in df.columns:
        print(f"No {PROBLEM_HASH_COLUMN} column: emitting set covers only")
        return {}

    # A row whose hash is missing, malformed or reserved contributes to the set
    # cover only, which is the pre-per-problem behaviour. attachProblemHashes.py
    # leaves the column empty for a problem it could not identify, so a partly
    # hashed file is expected input rather than an error.
    hashes = df[PROBLEM_HASH_COLUMN].astype(str).str.strip()
    valid = hashes.str.match(PROBLEM_HASH_PATTERN) & ~hashes.str.match(NO_PROBLEM_PATTERN)
    if not valid.all():
        unrecognized = sorted(set(hashes[~valid].astype(str)))
        print(f"WARNING: ignoring {(~valid).sum()} row(s) whose {PROBLEM_HASH_COLUMN} is not a "
              f"recordable hash, e.g. {unrecognized[:3]}")

    df = df.assign(**{PROBLEM_HASH_COLUMN: hashes})[valid]

    bests = {}
    for dtype in sorted(df['DataType'].unique()):
        df_typed = df[df['DataType'] == dtype]
        per_problem = {}
        for problem_hash, group in df_typed.groupby(PROBLEM_HASH_COLUMN):
            splits = group['PerfConfig'].apply(is_splitk)
            best = (pick_best(group[~splits]), pick_best(group[splits]))
            per_problem[int(problem_hash, 16)] = best
        bests[dtype] = per_problem

    return bests


# =============================================================================
# Shard Generation
# =============================================================================

COLUMN_LIMIT = 80

BANNER_RULE = "//===" + "-" * (COLUMN_LIMIT - 10) + "===//"

# Longest banner title that still leaves `//===- `, a separating dash and
# `===//` inside the column limit.
BANNER_TITLE_LIMIT = COLUMN_LIMIT - 14

LICENSE_HEADER = "\n".join([
    "//",
    "// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM",
    "// Exceptions. See https://llvm.org/LICENSE.txt for license information.",
    "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
    "//",
    BANNER_RULE,
])

INDEX_HEADER = """//===- QuickTuningShards.inc - index of quick-tuning shards ---------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The checked-in list of quick-tuning shards. A shard only reaches the binary
// if it is listed here, so adding one is an explicit, reviewable edit. This is
// deliberately not a CMake glob: a glob leaves deleted shards behind in an
// incremental build and bakes absolute paths into the build tree.
//
// Rewritten wholesale, from the shards on disk, by
// mlir/utils/performance/analysis/quickTuningGen.py.
//
// Included once per phase by QuickTuningShardDb.cpp, so it must contain
// nothing but `#include` directives, and must not carry an include guard.
// Keep the list sorted by file name.
//
//===----------------------------------------------------------------------===//
"""


def get_shard_dir():
    """Directory holding the per-key shards, relative to this script."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent.parent.parent / SHARD_DIR


def get_index_path():
    """The checked-in index that lists the shards."""
    return get_shard_dir().parent / "QuickTuningShards.inc"


def get_generator_path():
    """Get this script's path relative to the repo root for the header comment."""
    script_path = Path(__file__).resolve()
    # Find repo root (contains .git or mlir directory)
    for parent in script_path.parents:
        if (parent / ".git").exists() or (parent / "mlir").is_dir():
            try:
                return script_path.relative_to(parent)
            except ValueError:
                pass
    return script_path.name


def shard_suffix(arch, kernel_type, dtype):
    """The identifier suffix a shard's arrays share, e.g. `Gfx908GemmI8`.

    It is the lookup key with the separators dropped and each component
    capitalised, so key and suffix stay one edit apart.
    """
    return f"{arch.capitalize()}{kernel_type}{dtype.capitalize()}"


def lookup_key(arch, kernel_type, dtype):
    """The lookup key a shard answers to, as ParamLookupTable::makeKey spells it."""
    return f"{arch}_{kernel_type.lower()}_{dtype}"


def banner(text):
    """An LLVM-style file banner padded out to the column limit."""
    prefix = f"//===- {text} "
    return prefix + "-" * max(1, COLUMN_LIMIT - 5 - len(prefix)) + "===//"


def format_array(decl, entries):
    """Emit `decl = {...};`, on one line when it fits and indented otherwise."""
    one_line = f"{decl} = {{{', '.join(entries)}}};"
    if len(one_line) <= COLUMN_LIMIT:
        return one_line

    lines, current = [], ""
    for i, entry in enumerate(entries):
        piece = entry + ("," if i < len(entries) - 1 else "")
        if current and len(current) + 1 + len(piece) > COLUMN_LIMIT - 4:
            lines.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    lines.append(current)
    body = "\n".join(f"    {line}" for line in lines)
    return f"{decl} = {{\n{body}\n}};"


def fmt_index(index):
    """Spell a config index, using the sentinel's C++ spelling for `None`."""
    return f"0x{NO_CONFIG:04X}" if index is None else str(index)


def render_shard(key, suffix, configs, cover, problems):
    """Render a whole shard file.

    `configs` is the config pool, `cover` the set cover as indices into it in
    descending coverage order, and `problems` a list of
    ``(problem_hash, best_non_splitk_index, best_splitk_index)`` triples sorted
    by hash, either index possibly ``NO_CONFIG``.
    """
    if len(configs) > NO_CONFIG:
        raise ValueError(f"{key}: {len(configs)} configs overflows the uint16 shard indices")

    # Name the key in the banner, unless it is one of the long ones that leaves
    # no room for it within 80 columns; the entry below spells it out anyway.
    title = f"{suffix}.inc - quick-tuning shard"
    if len(title) + len(key) + len(" for ") <= BANNER_TITLE_LIMIT:
        title += f" for {key}"

    config_lines = ",\n".join(f'    "{cfg}"' for cfg in configs)
    lines = [
        banner(title),
        LICENSE_HEADER,
        "//",
        f"// Generated by {get_generator_path()}; do not edit.",
        "//",
        "// See QuickTuningShardDb.h for the entry layout and the include protocol.",
        "//",
        BANNER_RULE,
        "",
        "// clang-format off",
        "",
        "#ifdef QUICK_TUNING_DB_ARRAYS",
        "",
        "// The config pool. An array of separate string literals rather than one",
        "// concatenated blob: MSVC caps the length of a single string literal (C2026)",
        "// and this repo builds on Windows.",
        f"static const char *const kCfg{suffix}[] = {{",
        config_lines,
        "};",
        "",
        "// The set cover, in descending coverage order.",
        format_array(f"static const uint16_t kCover{suffix}[]", [str(i) for i in cover]),
    ]

    if problems:
        hashes = [f"0x{problem_hash:016x}ULL" for problem_hash, _, _ in problems]
        slots = [f"{fmt_index(non)}, {fmt_index(split)}" for _, non, split in problems]
        lines += [
            "",
            "// Problem hashes, ascending (see QuickTuningProblemKey.h).",
            f"static const uint64_t kProb{suffix}[] = {{",
            ",\n".join(f"    {h}" for h in hashes),
            "};",
            "",
            "// Problem i: [2*i] best non-split-K, [2*i+1] best split-K,",
            "// kQuickTuningNoConfig where none was measured.",
            f"static const uint16_t kProbCfg{suffix}[] = {{",
            ",\n".join(f"    {slot}" for slot in slots),
            "};",
        ]
        problem_fields = f" kProb{suffix}, kProbCfg{suffix},"
    else:
        problem_fields = " nullptr, nullptr,"

    lines += [
        "",
        "#endif // QUICK_TUNING_DB_ARRAYS",
        "",
        "#ifdef QUICK_TUNING_DB_ENTRIES",
        f'{{"{key}",',
        f" kCfg{suffix}, /*numConfigs=*/{len(configs)},",
        f" kCover{suffix}, /*numCover=*/{len(cover)},",
        f"{problem_fields} /*numProblems=*/{len(problems)}}},",
        "#endif // QUICK_TUNING_DB_ENTRIES",
        "",
    ]
    return "\n".join(lines)


def parse_shard(path):
    """Read back a shard as ``(key, configs, cover, num_problems)``.

    Only the generator reads shards: to carry a set cover into an alias, and to
    compare a shard against the one it is about to replace.
    """
    text = path.read_text()
    key = re.search(r'^\{"([^"]+)",', text, re.MULTILINE)
    pool = re.search(r'kCfg\w+\[\] = \{(.*?)\n\};', text, re.DOTALL)
    cover = re.search(r'kCover\w+\[\] = \{(.*?)\};', text, re.DOTALL)
    num_problems = re.search(r'/\*numProblems=\*/(\d+)\}', text)
    if not (key and pool and cover and num_problems):
        raise ValueError(f"{path} is not a shard this generator can read")

    configs = re.findall(r'"([^"]*)"', pool.group(1))
    indices = [int(i) for i in cover.group(1).replace("\n", " ").split(",") if i.strip()]
    return key.group(1), configs, indices, int(num_problems.group(1))


def warn_on_lost_problems(path, key, num_problems):
    """Warn when a shard is about to be rewritten with materially fewer problems.

    A run made with only part of the `.debug` set still produces a perfectly
    well-formed shard; the only symptom is that problems the last run knew
    about have quietly gone missing.
    """
    if not path.exists():
        return
    _, _, _, previous = parse_shard(path)
    if num_problems < previous * PROBLEM_LOSS_RATIO:
        print(f"WARNING: {key} drops from {previous} to {num_problems} measured problem(s); "
              "rerun with the full .debug set if that was not intended")


def write_shard(key, suffix, configs, cover, problems):
    """Write one shard, warning first if it loses problems the old one had."""
    path = get_shard_dir() / f"{suffix}.inc"
    warn_on_lost_problems(path, key, len(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_shard(key, suffix, configs, cover, problems))
    return path


def write_index():
    """Rewrite the shard index from the shards on disk.

    A shard only reaches the binary by being listed here, so every write of a
    shard has to be followed by a rewrite of the index.
    """
    path = get_index_path()
    names = sorted(p.name for p in get_shard_dir().glob("*.inc"))
    includes = "\n".join(f'#include "{SHARD_INCLUDE_DIR}/{name}"' for name in names)
    path.write_text(f"{INDEX_HEADER}\n{includes}\n")
    print(f"Indexed {len(names)} shard(s) in {path}")


def build_pool(cover_configs, bests):
    """Lay out a shard's config pool and index its cover and per-problem bests.

    The cover leads the pool, so its indices are the identity and a key with no
    measured problems is exactly its old list. The bests the cover does not
    already hold follow, sorted, so the pool depends on the data alone.

    Returns ``(pool, cover_indices, problems)``.
    """
    pool = list(cover_configs)
    indices = {cfg: i for i, cfg in enumerate(pool)}
    for cfg in sorted({c for pair in bests.values() for c in pair if c and c not in indices}):
        indices[cfg] = len(pool)
        pool.append(cfg)

    problems = [(problem_hash, indices.get(non_splitk), indices.get(splitk))
                for problem_hash, (non_splitk, splitk) in sorted(bests.items())]
    return pool, list(range(len(cover_configs))), problems


def write_shards(results, bests, arch, op):
    """Write a shard per data type covered by this run, then reindex."""
    kernel_type = OP_TO_KERNEL_TYPE[op]

    for dtype, cover_configs in results.items():
        key = lookup_key(arch, kernel_type, dtype)
        pool, cover, problems = build_pool(cover_configs, bests.get(dtype, {}))
        path = write_shard(key, shard_suffix(arch, kernel_type, dtype), pool, cover, problems)
        print(f"Wrote {path}: {len(cover)} cover config(s), {len(problems)} problem(s)")

    write_index()


def add_type_aliases(from_type, to_type):
    """Give `from_type` keys a shard carrying the `to_type` set cover.

    The cover transfers because it is what an untuned key falls back to
    anyway; the per-problem bests do not, since they are measurements of a
    different precision, so an aliased shard has no problems and behaves
    exactly as the pre-sharding table did.
    """
    shard_dir = get_shard_dir()
    if not shard_dir.is_dir():
        print(f"ERROR: {shard_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    aliases_added = 0
    for path in sorted(shard_dir.glob("*.inc")):
        key, configs, cover, _ = parse_shard(path)
        arch, kernel, dtype = key.split("_")
        if dtype != to_type:
            continue

        from_key = lookup_key(arch, kernel, from_type)
        kernel_type = OP_TO_KERNEL_TYPE[op_from_kernel(kernel)]
        suffix = shard_suffix(arch, kernel_type, from_type)

        # Don't overwrite existing shards - aliases are fallbacks only
        if (shard_dir / f"{suffix}.inc").exists():
            print(f"Skipping {from_key}: already exists")
            continue

        write_shard(from_key, suffix, [configs[i] for i in cover], list(range(len(cover))), [])
        print(f"Added: {from_key} -> {to_type}")
        aliases_added += 1

    if aliases_added > 0:
        write_index()
        print(f"Added {aliases_added} alias(es)")
    else:
        print("No aliases added")

    return True


# =============================================================================
# One-shot Conversion
# =============================================================================

MONOLITH_ENTRY_PATTERN = re.compile(r'\{"(gfx\w+)_(\w+)_(\w+)",\s*\{(\w+)::(\w+),')
MONOLITH_DEFINITION_PATTERN = re.compile(r'const StringRef \w+::(\w+)\[\] = \{(.*?)\n\};',
                                         re.DOTALL)


def convert_monolith(path):
    """Split the pre-sharding QuickTuningPerfconfigs.inc into shards.

    The measurements behind that table are long gone, so this reads the shipped
    lists themselves: every key becomes a shard whose pool is its old list and
    whose problem arrays are empty, which is bit-for-bit the behaviour it had.
    A key that already has a shard is left alone and only checked, so the
    conversion cannot undo work a real run has already done.

    Returns False if any key failed to carry over.
    """
    content = path.read_text()
    lists = {
        name: re.findall(r'"([^"]*)"', body)
        for name, body in MONOLITH_DEFINITION_PATTERN.findall(content)
    }

    converted, checked, failed = 0, 0, []
    for arch, kernel, dtype, _, param_name in MONOLITH_ENTRY_PATTERN.findall(content):
        key = f"{arch}_{kernel}_{dtype}"
        configs = lists.get(param_name)
        if not configs:
            failed.append(f"{key}: no definition of {param_name}")
            continue

        kernel_type = OP_TO_KERNEL_TYPE[op_from_kernel(kernel)]
        suffix = shard_suffix(arch, kernel_type, dtype)
        shard_path = get_shard_dir() / f"{suffix}.inc"

        if not shard_path.exists():
            write_shard(key, suffix, configs, list(range(len(configs))), [])
            converted += 1
        else:
            checked += 1

        # Read the shard back rather than trust what was just written: this is
        # the one run that has the old table to check the new ones against.
        shard_key, pool, cover, _ = parse_shard(shard_path)
        if shard_key != key:
            failed.append(f"{key}: {shard_path.name} answers to {shard_key}")
        elif [pool[i] for i in cover] != configs:
            failed.append(f"{key}: set cover in {shard_path.name} differs from {param_name}")

    write_index()
    print(f"Converted {converted} key(s), checked {checked} pre-existing shard(s)")
    for message in failed:
        print(f"ERROR: {message}", file=sys.stderr)
    return not failed


# =============================================================================
# Main
# =============================================================================


def print_results(results, arch):
    """Print selected perfconfigs for an architecture."""
    print(f"\n=== {arch} ===")
    for dtype, configs in results.items():
        print(f"\n{dtype}: {len(configs)} configs")
        for i, cfg in enumerate(configs, 1):
            print(f"{i:4d}: {cfg}")
    print()


def process_arch(df, arch, op, threshold, update):
    """Process data for a single architecture."""
    df_arch = df[df['Chip'] == arch]

    results = find_perfconfigs(df_arch, op, threshold)
    bests = find_problem_bests(df_arch)
    print_results(results, arch)

    if update:
        write_shards(results, bests, arch, op)


def main(args=None):
    parser = argparse.ArgumentParser(
        prog='quickTuningGen.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generate the quick-tuning shards from tuning data.',
        epilog='''
Examples:
    # Generate quick-tune lists from tuning data
    %(prog)s tuningData/*.debug --op conv --update
    %(prog)s gfx90a/*.debug gfx942/*.debug --op gemm --update
    cat data.debug | %(prog)s --op attention --update
    find . -name "*.debug" | xargs %(prog)s --op gemm --update

    # Add fallback type aliases (use f16 configs when there's no bf16 data)
    %(prog)s --alias bf16 f16

    # One-shot: split the pre-sharding table into shards. It was deleted once
    # the shards reproduced it, so take it from the history:
    #   git show <rev>:mlir/include/mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc > /tmp/old.inc
    %(prog)s --convert-monolith /tmp/old.inc
''')

    parser.add_argument(
        'files',
        nargs='*',
        metavar='FILE',
        help='.debug files produced by tuningRunner.py (reads TSV from stdin if none provided)')
    parser.add_argument('--op',
                        choices=['gemm', 'conv', 'attention', 'gemm_gemm', 'conv_gemm'],
                        help='Operation')
    parser.add_argument('--th',
                        type=float,
                        default=0.93,
                        metavar='THRESHOLD',
                        help='Coverage threshold (default: 0.93)')
    parser.add_argument('--update', action='store_true', help='Write the shards')
    parser.add_argument('--no-splitk', action='store_true', help='Exclude Split-K configurations')
    parser.add_argument('--alias',
                        nargs=2,
                        metavar=('FROM', 'TO'),
                        help='Add fallback: use TO configs for FROM type (e.g., --alias bf16 f16)')
    parser.add_argument('--convert-monolith',
                        metavar='FILE',
                        help='One-shot: split a pre-sharding QuickTuningPerfconfigs.inc into '
                        'shards, checking each set cover survives unchanged')

    pargs = parser.parse_args(args)

    if not pargs.op and not pargs.alias and not pargs.convert_monolith:
        parser.error('either --op, --alias or --convert-monolith must be specified')
        return 1

    # Convert the pre-sharding table
    if pargs.convert_monolith:
        if not convert_monolith(Path(pargs.convert_monolith)):
            return 1

    # Generate quick-tune lists
    if pargs.op:
        df = load_data(pargs.files, pargs.no_splitk)
        if not df.empty:
            archs = sorted(df['Chip'].unique())
            print(f"Processing {len(archs)} architecture(s): {', '.join(archs)}")
            for arch in archs:
                process_arch(df, arch, pargs.op, pargs.th, pargs.update)
        else:
            print("No data to process.")

    # Add type aliases
    if pargs.alias:
        from_type, to_type = pargs.alias
        print(f"Adding {from_type} -> {to_type} aliases...")
        add_type_aliases(from_type, to_type)

    return 0


if __name__ == '__main__':
    sys.exit(main())
