#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Quick Tuning Generator

Generates QuickTuningPerfconfigs.inc from tuning data produced by tuningRunner.py.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import subprocess
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

# Operations that share the attention (GemmGemm) tuning code path
GEMM_GEMM_OPS = {'attention', 'gemm_gemm', 'conv_gemm'}

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

PER_PROBLEM_TOP_N = 5

# Regex pattern for lookup table entries: {"arch_kernel_dtype", {Class::params, Class::count}}, // optional comment
LOOKUP_ENTRY_PATTERN = re.compile(r'\{("(gfx\w+)_(\w+)_(\w+)"),\s*(\{[^}]+\})\},(\s*//[^\n]*)?')

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


def get_instruction_type(arch, dtype, op):
    """Determine instruction type based on architecture, data type, and operation."""
    if op in GEMM_GEMM_OPS:
        return "GemmGemm"
    return "Gemm"


def get_class_name(arch, dtype, op):
    """Get the PopulateParams class name."""
    return f"PopulateParams{get_instruction_type(arch, dtype, op)}"


def get_param_names(arch, dtype, op):
    """Generate array and count variable names."""
    kernel_type = OP_TO_KERNEL_TYPE[op]
    base = f"initParameters{dtype.capitalize()}{kernel_type}{arch.capitalize()}"
    return base, f"n{base[0].upper()}{base[1:]}"


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
    """Extract the Split-K value (as a string) from a perfconfig string."""
    _, params = parse_perfconfig(perfconfig)
    value = params.get(SPLITK_KEY)
    return None if value is None else str(value)


def get_problem_priority(group):
    """Return the tier-1 priority for a problem group, if one is available."""
    if 'PerfPriority' not in group:
        return None
    priorities = pd.to_numeric(group['PerfPriority'], errors='coerce').dropna().unique()
    if len(priorities) == 0:
        return None
    # All measurements of one problem should have the same priority. If mixed
    # input is supplied, retain the stricter threshold rather than relaxing it.
    return int(max(priorities))


def threshold_for_priority(priority, threshold):
    """Relax coverage for low-priority tier-1 problems.

    Priorities 1, 2, and 3 lower the requested coverage threshold by fixed
    offsets of 3, 2, and 1 percentage points respectively. Priority 4 and above
    retain the requested threshold. With the default 93%, this permits gaps of
    10%, 9%, 8%, and 7% respectively.
    """
    if priority is None:
        return threshold
    relaxation = min(max(4 - priority, 0) * 0.01, 0.03)
    return max(0.0, threshold - relaxation)


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


def filter_split_k(df):
    """Drop configs with Split-K != 1."""
    before = len(df)
    df = df[df['PerfConfig'].apply(lambda x: get_splitk_value(x) in (None, '1'))]
    if len(df) < before:
        print(f"Filtered out {before - len(df)} out of {before} Split-K configs")
    return df


def load_data(files):
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

    return df


def build_coverage(df_typed, target_cols, op, threshold, use_perf_priority=False):
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
        problem_threshold = threshold
        if use_perf_priority:
            problem_threshold = threshold_for_priority(get_problem_priority(group), threshold)
        max_tflops = group['TFlops'].max()
        top = group[group['TFlops'] >= max_tflops * problem_threshold]['PerfConfig'].tolist()
        coverage[name, True] = top

        if op not in SPLIT_K_AWARE_OPS:
            continue

        is_split_k_free = group['PerfConfig'].apply(lambda config: get_splitk_value(config) in
                                                    (None, '1'))
        no_splitk = group[is_split_k_free]
        if no_splitk.empty:
            print(f"WARNING: no splitKFactor=1 config measured for {name}; the quick list "
                  "cannot cover it when split-K is illegal")
            continue

        cutoff = no_splitk['TFlops'].max() * problem_threshold
        top_no_splitk = no_splitk[no_splitk['TFlops'] >= cutoff]['PerfConfig'].tolist()
        if set(top_no_splitk) != set(top):
            coverage[name, False] = top_no_splitk

    return coverage


def solve_full_coverage(coverage, dtype):
    """Return the minimum config set and its coverage matrix."""
    problems = sorted(coverage.keys())
    configs = sorted({c for candidates in coverage.values() for c in candidates})
    config_idx = {config: i for i, config in enumerate(configs)}

    n_problems, n_configs = len(problems), len(configs)
    matrix = np.zeros((n_problems, n_configs), dtype=int)
    for i, problem in enumerate(problems):
        for config in coverage[problem]:
            matrix[i, config_idx[config]] = 1

    problem = pulp.LpProblem("SetCover", pulp.LpMinimize)
    selected = pulp.LpVariable.dicts("selected", range(n_configs), cat='Binary')
    problem += pulp.lpSum(selected[j] for j in range(n_configs))
    for i in range(n_problems):
        problem += pulp.lpSum(matrix[i, j] * selected[j] for j in range(n_configs)) >= 1

    status = problem.solve(pulp.PULP_CBC_CMD(msg=0))
    if status != pulp.LpStatusOptimal:
        status_name = pulp.LpStatus.get(status, "Unknown")
        raise RuntimeError(f"Set cover failed for {dtype}: {status_name}. "
                           f"This likely indicates corrupted input data or a bug.")

    chosen = [configs[j] for j in range(n_configs) if selected[j].varValue == 1]
    return chosen, problems, configs, config_idx, matrix


def solve_bounded_coverage(problems, problem_weights, configs, config_idx, matrix, max_configs):
    """Select at most ``max_configs`` configs using a precomputed coverage matrix."""
    n_problems, n_configs = len(problems), len(config_idx)

    problem = pulp.LpProblem("BoundedCoverage", pulp.LpMaximize)
    selected = pulp.LpVariable.dicts("selected", range(n_configs), cat='Binary')
    covered = pulp.LpVariable.dicts("covered", range(n_problems), cat='Binary')

    # The small tie-breaker prefers a shorter list without changing the primary
    # objective of maximizing weighted problem coverage.
    problem += (pulp.lpSum(problem_weights.get(p, 1) * covered[i] for i, p in enumerate(problems)) -
                1e-6 * pulp.lpSum(selected[j] for j in range(n_configs)))
    problem += pulp.lpSum(selected[j] for j in range(n_configs)) <= max_configs
    for i in range(n_problems):
        problem += covered[i] <= pulp.lpSum(matrix[i, j] * selected[j] for j in range(n_configs))

    status = problem.solve(pulp.PULP_CBC_CMD(msg=0))
    if status != pulp.LpStatusOptimal:
        status_name = pulp.LpStatus.get(status, "Unknown")
        raise RuntimeError(f"Bounded quick-tuning coverage failed: {status_name}.")

    return [configs[j] for j in range(n_configs) if selected[j].varValue == 1]


def find_perfconfigs(df, op, threshold, max_configs=40):
    """Find minimal covering set of perfconfigs using set cover optimization.

    For each problem (unique combination of problem dimensions), we identify
    configs that achieve >= threshold * best_tflops. We then solve a set cover
    problem to find the minimum number of configs that cover all problems. Perf
    priority is consulted only if that strict set exceeds ``max_configs``: low
    priorities first receive relaxed gaps, then a bounded solve maximizes
    priority-weighted coverage if full coverage still does not fit.

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
        grouping = target_cols + ['PerfConfig']
        if 'PerfPriority' in df_typed:
            df_typed = df_typed.groupby(grouping,
                                        as_index=False).agg(TFlops=('TFlops', 'max'),
                                                            PerfPriority=('PerfPriority', 'max'))
        else:
            df_typed = df_typed.groupby(grouping, as_index=False)['TFlops'].max()

        # First try the requested threshold uniformly. Perf priority must not
        # affect lists that already fit under the cap at full coverage.
        coverage = build_coverage(df_typed, target_cols, op, threshold)
        selected, problems, configs, config_idx, matrix = solve_full_coverage(coverage, dtype)

        has_priorities = ('PerfPriority' in df_typed and df_typed['PerfPriority'].notna().any())
        if max_configs is not None and len(selected) > max_configs and has_priorities:
            strict_count = len(selected)
            coverage = build_coverage(df_typed, target_cols, op, threshold, use_perf_priority=True)
            selected, problems, configs, config_idx, matrix = solve_full_coverage(coverage, dtype)
            print(f"{dtype}: strict {1 - threshold:.0%} gap needs {strict_count} configs; "
                  f"priority-aware gaps reduce it to {len(selected)}")

        # Extract selected configs, sorted by how many problems they cover.
        if max_configs is not None and len(selected) > max_configs:
            weights_by_name = {}
            for name, group in df_typed.groupby(target_cols):
                priority = get_problem_priority(group)
                weights_by_name[name] = max(priority, 1) if priority is not None else 1
            problem_weights = {problem: weights_by_name[problem[0]] for problem in coverage}

            print(f"WARNING: {dtype} needs {len(selected)} configs for full coverage; "
                  f"limiting quick tuning to {max_configs}")
            selected = solve_bounded_coverage(problems, problem_weights, configs, config_idx,
                                              matrix, max_configs)
            covered = [
                problem for problem, candidates in coverage.items()
                if any(config in candidates for config in selected)
            ]
            covered_weight = sum(problem_weights[problem] for problem in covered)
            total_weight = sum(problem_weights[problem] for problem in coverage)
            print(f"Capped list covers {len(covered)}/{len(coverage)} problem constraints "
                  f"and {covered_weight}/{total_weight} priority weight "
                  f"({covered_weight / total_weight:.1%})")

        counts = {config: int(matrix[:, config_idx[config]].sum()) for config in selected}
        results[dtype] = sorted(selected, key=lambda c: counts[c], reverse=True)

    return results


# =============================================================================
# File Generation
# =============================================================================


def get_output_path():
    """Get the output .inc file path relative to this script."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent.parent.parent / "include/mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"


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


def init_inc_file(path):
    """Create empty .inc file with required structure."""
    sections = ["Gemm", "GemmGemm"]
    lookup_table_sections = ["Gemm", "GemmGemm"]
    lines = [f"// Generated by: {get_generator_path()}", "", "// clang-format off", ""]
    for s in sections:
        lines += [f"#ifdef {s}_DEFINITIONS_GEN", "", f"#endif  // {s}_DEFINITIONS_GEN", ""]
        lines += [f"#ifdef {s}_DECLARATIONS_GEN", "", f"#endif  // {s}_DECLARATIONS_GEN", ""]
    for s in lookup_table_sections:
        lines += [f"#ifdef {s}_LOOKUP_TABLE_GEN", "", f"#endif  // {s}_LOOKUP_TABLE_GEN", ""]
    path.write_text("\n".join(lines))


def find_endif(content, section_name):
    """Find position of `#endif // SECTION_NAME` line, tolerant of whitespace.

    Matches `#endif`, any horizontal whitespace, `//`, any horizontal whitespace,
    then the section name. This survives clang-format normalizing two spaces to one.
    """
    pattern = re.compile(rf'^[ \t]*#endif[ \t]+//[ \t]*{re.escape(section_name)}[ \t]*$',
                         re.MULTILINE)
    match = pattern.search(content)
    return match.start() if match else -1


def ensure_section(content, section_name):
    """Ensure `#ifdef SECTION_NAME ... #endif // SECTION_NAME` exists; append if missing."""
    if find_endif(content, section_name) != -1:
        return content
    if not content.endswith("\n"):
        content += "\n"
    content += f"\n#ifdef {section_name}\n\n#endif  // {section_name}\n"
    print(f"Created missing section: {section_name}")
    return content


def replace_section(content, section_name, begin_marker, end_marker, new_content):
    """Replace content between begin/end markers inside the named #ifdef section.

    Creates the begin/end block if it doesn't exist, and creates the enclosing
    #ifdef/#endif section too if it's also missing.
    """
    pattern = re.compile(f'{re.escape(begin_marker)}.*?{re.escape(end_marker)}', re.DOTALL)

    if pattern.search(content):
        return pattern.sub(f'{begin_marker}\n{new_content}\n{end_marker}', content)

    content = ensure_section(content, section_name)
    insert_pos = find_endif(content, section_name)

    section = f'{begin_marker}\n{new_content}\n{end_marker}\n\n'
    return content[:insert_pos] + section + content[insert_pos:]


def add_lookup_entry(content, section_name, entry):
    """Add or replace a lookup table entry inside the named #ifdef section."""
    match = LOOKUP_ENTRY_PATTERN.match(entry)
    if not match:
        raise ValueError(f"Invalid lookup entry: {entry}")

    key = match.group(1)  # e.g., "gfx942_gemm_f16"

    # Check for existing entry
    remove_pattern = re.compile(r'\{' + re.escape(key) + r',\s*\{[^}]+\}\},?[^\n]*\n*')
    existing = remove_pattern.search(content)

    if existing:
        insert_pos = existing.start()
        content = content[:existing.start()] + content[existing.end():]
    else:
        content = ensure_section(content, section_name)
        insert_pos = find_endif(content, section_name)

    return content[:insert_pos] + f'{entry}\n\n' + content[insert_pos:]


def get_lookup_section(arch, op, dtype):
    """Get the appropriate lookup table section name."""
    if op in GEMM_GEMM_OPS:
        return "GemmGemm_LOOKUP_TABLE_GEN"
    return "Gemm_LOOKUP_TABLE_GEN"


def update_inc_file(results, arch, op):
    """Update the .inc file with results."""
    path = get_output_path()
    if not path.exists():
        init_inc_file(path)

    content = path.read_text()

    # Identifiers and section markers use the PascalCase KernelType; the lookup key uses its
    # lowercase form
    kernel_type = OP_TO_KERNEL_TYPE[op]

    for dtype, configs in results.items():
        instr = get_instruction_type(arch, dtype, op)
        class_name = get_class_name(arch, dtype, op)
        param_name, count_name = get_param_names(arch, dtype, op)

        # Generate definition. Perf configs are `prefix:key=value,...` strings
        # containing only identifier, digit, `-`, `=`, `,` and `:` characters,
        # so they embed directly into a C++ string literal without escaping.
        def_lines = [f"const StringRef {class_name}::{param_name}[] = {{"]
        for i, cfg in enumerate(configs):
            comma = "," if i < len(configs) - 1 else ""
            def_lines.append(f'    "{cfg}"{comma}')
        def_lines.append("};")

        content = replace_section(content, f"{instr}_DEFINITIONS_GEN",
                                  f"// BEGIN_{kernel_type.upper()}_{instr}_{dtype}_{arch}_DEFS",
                                  f"// END_{kernel_type.upper()}_{instr}_{dtype}_{arch}_DEFS",
                                  "\n".join(def_lines))

        # Generate declaration
        dec_lines = [
            f"static constexpr size_t {count_name} = {len(configs)};",
            f"static const StringRef {param_name}[{count_name}];"
        ]

        content = replace_section(content, f"{instr}_DECLARATIONS_GEN",
                                  f"// BEGIN_{kernel_type.upper()}_{instr}_{dtype}_{arch}_DECS",
                                  f"// END_{kernel_type.upper()}_{instr}_{dtype}_{arch}_DECS",
                                  "\n".join(dec_lines))

        # Add lookup entry
        section_name = get_lookup_section(arch, op, dtype)
        key = f"{arch}_{kernel_type.lower()}_{dtype}"
        value = f"{{{class_name}::{param_name}, {class_name}::{count_name}}}"
        entry = f'{{"{key}", {value}}},'
        content = add_lookup_entry(content, section_name, entry)

    path.write_text(content)


def add_type_aliases(from_type, to_type):
    """Add lookup entries for from_type that reference to_type's configs."""
    path = get_output_path()
    if not path.exists():
        print(f"ERROR: {path} does not exist", file=sys.stderr)
        sys.exit(1)

    content = path.read_text()

    aliases_added = 0
    for match in LOOKUP_ENTRY_PATTERN.finditer(content):
        arch = match.group(2)  # e.g., "gfx942"
        kernel = match.group(3)  # e.g., "gemm"
        dtype = match.group(4)  # e.g., "f16"
        value = match.group(5)  # e.g., "{PopulateParamsGemm::..., ...}"

        if dtype != to_type:
            continue

        from_key = f"{arch}_{kernel}_{from_type}"

        # Don't overwrite existing entries - aliases are fallbacks only
        if f'"{from_key}"' in content:
            print(f"Skipping {from_key}: already exists")
            continue

        op = op_from_kernel(kernel)  # e.g., "gemmelementwisegemm" -> "gemm_gemm"

        section_name = get_lookup_section(arch, op, from_type)
        entry = f'{{"{from_key}", {value}}},  // alias -> {to_type}'

        content = add_lookup_entry(content, section_name, entry)
        print(f"Added: {from_key} -> {to_type}")
        aliases_added += 1

    if aliases_added > 0:
        path.write_text(content)
        print(f"Added {aliases_added} alias(es)")
    else:
        print("No aliases added")

    return True


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


# =============================================================================
# Per-Problem Maps
# =============================================================================

PROBLEM_MAP_DIR = 'QuickTuningProblemMap'

CONFIG_CLASSES = {
    'gemm': 'GemmConfiguration',
    'conv': 'ConvConfiguration',
    'attention': 'AttentionConfiguration',
    'gemm_gemm': 'GemmGemmConfiguration',
    'conv_gemm': 'ConvGemmConfiguration',
}


def problem_key_hash(row, op, rocmlir_gen):
    """Ask the compiler for this problem's key hash.

    The key has one implementation, in C++. Rebuild the problem the way
    perfRunner would and let rocmlir-gen answer, rather than reproducing it.
    """
    # Imported here rather than at module scope: it pulls in the built
    # `amd_arch_db`, which only the per-problem path needs.
    import perfRunner
    conf_class = getattr(perfRunner, CONFIG_CLASSES[op])
    config = conf_class.from_table_entry(row, row['Chip'], int(row['numCU']),
                                         int(row['numChiplets']))
    # rocmlir-gen rejects --kernel-repeats without a host harness, and we are
    # not running anything.
    args = config.generate_problem_commandline(kernel_repeats=None).split()
    result = subprocess.run([str(rocmlir_gen), *args, '--emit-quick-tuning-problem-key-hash'],
                            capture_output=True,
                            check=False,
                            text=True)
    if result.returncode:
        raise RuntimeError(f'could not key {config.to_command_line()!r}: {result.stderr.strip()}')
    return int(result.stdout)


def positive_int(value):
    """An argparse type that rejects the top-N values select_perfconfigs cannot use."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f'must be at least 1, got {parsed}')
    return parsed


def select_perfconfigs(group, op, top_n):
    """The best measured perfconfigs, keeping one legal non-split-K slot."""
    ordered = group.sort_values(['TFlops', 'PerfConfig'], ascending=[False, True])
    perfconfigs = ordered.head(top_n)['PerfConfig'].tolist()
    if op not in SPLIT_K_AWARE_OPS or any(get_splitk_value(p) in (None, '1') for p in perfconfigs):
        return perfconfigs, False
    legal = [p for p in ordered['PerfConfig'] if get_splitk_value(p) in (None, '1')]
    if not legal:
        return perfconfigs, True
    perfconfigs[-1] = legal[0]
    return perfconfigs, False


def per_problem_perfconfigs(typed, op, top_n, rocmlir_gen):
    """Rank one data type's measurements into {problem key hash: perfconfigs}."""
    problem_cols = get_target_columns(op)
    groups = [rows for _, rows in typed.groupby(problem_cols, sort=True, dropna=False)]
    print(f"keying {len(groups)} problems ... ", end='', flush=True)
    with ThreadPoolExecutor() as pool:
        keys = pool.map(lambda rows: problem_key_hash(rows.iloc[0], op, rocmlir_gen), groups)

    problems = {}
    missing_non_split = 0
    short = 0
    for rows, key in zip(groups, keys):
        if key in problems:
            raise ValueError(
                f'{op}: two problems share key {key}, so one would be dropped. Either the '
                f'compiler keys on fewer fields than {problem_cols}, or these two hash '
                f'to the same value.')
        best = rows.groupby('PerfConfig', as_index=False)['TFlops'].max()
        perfconfigs, missing = select_perfconfigs(best, op, top_n)
        problems[key] = perfconfigs
        missing_non_split += missing
        short += len(perfconfigs) < top_n
    return problems, missing_non_split, short


def to_camel_case(key):
    return ''.join(part.capitalize() for part in key.split('_'))


def format_shard(key, op, problems):
    """Render one per-problem map shard.

    Perfconfigs are interned and each problem indexes a variable-length run of
    them, so lists need no padding.
    """
    suffix = to_camel_case(key)
    hashes = sorted(problems)
    perfconfigs = sorted({p for h in hashes for p in problems[h]})
    if len(perfconfigs) > 65535:
        raise ValueError(f'{key}: {len(perfconfigs)} perfconfigs do not fit in uint16_t')
    index_of = {p: i for i, p in enumerate(perfconfigs)}

    refs, indices = [], []
    for value in hashes:
        refs.append(f'{{{value}ULL, {len(indices)}, {len(problems[value])}}}')
        indices += [index_of[p] for p in problems[value]]

    section = 'GemmGemm' if op in GEMM_GEMM_OPS else 'Gemm'
    lines = [
        '// clang-format off', f'// {suffix}.inc -- generated by: {get_generator_path()}', '',
        f'#ifdef {section}_PER_PROBLEM_DEFINITIONS_GEN',
        f'static const QuickTuningProblemRef problems{suffix}[] = {{'
    ]
    lines += [f'    {ref},' for ref in refs]
    lines += [
        '};', f'static const uint16_t perfConfigIndices{suffix}[] = {{',
        '    ' + ', '.join(map(str, indices)) + ',', '};',
        f'static const StringRef perfConfigs{suffix}[] = {{'
    ]
    lines += [f'    {json.dumps(p)},' for p in perfconfigs]
    lines += [
        '};', f'#endif // {section}_PER_PROBLEM_DEFINITIONS_GEN', '',
        f'#ifdef {section}_PER_PROBLEM_LOOKUP_TABLE_GEN',
        f'{{"{key}", QuickTuningProblemMap(problems{suffix}, '
        f'perfConfigIndices{suffix}, perfConfigs{suffix})}},',
        f'#endif // {section}_PER_PROBLEM_LOOKUP_TABLE_GEN', ''
    ]
    return '\n'.join(lines)


def update_problem_maps(df_arch, arch, op, top_n, rocmlir_gen):
    """Write this architecture's per-problem map shards."""
    print(f"\n=== {arch} per-problem maps ===\n")
    shard_dir = get_output_path().with_name(PROBLEM_MAP_DIR)
    shard_dir.mkdir(parents=True, exist_ok=True)
    kernel_type = OP_TO_KERNEL_TYPE[op].lower()

    for dtype in sorted(df_arch['DataType'].unique()):
        print(f"{dtype}: ", end='')
        typed = df_arch[df_arch['DataType'] == dtype]
        problems, missing_non_split, short = per_problem_perfconfigs(typed, op, top_n, rocmlir_gen)
        if not problems:
            print("no problems")
            continue

        key = f'{arch}_{kernel_type}_{dtype}'
        name = f'{to_camel_case(key)}.inc'
        shard = format_shard(key, op, problems)
        (shard_dir / name).write_text(shard)
        perfconfigs = {p for row in problems.values() for p in row}
        print(f"{len(perfconfigs)} perfconfigs -> {PROBLEM_MAP_DIR}/{name}")
        if short:
            print(f"  {short} problem(s) measured fewer than {top_n} perfconfigs")
        if missing_non_split:
            print(f"  {missing_non_split} problem(s) have no measured splitKFactor=1 perfconfig")


def process_arch(df, arch, op, threshold, max_configs, update, top_n, no_splitk):
    """Process data for a single architecture."""
    df_arch = df[df['Chip'] == arch]

    # Split-K filtering shapes the set cover only. A per-problem list ranks
    # what was actually measured for that problem.
    cover_data = filter_split_k(df_arch) if no_splitk else df_arch
    results = find_perfconfigs(cover_data, op, threshold, max_configs)
    print_results(results, arch)

    if update:
        update_inc_file(results, arch, op)
        print(f"Updated {get_output_path()} for {arch}")
        update_problem_maps(df_arch, arch, op, top_n,
                            os.environ.get('ROCMLIR_GEN_PATH', 'rocmlir-gen'))


def main(args=None):
    parser = argparse.ArgumentParser(
        prog='quickTuningGen.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generate QuickTuningPerfconfigs.inc from tuning data.',
        epilog='''
Examples:
    # Generate quick-tune lists from tuning data
    %(prog)s tuningData/*.debug --op conv --update
    %(prog)s gfx90a/*.debug gfx942/*.debug --op gemm --update
    cat data.debug | %(prog)s --op attention --update
    find . -name "*.debug" | xargs %(prog)s --op gemm --update

    # Add fallback type aliases (use f16 configs when there's no bf16 data)
    %(prog)s --alias bf16 f16
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
    parser.add_argument('--max-configs',
                        type=int,
                        default=40,
                        metavar='COUNT',
                        help='Maximum configs per dtype (default: 40)')
    parser.add_argument('--update', action='store_true', help='Update QuickTuningPerfconfigs.inc')
    parser.add_argument('--no-splitk',
                        action='store_true',
                        help='Exclude Split-K configurations from the set cover')
    parser.add_argument('--per-problem-top-n',
                        type=positive_int,
                        default=PER_PROBLEM_TOP_N,
                        help=f'perfconfigs kept per problem (default: {PER_PROBLEM_TOP_N})')
    parser.add_argument('--alias',
                        nargs=2,
                        metavar=('FROM', 'TO'),
                        help='Add fallback: use TO configs for FROM type (e.g., --alias bf16 f16)')

    pargs = parser.parse_args(args)

    if pargs.max_configs < 1:
        parser.error('--max-configs must be at least 1')

    if not pargs.op and not pargs.alias:
        parser.error('either --op or --alias must be specified')
        return 1

    # Generate quick-tune lists
    if pargs.op:
        df = load_data(pargs.files)
        if not df.empty:
            archs = sorted(df['Chip'].unique())
            print(f"Processing {len(archs)} architecture(s): {', '.join(archs)}")
            for arch in archs:
                process_arch(df, arch, pargs.op, pargs.th, pargs.max_configs, pargs.update,
                             pargs.per_problem_top_n, pargs.no_splitk)
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
