#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Quick Tuning Generator

Generates QuickTuningPerfconfigs.inc and per-problem quick-tuning shards from
tuning data produced by tuningRunner.py.
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

# Regex pattern for lookup table entries: {"arch_kernel_dtype", {Class::params, Class::count}}, // optional comment
LOOKUP_ENTRY_PATTERN = re.compile(r'\{("(gfx\w+)_(\w+)_(\w+)"),\s*(\{[^}]+\})\},(\s*//[^\n]*)?')

PROBLEM_HASH_COLUMN = 'ProblemHash'
PROBLEM_HASH_PATTERN = re.compile(r'^0x[0-9a-f]{1,16}$')
NO_PROBLEM_PATTERN = re.compile(r'^0x0+$')
POSITIONAL_SPLITK_INDEX = 7
POSITIONAL_MAX_VERSION = 5
POSITIONAL_VERSION_PATTERN = re.compile(r'^v(\d+)$')

SHARD_INCLUDE_DIR = "mlir/Dialect/Rock/Tuning/QuickTuningShards"
SHARD_DIR = f"include/{SHARD_INCLUDE_DIR}"
NO_CONFIG = 0xFFFF
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
    """Extract Split-K from named and legacy positional perfconfigs."""
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


def find_problem_topn(df, top_n):
    """Return the descending-TFlops top-N configs for each recorded problem."""
    if PROBLEM_HASH_COLUMN not in df.columns:
        print(f"No {PROBLEM_HASH_COLUMN} column: emitting no problem shards")
        return {}

    hashes = df[PROBLEM_HASH_COLUMN].astype(str).str.strip()
    valid = hashes.str.match(PROBLEM_HASH_PATTERN) & ~hashes.str.match(NO_PROBLEM_PATTERN)
    if not valid.all():
        examples = sorted(set(hashes[~valid].astype(str)))[:3]
        print(f"WARNING: ignoring {(~valid).sum()} row(s) whose {PROBLEM_HASH_COLUMN} is not "
              f"a recordable hash, e.g. {examples}")
    df = df.assign(**{PROBLEM_HASH_COLUMN: hashes})[valid]

    result = {}
    for dtype in sorted(df['DataType'].unique()):
        typed = df[df['DataType'] == dtype]
        typed = typed.groupby([PROBLEM_HASH_COLUMN, 'PerfConfig'],
                              as_index=False)['TFlops'].max()
        problems = {}
        for problem_hash, group in typed.groupby(PROBLEM_HASH_COLUMN):
            ordered = group.sort_values(['TFlops', 'PerfConfig'],
                                        ascending=[False, True])
            configs = ordered['PerfConfig'].tolist()
            selected = configs[:top_n]
            if selected and not any(not is_splitk(config) for config in selected):
                non_splitk = ordered[~ordered['PerfConfig'].apply(is_splitk)]
                if non_splitk.empty:
                    print(f"WARNING: no splitKFactor=1 config measured for {problem_hash}")
                else:
                    reserved = non_splitk['PerfConfig'].iloc[0]
                    if len(selected) == top_n:
                        selected[-1] = reserved
                    else:
                        selected.append(reserved)
            problems[int(problem_hash, 16)] = selected
        result[dtype] = problems
    return result


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


def get_shard_dir():
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent.parent.parent / SHARD_DIR


def get_index_path():
    return get_shard_dir().parent / "QuickTuningShards.inc"


def shard_suffix(arch, kernel_type, dtype):
    return f"{arch.capitalize()}{kernel_type}{dtype.capitalize()}"


def format_array(decl, entries):
    one_line = f"{decl} = {{{', '.join(entries)}}};"
    if len(one_line) <= 80:
        return one_line
    return f"{decl} = {{\n" + ",\n".join(f"    {entry}" for entry in entries) + "\n};"


def banner(text):
    prefix = f"//===- {text} "
    return prefix + "-" * max(1, 80 - len(prefix) - 5) + "===//"


def render_shard(key, suffix, configs, problems, top_n):
    if len(configs) > NO_CONFIG:
        raise ValueError(f"{key}: config pool overflows uint16_t")
    config_lines = ",\n".join(f'    "{config}"' for config in configs)
    hashes = [f"0x{problem_hash:016x}ULL" for problem_hash, _ in problems]
    slots = [str(index) if index is not None else "kQuickTuningNoConfig"
             for _, indices in problems for index in indices]
    return f"""{banner(f"{suffix}.inc - generated quick-tuning shard")}
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Generated by {get_generator_path()}; do not edit.
//
//===----------------------------------------------------------------------===//

// clang-format off

#ifdef QUICK_TUNING_DB_ARRAYS
static const char *const kCfg{suffix}[] = {{
{config_lines}
}};

{format_array(f"static const uint64_t kProb{suffix}[]", hashes)}

{format_array(f"static const uint16_t kProbCfg{suffix}[]", slots)}
#endif // QUICK_TUNING_DB_ARRAYS

#ifdef QUICK_TUNING_DB_ENTRIES
{{"{key}",
 kCfg{suffix}, /*numConfigs=*/{len(configs)},
 kProb{suffix}, kProbCfg{suffix},
 /*numProblems=*/{len(problems)}, /*numTopN=*/{top_n}}},
#endif // QUICK_TUNING_DB_ENTRIES
"""


def parse_shard(path):
    text = path.read_text()
    key = re.search(r'^\{"([^"]+)",', text, re.MULTILINE)
    count = re.search(r'/\*numProblems=\*/(\d+)', text)
    if not (key and count):
        raise ValueError(f"{path} is not a shard this generator can read")
    return key.group(1), int(count.group(1))


def write_index():
    header = """//===- QuickTuningShards.inc - index of quick-tuning shards ---------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Generated by mlir/utils/performance/analysis/quickTuningGen.py.
//===----------------------------------------------------------------------===//
"""
    names = sorted(path.name for path in get_shard_dir().glob("*.inc"))
    includes = "\n".join(f'#include "{SHARD_INCLUDE_DIR}/{name}"' for name in names)
    get_index_path().write_text(f"{header}\n{includes}\n")


def write_shards(topn, arch, op, top_n):
    kernel_type = OP_TO_KERNEL_TYPE[op]
    for dtype, problem_lists in topn.items():
        if not problem_lists:
            continue
        key = f"{arch}_{kernel_type.lower()}_{dtype}"
        pool = sorted({config for configs in problem_lists.values() for config in configs})
        config_indices = {config: i for i, config in enumerate(pool)}
        problems = []
        for problem_hash, configs in sorted(problem_lists.items()):
            indices = [config_indices[config] for config in configs]
            indices += [None] * (top_n - len(indices))
            problems.append((problem_hash, indices))

        suffix = shard_suffix(arch, kernel_type, dtype)
        path = get_shard_dir() / f"{suffix}.inc"
        if path.exists():
            _, previous = parse_shard(path)
            if len(problems) < previous * PROBLEM_LOSS_RATIO:
                print(f"WARNING: {key} drops from {previous} to {len(problems)} measured "
                      "problem(s); rerun with the full .debug set if that was not intended")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_shard(key, suffix, pool, problems, top_n))
        print(f"Wrote {path}: {len(problems)} problem(s), top {top_n}")
    write_index()


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


def process_arch(df, arch, op, threshold, top_n, update):
    """Process data for a single architecture."""
    df_arch = df[df['Chip'] == arch]

    results = find_perfconfigs(df_arch, op, threshold)
    problem_topn = find_problem_topn(df_arch, top_n)
    print_results(results, arch)

    if update:
        update_inc_file(results, arch, op)
        print(f"Updated {get_output_path()} for {arch}")
        write_shards(problem_topn, arch, op, top_n)


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
    parser.add_argument('--update', action='store_true', help='Update QuickTuningPerfconfigs.inc')
    parser.add_argument('--top-n',
                        type=int,
                        default=5,
                        metavar='N',
                        help='Configs recorded per measured problem (default: 5)')
    parser.add_argument('--no-splitk', action='store_true', help='Exclude Split-K configurations')
    parser.add_argument('--alias',
                        nargs=2,
                        metavar=('FROM', 'TO'),
                        help='Add fallback: use TO configs for FROM type (e.g., --alias bf16 f16)')

    pargs = parser.parse_args(args)

    if pargs.top_n <= 0:
        parser.error('--top-n must be positive')

    if not pargs.op and not pargs.alias:
        parser.error('either --op or --alias must be specified')
        return 1

    # Generate quick-tune lists
    if pargs.op:
        df = load_data(pargs.files, pargs.no_splitk)
        if not df.empty:
            archs = sorted(df['Chip'].unique())
            print(f"Processing {len(archs)} architecture(s): {', '.join(archs)}")
            for arch in archs:
                process_arch(df, arch, pargs.op, pargs.th, pargs.top_n, pargs.update)
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
