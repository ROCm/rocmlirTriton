#!/usr/bin/env python3
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Attach quick-tuning problem hashes to data measured before rocmlir-gen emitted them.

The quick-tuning database records, per problem, the best config measured for
it, and identifies a problem by the hash rocmlir-gen prints for
``--emit-quick-tuning-hash``. C++ owns that hash and quickTuningGen.py only
ever groups by it, so a ``.debug`` file has to carry it in a ``ProblemHash``
column.

Re-measuring every tier-1 problem just to pick that column up would cost days
of GPU time, and it is not necessary: the hash is a function of the problem,
and the tier-1 config files describe exactly the problems that were measured.
So this script asks rocmlir-gen for the hash of every tier-1 config once, then
joins the answers onto existing rows through the problem columns those rows
already carry.

A row whose problem is not in the tier-1 config files keeps an empty
``ProblemHash``, which leaves it contributing to the set cover only -- the same
degradation a file with no hash column at all gets.

Examples:
    # Add ProblemHash to a directory of measurements, in place.
    %(prog)s --rocmlir-gen build/bin/rocmlir-gen tuningData/*.debug

    # Write the results elsewhere, leaving the inputs untouched.
    %(prog)s --rocmlir-gen build/bin/rocmlir-gen -o hashed/ tuningData/*.debug
"""

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

HASH_COLUMN = 'ProblemHash'

# The tier-1 config file and the perfRunner configuration class for each
# operation the bridge can hash. conv+gemm is absent because it has no tier-1
# config file to bridge from.
TIER1_CONFIGS = {
    'gemm': ('tier1-gemm-configs', 'GemmConfiguration'),
    'conv': ('tier1-conv-configs', 'ConvConfiguration'),
    'attention': ('tier1-attention-configs', 'AttentionConfiguration'),
    'gemm_gemm': ('tier1-gemmgemm-configs', 'GemmGemmConfiguration'),
}

# Columns of a tuning row that do not describe the problem. Architecture and
# data type are among them because the quick-tuning key drops both; the rest
# are the measurement rather than its subject.
NON_PROBLEM_COLUMNS = frozenset({
    'DataType', 'OutDataType', 'Chip', 'numCU', 'numChiplets', 'PerfConfig', 'LDSBankConflict',
    'TFlops', 'MeasurementsMs', 'Status', HASH_COLUMN
})


def load_perf_runner(rocmlir_gen):
    """perfRunner, which turns a tier-1 config line into rocmlir-gen arguments.

    Imported here rather than at module scope because it needs the amd_arch_db
    extension module, which is built next to rocmlir-gen. Taking the path off
    `--rocmlir-gen` means a caller does not have to set PYTHONPATH as well.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(rocmlir_gen).resolve().parent))
    import perfRunner
    return perfRunner


def config_class(perf_runner, op):
    return getattr(perf_runner, TIER1_CONFIGS[op][1])


def problem_columns(columns):
    """The problem-identifying subset of `columns`, in their original order."""
    return [column for column in columns if column not in NON_PROBLEM_COLUMNS]


def normalize(value):
    """The canonical string form of a problem-column value, for joining.

    The two sides of the join arrive from different directions -- one from a
    parsed config object, the other from a TSV cell -- so `1`, `1.0` and `"1"`
    all have to land on the same string, and an unset optional field has to
    look the same whether it shows up as `None` or as an empty cell.
    """
    if value is None:
        return ''
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return ''
    text = str(value).strip()
    if text.lower() in ('', 'nan', 'none'):
        return ''
    if text.lower() in ('true', 'false'):
        return text.capitalize()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def detect_op(columns):
    """The operation a tuning row of `columns` describes, or None if unknown."""
    if 'Direction' in columns:
        return 'conv'
    if 'SeqLenQ' in columns:
        return 'attention'
    if 'O' in columns:
        # conv+gemm shares gemm+gemm's second-gemm column but keeps the
        # convolution's layouts, and has no tier-1 config file.
        return None if 'FilterLayout' in columns else 'gemm_gemm'
    if 'M' in columns:
        return 'gemm'
    return None


def read_header(path):
    """The column names of a tuning file, or None if it has none."""
    with open(path, newline='') as handle:
        for row in csv.reader(handle, delimiter='\t'):
            return row
    return None


def hash_config(rocmlir_gen, config):
    """The quick-tuning problem hash rocmlir-gen prints for `config`."""
    command = [rocmlir_gen]
    command += config.generate_mlir_driver_commandline('', kernel_repeats=None).split()
    command += ['--emit-quick-tuning-hash']
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def build_hash_map(perf_runner, options, op):
    """Map each tier-1 problem of `op` to its hash, keyed on its problem columns.

    The tuning columns are not guaranteed to be as fine-grained as the hash --
    a convolution's group count, for one, has never been recorded in them -- so
    a key that two config lines disagree on is dropped rather than joined on a
    coin flip. Those rows keep the behaviour of unhashed data.
    """
    cls = config_class(perf_runner, op)
    path = Path(options.configs) / TIER1_CONFIGS[op][0]
    columns = problem_columns(cls.TABLE_COLUMNS)

    hashes = {}
    ambiguous = set()
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            config = cls.from_command_line(line.split(), options.arch, options.num_cu,
                                           options.num_chiplets)
        except ValueError as error:
            print(f"WARNING: {path}:{lineno}: {error}", file=sys.stderr)
            continue
        key = tuple(normalize(config.table_entry(1)[column]) for column in columns)
        try:
            problem_hash = hash_config(options.rocmlir_gen, config)
        except RuntimeError as error:
            print(f"WARNING: {path}:{lineno}: no hash: {error}", file=sys.stderr)
            continue
        if hashes.setdefault(key, problem_hash) != problem_hash:
            ambiguous.add(key)

    for key in ambiguous:
        del hashes[key]
    if ambiguous:
        print(
            f"WARNING: {path}: {len(ambiguous)} problem(s) the tuning columns cannot tell "
            f"apart; left unhashed",
            file=sys.stderr)
    print(f"{op}: hashed {len(hashes)} problem(s) from {path}")
    return columns, hashes


def attach(path, out_path, op, hash_maps):
    """Rewrite `path` to `out_path` with a ProblemHash column appended."""
    with open(path, newline='') as handle:
        rows = list(csv.reader(handle, delimiter='\t'))

    header = rows[0]
    columns, hashes = hash_maps[op]
    indices = [header.index(column) for column in columns]

    matched = 0
    out_rows = [header + [HASH_COLUMN]]
    for row in rows[1:]:
        # A row of the wrong width is a repeated header or a truncated write.
        # quickTuningGen.py drops those, so pass them through untouched.
        if len(row) != len(header):
            out_rows.append(row)
            continue
        problem_hash = hashes.get(tuple(normalize(row[index]) for index in indices), '')
        matched += bool(problem_hash)
        out_rows.append(row + [problem_hash])

    with open(out_path, 'w', newline='') as handle:
        csv.writer(handle, delimiter='\t', lineterminator='\n').writerows(out_rows)
    print(f"{out_path}: hashed {matched} of {len(rows) - 1} row(s)")


def main(args=None):
    default_configs = Path(__file__).resolve().parent.parent / 'configs'

    parser = argparse.ArgumentParser(prog='attachProblemHashes.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument('files',
                        nargs='+',
                        metavar='FILE',
                        help='.debug files produced by tuningRunner.py')
    parser.add_argument('--rocmlir-gen', required=True, help='path to the rocmlir-gen binary')
    parser.add_argument('--configs',
                        default=str(default_configs),
                        help=f'directory holding the tier1-*-configs files '
                        f'(default: {default_configs})')
    parser.add_argument('-o',
                        '--output-dir',
                        help='write the results here instead of rewriting the inputs in place')
    # The quick-tuning key carries neither the architecture nor the compute
    # unit counts, so these only have to name a target rocmlir-gen can generate
    # every tier-1 problem for.
    parser.add_argument('--arch',
                        default='gfx942',
                        help='architecture to generate with (default: gfx942)')
    parser.add_argument('--num-cu', type=int, default=304, help='compute units (default: 304)')
    parser.add_argument('--num-chiplets', type=int, default=1, help='chiplets (default: 1)')

    options = parser.parse_args(args)
    perf_runner = load_perf_runner(options.rocmlir_gen)

    # Decide what each file needs before hashing anything, so that a rejected
    # input is reported ahead of the rocmlir-gen runs rather than after them.
    ops = {}
    for path in options.files:
        header = read_header(path)
        if not header:
            print(f"WARNING: {path}: empty; skipped", file=sys.stderr)
            continue
        if HASH_COLUMN in header:
            print(f"WARNING: {path}: already has a {HASH_COLUMN} column; skipped", file=sys.stderr)
            continue
        op = detect_op(header)
        if op is None:
            print(f"WARNING: {path}: no tier-1 config file covers these columns; skipped",
                  file=sys.stderr)
            continue
        missing = [
            column for column in problem_columns(config_class(perf_runner, op).TABLE_COLUMNS)
            if column not in header
        ]
        if missing:
            print(f"WARNING: {path}: missing problem column(s) {', '.join(missing)}; skipped",
                  file=sys.stderr)
            continue
        ops[path] = op

    hash_maps = {op: build_hash_map(perf_runner, options, op) for op in sorted(set(ops.values()))}

    output_dir = Path(options.output_dir) if options.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    for path, op in ops.items():
        attach(path, output_dir / Path(path).name if output_dir else path, op, hash_maps)

    return 0


if __name__ == '__main__':
    sys.exit(main())
