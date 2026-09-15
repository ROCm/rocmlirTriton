#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Generate compact per-problem quick-tuning lists from tuning debug TSVs."""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perfCommonUtils import SPLITK_KEY, parse_perfconfig  # noqa: E402


TARGET_COLUMNS = {
    'gemm': ['TransA', 'TransB', 'TransO', 'G', 'M', 'K', 'N'],
    'conv': [
        'Direction', 'FilterLayout', 'InputLayout', 'OutputLayout', 'N', 'C', 'H', 'W', 'K', 'Y',
        'X', 'DilationH', 'DilationW', 'StrideH', 'StrideW', 'PaddingH', 'PaddingW'
    ],
    'attention': [
        'TransQ', 'TransK', 'TransV', 'TransO', 'Causal', 'ReturnLSE', 'SplitKV',
        'SlidingWindowLookBack', 'WithAttnScale', 'WithAttnBias', 'TransBias', 'G', 'SeqLenQ',
        'SeqLenK', 'NumHeadsQ', 'NumHeadsKV', 'HeadDimQK', 'HeadDimV'
    ],
    'gemm_gemm': ['TransA', 'TransB', 'TransC', 'TransO', 'G', 'M', 'K', 'N', 'O'],
    'conv_gemm': [
        'FilterLayout', 'InputLayout', 'TransC', 'TransO', 'N', 'C', 'H', 'W', 'K', 'Y', 'X',
        'DilationH', 'DilationW', 'StrideH', 'StrideW', 'PaddingH', 'PaddingW', 'O'
    ],
}

KERNEL_TYPES = {
    'gemm': 'gemm',
    'conv': 'conv',
    'attention': 'attention',
    'gemm_gemm': 'gemmelementwisegemm',
    'conv_gemm': 'convelementwisegemm',
}

GEMM_SECTION_OPS = {'gemm', 'conv'}
SPLIT_K_AWARE_OPS = {'gemm', 'conv', 'gemm_gemm', 'conv_gemm'}


def detect_operation(columns):
    """Identify the tuning operation from its debug-column schema."""
    columns = set(columns)
    if 'SeqLenQ' in columns:
        return 'attention'
    if 'O' in columns and 'FilterLayout' in columns:
        return 'conv_gemm'
    if 'O' in columns and 'TransC' in columns:
        return 'gemm_gemm'
    if 'Direction' in columns:
        return 'conv'
    if {'TransA', 'TransB', 'M', 'K', 'N'} <= columns:
        return 'gemm'
    raise ValueError('cannot identify operation from debug columns')


def as_bool(value):
    if isinstance(value, str):
        return value.lower() in ('1', 'true', 'yes')
    return bool(value)


def boolean(value):
    return 'true' if as_bool(value) else 'false'


def integer(value):
    return str(int(value))


def make_problem_name(row, op):
    """Match getQuickTuningProblemName's versioned, hardware-free spelling."""
    if op == 'gemm':
        fields = [
            '-transA', boolean(row.TransA), '-transB', boolean(row.TransB), '-transO',
            boolean(row.TransO)
        ]
        if 'ScaledGemm' in row.index and as_bool(row.ScaledGemm):
            fields += [
                '-scaledGemm', '-scale_a_dtype', str(row.ScaleADtype), '-scale_b_dtype',
                str(row.ScaleBDtype), '-transScaleA', boolean(row.TransScaleA), '-transScaleB',
                boolean(row.TransScaleB)
            ]
        fields += [
            '-g', integer(row.G), '-m', integer(row.M), '-n', integer(row.N), '-k', integer(row.K)
        ]
        return ' '.join(fields)

    if op == 'conv':
        direction = {'fwd': '1', 'bwd': '2', 'backward_data': '2'}.get(
            str(row.Direction).lower())
        if direction is None:
            raise ValueError(f'unsupported convolution direction: {row.Direction}')
        group = integer(row.G) if 'G' in row.index and not pd.isna(row.G) else '1'
        # Rock's internal convolution layout names use GEMM dimensions: filter
        # K is N, and output K is C. Match extractLayouts()'s compiler spelling.
        filter_layout = str(row.FilterLayout).lower().replace('k', 'n')
        output_layout = str(row.OutputLayout).lower().replace('k', 'c')
        return ' '.join([
            '-F', direction, '-f', filter_layout, '-I', str(row.InputLayout), '-O', output_layout,
            '-n', integer(row.N), '-c', integer(row.C), '-H', integer(row.H), '-W', integer(row.W),
            '-k', integer(row.K), '-y', integer(row.Y), '-x', integer(row.X), '-p',
            integer(row.PaddingH), '-q', integer(row.PaddingW), '-u', integer(row.StrideH), '-v',
            integer(row.StrideW), '-l', integer(row.DilationH), '-j', integer(row.DilationW), '-g',
            group
        ]).lower()

    if op == 'attention':
        fields = [
            '-transQ', boolean(row.TransQ), '-transK', boolean(row.TransK), '-transV',
            boolean(row.TransV), '-transO', boolean(row.TransO), '-causal', boolean(row.Causal),
            '-return_lse', boolean(row.ReturnLSE), '-split_kv', integer(row.SplitKV)
        ]
        sliding = row.SlidingWindowLookBack
        if not pd.isna(sliding) and int(sliding) > 0:
            fields += ['-sliding_window_look_back', integer(sliding)]
        fields += [
            '-num_heads_q', integer(row.NumHeadsQ), '-num_heads_kv', integer(row.NumHeadsKV), '-g',
            integer(row.G), '-seq_len_q', integer(row.SeqLenQ), '-seq_len_k', integer(row.SeqLenK),
            '-head_dim_qk', integer(row.HeadDimQK), '-head_dim_v', integer(row.HeadDimV),
            '-with-attn-scale', boolean(row.WithAttnScale), '-with-attn-bias',
            boolean(row.WithAttnBias), '-transBias', boolean(row.TransBias)
        ]
        return ' '.join(fields)

    if op == 'gemm_gemm':
        return ' '.join([
            '-transA', boolean(row.TransA), '-transB', boolean(row.TransB), '-transC',
            boolean(row.TransC), '-transO', boolean(row.TransO), '-g', integer(row.G), '-m',
            integer(row.M), '-n', integer(row.N), '-k', integer(row.K), '-gemmO', integer(row.O)
        ])

    group = integer(row.G) if 'G' in row.index and not pd.isna(row.G) else '1'
    return ' '.join([
        '-f', str(row.FilterLayout), '-I', str(row.InputLayout), '-transV', boolean(row.TransC),
        '-transO', boolean(row.TransO), '-n', integer(row.N), '-c', integer(row.C), '-H',
        integer(row.H), '-W', integer(row.W), '-k', integer(row.K), '-y', integer(row.Y), '-x',
        integer(row.X), '-p', integer(row.PaddingH), '-q', integer(row.PaddingW), '-u',
        integer(row.StrideH), '-v', integer(row.StrideW), '-l', integer(row.DilationH), '-j',
        integer(row.DilationW), '-g', group, '-gemmO', integer(row.O)
    ]).lower()


def is_non_split_k(perfconfig):
    _, params = parse_perfconfig(perfconfig)
    return str(params.get(SPLITK_KEY, 1)) == '1'


def select_problem_configs(group, op, top_n):
    """Select measured leaders, reserving one legal non-split-K slot."""
    ordered = group.sort_values(['TFlops', 'PerfConfig'], ascending=[False, True])
    configs = ordered.head(top_n)['PerfConfig'].tolist()
    missing_non_split = False
    if op in SPLIT_K_AWARE_OPS and not any(map(is_non_split_k, configs)):
        legal = [config for config in ordered['PerfConfig'] if is_non_split_k(config)]
        if legal:
            configs[-1] = legal[0]
        else:
            missing_non_split = True
    return configs, missing_non_split


def load_file(path):
    df = pd.read_csv(path, sep='\t', low_memory=False)
    op = detect_operation(df.columns)
    if 'TransBias' not in df.columns and op == 'attention':
        df['TransBias'] = False
    if 'SlidingWindowLookBack' not in df.columns and op == 'attention':
        df['SlidingWindowLookBack'] = -1
    df = df[df['DataType'] != 'DataType']
    df['TFlops'] = pd.to_numeric(df['TFlops'], errors='coerce')
    df = df.dropna(subset=['TFlops'])
    missing = set(TARGET_COLUMNS[op]) - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns: {sorted(missing)}')
    df['ProblemName'] = df.apply(make_problem_name, axis=1, op=op)
    return op, df


def collect_keys(paths, top_n):
    by_op = {}
    for path in paths:
        op, df = load_file(path)
        by_op.setdefault(op, []).append(df)

    keys = {}
    for op, frames in by_op.items():
        df = pd.concat(frames, ignore_index=True)
        for (arch, dtype), typed in df.groupby(['Chip', 'DataType'], sort=True):
            aggregate = typed.groupby(['ProblemName', 'PerfConfig'],
                                      as_index=False)['TFlops'].max()
            problems = {}
            missing_non_split = 0
            for name, group in aggregate.groupby('ProblemName', sort=True):
                configs, missing = select_problem_configs(group, op, top_n)
                problems[name] = configs
                missing_non_split += missing
            key = f'{arch}_{KERNEL_TYPES[op]}_{dtype}'
            keys[key] = (op, problems)
            if missing_non_split:
                print(f'WARNING: {key}: {missing_non_split} problem(s) have no measured '
                      'splitKFactor=1 config', file=sys.stderr)
    return keys


def identifier(key):
    return 'problem' + ''.join(part.capitalize() for part in re.split(r'[^A-Za-z0-9]+', key))


def format_values(values, indent='    ', width=100):
    lines = []
    current = indent
    for value in values:
        token = f'{value},'
        if len(current) + len(token) + 1 > width and current.strip():
            lines.append(current.rstrip())
            current = indent
        current += token + ' '
    if current.strip():
        lines.append(current.rstrip())
    return lines


def emit_key_definitions(key, problems):
    base = identifier(key)
    names = sorted(problems)
    configs = sorted({config for name in names for config in problems[name]})
    config_index = {config: index for index, config in enumerate(configs)}
    if len(configs) > 65535:
        raise ValueError(f'{key}: {len(configs)} configs do not fit in uint16_t')

    offsets = [0]
    indices = []
    for name in names:
        indices.extend(config_index[config] for config in problems[name])
        offsets.append(len(indices))

    lines = [f'static const StringRef {base}Names[] = {{']
    lines += [f'    {json.dumps(name)},' for name in names]
    lines += ['};', f'static const uint32_t {base}Offsets[] = {{']
    lines += format_values(offsets)
    lines += ['};', f'static const uint16_t {base}ConfigIndices[] = {{']
    lines += format_values(indices)
    lines += ['};', f'static const StringRef {base}Configs[] = {{']
    lines += [f'    {json.dumps(config)},' for config in configs]
    lines += ['};', '']

    string_bytes = sum(len(name) + 1 for name in names)
    string_bytes += sum(len(config) + 1 for config in configs)
    payload_bytes = string_bytes + 4 * len(offsets) + 2 * len(indices)
    return lines, base, len(names), len(configs), payload_bytes


def generate(keys):
    sections = {
        'Gemm_PROBLEM_DEFINITIONS_GEN': [],
        'Gemm_PROBLEM_LOOKUP_TABLE_GEN': [],
        'GemmGemm_PROBLEM_DEFINITIONS_GEN': [],
        'GemmGemm_PROBLEM_LOOKUP_TABLE_GEN': [],
    }
    summaries = []
    for key in sorted(keys):
        op, problems = keys[key]
        prefix = 'Gemm' if op in GEMM_SECTION_OPS else 'GemmGemm'
        definitions, base, num_problems, num_configs, payload_bytes = emit_key_definitions(
            key, problems)
        sections[f'{prefix}_PROBLEM_DEFINITIONS_GEN'] += definitions
        sections[f'{prefix}_PROBLEM_LOOKUP_TABLE_GEN'].append(
            f'{{"{key}", ProblemTuningData({base}Names, {base}Offsets, '
            f'{base}ConfigIndices, {base}Configs)}},')
        summaries.append((key, num_problems, num_configs, payload_bytes))

    lines = [
        '// Generated by: mlir/utils/performance/analysis/quickTuningProblemGen.py', '',
        '// clang-format off', ''
    ]
    for section, body in sections.items():
        lines += [f'#ifdef {section}']
        lines += body
        lines += [f'#endif // {section}', '']
    return '\n'.join(lines), summaries


def default_output_path():
    return (Path(__file__).resolve().parents[3] /
            'include/mlir/Dialect/Rock/Tuning/QuickTuningProblemPerfconfigs.inc')


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Generate per-problem quick-tuning data from .tsv.debug files.')
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--top-n', type=int, default=3,
                        help='measured configs retained per problem (default: 3)')
    parser.add_argument('--output', type=Path, default=default_output_path())
    pargs = parser.parse_args(args)
    if pargs.top_n <= 0:
        parser.error('--top-n must be positive')
    for path in pargs.files:
        if not path.is_file() or not str(path).endswith('.debug'):
            parser.error(f'{path} is not an existing .debug file')

    content, summaries = generate(collect_keys(pargs.files, pargs.top_n))
    pargs.output.parent.mkdir(parents=True, exist_ok=True)
    pargs.output.write_text(content)
    for key, num_problems, num_configs, payload_bytes in summaries:
        print(f'{key}: {num_problems} problems, {num_configs} configs, '
              f'{payload_bytes} payload bytes')
    print(f'Wrote {pargs.output} ({pargs.output.stat().st_size} source bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
