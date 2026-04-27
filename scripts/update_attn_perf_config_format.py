#!/usr/bin/env python3
"""
Script to update outdated attention perf_config formats in TOML and MLIR files.

Converts rocMLIR attn:v2 perf config format to rocmlirTriton's attn:v1 format.

Old format (rocMLIR):    attn:v2:mPerBlockG0,mPerBlockG1,nPerBlockG0,kpackPerBlock,mPerWave,nPerWave,kpack,splitKFactor,scheduleVersion,outputSwizzle,forceUnroll
New format (rocmlirTriton): attn:v1:mPerBlockG0,nPerBlockG0,kPerBlock,kpack,numCTAs,numWaves,matrixInstrNonkdim,splitKFactor,numStages,wavesPerEU,gridGroupSize

Conversion rules:
- v1.mPerBlockG0 = v2.nPerBlockG0  (M/N swap: rocMLIR computes transposed
- v1.nPerBlockG0 = v2.mPerBlockG0   attention, see GridwiseGemmToBlockwise.cpp)
- kPerBlock = kpackPerBlock * kpack (from v2)
- kpack = 1 (unrelated to v2 kpack which is a memory-packing concept)
- numWaves = (mPerBlockG0 * nPerBlockG0) / (mPerWave * nPerWave) (from v2)
- matrixInstrNonkdim = 0 (default)
- numCTAs = 1 (default)
- wavesPerEU = 0 (default)
- gridGroupSize = 0 (default)

Usage:
    python update_attn_perf_config_format.py <file_or_directory> [--dry-run] [--in-place]
    python update_attn_perf_config_format.py --check <directory>  # Check all files in directory
"""

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class OldAttnPerfConfig:
    """Parsed old attn:v2 perf config."""
    m_per_block_g0: int
    m_per_block_g1: int
    n_per_block_g0: int
    kpack_per_block: int
    m_per_wave: int
    n_per_wave: int  # The <combined> field
    kpack: int
    split_k_factor: int
    schedule_version: int
    output_swizzle: int
    force_unroll: int


@dataclass 
class NewAttnPerfConfig:
    """New attn:v1 perf config."""
    m_per_block_g0: int
    n_per_block_g0: int
    k_per_block: int
    kpack: int
    num_ctas: int
    num_waves: int
    matrix_instr_nonkdim: int
    split_k_factor: int
    num_stages: int
    waves_per_eu: int
    grid_group_size: int
    
    def to_string(self) -> str:
        return (f"attn:v1:{self.m_per_block_g0},{self.n_per_block_g0},{self.k_per_block},"
                f"{self.kpack},{self.num_ctas},{self.num_waves},{self.matrix_instr_nonkdim},"
                f"{self.split_k_factor},{self.num_stages},{self.waves_per_eu},{self.grid_group_size}")


def parse_old_attn_v2_config(config_str: str) -> Optional[OldAttnPerfConfig]:
    """Parse an old attn:v2 format perf config string."""
    match = re.match(r'^attn:v2:(\d+(?:,\d+)*)$', config_str.strip())
    if not match:
        return None
    
    parts = [int(x) for x in match.group(1).split(',')]
    if len(parts) != 11:
        return None
    
    return OldAttnPerfConfig(
        m_per_block_g0=parts[0],
        m_per_block_g1=parts[1],
        n_per_block_g0=parts[2],
        kpack_per_block=parts[3],
        m_per_wave=parts[4],
        n_per_wave=parts[5],
        kpack=parts[6],
        split_k_factor=parts[7],
        schedule_version=parts[8],
        output_swizzle=parts[9],
        force_unroll=parts[10],
    )


def is_new_format(config_str: str) -> bool:
    """Check if config is already in the new format."""
    return config_str.strip().startswith('attn:v1:')


def is_old_format(config_str: str) -> bool:
    """Check if config is in an old format that needs conversion."""
    return config_str.strip().startswith('attn:v2:')


def convert_to_new_format(old_config: OldAttnPerfConfig,
                          default_kpack: int = 1,
                          default_num_ctas: int = 1,
                          default_matrix_instr_nonkdim: int = 0,
                          default_num_stages: int = 1,
                          default_waves_per_eu: int = 0,
                          default_grid_group_size: int = 0) -> NewAttnPerfConfig:
    """
    Convert old attn perf config to new format.
    
    Conversion rules:
    - kPerBlock = kpackPerBlock * kpack (from old config)
    - numWaves = (mPerBlockG0 * nPerBlockG0) / (mPerWave * nPerWave)
    - kpack = 1 (hardcoded)
    - matrixInstrNonkdim = 0 (hardcoded)
    - numStages = 1 (hardcoded)
    """
    # Compute kPerBlock
    k_per_block = old_config.kpack_per_block * old_config.kpack
    
    # Compute numWaves
    numerator = old_config.m_per_block_g0 * old_config.n_per_block_g0
    denominator = old_config.m_per_wave * old_config.n_per_wave
    if denominator == 0:
        num_waves = 1  # Fallback to avoid division by zero
    else:
        num_waves = numerator // denominator
    
    # M/N swap: rocMLIR computes the transposed attention (M=seqK, N=seqQ) and
    # transposes the result before storing, so its mPerBlockG0 tiles seqK while
    # rocmlirTriton's mPerBlockG0 tiles seqQ. See transposeAttnOperand() in
    # rocMLIR's GridwiseGemmToBlockwise.cpp.
    return NewAttnPerfConfig(
        m_per_block_g0=old_config.n_per_block_g0,
        n_per_block_g0=old_config.m_per_block_g0,
        k_per_block=k_per_block,
        kpack=default_kpack,
        num_ctas=default_num_ctas,
        num_waves=num_waves,
        matrix_instr_nonkdim=default_matrix_instr_nonkdim,
        split_k_factor=old_config.split_k_factor,
        num_stages=default_num_stages,
        waves_per_eu=default_waves_per_eu,
        grid_group_size=default_grid_group_size,
    )


def convert_config_string(config_str: str) -> tuple[str, bool]:
    """
    Convert a single config string from old to new format.
    
    Returns:
        Tuple of (converted_string, was_converted)
    """
    stripped = config_str.strip()
    
    if is_new_format(stripped):
        return config_str, False
    
    if not is_old_format(stripped):
        return config_str, False
    
    old_config = parse_old_attn_v2_config(stripped)
    if old_config is None:
        return config_str, False
    
    new_config = convert_to_new_format(old_config)
    return new_config.to_string(), True


def process_file_content(content: str) -> tuple[str, int]:
    """
    Process file content and convert all old format configs to new format.
    
    Handles multiple patterns:
    1. Quoted configs: "attn:v2:..." or 'attn:v2:...'
    2. Inline configs: perf_config = "attn:v2:..." or perf_config="attn:v2:..."
    
    Returns:
        Tuple of (new_content, number_of_conversions)
    """
    conversions = 0
    
    def replace_config(match):
        nonlocal conversions
        config_str = match.group('config')
        new_config, was_converted = convert_config_string(config_str)
        if was_converted:
            conversions += 1
        return match.group(0).replace(config_str, new_config)
    
    # Pattern for quoted configs like "attn:v2:..." or 'attn:v2:...'
    pattern1 = re.compile(r'(["\'])(?P<config>attn:v2:\d+(?:,\d+)*)\1')
    
    # Pattern for inline configs like --perf_config attn:v2:... or -perf_config=attn:v2:...
    pattern2 = re.compile(r'(-{1,2}perf_config[= ])(?P<config>attn:v2:\d+(?:,\d+)*)')
    
    new_content = pattern1.sub(replace_config, content)
    new_content = pattern2.sub(replace_config, new_content)
    
    return new_content, conversions


def check_file(filepath: Path) -> tuple[bool, int]:
    """
    Check if a file contains outdated configs.
    
    Returns:
        Tuple of (has_outdated, count_outdated)
    """
    content = filepath.read_text()
    # Pattern for quoted configs
    pattern1 = re.compile(r'["\']attn:v2:\d+(?:,\d+)*["\']')
    # Pattern for inline configs
    pattern2 = re.compile(r'-{1,2}perf_config[= ]attn:v2:\d+(?:,\d+)*')
    
    matches1 = pattern1.findall(content)
    matches2 = pattern2.findall(content)
    total = len(matches1) + len(matches2)
    return total > 0, total


def process_file(filepath: Path, dry_run: bool = True, in_place: bool = False) -> int:
    """
    Process a single file.
    
    Returns:
        Number of conversions made
    """
    content = filepath.read_text()
    new_content, conversions = process_file_content(content)
    
    if conversions == 0:
        print(f"  {filepath}: No outdated attn configs found")
        return 0
    
    print(f"  {filepath}: {conversions} config(s) to convert")
    
    if dry_run:
        print("\n--- Proposed changes ---")
        old_lines = content.splitlines()
        new_lines = new_content.splitlines()
        for i, (old, new) in enumerate(zip(old_lines, new_lines), 1):
            if old != new:
                print(f"  Line {i}:")
                print(f"    - {old}")
                print(f"    + {new}")
        print("--- End of changes ---\n")
    elif in_place:
        filepath.write_text(new_content)
        print(f"  Updated {filepath}")
    else:
        print(new_content)
    
    return conversions


def main():
    parser = argparse.ArgumentParser(
        description='Update outdated attention perf_config formats in TOML/MLIR files'
    )
    parser.add_argument(
        'path',
        type=Path,
        help='Path to file or directory to process'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Only check for outdated configs, do not convert'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--in-place', '-i',
        action='store_true',
        help='Modify files in place'
    )
    
    args = parser.parse_args()
    
    if not args.path.exists():
        print(f"Error: {args.path} does not exist", file=sys.stderr)
        return 1
    
    # Collect files to process (TOML and MLIR files)
    if args.path.is_file():
        files = [args.path]
    else:
        toml_files = list(args.path.rglob('*.toml'))
        mlir_files = list(args.path.rglob('*.mlir'))
        files = toml_files + mlir_files
    
    if not files:
        print(f"No TOML or MLIR files found in {args.path}")
        return 0
    
    if args.check:
        print(f"Checking {len(files)} file(s) for outdated attn perf configs...\n")
        outdated_files = []
        for filepath in files:
            has_outdated, count = check_file(filepath)
            if has_outdated:
                outdated_files.append((filepath, count))
                print(f"  {filepath}: {count} outdated config(s)")
        
        if outdated_files:
            print(f"\nFound {len(outdated_files)} file(s) with outdated configs")
            return 1
        else:
            print("All attn configs are up to date!")
            return 0
    
    # Process files
    total_conversions = 0
    print(f"Processing {len(files)} file(s)...\n")
    
    for filepath in files:
        conversions = process_file(
            filepath,
            dry_run=args.dry_run or not args.in_place,
            in_place=args.in_place
        )
        total_conversions += conversions
    
    print(f"\nTotal: {total_conversions} config(s) {'would be ' if args.dry_run else ''}converted")
    
    if args.dry_run and total_conversions > 0:
        print("\nTo apply changes, run with --in-place flag")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
