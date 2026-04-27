#!/usr/bin/env python3
"""
Script to update outdated perf_config formats in TOML files.

Converts old v3/v4 perf config format to the new gemm:v1 format.

Old format: v3:MPerBlock,NPerBlock,KpackPerBlock,MPerWave,NPerWave,MnPerXdl,SplitKFactor,ScheduleVersion,...
New format: gemm:v1:mPerBlock,nPerBlock,kPerBlock,kpack,numCTAs,numWaves,matrixInstrNonkdim,splitKFactor,numStages,wavesPerEU,gridGroupSize

Usage:
    python update_perf_config_format.py <toml_file> [--dry-run] [--in-place]
    python update_perf_config_format.py --check <directory>  # Check all TOML files in directory
"""

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class OldPerfConfig:
    """Parsed old v3 perf config."""
    m_per_block: int
    n_per_block: int
    kpack_per_block: int
    m_per_wave: int
    n_per_wave: int
    mn_per_xdl: int
    split_k_factor: int
    schedule_version: int
    output_swizzle: int
    waves_per_eu: int
    grid_group_size: int


@dataclass 
class NewPerfConfig:
    """New gemm:v1 perf config."""
    m_per_block: int
    n_per_block: int
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
        return (f"gemm:v1:{self.m_per_block},{self.n_per_block},{self.k_per_block},"
                f"{self.kpack},{self.num_ctas},{self.num_waves},{self.matrix_instr_nonkdim},"
                f"{self.split_k_factor},{self.num_stages},{self.waves_per_eu},{self.grid_group_size}")


def parse_old_v3_config(config_str: str) -> Optional[OldPerfConfig]:
    """Parse an old v3 format perf config string."""
    match = re.match(r'^v3:(\d+(?:,\d+)*)$', config_str.strip())
    if not match:
        return None
    
    parts = [int(x) for x in match.group(1).split(',')]
    if len(parts) != 11:
        return None
    
    return OldPerfConfig(
        m_per_block=parts[0],
        n_per_block=parts[1],
        kpack_per_block=parts[2],
        m_per_wave=parts[3],
        n_per_wave=parts[4],
        mn_per_xdl=parts[5],
        split_k_factor=parts[6],
        schedule_version=parts[7],
        output_swizzle=parts[8],
        waves_per_eu=parts[9],
        grid_group_size=parts[10],
    )


def parse_old_v4_config(config_str: str) -> Optional[OldPerfConfig]:
    """Parse an old v4 format perf config string."""
    match = re.match(r'^v4:(\d+(?:,\d+)*)$', config_str.strip())
    if not match:
        return None
    
    parts = [int(x) for x in match.group(1).split(',')]
    if len(parts) < 12:
        return None
    
    return OldPerfConfig(
        m_per_block=parts[0],
        n_per_block=parts[1],
        kpack_per_block=parts[2],
        m_per_wave=parts[3],
        n_per_wave=parts[4],
        mn_per_xdl=parts[5],
        split_k_factor=parts[7],  # Note: position 6 is Kpack in v4
        schedule_version=parts[8],
        output_swizzle=parts[9],
        waves_per_eu=parts[10],
        grid_group_size=parts[11],
    )


def is_new_format(config_str: str) -> bool:
    """Check if config is already in the new format."""
    return config_str.strip().startswith(('gemm:v1:', 'attn:v1:'))


def is_old_format(config_str: str) -> bool:
    """Check if config is in an old format that needs conversion."""
    stripped = config_str.strip()
    return stripped.startswith('v3:') or stripped.startswith('v4:')


def convert_to_new_format(old_config: OldPerfConfig, 
                          default_kpack: int = 1,
                          default_num_ctas: int = 1,
                          default_num_waves: int = 4,
                          default_matrix_instr_nonkdim: int = 16,
                          default_num_stages: int = 2,
                          default_waves_per_eu: int = 0,
                          default_grid_group_size: int = 0) -> NewPerfConfig:
    """
    Convert old perf config to new format.
    
    The conversion uses:
    - splitKFactor from the old config
    - numStages is always set to 2 (not derived from old scheduleVersion)
    - Default values for other parameters based on typical usage patterns
    """
    return NewPerfConfig(
        m_per_block=old_config.m_per_block,
        n_per_block=old_config.n_per_block,
        k_per_block=old_config.kpackperblock*old_config.kpack,
        kpack=default_kpack,
        num_ctas=default_num_ctas,
        num_waves=(old_config.mperblock*old_config.nperblock)/(old_config.nperwave*old_config.mperwave),
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
    
    if stripped.startswith('v3:'):
        old_config = parse_old_v3_config(stripped)
    elif stripped.startswith('v4:'):
        old_config = parse_old_v4_config(stripped)
    else:
        return config_str, False
    
    if old_config is None:
        return config_str, False
    
    new_config = convert_to_new_format(old_config)
    return new_config.to_string(), True


def process_toml_content(content: str) -> tuple[str, int]:
    """
    Process TOML content and convert all old format configs to new format.
    
    Handles two patterns:
    1. Quoted configs: "v3:..." or 'v3:...'
    2. Inline configs: --perf_config v3:... or -perf_config=v3:...
    
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
        # Reconstruct the full match with the new config
        return match.group(0).replace(config_str, new_config)
    
    # Pattern 1: Quoted configs like "v3:..." or 'v3:...' (standalone in arrays)
    pattern1 = re.compile(r'(["\'])(?P<config>v[34]:\d+(?:,\d+)*)\1')
    
    # Pattern 2: Inline configs like --perf_config v3:... or -perf_config=v3:...
    pattern2 = re.compile(r'(-{1,2}perf_config[= ])(?P<config>v[34]:\d+(?:,\d+)*)')
    
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
    pattern1 = re.compile(r'["\']v[34]:\d+(?:,\d+)*["\']')
    # Pattern for inline configs (--perf_config v3:... or -perf_config=v3:...)
    pattern2 = re.compile(r'-{1,2}perf_config[= ]v[34]:\d+(?:,\d+)*')
    
    matches1 = pattern1.findall(content)
    matches2 = pattern2.findall(content)
    total = len(matches1) + len(matches2)
    return total > 0, total


def process_file(filepath: Path, dry_run: bool = True, in_place: bool = False) -> int:
    """
    Process a single TOML file.
    
    Returns:
        Number of conversions made
    """
    content = filepath.read_text()
    new_content, conversions = process_toml_content(content)
    
    if conversions == 0:
        print(f"  {filepath}: No outdated configs found")
        return 0
    
    print(f"  {filepath}: {conversions} config(s) to convert")
    
    if dry_run:
        print("\n--- Proposed changes ---")
        # Show diff-like output
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
        # Write to stdout
        print(new_content)
    
    return conversions


def main():
    parser = argparse.ArgumentParser(
        description='Update outdated perf_config formats in TOML files'
    )
    parser.add_argument(
        'path',
        type=Path,
        help='Path to TOML file or directory to process'
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
    
    # Collect files to process
    if args.path.is_file():
        files = [args.path]
    else:
        files = list(args.path.rglob('*.toml'))
    
    if not files:
        print(f"No TOML files found in {args.path}")
        return 0
    
    if args.check:
        print(f"Checking {len(files)} TOML file(s) for outdated perf configs...\n")
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
            print("All configs are up to date!")
            return 0
    
    # Process files
    total_conversions = 0
    print(f"Processing {len(files)} TOML file(s)...\n")
    
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
