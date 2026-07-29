#!/usr/bin/env python3
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Cross-platform companion to mlir/utils/widgets/rocm-run.

The bash version is the source of truth on Linux; this script exists
because Windows can't execute shebang-only files (lit's RUN line then fails
with WinError 193). The Lit substitution in mlir/test/lit.cfg.py wraps the
script with ``python.exe`` on Windows; Linux keeps using the bash version.

Library naming mirrors CMake's shared-library conventions:
  Linux   -> lib<name>.so in <llvm_build>/lib  and <rocmlir_build>/lib
  Windows -> <name>.dll   in <llvm_build>/bin and <rocmlir_build>/bin
"""

import os
import sys
import subprocess
from pathlib import Path


def find_rocmlir_build(script_dir: Path) -> Path:
    driver_name = 'rocmlir-driver.exe' if sys.platform == 'win32' else 'rocmlir-driver'
    configured_build = os.environ.get('ROCMLIR_BUILD_DIR')
    if configured_build:
        build_dir = Path(configured_build)
        if (build_dir / 'bin' / driver_name).exists():
            return build_dir
        sys.exit(f'rocm-run: rocMLIR build has no {driver_name}: {build_dir}')
    if (script_dir / driver_name).exists():
        return script_dir.parent
    repo_root = script_dir.parents[2]
    candidates = [
        repo_root / 'build',
        Path.cwd().parent,
        Path.cwd(),
        Path.cwd() / 'build',
        Path.home() / 'rocmlir' / 'build'
    ]
    for d in candidates:
        if (d / 'bin' / driver_name).exists():
            return d
    sys.exit('rocm-run: cannot locate rocMLIR build directory')


def find_llvm_build(rocmlir_build: Path) -> Path:
    runner_name = 'mlir-runner.exe' if sys.platform == 'win32' else 'mlir-runner'

    # In-tree subtree build: LLVM/MLIR are built under
    # <rocmlir_build>/external/llvm-project/llvm.
    in_tree_llvm = rocmlir_build / 'external' / 'llvm-project' / 'llvm'
    if (in_tree_llvm / 'bin' / runner_name).exists():
        return in_tree_llvm

    # Legacy submodule layout: LLVM built under external/triton/llvm-project.
    triton_llvm = rocmlir_build.parent / 'external' / 'triton' / 'llvm-project' / 'build'
    if (triton_llvm / 'bin' / runner_name).exists():
        return triton_llvm

    cache = rocmlir_build / 'CMakeCache.txt'
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith('LLVM_DIR:') and '=' in line:
                llvm_dir = Path(line.split('=', 1)[1].strip())
                candidate = llvm_dir.parent.parent.parent
                if (candidate / 'bin' / runner_name).exists():
                    return candidate
    sys.exit('rocm-run: cannot locate LLVM build directory')


def shlib_path(build: Path, name: str) -> str:
    if sys.platform == 'win32':
        return str(build / 'bin' / f'{name}.dll')
    return str(build / 'lib' / f'lib{name}.so')


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    rocmlir_build = find_rocmlir_build(script_dir)
    llvm_build = find_llvm_build(rocmlir_build)

    shared_libs = ','.join([
        shlib_path(llvm_build, 'mlir_rocm_runtime'),
        shlib_path(rocmlir_build, 'conv-validation-wrappers'),
        shlib_path(llvm_build, 'mlir_runner_utils'),
        shlib_path(llvm_build, 'mlir_float16_utils'),
        shlib_path(llvm_build, 'mlir_c_runner_utils'),
    ])

    runner = llvm_build / 'bin' / ('mlir-runner.exe' if sys.platform == 'win32' else 'mlir-runner')
    cmd = [
        str(runner), '-O2', f'--shared-libs={shared_libs}', '--entry-point-result=void',
        *sys.argv[1:]
    ]
    return subprocess.call(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == '__main__':
    sys.exit(main())
