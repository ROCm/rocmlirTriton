#!/usr/bin/env python3
""" A script to perform static tests for the mlir project.

This script runs clang-format and clang-tidy on the changes before a user
merges them to the master branch.

The code was extracted from https://github.com/google/llvm-premerge-checks.

Example usage:
~/rocMLIR#  ln -s build/compile_commands.json compile_commands.json
~/rocMLIR#  python3 ./mlir/utils/jenkins/static-checks/premerge-checks.py
"""

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import re
import subprocess
from typing import Tuple
import pathspec
import unidiff
import argparse
import multiprocessing
import git


def get_diff(base_commit, ignore_external_files: bool) -> Tuple[bool, str]:
    command = f"git-clang-format --diff {base_commit}"
    if ignore_external_files:
        # Restrict formatting to changed files outside the vendored external/
        # tree.
        command = (f"git-clang-format --diff {base_commit} -- "
                   f"$(git diff --name-only --diff-filter=d {base_commit} | grep -v '^external/')")
    diff_run = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    is_diff_run_succesful = diff_run.returncode <= 1
    diff = diff_run.stdout.decode()
    print(diff)
    return is_diff_run_succesful, diff


def check_external_file(filename: str) -> bool:
    print(filename)
    regex = '^external/'
    return re.search(regex, filename)


def run_clang_format(base_commit, ignore_config, ignore_external_files: bool = False):
    """Apply clang-format and return if no issues were found.
  Extracted from https://github.com/google/llvm-premerge-checks/blob/master/scripts/clang_format_report.py"""

    is_diff_run_succesful, patch = get_diff(base_commit, ignore_external_files)
    if not is_diff_run_succesful:
        print('git-clang-format returned an non-zero exit code')
        return False
    patches = unidiff.PatchSet(patch)
    ignore_lines = []

    if ignore_config is not None and os.path.exists(ignore_config):
        ignore_lines = open(ignore_config, 'r').readlines()
    ignore = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ignore_lines)
    patched_file: unidiff.PatchedFile
    success = True
    for patched_file in patches:
        # drop diff prefix
        patched_file_src = patched_file.source_file[2:]
        patched_file_tgt = patched_file.target_file[2:]
        if ignore.match_file(patched_file_src) or ignore.match_file(patched_file_tgt):
            continue
        if ignore_external_files:
            if check_external_file(patched_file_src) or check_external_file(patched_file_tgt):
                continue
        hunk: unidiff.Hunk
        for hunk in patched_file:
            success = False

    if not success:
        print(
            f'Please format your changes with clang-format by running `git-clang-format {base_commit}` or applying patch.'
        )
        return False
    return True


def remove_ignored(diff_lines, ignore_patterns_lines):
    ignore = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern,
                                          ignore_patterns_lines)
    good = True
    result = []
    for line in diff_lines:
        match = re.search(r'^diff --git (.*) (.*)$', line)
        if match:
            good = not (ignore.match_file(match.group(1)) and ignore.match_file(match.group(2)))
        if good:
            result.append(line)
    return result


def check_third_party_file(filename: str) -> bool:
    # We assume third party files to have absolute paths
    if filename[0] != '/':
        return False
    repo = git.Repo('.', search_parent_directories=True)
    regex = f'^{repo.working_tree_dir}'
    # Any file that does not belong to the repo, will be considered third party.
    return not re.search(regex, filename)


def run_clang_tidy(base_commit, ignore_config, ignore_external_files: bool = False):
    """Apply clang-tidy and return if no issues were found.
  Extracted from https://github.com/google/llvm-premerge-checks/blob/master/scripts/clang_tidy_report.py"""

    # Exclude the vendored upstream trees from the diff entirely so clang-tidy
    # is never invoked on external/ files. Without this, clang-tidy-diff.py can
    # timeout on large diffs.
    diff_command = f'git diff -U0 --no-prefix {base_commit}'
    if ignore_external_files:
        diff_command += " -- . ':!external'"
    r = subprocess.run(diff_command,
                       shell=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    diff = r.stdout.decode("utf-8", "ignore")
    if ignore_config is not None and os.path.exists(ignore_config):
        ignore = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern,
                                              open(ignore_config, 'r').readlines())
        diff = remove_ignored(diff.splitlines(keepends=True), open(ignore_config, 'r'))
    else:
        ignore = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, [])
    cpu_count = multiprocessing.cpu_count()
    # clang-tidy has no compile command for header files, so for any changed
    # header it interpolates a command from a "nearby" .cpp. This can give
    # errors unless we add the Triton include paths to the compile command.
    repo_root = git.Repo('.', search_parent_directories=True).working_tree_dir
    triton_include_roots = [
        os.path.join('external', 'triton', d) for d in ('include', 'third_party')
    ]
    triton_includes = []
    for root in triton_include_roots:
        triton_includes.append(os.path.join(repo_root, root))
        triton_includes.append(os.path.join(repo_root, 'build', root))
    extra_args = ['-extra-arg=-std=c++17']
    extra_args += [f'-extra-arg=-I{inc}' for inc in triton_includes]
    # TableGen outputs for rocmlirTriton (e.g. mlir/Conversion/CPU/Passes.h.inc)
    # land under build/mlir/include. Header-only diffs have no compile command, so
    # clang-tidy-diff must see that directory explicitly.
    extra_args.append(f'-extra-arg=-I{os.path.join(repo_root, "build", "mlir", "include")}')
    p = subprocess.Popen([
        './external/llvm-project/clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py', '-p0',
        '-quiet', '-j',
        str(cpu_count), *extra_args
    ],
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    a = ''.join(diff)
    out = p.communicate(input=a.encode())[0].decode()
    # Typical finding looks like:
    # [cwd/]clang/include/clang/AST/DeclCXX.h:3058:20: error: ... [clang-diagnostic-error]
    pattern = '^([^:]*):(\\d+):(\\d+): (.*): (.*)'
    errors_count = 0
    warn_count = 0
    for line in out.splitlines(keepends=False):
        line = line.strip()
        line = line.replace(os.getcwd() + os.sep, '')

        if len(line) == 0 or line == 'No relevant changes found.':
            continue
        match = re.search(pattern, line)
        if match:
            file_name = match.group(1)
            severity = match.group(4)
            if severity in ['warning', 'error']:
                if ignore.match_file(file_name):
                    print('{} is ignored by pattern and no comment will be added'.format(file_name))
                    continue
                if check_third_party_file(file_name):
                    print(
                        f'{file_name} is ignored as its a third-party file and no comment will be added'
                    )
                    continue
                if ignore_external_files:
                    if check_external_file(file_name):
                        print(
                            f'{file_name} is ignored as its a external (upstream) file and no comment will be added'
                        )
                        continue
                if severity == 'warning':
                    warn_count += 1
                if severity == 'error':
                    print('clang-tidy found error:', line)
                    errors_count += 1

    if errors_count + warn_count != 0:
        print('clang-tidy found {} errors and {} warnings.'.format(errors_count, warn_count))

    if errors_count != 0:
        return False

    return True


if __name__ == '__main__':
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="rocMLIR premerge checker",
        description="A helper script to invoke linters",
        allow_abbrev=False,
    )
    parser.add_argument("-b",
                        "--base-commit",
                        type=str,
                        default='origin/develop',
                        help="The base commit to lint against")
    parser.add_argument('--ignore-external', action='store_true', default=False)
    parsed_args = parser.parse_args(args)
    print(f"Running linters against base commit : {parsed_args.base_commit}")
    if not (run_clang_format(
            parsed_args.base_commit, './mlir/utils/jenkins/static-checks/clang-format.ignore',
            parsed_args.ignore_external) and run_clang_tidy(
                parsed_args.base_commit, './mlir/utils/jenkins/static-checks/clang-tidy.ignore',
                parsed_args.ignore_external)):
        exit(1)
