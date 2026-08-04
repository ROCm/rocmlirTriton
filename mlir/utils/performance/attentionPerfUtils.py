# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Shared helpers for portable attention performance comparisons."""

import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def normalize_option(option: str) -> str:
    """Normalize rocmlir-gen option spelling for reliable comparisons."""
    return option.lstrip("-").replace("-", "_").lower()


def parse_config(config: str) -> Dict[str, str]:
    """Parse a perf-runner config without importing the GPU-dependent runner."""
    tokens = shlex.split(config)
    result: Dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("-"):
            raise ValueError(f"Expected an option at token {index}: {token}")
        if "=" in token:
            option, value = token.split("=", 1)
        else:
            if index + 1 >= len(tokens):
                raise ValueError(f"Missing value for option: {token}")
            option = token
            value = tokens[index + 1]
            index += 1
        key = normalize_option(option)
        if key in result:
            raise ValueError(f"Duplicate option: {option}")
        result[key] = value
        index += 1
    return result


def canonical_config(config: str) -> str:
    """Return a stable, order-independent identity for a config."""
    parsed = parse_config(config)
    return " ".join(f"{key}={shlex.quote(parsed[key])}" for key in sorted(parsed))


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in ("1", "true"):
        return True
    if lowered in ("0", "false"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def read_config_lines(path: Path) -> List[Tuple[int, str]]:
    """Read non-empty, non-comment config lines with one-based line numbers."""
    result = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                result.append((line_number, stripped))
    return result


def atomic_write_text(path: Path, contents: str) -> None:
    """Replace a text file atomically, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(contents)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Mapping) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_lines(path: Path, lines: Iterable[str]) -> None:
    materialized = list(lines)
    suffix = "\n" if materialized else ""
    atomic_write_text(path, "\n".join(materialized) + suffix)


def option_value(options: Mapping[str, str],
                 name: str,
                 default: Optional[str] = None) -> Optional[str]:
    return options.get(normalize_option(name), default)


def remove_flag(argv: Sequence[str], flag: str) -> List[str]:
    return [argument for argument in argv if argument != flag]
