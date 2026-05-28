"""Shared path helpers for kpack-grouped GEMM problem config files."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KPACK_GROUPS = (4, 8, 16)


def normalize_arch_token(arch: str) -> str:
    """Return the short arch token used in repo-local filenames.

    Examples:
      gfx942 -> 942
      gfx942:sramecc+:xnack- -> 942
      90a -> 90a
    """
    token = arch.strip().lower()
    if token.startswith("gfx"):
        token = token[3:]
    token = token.split(":", 1)[0]
    if not token:
        raise ValueError(f"invalid arch: {arch!r}")
    return token


def kpack_report_path(arch: str, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / f"gemm-kpack-{normalize_arch_token(arch)}.txt"


def kpack_group_path(arch: str, kpack: int, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / f"{normalize_arch_token(arch)}-gemm-kpack-{kpack}"


def kpack_group_paths(
    arch: str,
    kpacks: Sequence[int] = DEFAULT_KPACK_GROUPS,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    return [kpack_group_path(arch, kpack, repo_root) for kpack in kpacks]


def kpack_sample_output_path(arch: str, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / f"{normalize_arch_token(arch)}-gemm-kpack-sample.txt"
