# Contributing to rocmlirTriton

Thanks for your interest in contributing.

## Reporting Issues

Use [GitHub Issues](../../issues) to report bugs or request features. Include a clear description, reproduction steps, and your environment: OS, ROCm version, GPU architecture (e.g. `gfx942`, `gfx950`, `gfx1100`).

## Pull Request Workflow

1. Clone the repository and create a branch from `develop`:
   ```bash
   git clone https://github.com/ROCm/rocmlirTriton.git
   cd rocmlirTriton
   git checkout -b feature/short-description
   ```
2. Make your change. Add tests under `mlir/test/` and update docs if behavior changes.
3. Format your C/C++ changes with `clang-format` (uses the repo's `.clang-format`):
   ```bash
   git clang-format .
   ```
4. Build and run the test suite locally:
   ```bash
   bash cmake.sh
   cd build && ninja check-rocmlir
   ```
5. Open a PR against `develop`. Describe *what* changed and *why*; link any related issue.
6. Ensure CI passes and request review from the relevant [CODEOWNERS](.github/CODEOWNERS).

By opening a PR, you agree your contribution is licensed under the terms in [LICENSE](LICENSE).

## Coding Standards

The codebase follows the [MLIR coding conventions](https://mlir.llvm.org/getting_started/DeveloperGuide/), which inherit the [LLVM coding standard](https://llvm.org/docs/CodingStandards.html) with one notable difference: variables, parameters, and class members use `camelBack` (lowerCamelCase) instead of LLVM's traditional `Capitalized` form. Functions remain `camelBack`; classes, enums, and unions are `PascalCase`.

Style is enforced via `.clang-format` (LLVM base style) and `.clang-tidy` at the repo root. Step 3 of the workflow above (`git clang-format .`) is the minimum expectation before opening a PR.

Python helpers (under `scripts/` and `mlir/utils/performance/`) follow [`yapf`](.style.yapf) and [`flake8`](.flake8). Format with `yapf -i <files>` and lint with `flake8 <files>` before committing changes there.

## External Contributors

This repo is part of the ROCm org. Non-AMD contributors need admin approval before being added as collaborators and must follow AMD's [open-source contribution guidelines](https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md).

## Security

For security issues, do **not** open a public issue -- see [SECURITY.md](SECURITY.md).
