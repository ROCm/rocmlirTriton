# rocmlirTriton Coding Standards

This is the authoritative coding-standards checklist for rocmlirTriton.
Both human reviewers and the `/review-rocmlir-triton-pr` auto-review skill
categorize findings against the **Critical / Major / Minor** tiers below;
a Critical finding here is one of the bullets in the "Critical" section,
a Major finding is one of the bullets in "Major", etc.

When you contribute, run `git clang-format --diff origin/develop` and
self-review your diff against this checklist. When you review, cite the
specific bullet so the author can look up the rationale.

## References

- [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html) -- the
  authoritative style guide; everything below derives from or extends it.
- [MLIR Developer Guide](https://mlir.llvm.org/getting_started/DeveloperGuide/)
  -- the one naming-convention deviation (`camelBack` for variables,
  parameters, and class members instead of LLVM's traditional `Capitalized`)
  lives there.
- `.clang-format` (LLVM base style) and `.clang-tidy` at the repo root --
  machine-enforced subset; the premerge `clang-format` job runs
  `git clang-format --diff origin/develop` and fails on any non-empty diff.
- Python helpers follow [`yapf`](.style.yapf) and [`flake8`](.flake8); format
  with `yapf -i <files>` and lint with `flake8 <files>` before committing.
- For rocmlirTriton-specific addenda that go beyond generic LLVM/MLIR
  standards (the `external/triton/` direct-edit ban + Triton-submodule-bump
  audit, the `rock::*` hardware-feature-detection rule, the bridge-pass
  conventions, and the **rocMLIR back-port check**) see Steps 4 and 5 of
  `.claude/skills/review-rocmlir-triton-pr/SKILL.md`. Those rules are
  enforced by the same review process as the tiers below.

## Critical (blocks merge)

- Unreleased hardware codenames, unannounced chip IDs, or NDA features in
  code, comments, commits, or docs. (rocmlirTriton is currently a private
  repo but treat it as if it could be open-sourced at any time; see the
  project's confidentiality policy.) It IS fine to reference unreleased
  `gfx*` IDs only when they are already mentioned upstream (in the pinned
  Triton submodule, the LLVM AMDGPU backend, or upstream rocMLIR).
- C++ exceptions (`throw`, `try`/`catch`); use `LogicalResult` /
  `emitOpError` / `signalPassFailure` instead. (Triton-side code
  additionally compiles with `-fno-exceptions -fno-rtti -Werror`; respect
  those flags.)
- RTTI (`dynamic_cast`, `typeid`); use LLVM's `isa`/`cast`/`dyn_cast`.
- Magic sentinel values (`-1`, `nullptr`) to signal failure; use
  `FailureOr<>` instead.
- `#include <iostream>`; use LLVM's `raw_ostream`.
- `using namespace std` at file scope or in headers.
- Static constructors/destructors (global objects with non-trivial
  ctors/dtors).
- Committed temp/generated files: build artifacts, `*.pyc`, editor swap
  files, secrets, profiler output (`.rocprofv3/`, `att_dump/`,
  `*.pftrace`), plan / scratch files, tuning DBs that don't belong in the
  repo.
- **Direct edits to `external/triton/`**. The Triton submodule is consumed
  as a pinned upstream tree; local changes must be captured as
  `triton-patches/*.patch` applied on top by `scripts/build-llvm.sh`. A PR
  that modifies files under `external/triton/` directly (without a
  corresponding bump of the submodule SHA) is wrong by construction.
- Breaking IR or C-API changes without documentation or a coordinated
  MIGraphX update.

## Major

- DRY/YAGNI/KISS violations: redundant code, dead code, unnecessarily
  complex algorithms, opportunities to use existing upstream
  LLVM/MLIR/Triton utilities instead of custom code.
- Raw `new`/`delete`; use MLIR allocation utilities, `std::unique_ptr`, or
  arena ownership.
- Inheritance where composition would do; CRTP only where MLIR/LLVM
  requires it.
- `std::string`/`std::vector` for non-owning parameters where
  `StringRef`/`ArrayRef`/`MutableArrayRef` would suffice.
- `std::vector` for small local collections where `SmallVector` is
  preferred.
- `std::map`/`std::unordered_map` where `llvm::DenseMap` is preferred.
- Missing `assert` with descriptive message on non-trivial preconditions;
  use `llvm_unreachable` for impossible paths (not `assert(false)`).
- C-style casts; use `static_cast`/`const_cast`.
- Visibility leaks: file-local helpers without `static` or anonymous
  namespace.
- `default:` label in a switch over an enum that already covers every case
  (defeats `-Wswitch`). Important when consuming Triton's `ISAFamily`
  enum -- a `default:` there silently hides new ISA families added in a
  Triton bump.
- `std::sort` instead of `llvm::sort` -- LLVM coding standard. `llvm::sort`
  wraps `std::sort` and, under `EXPENSIVE_CHECKS` builds, deterministically
  shuffles the input first to surface order-dependent bugs that would
  otherwise hide behind a libc++/libstdc++ implementation that happens to
  preserve input order. (Note: neither call is *stable*; if equal elements
  must keep their relative order, the fix is `llvm::stable_sort`, not
  `llvm::sort`. Don't suggest `llvm::sort` as a "stability" fix.)
- Naming: classes not `CamelCase`, functions/vars not `camelBack`.
- New op without `hasVerifier = 1` and a `verify()` implementation.
- New pass or op without positive E2E coverage and both positive and
  negative Lit tests with FileCheck.
- New optimization without a FileCheck test asserting the expected IR is
  produced.
- `LogicalResult` returned but ignored (not checked with `failed(...)`).
- `librockcompiler_deps.cmake` not updated when fat-library dependencies
  change (especially after Triton bumps -- this file is regenerated as
  part of the `triton-bump` workflow and is the contract the downstream
  MIGraphX `librockCompiler` fat library depends on).
- License header missing or wrong year on a new `.cpp`/`.h`/`.py` file
  (SPDX `Apache-2.0 WITH LLVM-exception` -- "we'll fix headers before
  going public" is not an acceptable plan; the headers must already be
  in place).
- `TODO` without an issue reference (`TODO(#issue-number)`).
- Architecture coverage: a new op/pass that should work on multiple GPU
  arch families (CDNA `gfx9xx` -- e.g. gfx908, gfx942, gfx950 -- and RDNA
  `gfx10xx`/`gfx11xx`/`gfx12xx`) is implemented for only one. Use the
  `mlir/test/e2e/` `.toml`/`.cfg` corpus as the reference for which arches
  are exercised in CI before approving "this is single-arch on purpose".
- Data type coverage: an op that should support multiple dtypes (f16/bf16/
  f32, fp8 variants `f8E4M3*`/`f8E5M2`, i8/i4) silently falls through for
  unhandled dtypes instead of returning `emitOpError`.
- Fusion-related changes that lack tests in `mlir/test/fusion/`,
  `mlir/test/fusion/pr-e2e/`, or (for end-to-end fixtures)
  `mlir/test/e2e/`. The legacy root-level `tests.sh` driver and the
  `fusion_*_with_host.mlir` top-level fixtures it ran are GONE -- they
  were migrated into LIT under `mlir/test/e2e/` (commit AIROCMLIR-760,
  PR #220) and `tests.sh` itself was removed in PR #243. A PR that adds a
  new top-level `fusion_*_with_host.mlir` at the repo root, or invokes
  `bash tests.sh`, is using a stale convention.
- Custom CMake targets that bypass `add_rocmlir_dialect_library` /
  `add_rocmlir_conversion_library` / `add_rocmlir_tool` /
  `add_rocmlir_unittest` (the CMake `project()` name is still `rocMLIR` so
  these helpers retain the `rocmlir-*` / `add_rocmlir_*` names).
- Downstream MIGraphX impact: changes to public IR, C API, or
  `librockcompiler` that need coordinated updates and aren't called out in
  the PR description.

## Minor

- Include order wrong: should be main module header, then local/private,
  then MLIR/LLVM, then stdlib (each group sorted lexicographically).
- Header lacks self-contained guards.
- Comments not English prose with proper capitalization; missing `///`
  Doxygen on public APIs.
- Missing early returns; `else` after `return`.
- Postincrement (`i++`) where preincrement (`++i`) would do.
- `for (auto it = c.begin(); it != c.end(); ++it)` re-evaluating `end()`;
  prefer range-based for.
- Braces around single-statement bodies (omit them); missing braces around
  multi-statement bodies.
- `auto` where the type isn't obvious; missing `auto &` / `auto *` causing
  copies.
- `inline` on a function defined inside the class body (already implicit).
- Spaces before parentheses in function calls (allowed only in control
  flow).
- File missing trailing newline; trailing whitespace.
- `LLVM_DEBUG` block missing `#define DEBUG_TYPE "rock-..."` at the top of
  the file.
- Lit test missing `// RUN:` line, `-verify-diagnostics`, or `FileCheck`
  prefix coverage.
- New `.toml` E2E config not registered in `mlir/test/e2e/CMakeLists.txt`.

## License-header reference (verify on every new file)

C++/header files (`.cpp`, `.h`):

```
//===- FileName.cpp - Brief description ----------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
```

Python files (`.py`):

```
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
```

Checklist on every new file:

- Header is present.
- Copyright year matches the current year (not copied from older files).
- SPDX identifier is exactly `Apache-2.0 WITH LLVM-exception`.
