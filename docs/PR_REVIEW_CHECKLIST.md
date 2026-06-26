# rocmlirTriton PR Review Checklist

This is the authoritative PR review checklist for rocmlirTriton.
Reviewers -- human and automated -- categorize findings against the
**Critical / Major / Minor** tiers below; a Critical finding here is one
of the bullets in the "Critical" section, a Major finding is one of the
bullets in "Major", etc.

When you contribute, run `git clang-format --diff origin/develop`
(or `upstream/develop` if you've forked and named the
`ROCm/rocmlirTriton` remote `upstream`) and self-review your diff
against this checklist. When you review, cite the specific bullet so
the author can look up the rationale.

## References

- [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html) -- the
  authoritative style guide; everything below derives from or extends it.
- [MLIR Developer Guide](https://mlir.llvm.org/getting_started/DeveloperGuide/)
  -- the one naming-convention deviation (`camelBack` for variables,
  parameters, and class members instead of LLVM's traditional `Capitalized`)
  lives there.
- `.clang-format` (LLVM base style) and `.clang-tidy` at the repo root --
  machine-enforced subset; the premerge `clang-format` job runs
  `git clang-format --diff origin/develop` (CI's checkout always names
  the `ROCm/rocmlirTriton` remote `origin`) and fails on any non-empty
  diff.
- Python helpers follow [`yapf`](../.style.yapf) and
  [`flake8`](../.flake8); format with `yapf -i <files>` and lint with
  `flake8 <files>` before committing.
- rocmlirTriton-specific review rules that don't fit the generic
  LLVM/MLIR tiers (Triton subtree updates, edits to the vendored
  `external/triton` / `external/llvm-project` trees, `rock::*`
  hardware-feature detection, bridge passes, fat-library + MIGraphX
  coordination, and the **rocMLIR back-port check** that flags
  shared-with-rocMLIR diffs) live in dedicated sections at the end of
  this document; apply them alongside the generic Critical / Major /
  Minor tiers.

## Critical (blocks merge)

- Unreleased hardware codenames, unannounced chip IDs, or NDA features in
  code, comments, commits, or docs. (rocmlirTriton is currently a private
  repo but treat it as if it could be open-sourced at any time; see the
  project's confidentiality policy.) It IS fine to reference unreleased
  `gfx*` IDs only when they are already mentioned upstream (in the
  vendored Triton subtree, the LLVM AMDGPU backend, or upstream rocMLIR).
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
- Breaking IR or C-API changes without documentation or a coordinated
  MIGraphX update.

## Major

- Redundant code, dead code, speculative abstractions, unnecessarily
  complex algorithms, or missed opportunities to use existing upstream
  LLVM/MLIR/Triton utilities instead of custom code.
- Raw `new`/`delete`; use MLIR allocation utilities, `std::unique_ptr`, or
  arena ownership.
- Inheritance where composition would do; curiously recurring template
  pattern (CRTP) only where MLIR/LLVM requires it.
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
- License header missing on a new `.cpp`/`.h`/`.py` file (SPDX
  `Apache-2.0 WITH LLVM-exception`; LLVM Project convention is no
  per-file copyright -- see the License-header reference below for the
  exact template). "We'll fix headers before going public" is not an
  acceptable plan; the headers must already be in place.
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

## rocmlirTriton-specific checks

rocmlirTriton lowers Rock dialect kernels through OpenAI Triton's
TTIR/TTGIR/LLIR pipeline to AMD GPU code. Several files and conventions
exist specifically to keep that integration safe and reproducible across
Triton subtree updates. Apply these checks **in addition to** the
generic Critical / Major / Minor tiers above; the severity of each rule
is documented inline below.

### Vendored Triton / LLVM subtrees (`external/triton`, `external/llvm-project`)

`external/triton` and `external/llvm-project` are vendored directly in
this repo via `git subtree`; downstream edits are committed directly on
top of the imported upstream trees. `triton-patches/*.patch` and
`llvm-patches/*.patch` are provenance and refresh aids for the next
upstream merge, not build-time inputs. Consequently:

- Editing files under `external/triton/` or `external/llvm-project/`
  directly is expected; do NOT flag the direct edit itself as a
  violation.
- **Major** -- a downstream change to the vendored trees lands WITHOUT a
  matching patch under `triton-patches/` or `llvm-patches/` and an entry
  in `triton-patches/triton-patch-content.txt` or
  `llvm-patches/llvm-patch-content.txt`. The patch must match the
  committed edit, and the PR description must either link the upstream
  issue/PR or explain why the change is a permanent fork.

### Triton / LLVM subtree updates (upstream bumps)

When a PR imports a new upstream Triton or LLVM revision into the
vendored subtrees, apply the review checks from
[`bump_triton_version.md`](bump_triton_version.md). Missing required bump
artifacts, such as replication-point audit notes,
`librockcompiler_deps.cmake` regeneration, or re-evaluation of the
downstream patches, should be reviewed as **Major** unless the PR
description explains why they do not apply.

### Hardware-feature detection (`rock::*` vs `triton::AMD::TargetInfo`)

Hardware-feature checks (whether the target supports a given MFMA
shape, dtype, LDS size, etc.) **must** go through `rock::*` helpers,
**not** through `triton::AMD::TargetInfo`. The `rock::*` helpers cover
the union of GPU architectures rocmlirTriton must support; using
`triton::AMD::TargetInfo` directly only sees what upstream Triton
tracks for its own backends and will silently miss arches Triton
hasn't taught about yet. **Major** finding on any new direct use of
`triton::AMD::TargetInfo` for a feature-detection check that has an
existing `rock::*` equivalent.

### Bridge passes between Rock and Triton

The Rock<->Triton bridge passes -- `RockToTTIRPass`,
`RockFuncToTritonFuncPass`, `RockSerializeHostFuncsPass`,
`TritonToHsacoPass`, the Triton-related code in
`rock::buildTritonPipeline` / `buildBackendPipeline` in
`Pipelines.cpp`, and `tritonUtils.cpp` -- are rocmlirTriton-specific.
Changes there must **not** be back-ported to rocMLIR (they appear in
the rocmlirTriton-only path list in the "rocMLIR back-port check"
section below) but they **do** need lit + E2E coverage like any other
pass change (already covered by the "New pass or op without positive
E2E coverage..." Major bullet above).

### Fat library + downstream MIGraphX

The vendored Triton subtree is wired into the `librockCompiler` fat library
that MIGraphX consumes. Any change that alters the public C-API
surface, IR ingest/emit shape, or default Triton-pipeline behavior is
a downstream coordination point with MIGraphX -- the PR description
must call this out; **Major** otherwise. (This is the rocmlirTriton-
specific application of the generic "Downstream MIGraphX impact"
Major bullet above; the Triton-pipeline angle is the addition.)

## rocMLIR back-port check

rocmlirTriton was forked from rocMLIR (private fork) and most of
`mlir/` is still shared between the two trees. For every change that
is **not** exclusively in the Triton-only paths listed below, ask:
*"does this fix or feature also need to land in `ROCm/rocMLIR`?"* A
missing back-port silently causes drift between the two trees and is
one of the most common review escapes.

### Likely needs a back-port (shared with rocMLIR)

A diff is "shared" if its file path matches any of:

- `mlir/lib/Dialect/Rock/` -- **except** the rocmlirTriton-only
  bridge-pass source files in `mlir/lib/Dialect/Rock/Transforms/`:
  `RockToTTIR.cpp`, `FuncToTritonFunc.cpp`, `SerializeHostFuncs.cpp`
  (and any future Triton-only restore/serialize pass added beside
  them).
- `mlir/lib/Dialect/MIGraphX/`.
- `mlir/lib/Conversion/MIGraphXTo*/`.
- `mlir/tools/{rocmlir-gen,rocmlir-driver,rocmlir-opt,rocmlir-tuning-driver,rocmlir-lsp-server}/`
  -- excluding any Triton-pipeline registrations inside those tools.
- `mlir/utils/performance/`, `mlir/utils/jenkins/static-checks/`.
- Generic `cmake/` helpers and root-CMake options that aren't
  Triton-specific.

Note: `mlir/lib/Analysis/` exists in rocMLIR (e.g.
`BufferDependencyAnalysis.cpp`) but has been refactored away in
rocmlirTriton. If a PR re-introduces that path locally, treat it as
shared and apply the back-port check; otherwise it cannot appear in a
rocmlirTriton diff.

### rocmlirTriton-only (no back-port needed)

- `external/triton/`, `external/llvm-project/`, `triton-patches/`,
  `llvm-patches/`, `cmake/triton.cmake`, `cmake.sh`.
- The bridge-pass source files in `mlir/lib/Dialect/Rock/Transforms/`:
  `RockToTTIR.cpp` (pass `RockToTTIRPass`), `FuncToTritonFunc.cpp`
  (pass `RockFuncToTritonFuncPass`), `SerializeHostFuncs.cpp` (pass
  `RockSerializeHostFuncsPass`); plus the Triton portion of
  `rock::buildTritonPipeline` / `buildBackendPipeline` in
  `mlir/lib/Dialect/Rock/Pipelines/Pipelines.cpp` and
  `mlir/lib/Dialect/Rock/utility/tritonUtils.cpp`.
- `mlir/lib/Translation/TritonToHsaco/TritonToHsaco.cpp` (pass
  `TritonToHsacoPass`).
- `mlir/tools/rocmlir-lib/librockcompiler_deps.cmake`,
  [`docs/bump_triton_version.md`](bump_triton_version.md),
  `mlir/utils/jenkins/Jenkinsfile.downstream`.

### Verdict

If the PR touches any of the "shared with rocMLIR" paths above, the
PR description must EITHER:

(a) link to a parallel `ROCm/rocMLIR` PR, OR
(b) include a one-line note explaining why the divergence is
    intentional (for example: "this fix depends on Triton-only
    changes and is not applicable upstream"), OR
(c) confirm the file no longer exists / has been refactored on the
    rocMLIR side (with a short justification).

If **none** of (a)/(b)/(c) is in the PR description, a reviewer
should raise a **Major** finding anchored to one of the touched
shared files. The body should list each shared path the PR modifies
and ask the author to add the back-port note or open the parallel
rocMLIR PR.

If the PR is exclusively in rocmlirTriton-only paths, no back-port
note is needed.

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
- License attribution line is exactly `Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.` (no per-file copyright -- the LLVM Project convention is that `LICENSE.TXT` is the single source).
- SPDX identifier is exactly `Apache-2.0 WITH LLVM-exception`.
