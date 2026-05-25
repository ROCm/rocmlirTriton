# Building rocmlirTriton

This is the canonical reference for how rocmlirTriton is built. It covers:

- The two distinct build flows the project supports (standalone dev, MIGraphX
  via cget) and why we have both
- The design of the standalone build (this is the active redesign)
- The day-to-day selective-clean workflow that avoids accidentally throwing
  away the LLVM build
- The reasoning behind the choices, so the design can be revisited from a
  shared starting point rather than re-derived from scratch

The MIGraphX-specific cget contract has its own page, see
[`docs/migraphx_build.md`](migraphx_build.md). This document owns everything
else.

## Two consumers, one CMake graph

rocmlirTriton has two distinct consumers, each with very different ergonomic
needs:

| Consumer | How it builds | What it needs |
|---|---|---|
| Standalone development | `cmake -B build && ninja` | Fast iteration, incremental rebuilds, no toolchain manager in the loop |
| MIGraphX integration | `cget install ROCm/rocmlirTriton` | Reproducible, offline-friendly, dependencies declared as data |

Both consumers compile **the same** rocmlirTriton sources against **the same**
LLVM/MLIR commit and **the same** Triton commit. Pinning lives in
[`.gitmodules`](../.gitmodules) — each submodule's HEAD is the authoritative
reference, no separate hash file. The two flows differ only in *where* LLVM,
MLIR, and Triton come from at configure time — not in what gets compiled.

- The **standalone** flow vendors LLVM and Triton as git submodules under
  `external/`, and adds them into the same CMake project via
  `add_subdirectory`. One `cmake` invocation processes the entire dependency
  graph.
- The **MIGraphX** flow declares the same dependencies through cget
  ([`requirements.txt`](../requirements.txt) at the top level and another
  inside the Triton package). cget builds and installs each dependency in
  isolation, then rocmlirTriton consumes the install via `find_package`.

Both rely on the same `cmake/triton.cmake` glue (see [Source layout](#source-layout)),
which is written to detect whether LLVM/MLIR is being added in-tree or
imported from an install prefix and to behave accordingly.

## Design of the standalone build

The standalone build follows the same shape as
[upstream rocMLIR](https://github.com/ROCm/rocMLIR): a single monolithic CMake
project that pulls LLVM in via `add_subdirectory`. This is sometimes called
the *option A* layout in design discussions; the alternatives are documented
in [Design alternatives](#design-alternatives-considered-and-rejected) below.

### What it looks like

```
$ cd rocmlirTriton
$ cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
$ ninja -C build
```

That's the entire build. There is no separate "build LLVM first" step from
the user's point of view: the `add_subdirectory(external/llvm-project/llvm)`
call inside the top-level `CMakeLists.txt` brings LLVM, MLIR, and LLD into
the same CMake graph, and Ninja figures out the build order across all
targets.

### Source layout

| Path | Role |
|---|---|
| `external/llvm-project/` | LLVM/MLIR/LLD submodule (HEAD pinned in `.gitmodules`) |
| `external/triton/` | Triton submodule (HEAD pinned in `.gitmodules`) |
| `triton-patches/*.patch` | Historical local fixups for `external/triton/`; on a path to being upstreamed into the fork submodule (see [Patches on vendored sources](#patches-on-vendored-sources)) |
| `llvm-patches/*.patch` | Historical local fixups for `external/llvm-project/`; same direction |
| `cmake/submodules.cmake` | `rocmlir_ensure_submodules()` (runs `git submodule update --init --recursive` at configure time) |
| `cmake/triton.cmake` | The build glue that wires LLVM/MLIR/LLD/Triton into the rocmlirTriton project |
| `mlir/` | rocmlirTriton's own sources (Rock dialect, conversions, tools, tests) |

Build artifacts go under `build/` by default. With the monolithic layout
they land at:

| Artifact | Path |
|---|---|
| rocmlirTriton libraries/binaries | `build/lib/`, `build/bin/` |
| LLVM/MLIR/LLD libraries | `build/external/llvm-project/llvm/lib/` etc. |
| Build-tree `MLIRConfig.cmake` | `build/lib/cmake/mlir/MLIRConfig.cmake` |

That last row matters for the selective-clean workflow — see
[Selective-clean workflow](#selective-clean-workflow).

### How `find_package(MLIR)` is handled

Triton's `external/triton/CMakeLists.txt` calls
`find_package(MLIR REQUIRED CONFIG)` unconditionally. That works when LLVM is
imported from an install prefix (the cget flow), but it collides with the
in-tree LLVM targets created by `add_subdirectory(external/llvm-project/llvm)`:
CMake errors out the second time it sees `add_library(MLIRSupport IMPORTED)`.

To make Triton compatible with both flows, we keep a tiny patch in
`triton-patches/` that guards the call:

```cmake
if(NOT TARGET MLIRSupport)
  if(NOT MLIR_DIR)
    set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
  endif()
  if(NOT LLD_DIR)
    set(LLD_DIR ${LLVM_LIBRARY_DIR}/cmake/lld)
  endif()
  find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})
endif()
```

When LLVM is in-tree the `if` is false and Triton consumes the in-tree
targets directly; when LLVM is imported, the body runs as upstream Triton
expects. This patch is small, mechanical, and is on a path to upstream.

Because we skip `find_package(MLIR)` in the in-tree case, our top-level
`CMakeLists.txt` is responsible for populating the variables that Triton's
CMake reads afterwards (`MLIR_INCLUDE_DIRS`, `LLVM_INCLUDE_DIRS`,
`LLVM_LIBRARY_DIR`, `MLIR_CMAKE_DIR`, `LLVM_CMAKE_DIR`, `LLVM_LIBDIR_SUFFIX`).
These are set explicitly after the LLVM `add_subdirectory`, mirroring what
upstream rocMLIR does in its own `cmake/llvm-project.cmake`.

### Why option A and not "monolithic + separate LLVM build dir"

It is technically possible to keep a single `cmake` invocation while keeping
LLVM's build artifacts in a sticky `external/llvm-project/build/` directory
(by passing an absolute `binary_dir` to `add_subdirectory`). That layout is
called *option B* in the design discussion. We deliberately do not use it.
See [Design alternatives](#design-alternatives-considered-and-rejected) for
the full reasoning; the short version is:

- The build-tree `MLIRConfig.cmake` is still written to `${CMAKE_BINARY_DIR}`
  (= `build/lib/cmake/mlir/`) regardless of where LLVM's object files live,
  because MLIR's own CMakeLists hard-codes `${CMAKE_BINARY_DIR}` in
  [`mlir/cmake/modules/CMakeLists.txt`](../external/llvm-project/mlir/cmake/modules/CMakeLists.txt).
  So the "sticky" property is leaky: `rm -rf build` still loses the config
  file the rebuild needs to find.
- The sticky layout disallows parallel rocmlirTriton build configurations
  (e.g. `build-debug/` + `build-release/`) because both would write into the
  same `external/llvm-project/build/`.
- It diverges from upstream rocMLIR, so any future MLIR/LLVM CMake convention
  changes would land on us first instead of on rocMLIR.
- The motivating pain — preserving LLVM across a `rm -rf build` — is rare in
  practice and has cheaper mitigations (see the next section).

## Selective-clean workflow

The reflex `rm -rf build` is expensive in the monolithic layout: it deletes
LLVM, MLIR, and LLD, which together cost a full LLVM rebuild (~30 minutes
even on a fast workstation). Don't reach for it.

When CMake gets confused and you want to start over, use one of the
narrower options below depending on what you actually need to invalidate.

### Force a re-configure, keep all compiled artifacts

```bash
rm build/CMakeCache.txt
rm -rf build/CMakeFiles
cmake -B build  # re-run with whatever flags you used originally
ninja -C build
```

This wipes the CMake cache and the per-directory `CMakeFiles/`
bookkeeping, so the next `cmake -B build` re-runs the configure step from
scratch. Object files, libraries, and binaries (rocmlirTriton's *and*
LLVM's) survive untouched. Ninja will compare timestamps after the
re-configure and rebuild only what genuinely needs rebuilding.

This is the workhorse for "I changed CMake flags or my CMake glue and want a
clean configure" situations.

### Clean only rocmlirTriton's compiled output

```bash
rm -rf build/mlir build/lib build/bin
rm build/CMakeCache.txt && rm -rf build/CMakeFiles
cmake -B build
ninja -C build
```

Use this when you want to force-rebuild rocmlirTriton's own libraries and
binaries from scratch (for example, to chase down a stale-link-order bug)
without touching the LLVM subtree at `build/external/llvm-project/`. The
re-configure step regenerates the `MLIRConfig.cmake` file that lives at
`build/lib/cmake/mlir/MLIRConfig.cmake` — that file is produced by MLIR's
own `add_subdirectory` invocation, so it costs effectively nothing to
regenerate.

### Clean only the dependency graph, keep all object code

```bash
ninja -C build -t clean
```

Asks Ninja to delete the files it knows about — i.e. the outputs of every
target in the current build graph. This includes LLVM/MLIR targets, so it is
*not* a replacement for the first option above when your goal is "rebuild
fast." Use it only when you genuinely want to force a from-scratch link of
every target.

### When `rm -rf build` is actually the right call

Use the nuclear option deliberately, when one of these is true:

- You bumped the `external/llvm-project` submodule and want a fully clean
  LLVM build with the new sources (see [`docs/bump_triton_version.md`](bump_triton_version.md))
- You changed an LLVM-affecting flag (e.g. `LLVM_ENABLE_ASSERTIONS`,
  `BUILD_SHARED_LIBS`, `LLVM_TARGETS_TO_BUILD`) and want to be 100% certain
  no stale LLVM object files are reused
- CMake or Ninja are in an unrecoverable state and the selective options
  above don't unstick the build

If you find yourself reaching for `rm -rf build` more than once a week,
the right fix is probably a bug report against this build system rather
than a habit.

## Standalone build: current state

Option A (monolithic `cmake -B build && ninja`) is now wired into the build
system. [`cmake/triton.cmake`](../cmake/triton.cmake) implements a dual-path
discovery:

- **Imported flow** — used when an externally-built MLIR is available
  (cache `MLIR_DIR`, `MLIR_DIR` env var, `CMAKE_PREFIX_PATH`, or the legacy
  `external/llvm-project/build/` layout from `scripts/build-llvm.sh`).
  It calls `find_package(MLIR CONFIG)` as the cget flow has always done.
- **In-tree flow** — used when no external MLIR is found. It
  `add_subdirectory`s `external/llvm-project/llvm/` into the rocmlirTriton
  build, sets LLVM build flags (`LLVM_ENABLE_PROJECTS=mlir;lld`,
  `LLVM_INSTALL_UTILS=ON`, `LLVM_TARGETS_TO_BUILD=X86;AMDGPU`,
  `LLVM_ENABLE_ASSERTIONS=ON`) ahead of the call, and manually populates
  the `MLIR_*` / `LLVM_*` discovery variables so Triton's CMake (with the
  guard described below) finds what it needs.

The guard on Triton's `find_package(MLIR)` is applied in-place to
`external/triton/CMakeLists.txt`:

```cmake
if(NOT TARGET MLIRSupport)
  # ... original Triton find_package + module path + includes ...
endif()
```

This change lives directly in the submodule source so that no build-time
patch step is required for either flow. It is intended to be committed to
the Triton fork submodule alongside the other in-place fixups already
present there. The guard is forward+backward compatible: in the imported
flow the `MLIRSupport` target doesn't exist yet and the body runs exactly
as upstream Triton expects.

Two further symmetrical adjustments live in `mlir/` and `cmake/`:

- [`mlir/lib/Translation/TritonToHsaco/CMakeLists.txt`](../mlir/lib/Translation/TritonToHsaco/CMakeLists.txt)
  wraps its `find_package(LLD CONFIG QUIET)` with `if(TARGET lldELF)` so it
  consumes the in-tree LLD targets directly in the monolithic flow, and
  only calls `find_package` when LLD is imported.
- In the in-tree branch of `cmake/triton.cmake` we set
  `LLVM_INSTALL_TOOLCHAIN_ONLY=ON`. Standalone dev does not install MLIR
  (we consume it from the build tree), and turning this on also avoids
  CMake's `install(EXPORT MLIRTargets)` complaint that Triton's
  first-class in-tree targets (`TritonGPUTransforms`, etc.) are not in
  MLIR's export set when our Rock libraries link them.
- In the same in-tree branch we set `CMAKE_DISABLE_PRECOMPILE_HEADERS=ON`.
  LLVM's CMake (see `llvm/lib/Support/CMakeLists.txt` and the
  `PRECOMPILE_HEADERS` branch of `add_llvm_library`) wires every library
  that links `LLVMSupport` to reuse its precompiled header via
  `target_precompile_headers(... REUSE_FROM LLVMSupport)`. That
  propagates transitively to our Rock libraries and to
  `mlir/lib/ExecutionEngine/conv-validation-wrappers.cpp`. The PCH is
  built with LLVM's own flags (`-std=c++17`, no GNU extensions), but our
  targets compile with the CMake default of `-std=gnu++17`. Clang then
  refuses to load the PCH with `"GNU extensions was disabled in AST file
  ... but is currently enabled"`. The knob is the standard CMake variable
  `CMAKE_DISABLE_PRECOMPILE_HEADERS`, which
  `llvm/cmake/modules/HandleLLVMOptions.cmake` honors via
  `if(NOT DEFINED CMAKE_DISABLE_PRECOMPILE_HEADERS)`; setting it before
  `add_subdirectory` keeps LLVM's auto-detection from re-enabling PCH.
  Disabling PCH costs ~10–20 % of LLVM build time but is otherwise
  invisible. The cleaner long-term alternative would be to set
  `CMAKE_CXX_EXTENSIONS=OFF` project-wide (matching upstream rocMLIR and
  upstream LLVM) so the PCH and its consumers agree on the standard; this
  is intentionally deferred until the monolithic flow is fully operational
  on every developer machine.

### Known follow-ups

These items are tracked but intentionally out of scope for the dual-path
wiring above:

- **`llvm-patches/*.patch` are not yet applied in-tree.** The historical
  `scripts/build-llvm.sh` flow applies them at build time. For the
  monolithic in-tree flow to compile from a clean `external/llvm-project/`,
  apply them once with `for p in llvm-patches/*.patch; do git -C external/llvm-project apply "$p"; done`,
  or wait until they are committed in the LLVM fork submodule.
- **Drop `scripts/build-llvm.sh` and `scripts/dev-build.sh`.** These are
  retained today so that the imported flow (and any developer with an
  existing `external/llvm-project/build/`) keeps working unchanged.
  They should be removed once the monolithic in-tree flow is the
  documented standalone entry point and the LLVM fork has the patches
  baked in.
- **Latent missing `LINK_LIBS` exposed by `-Wl,-z,defs`.** LLVM's
  `HandleLLVMOptions.cmake` appends `-Wl,-z,defs` to
  `CMAKE_SHARED_LINKER_FLAGS` when building shared libraries on ELF, which
  forces every `.so` to resolve all its referenced symbols at link time.
  The monolithic in-tree flow triggers this (we `include(HandleLLVMOptions)`
  in [`mlir/CMakeLists.txt`](../mlir/CMakeLists.txt)); the legacy
  `find_package(MLIR)` flow did not, because importing an already-built
  MLIR does not re-run that module against a built tree. As a result, the
  monolithic build reveals pre-existing missing dependency declarations
  that the dynamic linker used to paper over at runtime:

  - `MLIRRockOps` invokes `rock::getAccType`, `rock::getMfmaVersion`,
    `rock::getWmmaVersion`, `rock::backwardDataKernelIds` — all defined
    in `MLIRRockUtility` — without listing it in `LINK_LIBS` (would
    introduce a CMake cycle since `MLIRRockUtility` `PUBLIC`-links
    `MLIRRockOps`).
  - `MLIRRockOps` uses `mlir::triton::AMD::TargetInfo` and
    `mlir::triton::AMD::deduceISAFamily` without linking
    `TritonAMDGPUToLLVM` (where both are defined).

  To unblock the monolithic flow without a large dep-graph refactor, we
  strip `-Wl,-z,defs` (and the matching `-Wl,-z,nodelete`) from
  `CMAKE_SHARED_LINKER_FLAGS` in `mlir/CMakeLists.txt` immediately after
  the `include(HandleLLVMOptions)` call. This matches what the develop
  branch was effectively doing. The proper fixes are to either declare
  the missing `LINK_LIBS` and accept the resulting cycle (ELF dynamic
  linking handles cross-library cycles fine), or to lift the `rock::*`
  helpers that `MLIRRockOps` consumes into `MLIRRockOps` itself so the
  IR layer never depends on the utility layer. Either fix should land
  before we re-enable `-Wl,-z,defs`.

The cget flow (and [`docs/migraphx_build.md`](migraphx_build.md)) is not
affected by any of these: `find_package(MLIR)` against `CMAKE_PREFIX_PATH`
continues to be the first thing the imported flow tries.

## Patches on vendored sources

We carry two local patch directories, both on a path to being **committed
in the fork submodules** so that any flow consuming the submodule sees the
patched state without a build-time `git apply` step.

- **`triton-patches/`** — historical fixups for `external/triton/`. Most of
  these are already applied in-place in the `external/triton/` submodule
  (the working tree shows them as `git status` modifications). They are
  intended to be committed in the Triton fork submodule. The
  `find_package(MLIR)` guard described in
  [Standalone build: current state](#standalone-build-current-state) is one
  of these.
- **`llvm-patches/`** — fixups for `external/llvm-project/`. Not yet
  applied in-place; today they are applied at build time by
  `scripts/build-llvm.sh`. The monolithic in-tree flow needs them applied
  to the submodule before the first build (see
  [Known follow-ups](#known-follow-ups)).

We avoid in-source CMake patches applied via `sed` in
`scripts/build-llvm.sh` where we can; the long-term direction is "everything
that needs to be patched is patched in the fork submodule."

## MIGraphX (cget) build

MIGraphX consumes rocmlirTriton through cget, which builds and installs each
dependency in isolation. The contract is declared in two
[`requirements.txt`](../requirements.txt) files (one in this repo, one in
the Triton package). See [`docs/migraphx_build.md`](migraphx_build.md) for
the full description, including:

- The cget dependency chain (`rocMLIR → Triton → TritonLLVM`)
- Which CMake flags belong on which `requirements.txt` line and why
- Environment hygiene (`LLVM_BUILD_DIR`, `MLIR_DIR`, `LLVM_SYSPATH`) that
  must be unset before invoking cget so previous standalone builds don't
  leak in

The cget flow does **not** use `add_subdirectory(external/llvm-project/llvm)`;
it consumes the installed `MLIRConfig.cmake` via `CMAKE_PREFIX_PATH`. The
fork between the two flows is in `cmake/triton.cmake`.

## Design alternatives considered and rejected

This section records the alternatives we considered while designing the
standalone build, so future revisits start from a shared baseline.

### Option B: monolithic + override `add_subdirectory` binary_dir

Override LLVM's binary dir to an absolute path inside the source tree:

```cmake
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project/llvm"
  "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project/build"
  EXCLUDE_FROM_ALL
)
```

LLVM's `.so` / `.a` / `mlir-tblgen` artifacts then land at
`external/llvm-project/build/` and survive `rm -rf build`. We rejected this
because:

- MLIR's CMake writes `MLIRConfig.cmake` to `${CMAKE_BINARY_DIR}` (top-level,
  = `build/`), not to its own `CMAKE_CURRENT_BINARY_DIR`. So `rm -rf build`
  still loses the config file; the rebuild has to regenerate it before any
  artifact reuse can kick in. The "best of both worlds" property is leaky.
- Multiple top-level rocmlirTriton build directories (`build-debug/`,
  `build-release/`) would all point at the same
  `external/llvm-project/build/` and clobber each other.
- It diverges from upstream rocMLIR's layout. Any future MLIR/LLVM CMake
  convention change lands on us first instead of on upstream.

### Option C: keep the current two-script flow indefinitely

Keep [`scripts/build-llvm.sh`](../scripts/build-llvm.sh) and
[`scripts/dev-build.sh`](../scripts/dev-build.sh) as the permanent standalone
entry points. Rejected because:

- It is a second build system to maintain alongside the cget contract.
- New contributors have to learn an extra step that upstream rocMLIR
  developers do not.
- The shell-script patch-injection logic in `build-llvm.sh` (the `sed`-based
  `MLIR_ENABLE_ROCM_RUNNER` / `LLVM_INSTALL_UTILS` rewrites and the
  `llvm-patches` hook) is fragile and is itself a candidate for removal once
  patches live in the submodule branches.

### CMake superbuild (`ExternalProject_Add` for LLVM)

Treat LLVM as an external project built at *build* time, with its own
binary dir, then consume the install via `find_package`. Rejected because:

- ExternalProject targets are not visible at the parent project's configure
  time, so our own rocmlirTriton libraries that link MLIR targets directly
  would need a second-stage configure — i.e. a superbuild orchestrator on
  top of CMake. That is a lot of complexity for marginal benefit.
- Upstream rocMLIR demonstrates the monolithic `add_subdirectory` approach
  works fine without a superbuild, so paying that complexity tax here would
  be unjustified.

## Reference: key files in the build system

| File | What it does |
|---|---|
| [`CMakeLists.txt`](../CMakeLists.txt) | Top-level project entry; sets options and includes the Triton/LLVM glue |
| [`cmake/triton.cmake`](../cmake/triton.cmake) | LLVM/MLIR discovery + `add_subdirectory(triton)` + helper functions |
| [`cmake/submodules.cmake`](../cmake/submodules.cmake) | `rocmlir_ensure_submodules()` (configure-time `git submodule update --init --recursive`) |
| [`requirements.txt`](../requirements.txt) | cget dependency line for MIGraphX |
| [`scripts/build-llvm.sh`](../scripts/build-llvm.sh) | (Transitional) builds LLVM into `external/llvm-project/build/` |
| [`scripts/dev-build.sh`](../scripts/dev-build.sh) | (Transitional) chains `build-llvm.sh` + the rocmlirTriton cmake/ninja invocation |
| [`triton-patches/`](../triton-patches) | Local fixups applied on top of the Triton submodule |
| [`llvm-patches/`](../llvm-patches) | Local fixups applied on top of the LLVM submodule |

The test entry point is `ninja check-rocmlir` (run from the build directory).
