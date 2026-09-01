# Guide: Bumping the Triton Version in rocmlirTriton

## Overview

rocmlirTriton vendors Triton under `external/triton/` and the
Triton-pinned LLVM/MLIR tree under `external/llvm-project/`, both via
`git subtree`. Several Triton Python functions are replicated in C++
within this project, so every upstream Triton/LLVM import must reconcile
those C++ implementations and the downstream patch records under
`triton-patches/` and `llvm-patches/`.

Note that we want to use Triton from https://github.com/triton-lang/triton-windows, as that includes Windows support not included in https://github.com/triton-lang/triton

## Step 1: Import new upstream revisions

Use the Triton-bump commit-message convention for upstream subtree
imports, such as `[TRITON-BUMP]` or a Jira-tracked
`[AIROCMLIR-NNN]` prefix. The `[EXTERNAL]` prefix is reserved for
downstream vendored-tree patch commits that modify `external/triton/`
or `external/llvm-project/` outside an upstream import / bump commit.

### 1.1 Record the current repository commit

```bash
export OLD_REPO=$(git rev-parse HEAD)
```
### 1.2 Drop the previously applied patches

Before importing new upstream revisions, reverse-apply every downstream
patch so the vendored trees match clean upstream again. This turns the
subsequent `git subtree pull` into a clean upstream-to-upstream merge
instead of one that has to reconcile our local edits.

Reverse-apply every `*.patch` in each directory from its subtree root.
Order matters for stacked patches (ones that touch the same file build on
each other), so reverse newest-first; `sort -r` does this for our
PR-numbered patch names (e.g. `patch10498` is reversed before the
`patch10450` it stacks on):

```bash
# LLVM patches -> external/llvm-project
cd external/llvm-project
for p in $(ls ../../llvm-patches/*.patch | sort -r); do
  patch -p1 -R --force < "$p"
done
cd ../..

# Triton patches -> external/triton
cd external/triton
for p in $(ls ../../triton-patches/*.patch | sort -r); do
  patch -p1 -R --force < "$p"
done
cd ../..
```

### 1.3 Pull the new versions

Import the desired upstream Triton and LLVM revisions into their vendored
subtrees from the repository root. Replace `<triton-ref>` with the latest
commit hash from https://github.com/triton-lang/triton-windows/tree/main-windows,
and replace `<llvm_ref>` with the commit hash in llvm-info.json on `main-windows`:

Each `git subtree pull --squash` creates two commits: a synthetic
`Squashed '<prefix>/' changes from <old>..<new>` commit (the merge's
second parent, whose message carries a long per-upstream-commit list plus
the `git-subtree-dir:` / `git-subtree-split:` trailers) and a merge commit
that joins it into the branch. Collapse each pair into a single commit
with a clean message, keeping only the two `git-subtree-*` trailers (drop
the bullet list):

```bash
# Triton
git subtree pull --prefix=external/triton \
  https://github.com/triton-lang/triton-windows.git <triton-ref> --squash
split=$(git log -1 --format=%b HEAD^2 | sed -n 's/^git-subtree-split: //p')
git reset --soft HEAD^1
git commit -m "Bump external/triton to ${split:0:12}" \
  -m "git-subtree-dir: external/triton
git-subtree-split: $split"

# LLVM
git subtree pull --prefix=external/llvm-project \
  https://github.com/llvm/llvm-project.git <llvm-ref> --squash
split=$(git log -1 --format=%b HEAD^2 | sed -n 's/^git-subtree-split: //p')
git reset --soft HEAD^1
git commit -m "Bump external/llvm-project to ${split:0:12}" \
  -m "git-subtree-dir: external/llvm-project
git-subtree-split: $split"
```

Note: Always use `--squash`. The `triton-windows` `main-windows` branch is
rebased onto upstream Triton on every release, which rewrites its commit
SHAs. A non-squash `git subtree pull` relies on shared commit ancestry, so a
rebased upstream makes the entire branch look like new work and produces
spurious conflicts (or `fatal: refusing to merge unrelated histories`).
`--squash` sidesteps this.

### 1.4 Clean the git history

Step 1.2 has to be its own commit, because `git subtree pull` refuses to
run with a dirty working tree. Once the bumps are in, the history looks
like:

```
<OLD_REPO>
Remove previously applied patches
Bump external/triton to <triton-sha>
Bump external/llvm-project to <llvm-sha>
```

We don't want to carry the additional "Remove previously applied
patches" commit, so we fold it into the bump with an interactive rebase:

```bash
git rebase -i "$OLD_REPO"
```

In the todo list, mark the Triton bump as `squash` so it merges into the
patch-removal commit directly above it:

```
pick   <hash> Remove previously applied patches
squash <hash> Bump external/triton to <triton-sha>
pick   <hash> Bump external/llvm-project to <llvm-sha>
```

When the combined-message editor opens, keep the `Bump external/triton to
<triton-sha>` subject and its `git-subtree-dir:` / `git-subtree-split:`
trailers, and delete everything else. The result is a single commit that
both bumps Triton and drops the now-obsolete patches.

Capture the tip after any folding, since the rebase rewrites hashes:

```bash
export NEW_REPO=$(git rev-parse HEAD)
```

## Step 2: Reconcile downstream patch records and re-apply the survivors

We keep downstream patch records under `triton-patches/` and
`llvm-patches/` so the next upstream import can re-apply or reconcile
local divergence. After Step 1 the vendored trees are at clean upstream
of the new revision (Step 1.2 reverse-applied every patch), so a patch's
change being present in the tree right now means it landed upstream.

Re-evaluate every patch and sort it into one of three buckets: drop
(now upstream), apply (still needed, applies cleanly), or update
(still needed but upstream context drifted, so the hunks must be refreshed
first). Keep the human-readable patch indexes in sync at the same time:
`triton-patches/triton-patch-content.txt` and
`llvm-patches/llvm-patch-content.txt` should drop removed patches, rename
survivors, and update the "drop this patch" guidance for the new pins.

### 2.1 Classify each patch

For each patch, check the files that are modified and do:

```bash
cd external/triton
git diff ${OLD_REPO}..${NEW_REPO} -- FILE
```

Compare the upstream changes with the corresponding `.patch` file.

If the changes are already in `external/triton`, then remove the `.patch` file
and its entry in the matching `*-patch-content.txt` index.
If they are still needed, then update the .patch contents accordingly.

### 2.2 Apply the survivors to the vendored tree

The records are not applied at build time; the edits must be committed
into the vendored tree. Apply every still required patch from its
subtree root and confirm there are no rejects:

```bash
( cd external/triton      && patch -p1 < ../../triton-patches/PATCH.patch )
( cd external/llvm-project && patch -p1 < ../../llvm-patches/PATCH.patch )
```

Offsets and small fuzz are expected (upstream shifted the surrounding
lines); rejects are not.

## Step 3: Rebuild LLVM/MLIR

The LLVM version must remain compatible with the imported Triton tree
(`external/triton/cmake/llvm-hash.txt` is a useful cross-check). Build
from the vendored trees:

```bash
bash cmake.sh
```

Patch files are not applied during configure or build; they are records
of downstream cherry-picks that must be kept in sync with the committed
vendored trees.

## Step 4: Analyze Upstream Changes

Generate a diff between `OLD_REPO` and `NEW_REPO` for the key files that
need synchronization. Those are:

```bash
for f in \
    third_party/amd/backend/compiler.py \
    third_party/amd/backend/driver.c \
    third_party/amd/python/triton_amd.cc \
    python/src/llvm.cc \
    third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp \
    third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h \
    CMakeLists.txt \
    python/build_helpers.py; do
  git diff "$OLD_REPO..$NEW_REPO" --function-context -- "external/triton/$f" > "$(basename "$f").diff"
done
```

## Step 5: Synchronize C++ Implementations

The following tables map Python functions to their C++ equivalents. Each must be reviewed and updated when upstream changes.

### 5.1 Pipeline Functions (from `compiler.py`)

| Python Function | C++ Location |
|----------------|--------------|
| `make_ttir()` | `mlir/lib/Dialect/Rock/Pipelines/Pipelines.cpp` |
| `make_ttgir()` | `mlir/lib/Dialect/Rock/Pipelines/Pipelines.cpp` |
| `make_llir()` Part 1 | `mlir/lib/Dialect/Rock/Pipelines/Pipelines.cpp` |
| `make_llir()` Part 2 + `make_amdgcn()` + `make_hsaco()` | `mlir/lib/Translation/TritonToHsaco/TritonToHsaco.cpp` |

The `TRITON` CHECK prefix in `mlir/test/rocmlir-driver/pipelines.mlir` pins the exact ordering of every pass added by `rock::buildTritonPipeline` (i.e. the three functions above).

A Triton bump that drops, renames, reorders, or changes the options on any pass in `makeTTIR` / `makeTTGIR` / `makeLLIR` will fail it. That is the desired behaviour: it forces an explicit review of the new pass ordering against upstream Triton's `compiler.py`. If the new behaviour is correct, regenerate the expected output and update the CHECK lines:

If a Triton bump introduces a new arch-conditional branch in `Pipelines.cpp` that gfx942 doesn't exercise, prefer adding a second prefix (e.g. `TRITON_GFX1250`) for the relevant arch over weakening the existing one.

##### Understanding Pass Bindings (from `triton_amd.cc`)

The file `external/triton/third_party/amd/python/triton_amd.cc` contains the Python bindings for Triton passes. When `compiler.py` adds a new pass call, check `triton_amd.cc` to find the actual C++ pass creation function.

### 5.2 LLVM Functions (from `llvm.cc`)

| Python/C++ Function in Triton | C++ Location in rocmlirTriton |
|------------------------------|-------------------------------|
| `init_targets()` | `TritonToHsaco.cpp::initializeLLVMTargets()` |
| `createTargetMachine()` | `TritonToHsaco.cpp::createTargetMachine()` |
| `optimize_module()` | `TritonToHsaco.cpp::optimizeModule()` |

### 5.3 Triton Utility Functions (from `AccelerateAMDMatmul.cpp`)

All Triton-internal helper functions that we replicate are centralized in a
single module for easy updating:

| Triton Function | C++ Location in rocmlirTriton |
|----------------|-------------------------------|
| `getMfmaVersion()` | `mlir/lib/Dialect/Rock/utility/tritonUtils.cpp` |
| `getWmmaVersion()` | `mlir/lib/Dialect/Rock/utility/tritonUtils.cpp` |
| `mlirTypeToScaledElemType()` | `mlir/lib/Dialect/Rock/utility/tritonUtils.cpp` (as `mlirTypeToScaleDotElemType`, extended with BF16/FP16) |

Header: `mlir/include/mlir/Dialect/Rock/utility/tritonUtils.h`

If there are any new architecture not handled by our rocmlirTriton functions we should see warnings/errors because the switch would not be handling all cases.

We also carry a **downstream patch** to this file: `triton-patches/patch-wmma-preserve-discardable-attrs.patch`. It adds a `copyDiscardableAttrs()` helper that the accelerate-matmul rewrite patterns call when creating the new (WMMA/MFMA/scaled) dot, so that the `rock.o_transposed` metadata set by `rock-add-triton-metadata` survives onto the rewritten dot. `mlir/lib/Dialect/Rock/Transforms/SetMatmulOutputTranspose.cpp` reads that metadata back via `getDiscardableAttr(rock::OTransposedAttr::getNameStr())`, so dropping the patch silently breaks `rock-set-matmul-output-transpose` (no compile error, just lost output-transpose tuning).

Because this is a downstream patch, the `AccelerateAMDMatmul.cpp.diff` generated in Step 4 will show `copyDiscardableAttrs` and its call sites as *removed* (our patched old tree vs. pristine new upstream) — that is expected and is the signal to **re-apply the patch**, not a deletion to accept. After a bump, confirm `copyDiscardableAttrs` is present in the vendored `AccelerateAMDMatmul.cpp` with a call at every dot-creation site, and that `accelerate-matmul-preserve-rock-metadata*.mlir` / `set-matmul-output-transpose.mlir` still pass.

### 5.3.1 Kernel Launch Wrapper (from `driver.c`)

The tuning driver has a local launch helper that mirrors Triton's AMD backend
launcher:

| Triton Function | C++ Location in rocmlirTriton |
|----------------|-------------------------------|
| `_launch()` in `external/triton/third_party/amd/backend/driver.c` | `mlir/tools/rocmlir-tuning-driver/rocmlir-tuning-driver.cpp` (as `launchKernel()`) |

Review this mapping on every Triton bump, especially changes to cluster launch
attributes, cooperative launch handling, and `hipDrvLaunchKernelEx` usage. The
helper intentionally lives in the tuning driver, not `tritonUtils.cpp`, so the
shared Rock libraries do not pick up a HIP runtime dependency.

**Example mapping:**

Python call in `compiler.py`:
```python
amd.passes.ttgpuir.add_move_up_prologue_loads(pm)
```

Definition in `triton_amd.cc`:
```cpp
m.def("add_move_up_prologue_loads", [](mlir::PassManager &pm) {
  pm.addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUMoveUpPrologueLoads());
});
```

Corresponding C++ in `Pipelines.cpp`:
```cpp
pm->addNestedPass<mlir::triton::FuncOp>(
    mlir::createTritonAMDGPUMoveUpPrologueLoads());
```

### 5.3.2 KV-cache attention LLVM workaround

`GridwiseAttnToBlockwise.cpp` clamps the KV-cache N-loop trip count to the
static K/V block count. This is a workaround for an LLVM AMDGPU raw-buffer
bounds-checking bug that can make an out-of-contract `lastValidKVIndex` read
past the K/V allocation. The LLVM issue is tracked by ROCM-28757.

On every LLVM bump, check whether the new pinned LLVM revision contains the
upstream fix and whether the workaround is still necessary. Do not remove the
clamp based only on the revision change: also verify the behavior with
`gridwise-attention-kvcache-clamp.mlir` and
`mixr-attention-kvcache.mlir`.

### 5.4 Mirrored Enums / Attributes (from `TritonAttrDefs.td`)

Some Triton enums are hand-replicated in the Rock dialect so we can carry the
value through Rock IR (and map it back onto Triton when lowering) without taking
a TableGen dependency on the Triton dialect. These are **manual copies** and
will silently drift if upstream changes them, so diff the source on every bump:

| Rock copy | Upstream source | What to check |
|-----------|-----------------|---------------|
| `CacheModifier` / `CacheModifierAttr` in `mlir/include/mlir/Dialect/Rock/IR/RockAttrDefs.td` | `TT_CacheModifierAttr` in `external/triton/include/triton/Dialect/Triton/IR/TritonAttrDefs.td` | Names **and** integer values must match one-to-one (currently `none=1, ca=2, cg=3, wb=4, cs=5, wt=6, cv=7`). The Rock->Triton lowering relies on the integer values lining up. |

```bash
git diff "$OLD_REPO..$NEW_REPO" -- external/triton/include/triton/Dialect/Triton/IR/TritonAttrDefs.td
```

If upstream adds, renames, or renumbers a `CacheModifier` case, update the Rock
copy to match. If a new case is added, also extend `verifyLoadCacheModifier()`
in `mlir/lib/Dialect/Rock/IR/RockDialect.cpp` (the `switch` is exhaustive and
classifies each modifier as load-legal or store-only) and the
`getNameForCacheModifier()` helper.

### 5.4.1 Mirrored constants (from Triton IR traits)

Some Triton compile-time constants are hand-copied into rocMLIR so the tuning /
lowering logic can reason about a Triton limit without taking a link/header
dependency on the Triton dialect. Like the mirrored enums above, these are
**manual copies** that will silently drift if upstream changes the value, so
diff the source on every bump:

| Rock copy | Upstream source | What to check |
|-----------|-----------------|---------------|
| `kTritonMaxTensorNumElements` in `mlir/lib/Dialect/Rock/Tuning/RockTuningImpl.cpp` | `maxTensorNumElements` in `external/triton/include/triton/Dialect/Triton/IR/Traits.h` | Integer value must match exactly (currently `1048576`). It gates the tuning space (`exceedsTritonTensorCap`) against Triton's per-tensor element cap enforced by `verifyTensorSize`; if it drifts high, tuning emits configs that fail Triton verification, and if it drifts low, valid configs get dropped. |

```bash
git diff "$OLD_REPO..$NEW_REPO" -- external/triton/include/triton/Dialect/Triton/IR/Traits.h
```

### 5.5 Architecture Database (`AmdArchDb.cpp`)

`mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp` maps AMD GPU architectures to hardware
capabilities using Triton APIs (`ISAFamily`, `MfmaIntrinsic::selectFor`,
`WmmaIntrinsic::selectFor`, `TargetInfo`). When a new architecture is added
upstream (e.g. a hypothetical RDNA5 / CDNA5), this file needs review:

| Function | What to check |
|----------|---------------|
| `getMatrixAccelKind()` | Does the new arch support MFMA, WMMA, or scaled variants? Update the selection logic (version thresholds, `isF8F6F4`, `isScaledWmmaType`). |
| `getMaxNumChiplets()` | Update if the new arch has multi-chiplet GPUs. |
| `getMinNumCU()` | Add the new `ISAFamily` case with the minimum CU count. |
| `getMaxWavesPerEU()` | Add the new `ISAFamily` case with the correct occupancy limit. |
| `getWaveSize()` / `getLDSSize()` | These delegate to `TargetInfo`, so they should work automatically if Triton adds the arch. Verify. |
| `supportsTDM()` | Delegates to `TargetInfo`. Verify it returns the correct value for the new arch. |
| `isCDNA()` / `isRDNA()` | Delegate to `triton::amdgpu::isCDNA` / `isRDNA`, and `isCDNA()` picks the tuning space. Re-check the classification on every bump: a family moving between the two switches resizes the space with no build error. |

Also check that `tritonUtils.cpp::getMfmaVersion()` and
`tritonUtils.cpp::getWmmaVersion()` handle the new `ISAFamily` / chip string.
Additionally, `mlir/test/common_utils/amd_arch_db/binding.cpp` will need to have
it's `py::enum_<ISAFamily>(...).value(...)` enum updated as well. That module is
also how the performance scripts reach `AmdArchDb.cpp`, so a new `rock` arch
predicate that Python needs (`is_cdna` / `is_rdna`, say) has to be exported
there too -- see section 5.6.

When a new ISA family (not just a new chip in an existing family) gains support,
add a representative chip to `DEFAULT_ARCHES` in
`mlir/utils/performance/generateLDSBlacklist.py`. The LDS-overflow blacklist is
keyed per ISA family because `ttg.shared` depends on the matmul lowering (MFMA
vs WMMA vs non-accel), so a new family with no entry gets *no* blacklist and its
overflowing tiles are never pruned during tuning. Existing chips that fall into
an already-listed family are covered automatically by the lookup-time fallback
(see `LdsBlacklist.cpp`), so they do not need a new entry. After editing the
list, regenerate the `.inc` (Step 11).

**How to detect needed changes:** Triton uses exhaustive `switch` statements over
`ISAFamily`. If a new variant is added, our switches (which use `default:`) will
silently return a fallback value. The `ISAFamily` enum is defined in
`TargetFeatures.h` (it moved out of the now-removed
`TritonAMDGPUToLLVM/TargetUtils.h`); diff it for new entries, along with the
`.cpp` that holds the `isCDNA` / `isRDNA` switch bodies (a reclassification does
not touch the header):

```bash
git diff "$OLD_REPO..$NEW_REPO" -- \
  external/triton/third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h \
  external/triton/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp
```

### 5.6 Hardware feature detection
Python code will usually add certain passes only if hardware supports it. For example:

```python
if not amd.supports_tdm(options.arch):
```

In these cases, ALWAYS use `rock` functions to check for hardware features.

For example, instead of:

```cpp
if (!triton::AMD::TargetInfo(arch.str()).supportsTDM())
```

use the `rock` function:

```cpp
if (rock::supportsTDM(arch))
```

If there is no `rock` equivalent function to check that hardware feature, then implement a new function in `AmdArchDb.cpp` and use it.

The same rule holds for the performance scripts under
`mlir/utils/performance/`: classify an arch with the `amd_arch_db` pybind
module (`amd_arch_db.is_rdna(arch)`, `amd_arch_db.get_isa_family(arch)`, ...)
rather than with a `gfx` number range or a hand-maintained chip list. A range
check like `0x1100 <= n < 0x1250` compiles and runs fine while silently
mis-bucketing whichever family lands inside it next.

## Step 6: Regenerate Fat Library Dependencies

The file `mlir/tools/rocmlir-lib/librockcompiler_deps.cmake` lists all LLVM/MLIR and rocMLIR libraries that get merged into `librockCompiler.a`. A Triton bump can add or remove library dependencies, so this file must be regenerated after a successful build.

From the **build directory**, run:

```bash
perl ../mlir/utils/jenkins/static-checks/get_fat_library_deps_list.pl > ../mlir/tools/rocmlir-lib/librockcompiler_deps.cmake
```

## Step 7: Env variables
Search for `triton::tools::getStrEnv` in rocmlirTriton and make sure that the variable names are up to date since they might have been renamed or removed during Triton bump.

## Step 8: Features Intentionally NOT Implemented

The following Python features are **intentionally omitted** from the C++ implementation. Do NOT add them when synchronizing:

| Feature | Python Location | Reason |
|---------|-----------------|--------|
| `HIPBackend.instrumentation.patch()` | `compiler.py` make_llir, make_ttgir | Not needed for our use case |
| `passes.llvmir.add_di_scope()` | `compiler.py` make_llir | TODO, not critical |
| `llvm.translate_to_mir()` | `compiler.py` make_amdgcn | Simplified implementation |
| `llvm.dump_sched_dag()` | `compiler.py` make_amdgcn | Debugging feature not needed |
| `knobs.amd.swap_mir` | `compiler.py` make_amdgcn | Debugging feature not needed |
| `knobs.compilation.dump_ir_*` | `compiler.py` make_llir | Debugging feature not needed |
| FPSan instrumentation mode | `compiler.py` make_ttgir | Not implemented |

When reviewing diffs, **skip changes** related to these features.

### 8.1 Knobs that we mirror

On a bump, review the upstream diff and propagate any default/semantic changes:

| Upstream (`compiler.py`)                          | perfConfig field                      | Pipeline option                                                |
|---------------------------------------------------|---------------------------------------|----------------------------------------------------------------|
| `knobs.amd.use_async_copy`                        | `useAsyncCopy`                        | `TritonOptions::useAsyncCopy`                                  |
| `knobs.amd.use_block_pingpong`                    | `useBlockPingpong`                    | `TritonOptions::useBlockPingpong`                              |
| `knobs.amd.use_in_thread_transpose`               | `useInThreadTranspose`                | `TritonOptions::useInThreadTranspose`                          |
| `knobs.amd.use_buffer_ops`                        | `useBufferOps`                        | `TritonOptions::useBufferOps`                                  |
| `knobs.amd.use_buffer_atomics`                    | `useBufferAtomics`                    | `TritonOptions::useBufferAtomics`                              |
| `knobs.amd.buffer_ops_analyze_small_tensor_range` | (not in perfConfig -- debug-only)     | `TritonOptions::bufferOpsAnalyzeSmallTensorRange`              |
| `knobs.amd.use_expert_scheduling`                 | (not in perfConfig -- debug-only)     | `BackendOptions::useExpertScheduling` -> `TritonToHsacoOptions::useExpertScheduling` (backend/HSACO stage, not the Triton MLIR pipeline) |

`knobs.amd.use_expert_scheduling` deliberately diverges from upstream Triton's
implementation at the final LLVM codegen step. Upstream appends
`"amdgpu-expert-scheduling-mode"` to the `translate_to_asm` flags, and
`python/src/llvm.cc` applies those flags by mutating LLVM's process-global
command-line options. `rocmlir-tuning-driver` compiles perf configs in parallel
worker threads, so `TritonToHsaco.cpp` must instead stamp the LLVM function
attribute `amdgpu-expert-scheduling-mode=true/false` on every defined function.
LLVM's AMDGPU backend reads this attribute when no process command-line
occurrence of the global option exists; do not replace it with the upstream
global-option path during a Triton bump.

If upstream adds a new `knobs.amd.*` switch around an existing pass we
already replicate, decide whether it's a *tuner* knob (per-arch defaults
vary, plausibly affects perf-tunable shapes) or a *debug* knob (universal
default, no tuning value). For a tuner knob: add a corresponding
`Option<int>` to `TritonOptions` (use the `kKnobDefault = -1` tri-state
sentinel), add a matching perfConfig field (see section 8.2), and propagate
the value through `compileUtils.cpp`. For a debug knob: only add the
`TritonOptions` field (and document it like
`bufferOpsAnalyzeSmallTensorRange`); skip the perfConfig schema entirely.

`HIPOptions.schedule_hint` is intentionally not mirrored: upstream gutted
it (the field is now a no-op default `''`, and the TTGIR/LLIR sched-hint
passes plus the `memory-bound-attention -> amdgpu-sched-strategy=iterative-ilp`
mapping were all removed).

### 8.2 Adding or removing a perfConfig field

A bump that gains or loses a knob changes the perfConfig schema, and
perfConfig strings outlive the schema they were written against: they are
pasted into bug reports, committed into tuning databases and quick-tuning
lists, and saved by hand during performance investigations. A config
written months ago must therefore still be usable after the schema moves,
and the failure mode when it isn't must be a diagnostic and not a crash.

The schema lives in a single field list per attribute in
`mlir/include/mlir/Dialect/Rock/IR/RockAttrDefs.td`: the tile fields passed
to `Rock_PerfConfigSchema` plus the shared `kRockCommonPerfConfigFields`.
The serializer, the key/default arrays the parser validates against, and
the `getFromPerfConfigValues()` builder it feeds the resolved values into
are all generated from that one list, and a TableGen assert fails the build
if it disagrees with the attribute's `let parameters` dag. There is no
schema version to bump: the canonical output form is the named
`gemm:key=value,...` / `attn:key=value,...` one, and the legacy positional
`prefix:vN:` versions are frozen read-only decoders.

**Adding a field.** Add the parameter to `kRockGemmParams` /
`kRockGemmGemmParams` and a matching
`RockPerfConfigField<"theKey", theDefault>` in the same position. Older
configs simply omit the key and decode to `theDefault`, so pick a default
that reproduces the pre-change behaviour (that is why the knobs use the
`-1` "arch default" sentinel and why `nPerBlockG1` defaults to `0`, i.e.
untiled). Nothing else has to change for back-compat.

**Removing a field.** Deleting it from the parameter dag and the schema is
not enough: every previously saved config that still spells the key would
then be rejected with `unknown field`, because `parseNamedPerfConfig()` in
`mlir/lib/Dialect/Rock/IR/RockDialect.cpp` rejects keys outside the
schema. Also add the retired key to the ignore path in that function, next
to the `scheduleHint` case, so the entry is dropped with a warning instead
of failing the parse. If the field was also part of a legacy positional
version, keep that version decoding and discarding its token (see the
`version == 2` `scheduleHint` branch and `getExpectedPerfConfigFieldCount()`);
never renumber or shrink an existing `vN`, since strings in that form are
already in the wild. Renaming a field is a removal plus an addition, and
needs both halves.

So, for a perfConfig saved before the change: added fields fall back to
their defaults, removed fields are ignored with a warning, and a key that
is in neither the current schema nor the retired set is a hard error --
`get()` returns a null attribute and the pass that asked for the config
fails with a diagnostic, rather than crashing. Cover the new behaviour in
`mlir/unittests/Dialect/Rock/PerfConfigParsingTests.cpp`, which already has
named-form and per-version positional back-compat cases to copy from.

## Step 9: Handling Pass Interface Changes

When Triton changes pass interfaces (arguments, types), follow these steps:

1. **Identify the change** in `compiler.py` or `triton_amd.cc`
2. **Find the pass creation function** in Triton headers (e.g., `TritonAMDGPUTransforms/Passes.h`)
3. **Update the C++ call** in `Pipelines.cpp` or `TritonToHsaco.cpp` to match new arguments
4. **Build and fix** any compilation errors

**Common patterns:**

- New boolean flag added → add parameter to pass creation call
- New string parameter (usually arch) → pass `options.arch` or equivalent
- New integer parameter → check what value Triton passes and match it

## Step 10: Rebuild rocmlirTriton and fix changes (due to upstream MLIR changes)

After making all synchronization changes:

```bash
# From project root
bash cmake.sh
```

This may fail after an upstream import. Resolve build errors caused by
Triton or LLVM API changes, then rebuild.

### Watch for new Triton build-system requirements

Upstream occasionally adds new required CMake variables or download hooks to
`external/triton/CMakeLists.txt` (and its helpers in
`external/triton/python/build_helpers.py`). 

Because we embed Triton via
`add_subdirectory` in `cmake/triton.cmake`, but Triton does it through `setup.py`, any change must be wired up on our side or the build will fail or start downloading things unnecessarily.

In particular, when `external/triton/CMakeLists.txt` or
`external/triton/python/build_helpers.py` changes, check whether
`cmake/triton.cmake` needs matching cache options or path fixes. New
`TRITON_BUILD_*` options should be explicitly set when we rely on disabling
optional Triton components, and generated third-party CMake cache variables
must not leave Triton's `find_package(MLIR)` pointed at a prebuilt LLVM/MLIR
layout instead of rocmlirTriton's in-tree build.

## Step 11: Regenerate the LDS overflow blacklist

The compiled-in blacklist of GEMM perf configs that overflow LDS
(`mlir/include/mlir/Dialect/Rock/Tuning/LdsBlacklistPerfconfigs.inc`, consumed
via `LdsBlacklist.h`) is an *empirical* artifact: it records which tile shapes
exceed each arch's LDS budget under the current Rock->Triton lowering. A Triton
(or Triton-pinned LLVM) bump can change per-block shared-memory usage, so the
table must be regenerated against the freshly built compiler — otherwise it
drifts out of sync and the nightly drift-detection test
(`generateLDSBlacklist.py --verify`) fails.

If this bump added a new ISA family, first add a representative chip to
`DEFAULT_ARCHES` in `generateLDSBlacklist.py` (see section 5.5) so the new family
gets a blacklist at all.

The generator is copied into `build/bin` (by the `ci-performance-scripts`
target), so run it from there to pick up the just-built tools:

```bash
cd build/bin
python3 generateLDSBlacklist.py
```

This rewrites `LdsBlacklistPerfconfigs.inc` in place for the full arch/dtype
matrix. Because the table is compiled into the library, rebuild afterward
(`bash cmake.sh` from the project root) and commit the regenerated `.inc`.

## Step 12: Run Tests

```bash
cd build && ninja check-mlir && ninja check-rocmlir
```

`check-mlir` runs the upstream MLIR suite from `external/llvm-project`, which is
the main signal that a fresh upstream import plus our re-applied `llvm-patches/`
did not regress MLIR itself. Run it before `check-rocmlir`: it is far quicker, so
regressions surface earlier. The nightly pipeline runs it too, and there with
`MLIR_INCLUDE_INTEGRATION_TESTS=ON`, so a bump that lands on `develop` is covered
even if this step is skipped.

## Checklist Summary

Use this checklist to track progress:

- [ ] Record `OLD_REPO`
- [ ] Import new Triton / LLVM upstream revisions and record `NEW_REPO`
- [ ] Build with `cmake.sh`
- [ ] Generate diff for `third_party/amd/backend/compiler.py`
- [ ] Generate diff for `python/src/llvm.cc`
- [ ] Generate diff for `third_party/amd/python/triton_amd.cc`
- [ ] Generate diff for `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`
- [ ] Generate diff for `third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h`
- [ ] Generate diff for `CMakeLists.txt` and `python/build_helpers.py`, then update `cmake/triton.cmake` for any new build options, downloads, or generated cache variables
- [ ] Generate diff for `include/triton/Dialect/Triton/IR/TritonAttrDefs.td` and reconcile the mirrored `CacheModifier` enum (see section 5.4)
- [ ] Generate diff for `include/triton/Dialect/Triton/IR/Traits.h` and reconcile the mirrored `kTritonMaxTensorNumElements` constant (see section 5.4.1)
- [ ] Check whether the pinned LLVM revision fixes the KV-cache raw-buffer bounds-checking bug and re-evaluate the N-loop clamp (see section 5.3.2)
- [ ] Update `Pipelines.cpp::makeTTIR()` for `make_ttir()` changes
- [ ] Update `Pipelines.cpp::makeTTGIR()` for `make_ttgir()` changes
- [ ] Update `Pipelines.cpp::makeLLIR()` for `make_llir()` Part 1 changes
- [ ] Refresh the `TRITON` prefix in `mlir/test/rocmlir-driver/pipelines.mlir` if any of `makeTTIR` / `makeTTGIR` / `makeLLIR` changed (see section 5.1)
- [ ] Update `TritonToHsaco.cpp::translateTritonToHsaco()` for `make_llir()` Part 2 changes
- [ ] Update `TritonToHsaco.cpp` for LLVM function changes (`initializeLLVMTargets`, `createTargetMachine`, `optimizeModule`)
- [ ] Update `tritonUtils.cpp::getMfmaVersion()` if changed
- [ ] Update `tritonUtils.cpp::getWmmaVersion()` if changed
- [ ] Update `tritonUtils.cpp::mlirTypeToScaleDotElemType()` if changed
- [ ] Update `AmdArchDb.cpp` if new `ISAFamily` added (see section 5.5)
- [ ] Diff `TargetFeatures.cpp` and re-check the `isCDNA` / `isRDNA` classification that picks the tuning space (see section 5.5)
- [ ] Add a representative chip to `DEFAULT_ARCHES` in `generateLDSBlacklist.py` if a new ISA family was added (see section 5.5)
- [ ] Mirror new or dropped `knobs.amd.*` switches, and if the perfConfig gained or lost a field, keep configs saved against the old schema readable (see sections 8.1 and 8.2)
- [ ] Refresh `triton-patches/` and `llvm-patches/` records and indexes
- [ ] Build project with `cmake.sh`
- [ ] Regenerate `librockcompiler_deps.cmake` with `get_fat_library_deps_list.pl`
- [ ] Regenerate the LDS blacklist (`generateLDSBlacklist.py` from `build/bin`), rebuild, and commit the updated `.inc`
- [ ] Run tests with `cd build && ninja check-mlir && ninja check-rocmlir`
- [ ] All tests pass
- [ ] Commit all changes

---

## Troubleshooting

### Build Failures Due to Missing Pass

If a new pass is added in upstream Triton and called in `compiler.py`:

1. Find the pass definition in `triton_amd.cc` to understand its C++ signature
2. Add the corresponding `pm->addPass()` call in your C++ pipeline
3. Include any necessary headers from Triton

### API Changes in Triton Passes

If a pass signature changes:

1. Check the new signature in Triton headers (e.g., `external/triton/third_party/amd/include/TritonAMDGPUTransforms/Passes.h`)
2. Update the C++ call to match the new parameters
3. Handle any new required arguments

### Test Failures

If tests fail after bumping:

1. Check if the failure is due to a behavioral change in Triton
2. Review the Triton changelog or commit messages for breaking changes
3. Adjust test expectations or C++ implementation as needed
4. Resolve conflicts manually

### Header Include Errors

If new Triton headers are needed:

1. Check what headers are included in `triton_amd.cc` or other Triton files
2. Add corresponding includes to your C++ files
3. Ensure CMake finds the headers (check `CMakeLists.txt` if needed)

---

## File Quick Reference

| Purpose | File Path |
|---------|-----------|
| TTIR/TTGIR/LLIR pipelines | `mlir/lib/Dialect/Rock/Pipelines/Pipelines.cpp` |
| HSACO translation | `mlir/lib/Translation/TritonToHsaco/TritonToHsaco.cpp` |
| Architecture database | `mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp` |
| Triton utility replicas | `mlir/lib/Dialect/Rock/utility/tritonUtils.cpp` |
| Mirrored `CacheModifier` enum | `mlir/include/mlir/Dialect/Rock/IR/RockAttrDefs.td` |
| perfConfig field list / schema | `mlir/include/mlir/Dialect/Rock/IR/RockAttrDefs.td` |
| perfConfig parser | `mlir/lib/Dialect/Rock/IR/RockDialect.cpp` |
| Triton `CacheModifier` source | `external/triton/include/triton/Dialect/Triton/IR/TritonAttrDefs.td` |
| Mirrored `kTritonMaxTensorNumElements` constant | `mlir/lib/Dialect/Rock/Tuning/RockTuningImpl.cpp` |
| Triton `maxTensorNumElements` source | `external/triton/include/triton/Dialect/Triton/IR/Traits.h` |
| Triton compiler.py | `external/triton/third_party/amd/backend/compiler.py` |
| Triton llvm.cc | `external/triton/python/src/llvm.cc` |
| Triton pass bindings | `external/triton/third_party/amd/python/triton_amd.cc` |
| Triton AccelerateMatmul | `external/triton/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp` |
| Build script | `cmake.sh` |
| Fat library deps generator | `mlir/utils/jenkins/static-checks/get_fat_library_deps_list.pl` |
| Fat library deps list | `mlir/tools/rocmlir-lib/librockcompiler_deps.cmake` |
| LDS blacklist generator | `mlir/utils/performance/generateLDSBlacklist.py` |
| LDS blacklist table | `mlir/include/mlir/Dialect/Rock/Tuning/LdsBlacklistPerfconfigs.inc` |
