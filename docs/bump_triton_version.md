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

### 1.1 Record the current repository commit

```bash
export OLD_REPO=$(git rev-parse HEAD)
```

### 1.2 Pull the new versions

Import the desired upstream Triton and LLVM revisions into their vendored
subtrees from the repository root. Replace `<triton-ref>` and `<llvm-ref>`
with the branch, tag, or commit selected for the bump:

```bash
git subtree pull --prefix=external/triton \
  https://github.com/triton-lang/triton-windows.git <triton-ref>

git subtree pull --prefix=external/llvm-project \
  https://github.com/llvm/llvm-project.git <llvm-ref>

export NEW_REPO=$(git rev-parse HEAD)
```

`external/triton` and `external/llvm-project` are directories inside this
repository, not nested Git repositories. Do not use `git -C
external/triton rev-parse HEAD` to record the upstream revision; it
returns the main repository's `HEAD`. Use the `OLD_REPO..NEW_REPO`
bracket below to inspect what the subtree pulls changed under each
prefix.

## Step 2: Rebuild LLVM/MLIR

The LLVM version must remain compatible with the imported Triton tree
(`external/triton/cmake/llvm-hash.txt` is a useful cross-check). Build
from the vendored trees:

```bash
bash cmake.sh
```

Patch files are not applied during configure or build; they are records
of downstream cherry-picks that must be kept in sync with the committed
vendored trees.

## Step 3: Check if downstream patch records are still needed

We keep downstream patch records under `triton-patches/` and
`llvm-patches/` so the next upstream import can re-apply or reconcile
local divergence. Re-evaluate each one; it may no longer be needed if the
change was merged upstream.

For each Triton patch, check the files that are modified and do:

```bash
git diff "$OLD_REPO..$NEW_REPO" -- external/triton/FILE
```

For each LLVM patch, use the same pattern in the LLVM subtree:

```bash
git diff "$OLD_REPO..$NEW_REPO" -- external/llvm-project/FILE
```

Compare the upstream changes with the corresponding `.patch` file.

If the changes are already in the imported upstream tree, remove the
patch file and its entry in the matching `*-patch-content.txt` index.
Otherwise, refresh the patch so it matches the committed vendored-tree
edit.

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
    third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h; do
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

### 5.5 Architecture Database (`AmdArchDb.cpp`)

`mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp` maps AMD GPU architectures to hardware
capabilities using Triton APIs (`ISAFamily`, `MfmaIntrinsic::selectFor`,
`WmmaIntrinsic::selectFor`, `TargetInfo`). When a new architecture is added
upstream (e.g. a hypothetical RDNA5 / CDNA5), this file needs review:

| Function | What to check |
|----------|---------------|
| `getMatrixAccelKind()` | Does the new arch support MFMA, WMMA, or scaled variants? Update the selection logic (version thresholds, `isF8F6F4`, `isScaledWmmaType`). |
| `isFastAtomicAddSupported()` | Add the new `ISAFamily` case if atomic f32/f16/bf16 adds are supported. |
| `isFastAtomicMaxSupported()` | Add the new `ISAFamily` case if atomic f32 max is supported. |
| `getMaxNumChiplets()` | Update if the new arch has multi-chiplet GPUs. |
| `getMinNumCU()` | Add the new `ISAFamily` case with the minimum CU count. |
| `getMaxWavesPerEU()` | Add the new `ISAFamily` case with the correct occupancy limit. |
| `getWaveSize()` / `getLDSSize()` | These delegate to `TargetInfo`, so they should work automatically if Triton adds the arch. Verify. |
| `supportsTDM()` | Delegates to `TargetInfo`. Verify it returns the correct value for the new arch. |

Also check that `tritonUtils.cpp::getMfmaVersion()` and
`tritonUtils.cpp::getWmmaVersion()` handle the new `ISAFamily` / chip string.
Additionally, `mlir/test/common_utils/amd_arch_db/binding.cpp` will need to have
it's `py::enum_<ISAFamily>(...).value(...)` enum updated as well.

**How to detect needed changes:** Triton uses exhaustive `switch` statements over
`ISAFamily`. If a new variant is added, our switches (which use `default:`) will
silently return a fallback value. The `ISAFamily` enum is defined in
`TargetFeatures.h` (it moved out of the now-removed
`TritonAMDGPUToLLVM/TargetUtils.h`); diff it for new entries:

```bash
git diff "$OLD_REPO..$NEW_REPO" -- external/triton/third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h
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

### Knobs that we mirror

On a bump, review the upstream diff and propagate any default/semantic changes:

| Upstream (`compiler.py`)                          | perfConfig field (v2 trailing block)  | Pipeline option                                                |
|---------------------------------------------------|---------------------------------------|----------------------------------------------------------------|
| `knobs.amd.use_async_copy`                        | `useAsyncCopy`                        | `TritonOptions::useAsyncCopy`                                  |
| `knobs.amd.use_block_pingpong`                    | `useBlockPingpong`                    | `TritonOptions::useBlockPingpong`                              |
| `knobs.amd.use_in_thread_transpose`               | `useInThreadTranspose`                | `TritonOptions::useInThreadTranspose`                          |
| `knobs.amd.use_buffer_ops`                        | `useBufferOps`                        | `TritonOptions::useBufferOps`                                  |
| `knobs.amd.use_buffer_atomics`                    | `useBufferAtomics`                    | `TritonOptions::useBufferAtomics`                              |
| `knobs.amd.buffer_ops_analyze_small_tensor_range` | (not in perfConfig -- debug-only)     | `TritonOptions::bufferOpsAnalyzeSmallTensorRange`              |
| `HIPOptions.schedule_hint`                        | `scheduleHint`                        | `TritonOptions::scheduleHint` / `BackendOptions::scheduleHint` |

If upstream adds a new `knobs.amd.*` switch around an existing pass we
already replicate, decide whether it's a *tuner* knob (per-arch defaults
vary, plausibly affects perf-tunable shapes) or a *debug* knob (universal
default, no tuning value). For a tuner knob: add a corresponding
`Option<int>` to `TritonOptions` (use the `kKnobDefault = -1` tri-state
sentinel), append a matching `int64_t` field to `Rock_GemmParamsAttr` /
`Rock_GemmGemmParamsAttr` in `RockAttrDefs.td`, extend the `v2` parser in
`RockDialect.cpp` (and its range validator), and propagate the value
through `compileUtils.cpp`. For a debug knob: only add the
`TritonOptions` field (and document it like
`bufferOpsAnalyzeSmallTensorRange`); skip the perfConfig schema entirely.

If upstream adds a new `SchedHint` enum entry, claim a new bit in
`kScheduleHintBitTable` in `KnobUtils.cpp` and document it in
the `RockAttrDefs.td` docstring; `expandScheduleHintBitfield` will then
pick it up automatically. The LLIR-only `memory-bound-attention`
literal already has its own bit and lives next to the table.

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

## Step 11: Run Tests

```bash
cd build && ninja check-rocmlir
```
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
- [ ] Generate diff for `include/triton/Dialect/Triton/IR/TritonAttrDefs.td` and reconcile the mirrored `CacheModifier` enum (see section 5.4)
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
- [ ] Refresh `triton-patches/` and `llvm-patches/` records and indexes
- [ ] Build project with `cmake.sh`
- [ ] Regenerate `librockcompiler_deps.cmake` with `get_fat_library_deps_list.pl`
- [ ] Run tests with `cd build && ninja check-rocmlir`
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
| `schedule_hint` parser | `mlir/lib/Dialect/Rock/utility/KnobUtils.cpp` |
| Mirrored `CacheModifier` enum | `mlir/include/mlir/Dialect/Rock/IR/RockAttrDefs.td` |
| Triton `CacheModifier` source | `external/triton/include/triton/Dialect/Triton/IR/TritonAttrDefs.td` |
| Triton compiler.py | `external/triton/third_party/amd/backend/compiler.py` |
| Triton llvm.cc | `external/triton/python/src/llvm.cc` |
| Triton pass bindings | `external/triton/third_party/amd/python/triton_amd.cc` |
| Triton AccelerateMatmul | `external/triton/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp` |
| Build script | `cmake.sh` |
| Fat library deps generator | `mlir/utils/jenkins/static-checks/get_fat_library_deps_list.pl` |
| Fat library deps list | `mlir/tools/rocmlir-lib/librockcompiler_deps.cmake` |
