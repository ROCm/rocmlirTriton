# Guide: Bumping the Triton Version in rocmlirTriton

## Overview

The rocmlirTriton project embeds Triton as a git submodule at `external/triton/`. Several Triton Python functions are replicated in C++ within this project. When the Triton version is bumped, these C++ implementations must be synchronized with the upstream changes.

Note that we want to use Triton from https://github.com/triton-lang/triton-windows, as that includes Windows support not included in https://github.com/triton-lang/triton

## Step 1: Update the Triton Submodule

### 1.1 Record the Current Commit

```bash
cd external/triton
export OLD_COMMIT=$(git rev-parse HEAD)
```

### 1.2 Pull the New Version

```bash
cd external/triton
git pull
export NEW_COMMIT=$(git rev-parse HEAD)
```

### 1.3 Update Submodule Reference

```bash
git add external/triton
```

## Step 2: Rebuild LLVM

The LLVM version is tied to the Triton version (specified in `external/triton/cmake/llvm-hash.txt`). After bumping Triton, LLVM must be rebuilt.

```bash
bash scripts/build-llvm.sh
```

This wrapper script handles submodule init, applying `triton-patches/*.patch`, patching `MLIR_ENABLE_ROCM_RUNNER=ON`, and building LLVM/MLIR.

## Step 3: Check if our local patches are still needed
We may have local patches (`.patch` files) under `./triton-patches`, which are applied using the `cmake.sh` script.
These might not be needed if the changes in the patch were merged into upstream.

For each patch, check the files that are modified and do:

```bash
cd external/triton
git diff ${OLD_COMMIT}..${NEW_COMMIT} -- FILE
```

And compare the changes with the corresponding `.patch` file.

If the changes are already in `external/triton`, then remove the `.patch` file.

## Step 4: Analyze Upstream Changes

Generate a diff between the old and new commits for the key files that need synchronization. Those are:

```bash
cd external/triton
for f in \
    third_party/amd/backend/compiler.py \
    third_party/amd/python/triton_amd.cc \
    python/src/llvm.cc \
    third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp \
    third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h; do
  git diff ${OLD_COMMIT}..${NEW_COMMIT} --function-context -- "$f" > "$(basename "$f").diff"
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

**How to detect needed changes:** Triton uses exhaustive `switch` statements over
`ISAFamily`. If a new variant is added, our switches (which use `default:`) will
silently return a fallback value. Diff `TargetUtils.h` for new `ISAFamily`
entries:

```bash
cd external/triton
git diff ${OLD_COMMIT}..${NEW_COMMIT} -- third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h
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

## Step 6: Features Intentionally NOT Implemented

The following Python features are **intentionally omitted** from the C++ implementation. Do NOT add them when synchronizing:

| Feature | Python Location | Reason |
|---------|-----------------|--------|
| `HIPBackend.instrumentation.patch()` | `compiler.py` make_llir, make_ttgir | Not needed for our use case |
| `knobs.*` configuration | Throughout `compiler.py` | Hardcoded values are sufficient |
| `passes.llvmir.add_di_scope()` | `compiler.py` make_llir | TODO, not critical |
| `llvm.translate_to_mir()` | `compiler.py` make_amdgcn | Simplified implementation |
| `llvm.dump_sched_dag()` | `compiler.py` make_amdgcn | Debugging feature not needed |
| `knobs.amd.swap_mir` | `compiler.py` make_amdgcn | Debugging feature not needed |
| `knobs.compilation.dump_ir_*` | `compiler.py` make_llir | Debugging feature not needed |
| FPSan instrumentation mode | `compiler.py` make_ttgir | Not implemented |
| `schedule_hint` loop processing | `compiler.py` make_ttgir | Partially hardcoded |

When reviewing diffs, **skip changes** related to these features.

## Step 7: Handling Pass Interface Changes

When Triton changes pass interfaces (arguments, types), follow these steps:

1. **Identify the change** in `compiler.py` or `triton_amd.cc`
2. **Find the pass creation function** in Triton headers (e.g., `TritonAMDGPUTransforms/Passes.h`)
3. **Update the C++ call** in `Pipelines.cpp` or `TritonToHsaco.cpp` to match new arguments
4. **Build and fix** any compilation errors

**Common patterns:**

- New boolean flag added → add parameter to pass creation call
- New string parameter (usually arch) → pass `options.arch` or equivalent
- New integer parameter → check what value Triton passes and match it

## Step 8: Rebuild rocmlirTriton and fix changes (due to upstream MLIR changes)

After making all synchronization changes:

```bash
# From project root
bash cmake.sh
```

Which will probably fail due to LLVM being also bumped with Triton version.
For this, we need to manually resolve the errors due to upstream LLVM changes.

### 8.1 Watch for new Triton build-system requirements

Upstream occasionally adds new required CMake variables or download hooks to
`external/triton/CMakeLists.txt` (and its helpers in
`external/triton/python/build_helpers.py`). 

Because we embed Triton via
`add_subdirectory` in `cmake/triton.cmake`, but Triton does it through `setup.py`, any change must be wired up on our side or the build will fail or start downloading things unnecessarily.

## Step 9: Regenerate Fat Library Dependencies

The file `mlir/tools/rocmlir-lib/librockcompiler_deps.cmake` lists all LLVM/MLIR and rocMLIR libraries that get merged into `librockCompiler.a`. A Triton bump can add or remove library dependencies, so this file must be regenerated after a successful build.

From the **build directory**, run:

```bash
perl ../mlir/utils/jenkins/static-checks/get_fat_library_deps_list.pl > ../mlir/tools/rocmlir-lib/librockcompiler_deps.cmake
```

## Step 10: Run Tests

```bash
bash tests.sh
```
## Checklist Summary

Use this checklist to track progress:

- [ ] Record old Triton commit (OLD_COMMIT)
- [ ] Update Triton submodule to new commit (NEW_COMMIT)
- [ ] Rebuild LLVM with `scripts/build-llvm.sh`
- [ ] Generate diff for `third_party/amd/backend/compiler.py`
- [ ] Generate diff for `python/src/llvm.cc`
- [ ] Generate diff for `third_party/amd/python/triton_amd.cc`
- [ ] Generate diff for `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`
- [ ] Generate diff for `third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h`
- [ ] Update `Pipelines.cpp::makeTTIR()` for `make_ttir()` changes
- [ ] Update `Pipelines.cpp::makeTTGIR()` for `make_ttgir()` changes
- [ ] Update `Pipelines.cpp::makeLLIR()` for `make_llir()` Part 1 changes
- [ ] Update `TritonToHsaco.cpp::translateTritonToHsaco()` for `make_llir()` Part 2 changes
- [ ] Update `TritonToHsaco.cpp` for LLVM function changes (`initializeLLVMTargets`, `createTargetMachine`, `optimizeModule`)
- [ ] Update `tritonUtils.cpp::getMfmaVersion()` if changed
- [ ] Update `tritonUtils.cpp::getWmmaVersion()` if changed
- [ ] Update `tritonUtils.cpp::mlirTypeToScaleDotElemType()` if changed
- [ ] Update `AmdArchDb.cpp` if new `ISAFamily` added (see section 5.5)
- [ ] Build project with `cmake.sh`
- [ ] Regenerate `librockcompiler_deps.cmake` with `get_fat_library_deps_list.pl`
- [ ] Run tests with `tests.sh`
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
| Triton compiler.py | `external/triton/third_party/amd/backend/compiler.py` |
| Triton llvm.cc | `external/triton/python/src/llvm.cc` |
| Triton pass bindings | `external/triton/third_party/amd/python/triton_amd.cc` |
| Triton AccelerateMatmul | `external/triton/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp` |
| Build script | `cmake.sh` |
| Test script | `tests.sh` |
| LLVM build wrapper | `scripts/build-llvm.sh` |
| Fat library deps generator | `mlir/utils/jenkins/static-checks/get_fat_library_deps_list.pl` |
| Fat library deps list | `mlir/tools/rocmlir-lib/librockcompiler_deps.cmake` |
