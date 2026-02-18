# Guide: Bumping the Triton Version in rocmlirTriton

## Overview

The rocmlirTriton project embeds Triton as a git submodule at `external/triton/`. Several Triton Python functions are replicated in C++ within this project. When the Triton version is bumped, these C++ implementations must be synchronized with the upstream changes.

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
cd external/triton/scripts/
bash build-llvm-project.sh
```

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
git diff ${OLD_COMMIT}..${NEW_COMMIT} -- third_party/amd/backend/compiler.py > compiler.py.diff
git diff ${OLD_COMMIT}..${NEW_COMMIT} -- python/src/llvm.cc > llvm.cc.diff
git diff ${OLD_COMMIT}..${NEW_COMMIT} -- third_party/amd/python/triton_amd.cc > triton_amd.cc.diff
git diff ${OLD_COMMIT}..${NEW_COMMIT} -- third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp > AccelerateAMDMatmul.cpp.diff
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

### 5.2 Understanding Pass Bindings (from `triton_amd.cc`)

The file `external/triton/third_party/amd/python/triton_amd.cc` contains the Python bindings for Triton passes. When `compiler.py` adds a new pass call, check `triton_amd.cc` to find the actual C++ pass creation function.

### 5.3 LLVM Functions (from `llvm.cc`)

| Python/C++ Function in Triton | C++ Location in rocmlirTriton |
|------------------------------|-------------------------------|
| `init_targets()` | `TritonToHsaco.cpp::initializeLLVMTargets()` |
| `createTargetMachine()` | `TritonToHsaco.cpp::createTargetMachine()` |
| `optimize_module()` | `TritonToHsaco.cpp::optimizeModule()` |

### 5.4 Architecture Database (from `AccelerateAMDMatmul.cpp`)

| Triton Function | C++ Location in rocmlirTriton |
|----------------|-------------------------------|
| `getMfmaVersion()` | `mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp` |
| `getWmmaVersion()` | `mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp` |

If there are any new architecture not handled by our rocmlirTrion functions we should see warnings/errors because the switch would not be handling all cases.

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

### 5.5 Hardware feature detection
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

## Step 8: Build the Project

After making all synchronization changes:

```bash
# From project root
bash cmake.sh
```
## Step 9: Run Tests

```bash
bash tests.sh
```
## Checklist Summary

Use this checklist to track progress:

- [ ] Record old Triton commit (OLD_COMMIT)
- [ ] Update Triton submodule to new commit (NEW_COMMIT)
- [ ] Rebuild LLVM with `external/triton/scripts/build-llvm-project.sh`
- [ ] Generate diff for `third_party/amd/backend/compiler.py`
- [ ] Generate diff for `python/src/llvm.cc`
- [ ] Generate diff for `third_party/amd/python/triton_amd.cc`
- [ ] Generate diff for `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`
- [ ] Update `Pipelines.cpp::makeTTIR()` for `make_ttir()` changes
- [ ] Update `Pipelines.cpp::makeTTGIR()` for `make_ttgir()` changes
- [ ] Update `Pipelines.cpp::makeLLIR()` for `make_llir()` Part 1 changes
- [ ] Update `TritonToHsaco.cpp::translateTritonToHsaco()` for `make_llir()` Part 2 changes
- [ ] Update `TritonToHsaco.cpp` for LLVM function changes (`initializeLLVMTargets`, `createTargetMachine`, `optimizeModule`)
- [ ] Update `AmdArchDb.cpp::getMfmaVersion()` if changed
- [ ] Update `AmdArchDb.cpp::getWmmaVersion()` if changed
- [ ] Build project with `cmake.sh`
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
| Triton compiler.py | `external/triton/third_party/amd/backend/compiler.py` |
| Triton llvm.cc | `external/triton/python/src/llvm.cc` |
| Triton pass bindings | `external/triton/third_party/amd/python/triton_amd.cc` |
| Triton AccelerateMatmul | `external/triton/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp` |
| Build script | `cmake.sh` |
| Test script | `tests.sh` |
| LLVM build script | `external/triton/scripts/build-llvm-project.sh` |
