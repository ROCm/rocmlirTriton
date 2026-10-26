# Guide: Bumping the Triton Version in rocmlirTriton

## Overview

rocmlirTriton no longer vendors `external/triton` and
`external/llvm-project` in-tree. It now fetches pinned external branches at
configure time:

- `cmake/triton.cmake` (FetchContent)
  - `ROCMLIR_TRITON_GIT_TAG` (`rmt-<date>`)
  - `ROCMLIR_LLVM_GIT_TAG` (`rmt-core-<date>`)
- `cmake/triton_amd_codegen.cmake` (ExternalProject_Add)
  - `ROCMLIR_AMD_CODEGEN_LLVM_GIT_TAG` (`rmt-amd-<date>`)

Downstream patches are no longer stored as `*.patch` files in this repo.
They live as commits on the external branch tips.

We still synchronize the same Triton/LLVM source files into rocmlirTriton C++
mirrors (pipelines, LLVM lowering, arch DB, etc.); only the *update mechanism*
changed.

Note: Triton updates are based on
https://github.com/triton-lang/triton-windows (`main-windows`).

## Step 1: Capture current pins and create new external update branches

### 1.1 Record current rocmlirTriton baseline

```bash
export OLD_REPO=$(git rev-parse HEAD)
```

### 1.2 Record current external pins from CMake

Read the current branch/tag pins from:

- `cmake/triton.cmake`
- `cmake/triton_amd_codegen.cmake`

Set the date suffix you plan to bump to:

```bash
export BUMP_DATE=$(date +%F)   # e.g. 2026-10-26
export NEW_TRITON_BRANCH="rmt-${BUMP_DATE}"
export NEW_LLVM_CORE_BRANCH="rmt-core-${BUMP_DATE}"
export NEW_LLVM_AMD_BRANCH="rmt-amd-${BUMP_DATE}"
```

## Step 2: Update external repositories (not rocmlirTriton)

Create/update the branch tips in the external repos first, then point
rocmlirTriton at those new branch names.

### 2.1 Triton branch (`rmt-<date>`)

In your `ROCm/triton` clone:

1. Create `rmt-<date>` from the prior `rmt-*` baseline.
2. Pull/cherry-pick desired upstream Triton-Windows commits.
3. Cherry-pick any remaining downstream commits still needed.
4. Drop downstream commits that are now upstream.

### 2.2 LLVM branches (`rmt-core-<date>`, `rmt-amd-<date>`)

In your `ROCm/llvm-project` clone:

1. Create `rmt-core-<date>` and `rmt-amd-<date>` from the prior branch tips.
2. Pull/cherry-pick required LLVM commits for each line (core vs AMD codegen).
3. Cherry-pick any remaining downstream commits still needed.
4. Drop no-longer-needed downstream commits.

## Step 3: Point rocmlirTriton at the new external branches

After pushing the new external branch tips, update the branch names in:

- `cmake/triton.cmake`
  - `ROCMLIR_TRITON_GIT_TAG`
  - `ROCMLIR_LLVM_GIT_TAG`
- `cmake/triton_amd_codegen.cmake`
  - `ROCMLIR_AMD_CODEGEN_LLVM_GIT_TAG`
  - (and `ROCMLIR_AMD_CODEGEN_LLVM_REVISION` if your policy bumps it)

Then build:

```bash
bash cmake.sh
```

## Step 4: Analyze upstream changes

Use the external repo SHAs/refs for old vs new comparisons.

Example setup:

```bash
export TRITON_REPO=/path/to/ROCm/triton
export LLVM_REPO=/path/to/ROCm/llvm-project

export OLD_TRITON=<old-rmt-triton-commit>
export NEW_TRITON=<new-rmt-triton-commit>
export OLD_LLVM_CORE=<old-rmt-core-commit>
export NEW_LLVM_CORE=<new-rmt-core-commit>
```

For the rest of this doc, paths like `external/triton/...` and
`external/llvm-project/...` refer to paths *inside those external repositories*
(i.e. under `$TRITON_REPO` / `$LLVM_REPO`), not paths committed in
rocmlirTriton.

Generate diffs for the key Triton files (same sync targets as before):

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
  git -C "$TRITON_REPO" diff "$OLD_TRITON..$NEW_TRITON" --function-context -- "$f" > "$(basename "$f").diff"
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

##### Downstream LDS check (`rock-check-lds-limit`)

`makeLLIR` runs the downstream `rock-check-lds-limit` pass right before `convert-triton-amdgpu-to-llvm`, so a kernel that overflows LDS is rejected before that expensive lowering. The pass reads `ttg.shared`, so it must stay after every pass in our pipeline that writes it. Today that is only `allocate-amdgpu-shared-memory`; Proton and the NVIDIA passes also write it, but our pipeline doesn't run them. If a bump adds or moves a pass that writes `ttg.shared` or otherwise changes the shared-memory allocation, move `rock-check-lds-limit` after it and keep it before `convert-triton-amdgpu-to-llvm`.

Keep the check in `resolve-kernel-launch-params` too: that pass still sizes `@global_smem` from `ttg.shared`, and it is the only LDS check when it runs on its own. Both passes report an overflow through `rock::checkSharedMemoryFitsLDS`, whose `exceeds LDS limit` wording `generateLDSBlacklist.py` matches (`LDS_OVERFLOW_MARKER`), so keep the two in sync.

##### Understanding Pass Bindings (from `triton_amd.cc`)

The file `external/triton/third_party/amd/python/triton_amd.cc` contains the Python bindings for Triton passes. When `compiler.py` adds a new pass call, check `triton_amd.cc` to find the actual C++ pass creation function.

### 5.2 LLVM Functions (from `llvm.cc`)

| Python/C++ Function in Triton | C++ Location in rocmlirTriton |
|------------------------------|-------------------------------|
| `init_targets()` | `TritonToHsaco.cpp::initializeLLVMTargets()` |
| `createTargetMachine()` | `TritonToHsaco.cpp::createTargetMachine()` |
| `optimize_module()` | `TritonToHsaco.cpp::optimizeModule()` |

`TritonToHsaco.cpp::setModuleFunctionAttributes()` deliberately diverges from
upstream on **`allow_flush_denorm`**: after linking device libraries, stamp the
`denormal_fpenv` enum attribute on every definition, not upstream's legacy
`"denormal-fp-math-f32"` string (see section 8.1).

`TritonToHsaco.cpp` also deliberately diverges on **LLVM IR verification**.
Upstream verifies unconditionally in two places: `to_module()` calls
`translateModuleToLLVMIR` with verification enabled, and the codegen pass
manager adds `createVerifierPass()` after the always-inliner (llvm.cc lines
333-344, mirrored by `emitMachineCode()`). Each is a full IR walk on every
compile over IR our own MLIR lowering just produced. We gate both on the
`kVerifyLLVMIR` constant, which follows `NDEBUG`, so assert-enabled builds
verify and release builds do not. A bump that reintroduces either call
unconditionally should be re-gated on `kVerifyLLVMIR` rather than taking the
upstream spelling.

`TritonToHsaco.cpp::disableHighDuplicationDeviceLibInlining()` is another
deliberate downstream-only step with no upstream Triton counterpart. Before
the default LLVM optimization pipeline, it applies call-site `noinline` to
dense groups of sufficiently large linked OCML/OCKL functions. Preserve both
the call from `translateTritonToHsaco()` and its per-basic-block,
per-device-library-callee behavior when reconciling `make_llir()` with
upstream. Regression coverage lives in the
`triton-to-hsaco-*-ocml.mlir` tests.

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

We also carry a **downstream commit** on the ROCm/triton `rmt-*` branch for
this file: it adds a `copyDiscardableAttrs()` helper that the
accelerate-matmul rewrite patterns call when creating the new
(WMMA/MFMA/scaled) dot, so that the `rock.o_transposed` metadata set by
`rock-add-triton-metadata` survives onto the rewritten dot.
`mlir/lib/Dialect/Rock/Transforms/SetMatmulOutputTranspose.cpp` reads that
metadata back via `getDiscardableAttr(rock::OTransposedAttr::getNameStr())`,
so dropping this downstream commit silently breaks
`rock-set-matmul-output-transpose` (no compile error, just lost
output-transpose tuning).

Because this is downstream-only, `AccelerateAMDMatmul.cpp.diff` in Step 4 may
show `copyDiscardableAttrs` and its call sites as removed when comparing
patched-old vs fresh-upstream. Treat that as a replay signal (cherry-pick onto
new `rmt-*`), not an automatic deletion. After a bump, confirm
`copyDiscardableAttrs` is present in the Triton branch tip used by
rocmlirTriton and that `accelerate-matmul-preserve-rock-metadata*.mlir` /
`set-matmul-output-transpose.mlir` still pass.

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

### 5.3.3 gfx950 packed-FP32 / MFMA hazard workaround

`mlir/test/fusion/pr-e2e/attention/mixr-attention-flash-decoding-3d-output-fusion.mlir`
pins a `perf_config` on its second `migraphx.dot` instead of using the gfx950
default. The default config miscompiles on gfx950: LLVM inserts no wait
between a `v_pk_fma_f32` that uses `op_sel` on src1 and an immediately
following MFMA that reuses the previous MFMA's srcA, so lanes 48-63 of that
MFMA produce wrong results. The LLVM issue is tracked by ROCM-32038.

On every LLVM bump, check whether the new pinned LLVM revision contains the
upstream fix. Remove the pin only when both hold: the revision contains the
fix, and the unpinned test passes on gfx950. Neither is enough on its own. A
passing test can just mean a scheduling change moved the two instructions
apart, which hides the bug without fixing it.

### 5.3.4 Parameter-sweep skip list for the InlineSpiller spill-slot bug

`SKIPPED_PERF_CONFIGS` in `mlir/utils/performance/parameterSweeps.py` lists
(chip, sweep kind, perf_config) combinations that `parameterSweeps.py` and
`attentionSweeps.py` skip. LLVM's InlineSpiller stores a partially defined
register full-width into a spill slot that still holds a live buffer
descriptor, so those kernels fault the GPU or return wrong results. The LLVM
issue is tracked by ROCM-32030 and upstream as llvm/llvm-project#225054.

On every LLVM bump, check whether the new pinned LLVM revision contains a fix
for that issue. Do not drop an entry based only on the revision change: also
run the config with the entry removed on the listed chip and check that it
passes. All entries are gfx950 attention configs, found as these samples of
`attentionSweeps.py attention`:

| Seed | Sample | Data type | Failure without a fix |
|------|--------|-----------|-----------------------------------------------|
| 40   | 259    | bf16      | Memory access fault |
| 44   | 208    | i8        | Memory access fault |
| 44   | 246    | bf16      | Memory access fault; wrong results with `-rand` |
| 45   | 8      | i8        | Wrong results with `-rand 3` |

The default fixed input pattern can hide the wrong results, so also run each
config with a few `-rand` seeds.

The entries are compared as exact strings, so `parameterSweeps.py` refuses to
load if an entry's keys differ from `PERF_CONFIG_FIELD_NAMES` in
`perfCommonUtils.py`. After a field rename, renaming the key in each entry is
enough, because the seeds still draw the same configs. Any other change to the
field list changes which configs the seeds draw, so rewriting the strings is
not enough: the seeds in the table have to be tested again.

### 5.4 Mirrored Enums / Attributes (from `TritonAttrDefs.td`)

Some Triton enums are hand-replicated in the Rock dialect so we can carry the
value through Rock IR (and map it back onto Triton when lowering) without taking
a TableGen dependency on the Triton dialect. These are **manual copies** and
will silently drift if upstream changes them, so diff the source on every bump:

| Rock copy | Upstream source | What to check |
|-----------|-----------------|---------------|
| `CacheModifier` / `CacheModifierAttr` in `mlir/include/mlir/Dialect/Rock/IR/RockAttrDefs.td` | `TT_CacheModifierAttr` in `external/triton/include/triton/Dialect/Triton/IR/TritonAttrDefs.td` | Names **and** integer values must match one-to-one (currently `none=1, ca=2, cg=3, wb=4, cs=5, wt=6, cv=7`). The Rock->Triton lowering relies on the integer values lining up. |

```bash
git -C "$TRITON_REPO" diff "$OLD_TRITON..$NEW_TRITON" -- \
  include/triton/Dialect/Triton/IR/TritonAttrDefs.td
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
git -C "$TRITON_REPO" diff "$OLD_TRITON..$NEW_TRITON" -- \
  include/triton/Dialect/Triton/IR/Traits.h
```

### 5.4.2 Mirrored libdevice symbols and the tanh arch gate

`mlir/lib/Dialect/Rock/Transforms/LegalizeMathForTriton.cpp` emits
`tt.extern_elementwise` calls to OCML entry points directly, spelling the symbol
names as string constants rather than going through the Triton Python frontend
that would normally pick them. It also decides *whether* to emit a tanh call at
all by mirroring the architecture gate Triton puts on its own rewrite of that
call. Both are **manual copies** that will silently drift if upstream renames an
entry point, changes which one a dtype maps to, or moves the gate, so diff the
source on every bump:

| Rock copy | Upstream source | What to check |
|-----------|-----------------|---------------|
| `kOcmlTanhF32` / `kOcmlTanhF64` in `mlir/lib/Dialect/Rock/Transforms/LegalizeMathForTriton.cpp` | `tanh()` in `external/triton/third_party/amd/language/hip/libdevice.py` | Symbol strings must match, and the dtype split must stay f32/f64 (the pass computes anything narrower than f32 at f32). `kOcmlTanhF32` must also stay in sync with the `calleeName == "__ocml_tanh_f32"` match in `third_party/amd/lib/TritonAMDGPUToLLVM/BuiltinFuncToLLVM.cpp`, which is what turns the call into `llvm.amdgcn.tanh.f32`. |
| `kOcmlPowF32` / `kOcmlPowF64` | `pow()` in the same file | Symbol strings and the same f32/f64 split. The `__ocml_pown_*` integer-exponent entries are deliberately not mirrored: the pass only lowers `math.powf`, whose exponent is a float. |
| `tritonLowersTanhToNativeInst()` in `mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp` | the `getISAFamily() == ISAFamily::CDNA5` (historically `GFX1250`) guard on the `__ocml_tanh_f32` branch of `BuiltinFuncToLLVM.cpp` | The set of families Triton rewrites on. Note this is deliberately *not* LLVM's `FeatureTanhInsts`, which is wider: LLVM gives gfx13 `v_tanh_f32` but Triton does not rewrite there, so querying the LLVM feature would pick a full OCML library call believing it to be one instruction. If upstream widens the guard to a feature check or to more families, widen the switch to match; the point of choosing the call on these targets is that it stops being a call. |

```bash
git -C "$TRITON_REPO" diff "$OLD_TRITON..$NEW_TRITON" -- \
  third_party/amd/language/hip/libdevice.py \
  third_party/amd/lib/TritonAMDGPUToLLVM/BuiltinFuncToLLVM.cpp
```

A renamed symbol does not fail the build, and the lit tests that pin these
strings (`lowering_rock_legalize_math_for_triton.mlir`,
`fastmath-through-triton-transcendentals.mlir`) keep passing because they check
the same copies:
the reference stays unresolved until the `ocml.bc` link in `TritonToHsaco`, so it
surfaces as a link failure there or, for the gfx1250 / CDNA5 rewrite above, as a silent
loss of `v_tanh_f32` that `mlir/test/rocmlir-driver/tanh-isa.mlir` catches.

A moved gate is quieter still, since both sides keep compiling and the call is
valid either way. The one thing that does speak up is a bump adding an
`ISAFamily`: `tritonLowersTanhToNativeInst()` switches exhaustively over Triton's
enum precisely so that this becomes a build error and someone has to go read the
guard.

### 5.4.3 Mirrored assertion flags (from `HandleLLVMOptions.cmake`)

`LLVM_ENABLE_ASSERTIONS` is not just a switch on `assert()`: `HandleLLVMOptions`
turns it into `-UNDEBUG`, `-D_DEBUG`, `-D_GLIBCXX_ASSERTIONS` and a libc++
hardening mode. Triton's `CMakeLists.txt` never includes that module, and we
cannot `include()` it there (it would impose LLVM's whole flag policy on a
subtree that builds `-Werror` / `/WX`), so `cmake/triton.cmake` hand-copies the
block before `add_subdirectory(external/triton)`. Drift is silent -- a missed
define leaves Triton as the one subtree still on `NDEBUG` in an assertions
build, with nothing failing to announce it. This **manual copy** is keyed to the
pinned LLVM rather than to Triton, so diff it on every LLVM bump:

| Rock copy | Upstream source | What to check |
|-----------|-----------------|---------------|
| the `if(LLVM_ENABLE_ASSERTIONS)` block in `cmake/triton.cmake` | the `if( LLVM_ENABLE_ASSERTIONS )` block at `external/llvm-project/llvm/cmake/modules/HandleLLVMOptions.cmake:113-155` | Every define and option upstream adds under that `if` must appear in the copy, with the same build-type guard on `-UNDEBUG`. Three deviations are deliberate: the guard reads our `rocmlir_uppercase_build_type`, the `CHECK_CXX_SOURCE_COMPILES` probe is replaced by the `SUPPORTS_LIBCXX_HARDENING_MODE` that LLVM's own scope already cached, and the `LIBCXX_HARDENING_MODE` override warning is dropped as LLVM has already emitted it. |

```bash
git -C "$LLVM_REPO" diff "$OLD_LLVM_CORE..$NEW_LLVM_CORE" -- \
  llvm/cmake/modules/HandleLLVMOptions.cmake
```

### 5.5 Architecture Database (`AmdArchDb.cpp`)

`mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp` maps AMD GPU architectures to hardware
capabilities using Triton APIs (`ISAFamily`, `MfmaIntrinsic::selectFor`,
`WmmaIntrinsic::selectFor`, `TargetInfo`). When architecture support changes
upstream (new family, removed family, or family rename/reclassification), this
file needs review.

In particular, if upstream removes unsupported families/targets (e.g. `GCN5_1`,
`CDNA1`) or renames/reclassifies one (e.g. `GFX1250` -> `CDNA5`), make the same
cleanup in rocmlirTriton and dependent tests/scripts so behavior stays aligned.

For new-family additions (e.g. a hypothetical RDNA5 / CDNA5), use this checklist:

| Function | What to check |
|----------|---------------|
| `getMatrixAccelKind()` | Does the new arch support MFMA, WMMA, or scaled variants? Update the selection logic (version thresholds, `isF8F6F4`, `isScaledWmmaType`). |
| `getMaxNumChiplets()` | Update if the new arch has multi-chiplet GPUs. |
| `getMinNumCU()` | Add the new `ISAFamily` case with the minimum CU count. |
| `getMaxWavesPerEU()` | Add the new `ISAFamily` case with the correct occupancy limit. |
| `getWaveSize()` / `getLDSSize()` | These delegate to `TargetInfo`, so they should work automatically if Triton adds the arch. Verify. |
| `supportsTDM()` | Delegates to `TargetInfo`. Verify it returns the correct value for the new arch. |
| `isCDNA()` / `isRDNA()` | Delegate to `triton::amdgpu::isCDNA` / `isRDNA`, and `isCDNA()` picks the tuning space. Re-check the classification on every bump: a family moving between the two switches resizes the space with no build error. |
| `tritonLowersTanhToNativeInst()` | Not a hardware property but a mirror of a Triton guard; see section 5.4.2 before adding the new family to either arm of the switch. |

Also check that `tritonUtils.cpp::getMfmaVersion()` and
`tritonUtils.cpp::getWmmaVersion()` handle the new `ISAFamily` / chip string.
Additionally, `mlir/test/common_utils/amd_arch_db/binding.cpp` will need to have
its `py::enum_<ISAFamily>(...).value(...)` enum updated as well. That module is
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
git -C "$TRITON_REPO" diff "$OLD_TRITON..$NEW_TRITON" -- \
  third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h \
  third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp
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
| `HIPOptions.allow_flush_denorm`                   | (not in perfConfig)                   | `BackendOptions::allowFlushDenorm` -> `TritonToHsacoOptions::allowFlushDenorm` and `RockPrepareLLVMPassOptions::allowFlushDenorm` |

`knobs.amd.use_expert_scheduling` deliberately diverges from upstream Triton's
implementation at the final LLVM codegen step. Upstream appends
`"amdgpu-expert-scheduling-mode"` to the `translate_to_asm` flags, and
`python/src/llvm.cc` applies those flags by mutating LLVM's process-global
command-line options. `rocmlir-tuning-driver` compiles perf configs in parallel
worker threads, so `TritonToHsaco.cpp` must instead stamp the LLVM function
attribute `amdgpu-expert-scheduling-mode=true/false` on every defined function
after linking device libraries. LLVM's AMDGPU backend reads this attribute when
no process command-line occurrence of the global option exists; do not replace
it with the upstream global-option path during a Triton bump.

`HIPOptions.allow_flush_denorm` deliberately diverges from upstream at the
final LLVM codegen step. Upstream `compiler.py` stamps the legacy
`"denormal-fp-math-f32"` string attribute (`"preserve-sign"` when
`allow_flush_denorm` is true, `"ieee"` otherwise). LLVM has since moved
denormal controls to the `denormal_fpenv` enum attribute and only
auto-upgrades the string spelling when a module is **parsed** from IR text.
`TritonToHsaco.cpp` builds the LLVM module in memory, so copying upstream's
string attribute would leave it inert: f32 denormals would stay unflushed and
the AMDGPU backend would insert denormal range guards around every
`llvm.exp2`/`llvm.log2`. Instead, `setModuleFunctionAttributes()` stamps
`denormal_fpenv(float: preservesign)` when `allowFlushDenorm` is true (the
rocmlir default) and IEEE otherwise on every definition after device-library
linking. This keeps outlined device-library bodies consistent with the kernel's
hardware mode. Do not replace this with upstream's string attribute during a
Triton bump. Regression coverage: `triton-to-hsaco-denormal-mode.mlir` and
`triton-to-hsaco-outline-duplicated-ocml.mlir`.

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
cd build && ninja check-mlir && ninja check-triton-lit-tests && ninja check-rocmlir
```

`check-mlir` runs the upstream MLIR suite from the fetched LLVM tree and
`check-triton-lit-tests` runs Triton's lit suite from the fetched Triton tree.
Together they are the main signal that the new external branch pins plus our
mirrored rocmlirTriton updates still compose correctly. Run both before
`check-rocmlir`: they are far quicker, so regressions surface earlier. The
nightly pipeline runs both too (`check-mlir` there with
`MLIR_INCLUDE_INTEGRATION_TESTS=ON`), so a bump that lands on `develop` is
covered even if this step is skipped.

## Checklist Summary

Use this checklist to track progress:

- [ ] Record `OLD_REPO`
- [ ] Create/update external branches: `rmt-<date>`, `rmt-core-<date>`, `rmt-amd-<date>`
- [ ] Pull upstream commits into those branches and cherry-pick surviving downstream commits
- [ ] Update `ROCMLIR_TRITON_GIT_TAG`, `ROCMLIR_LLVM_GIT_TAG`, `ROCMLIR_AMD_CODEGEN_LLVM_GIT_TAG` in CMake files
- [ ] Build with `cmake.sh`
- [ ] Generate diff for `third_party/amd/backend/compiler.py`
- [ ] Generate diff for `python/src/llvm.cc`
- [ ] Generate diff for `third_party/amd/python/triton_amd.cc`
- [ ] Generate diff for `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`
- [ ] Generate diff for `third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h`
- [ ] Generate diff for `CMakeLists.txt` and `python/build_helpers.py`, then update `cmake/triton.cmake` for any new build options, downloads, or generated cache variables
- [ ] Generate diff for `include/triton/Dialect/Triton/IR/TritonAttrDefs.td` and reconcile the mirrored `CacheModifier` enum (see section 5.4)
- [ ] Generate diff for `include/triton/Dialect/Triton/IR/Traits.h` and reconcile the mirrored `kTritonMaxTensorNumElements` constant (see section 5.4.1)
- [ ] Generate diff for `third_party/amd/language/hip/libdevice.py` and `BuiltinFuncToLLVM.cpp`, and reconcile the mirrored `__ocml_*` symbol names in `LegalizeMathForTriton.cpp` and the arch gate in `tritonLowersTanhToNativeInst()` (see section 5.4.2)
- [ ] Generate diff for `llvm/cmake/modules/HandleLLVMOptions.cmake` and reconcile the mirrored `LLVM_ENABLE_ASSERTIONS` flag block in `cmake/triton.cmake` (see section 5.4.3)
- [ ] Check whether the pinned LLVM revision fixes the KV-cache raw-buffer bounds-checking bug and re-evaluate the N-loop clamp (see section 5.3.2)
- [ ] Check whether the pinned LLVM revision fixes the gfx950 packed-FP32 / MFMA hazard and re-evaluate the perfConfig pin in `mixr-attention-flash-decoding-3d-output-fusion.mlir` (see section 5.3.3)
- [ ] Check whether the pinned LLVM revision fixes llvm/llvm-project#225054 and re-evaluate `SKIPPED_PERF_CONFIGS` in `parameterSweeps.py` (see section 5.3.4)
- [ ] Update `Pipelines.cpp::makeTTIR()` for `make_ttir()` changes
- [ ] Update `Pipelines.cpp::makeTTGIR()` for `make_ttgir()` changes
- [ ] Update `Pipelines.cpp::makeLLIR()` for `make_llir()` Part 1 changes
- [ ] Refresh the `TRITON` prefix in `mlir/test/rocmlir-driver/pipelines.mlir` if any of `makeTTIR` / `makeTTGIR` / `makeLLIR` changed (see section 5.1)
- [ ] Keep `rock-check-lds-limit` after the last pass that writes `ttg.shared` and before `convert-triton-amdgpu-to-llvm`, and keep the LDS check in `resolve-kernel-launch-params` (see section 5.1)
- [ ] Update `TritonToHsaco.cpp::translateTritonToHsaco()` for `make_llir()` Part 2 changes
- [ ] Preserve `TritonToHsaco.cpp::setModuleFunctionAttributes()` stamping of linked definitions for expert scheduling and `denormal_fpenv` (do not copy upstream's process-global expert-scheduling option or `"denormal-fp-math-f32"` string; see section 8.1)
- [ ] Keep LLVM IR verification in `TritonToHsaco.cpp` gated on `kVerifyLLVMIR` (upstream verifies unconditionally in `to_module()` and in the codegen pass manager; see section 5.2)
- [ ] Preserve `TritonToHsaco.cpp::disableHighDuplicationDeviceLibInlining()` and its call from `translateTritonToHsaco()` (see section 5.2)
- [ ] Update `TritonToHsaco.cpp` for LLVM function changes (`initializeLLVMTargets`, `createTargetMachine`, `optimizeModule`)
- [ ] Update `tritonUtils.cpp::getMfmaVersion()` if changed
- [ ] Update `tritonUtils.cpp::getWmmaVersion()` if changed
- [ ] Update `tritonUtils.cpp::mlirTypeToScaleDotElemType()` if changed
- [ ] Update `AmdArchDb.cpp` if new `ISAFamily` added (see section 5.5)
- [ ] Diff `TargetFeatures.cpp` and re-check the `isCDNA` / `isRDNA` classification that picks the tuning space (see section 5.5)
- [ ] Add a representative chip to `DEFAULT_ARCHES` in `generateLDSBlacklist.py` if a new ISA family was added (see section 5.5)
- [ ] Mirror new or dropped `knobs.amd.*` switches, and if the perfConfig gained or lost a field, keep configs saved against the old schema readable (see sections 8.1 and 8.2)
- [ ] Reconcile downstream-only commits on the new external branch tips (drop upstreamed commits, keep required ones)
- [ ] Keep architecture removals/renames/reclassification in sync with upstream (`AmdArchDb.cpp`, tests, perf scripts; see section 5.5), e.g. `GCN5_1`/`CDNA1` removal and `GFX1250 -> CDNA5`
- [ ] Build project with `cmake.sh`
- [ ] Regenerate `librockcompiler_deps.cmake` with `get_fat_library_deps_list.pl`
- [ ] Regenerate the LDS blacklist (`generateLDSBlacklist.py` from `build/bin`), rebuild, and commit the updated `.inc`
- [ ] Run tests with `cd build && ninja check-mlir && ninja check-triton-lit-tests && ninja check-rocmlir`
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

1. Check the new signature in Triton headers (e.g.,
   `$TRITON_REPO/third_party/amd/include/TritonAMDGPUTransforms/Passes.h`)
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

(For entries under `external/triton/...` and `external/llvm-project/...`, read
those as paths inside your external repo checkouts: `$TRITON_REPO` /
`$LLVM_REPO`.)

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
| Mirrored `__ocml_*` symbol names | `mlir/lib/Dialect/Rock/Transforms/LegalizeMathForTriton.cpp` |
| Triton libdevice symbol source | `external/triton/third_party/amd/language/hip/libdevice.py` |
| Mirrored tanh rewrite arch gate | `mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp` (`tritonLowersTanhToNativeInst`) |
| Triton tanh rewrite source | `external/triton/third_party/amd/lib/TritonAMDGPUToLLVM/BuiltinFuncToLLVM.cpp` |
| Mirrored `LLVM_ENABLE_ASSERTIONS` flag block | `cmake/triton.cmake` |
| LLVM assertion flag source | `external/llvm-project/llvm/cmake/modules/HandleLLVMOptions.cmake` |
| Triton compiler.py | `external/triton/third_party/amd/backend/compiler.py` |
| Triton llvm.cc | `external/triton/python/src/llvm.cc` |
| Triton pass bindings | `external/triton/third_party/amd/python/triton_amd.cc` |
| Triton AccelerateMatmul | `external/triton/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp` |
| Build script | `cmake.sh` |
| Fat library deps generator | `mlir/utils/jenkins/static-checks/get_fat_library_deps_list.pl` |
| Fat library deps list | `mlir/tools/rocmlir-lib/librockcompiler_deps.cmake` |
| LDS blacklist generator | `mlir/utils/performance/generateLDSBlacklist.py` |
| LDS blacklist table | `mlir/include/mlir/Dialect/Rock/Tuning/LdsBlacklistPerfconfigs.inc` |
