# Guide: Bumping the Triton Version in rocmlirTriton

## Overview

rocmlirTriton no longer vendors `external/triton` and
`external/llvm-project` in-tree. It now fetches pinned external branches at
configure time:

- `cmake/triton.cmake` (FetchContent)
  - `ROCMLIR_TRITON_GIT_BRANCH` (`rmt-<date>`)
  - `ROCMLIR_LLVM_GIT_BRANCH` (`rmt-core-<date>`)
  - `ROCMLIR_AMD_CODEGEN_LLVM_GIT_BRANCH` (`rmt-amd-<date>`)

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

Read the current branch/tag pins from `cmake/triton.cmake`

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
  - `ROCMLIR_TRITON_GIT_BRANCH`
  - `ROCMLIR_LLVM_GIT_BRANCH`
  - `ROCMLIR_AMD_CODEGEN_LLVM_GIT_BRANCH`
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
    third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp \
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
Upstream verifies unconditionally in two places: the `to_module()` binding calls
`translateModuleToLLVMIR`, and `translateLLVMIRToASM()` adds
`createVerifierPass()` after the always-inliner (both in `python/src/llvm.cc`,
the latter mirrored by `emitMachineCode()`). Each is a full IR walk on every
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

### 5.3.5 gfx11 WMMA + InlineSpiller "corrupted sub-interval"

Distinct from the spill-slot bug in 5.3.4: this one is an assertion, not a
miscompile, and it fires at compile time.

```
InlineSpiller.cpp: Assertion `OrigVNI && "corrupted sub-interval"' failed.
```

in `InlineSpiller::reMaterializeFor`.

The cause is a stale early-clobber kill slot. A two-address pseudo whose def is
`@earlyclobber` makes LiveIntervals end the *tied use* at the early-clobber slot
rather than the regular one (`LiveIntervalCalc::extendToUses`, the
`isRegTiedToDefOperand` branch), which is correct while the tie exists.
`TwoAddressInstructionPass::convertInstTo3Addr` can then rewrite the op into its
three-address form, dropping the tie -- AMDGPU:
`V_WMMA_*_twoaddr` -> `V_WMMA_*_threeaddr`, i.e.
`@earlyclobber $vdst,$vdst = $src2` -> `@earlyclobber $vdst`.
`SIInstrInfo::convertToThreeAddress` fixes up the live range of the new
early-clobber *def* but not of the formerly tied *use*, which keeps ending at a
slot nothing redefines any more. `-verify-machineinstrs` reports it directly:

```
*** Bad machine code: Live segment ending at early clobber slot must be
    redefined by an EC def in the same instruction ***
```

It stays latent until that register is spilled, because `reMaterializeFor` is
the only reader that queries the *original* interval at `getRegSlot(/*EC=*/true)`.
So three things must coincide: the converted WMMA is the last use of its `src2`,
that register gets spilled, and the target has early-clobber WMMA. Both GFX11
and GFX12 WMMA carry `@earlyclobber $vdst` with `isConvertibleToThreeAddress`,
so neither is immune; only the register pressure differs.

Observed on gfx1151 (and gfx1100) with the `attn:v1:32,32,256,...` perf config,
whose `kPerBlock=256` exceeds the first GEMM's K and so unrolls the whole
padded K into one `tt.dot`, spilling hard. A standalone repro:

```bash
rocmlir-gen -m 384 -n 384 -k 128 -gemmO 128 --transA=true --transB=true \
  --transC=false --transO=true \
  -perf_config=attn:v1:32,32,256,1,1,2,0,4,1,0,0 \
  --operation gemm_gemm -t f16 --arch gfx1151 -pv \
| rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c
```

The fix is to recompute the formerly tied register's live range once the tie is
gone, at the end of `convertInstTo3Addr` (after the old instruction is erased --
doing it inside the target hook trips over the stale MI still on the use list):

```cpp
  if (LIS && RegB.isVirtual() && LIS->hasInterval(RegB))
    LIS->shrinkToUses(&LIS->getInterval(RegB));
```

Validated against the whole e2e + fusion suite. It has to land on **both**
branch tips: `rmt-amd-<date>` is what `rocmlir-driver` links and therefore what
fixes the failures, `rmt-core-<date>` keeps `llc`/`opt` consistent. On every
bump, re-check whether upstream has fixed it before carrying the patch again.

### 5.4 Mirrored Enums / Attributes (from `TritonAttrEnums.td`)

Some Triton enums are hand-replicated in the Rock dialect so we can carry the
value through Rock IR (and map it back onto Triton when lowering) without taking
a TableGen dependency on the Triton dialect. These are **manual copies** and
will silently drift if upstream changes them, so diff the source on every bump:

| Rock copy | Upstream source | What to check |
|-----------|-----------------|---------------|
| `CacheModifier` / `CacheModifierAttr` in `mlir/include/mlir/Dialect/Rock/IR/RockAttrDefs.td` | `TT_CacheModifierAttr` in `external/triton/include/triton/Dialect/Triton/IR/TritonAttrEnums.td` | Names **and** integer values must match one-to-one (currently `none=1, ca=2, cg=3, wb=4, cs=5, wt=6, cv=7`). The Rock->Triton lowering relies on the integer values lining up. The enum lived in `TritonAttrDefs.td` until the 2026-11 bump split the plain `I32EnumAttr`s out into `TritonAttrEnums.td`; diff both, since a future split can move it again and a diff of the wrong file reports no change. |

```bash
git -C "$TRITON_REPO" diff "$OLD_TRITON..$NEW_TRITON" -- \
  include/triton/Dialect/Triton/IR/TritonAttrEnums.td \
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
| `kOcmlTanhF32` / `kOcmlTanhF64` / `kTritonFastTanhF32` in `mlir/lib/Dialect/Rock/Transforms/LegalizeMathForTriton.cpp` | `tanh()` and `fast_tanhf()` in `external/triton/third_party/amd/language/hip/libdevice.py` | Symbol strings must match, and the dtype split must stay f32/f64 (the pass computes anything narrower than f32 at f32). The two f32 spellings are not interchangeable: `kTritonFastTanhF32` must stay in sync with the `calleeName == "__triton_hip_fast_tanhf"` match in `third_party/amd/lib/TritonAMDGPUToLLVM/BuiltinFuncToLLVM.cpp`, which is what turns the call into `llvm.amdgcn.tanh.f32` on the families gated below *and expands it inline everywhere else*, while `kOcmlTanhF32` must stay a symbol that file does **not** match, so that `--disable-fast-math` keeps getting the real library body from the `ocml.bc` link. Swapping one for the other still builds and runs: `mlir/test/rocmlir-driver/tanh-isa.mlir` is what catches it, through the `NATIVE` checks one way and the `OCML` checks the other. |
| `kOcmlPowF32` / `kOcmlPowF64` | `pow()` in the same file | Symbol strings and the same f32/f64 split. The `__ocml_pown_*` integer-exponent entries are deliberately not mirrored: the pass only lowers `math.powf`, whose exponent is a float. |
| `tritonLowersTanhToNativeInst()` in `mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp` | the `getISAFamily() == ISAFamily::CDNA5` (historically `GFX1250`) guard on the `__triton_hip_fast_tanhf` branch of `BuiltinFuncToLLVM.cpp` | The set of families Triton rewrites on. Note this is deliberately *not* LLVM's `FeatureTanhInsts`, which is wider: LLVM gives gfx13 `v_tanh_f32` but Triton does not rewrite there, so querying the LLVM feature would pick a full OCML library call believing it to be one instruction. If upstream widens the guard to a feature check or to more families, widen the switch to match; the point of choosing the call on these targets is that it stops being a call. |

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

The 2026-11 bump is the worked example: the rewrite moved off `__ocml_tanh_f32`
onto `__triton_hip_fast_tanhf`, so gfx1250 quietly stopped emitting
`v_tanh_f32`. Renaming `kOcmlTanhF32` to follow it fixes that and breaks the
other direction, because the fast spelling expands inline on targets without
the instruction and `--disable-fast-math` stops reaching OCML at all; hence the
two separate constants.

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

### 5.4.4 Mirrored IR shapes (from `BufferOpsEmitter.cpp`)

`rock-fold-oob-buffer-ops` reads Triton's own output rather than a symbol or a
constant: it recognises the offset the buffer-op emitter hands a masked-off
lane, proves it past `numRecords`, and drops the access so the stored value and
everything feeding it can die. That makes the emitter's **code shape** a mirrored
copy like the ones above, and a quieter one -- a shape change does not fail to
build or to run, it just stops the fold from firing, and the only symptom is the
padding epilogue staying alive.

| Rock copy | Upstream source | What to check |
|-----------|-----------------|---------------|
| `classifyOobOffset()` in `mlir/lib/Dialect/Rock/Transforms/FoldOobBufferOps.cpp` | `BufferEmitter::fillCommonArgs()` in `external/triton/third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp` | How a masked lane's `voffset` is formed and what reaches `soffset`. Today the emitter selects a bare `-1` (or `-accessBytes` for atomics, which fault on a misaligned address even out of bounds) and passes `soffset` through, so a zero-`soffset` bound on `voffset` decides on its own. If the emitter starts relating the two again, the pass declines every such access and silently stops reclaiming. |

```bash
git -C "$TRITON_REPO" diff "$OLD_TRITON..$NEW_TRITON" -- \
  third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp
```

The 2026-11 bump is the worked example in the other direction. Upstream used to
lift a uniform part of the offset into an soffset SGPR, which made a bare
sentinel unsound (the hardware checks `voffset + soffset`), so it emitted a
correlated pair that the pass matched structurally:

```
%c    = llvm.icmp "sge" %uniform, 0
soff  = llvm.select %c, %uniform, 0
%oob  = llvm.select %c, llvm.sub(-1, %uniform), -2147483648
```

That scheme is gone, together with the `annotate-buffer-op-split-safety` pass
behind it: a bare `-1` ends past 2^32 and so is out of bounds for any 32-bit
`numRecords` whatever the soffset adds. gfx1100 and gfx1200 stopped taking the
split path, their fold went from 24 of 32 stores to 31, and the matcher was
dropped as unreachable.

The counts in `mlir/test/rocmlir-driver/oob-buffer-store-fold{,-padded-tile}.mlir`
are what makes any of this visible, so treat a count moving there as a signal to
re-read the emitter rather than as a number to refresh. Keep the
`-NOT: llvm.icmp "sge"` lines: they are the tripwire for the correlated form
coming back.

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
`ISAFamily`, and so do most of ours, so a new variant is usually a build error in
`AmdArchDb.cpp` rather than a silent fallback. Two functions still end in
`default:` and will quietly bucket a new family: `getMaxNumChiplets()` (falls back
to 1 chiplet) and `getMaxWavesPerEU()` (falls back to 1 wave, which would
drastically misreport occupancy). Check those two by hand on every bump, and
prefer making a switch exhaustive over adding a `default:` when editing. The
`ISAFamily` enum is defined in `TargetFeatures.h` (it moved out of the
now-removed `TritonAMDGPUToLLVM/TargetUtils.h`); diff it for new entries, along
with the `.cpp` that holds the `isCDNA` / `isRDNA` switch bodies (a
reclassification does not touch the header):

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

From the **build directory**, after a successful build, run:

```bash
../scripts/get_fat_library_deps.sh > ../mlir/tools/rocmlir-lib/librockcompiler_deps.cmake
```

The script shells out to `ninja -t deps bin/mlir-mixr-fullc-test`, so it must run
with the build directory as the working directory (that is where `build.ninja`
lives), and the build must have completed, since it reads the dependency closure
ninja recorded for that link step. The archive paths it gets back are relative to
the build directory, and it splits them into `__mlir_libs` (from
`external/llvm-project/llvm/lib/`) and `__rocmlir_libs` (from `lib/`).

Two things to keep in mind when reviewing the regenerated file:

- The `LLVMX86*` backend libraries are **deliberately filtered out**. Host
  codegen is only needed by the e2e / JIT-runner path, which `MLIR_ENABLE_ROCM_RUNNER`
  builds pull in on their own; `librockCompiler.a` only ever targets AMDGPU. If a
  regeneration puts `LLVMX86CodeGen` & co. back into the list, something started
  linking the X86 backend into `TritonToHsacoTranslation` again -- fix that
  instead of committing the entries.
- The list is sorted with plain `sort`, so the ordering follows the ambient
  locale. Regenerate with the same locale the file was last generated under
  (`en_US.UTF-8`) or the diff will be pure reordering noise.

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
| FPSan instrumentation mode | `compiler.py` make_ttgir | Not implemented (the call site is kept commented out in `Pipelines.cpp::makeTTGIR()`) |
| ConSan (concurrency sanitizer) | `compiler.py` make_llir (`prepare_consan_captures`, `concurrency_sanitizer`, and the `gluon` canonicalizer/CSE pair that follows them) | Not implemented; upstream runs the whole group only under `is_enabled(options, "consan")` |
| `gluon_to_ttgir()` and every `passes.gluon.*` it adds | `compiler.py` | A separate frontend entry point (wired in as the `ttgir` stage only when the source is Gluon), not a stage of `make_ttgir`. The one `passes.gluon.add_inliner` on the normal path, in `make_llir`, *is* mirrored (`createGluonInline`). Do not read `add_warp_pipeline` / `add_allocate_warp_groups` / `add_resolve_auto_encodings` out of `gluon_to_ttgir` as missing TTGIR passes. |

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
| `knobs.amd.use_coexec_scheduler`                  | (not in perfConfig -- debug-only)     | not mirrored as a knob; `TritonToHsaco.cpp` mirrors the `is_coexec_scheduler_enabled()` arch gate and the `num_warps <= 4` condition only |
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

`HIPOptions.schedule_hint` is **not** mirrored, but it is no longer the dead
field this guide used to describe. The 2025 gutting it records did happen: the
TTGIR/LLIR sched-hint passes and the
`memory-bound-attention -> amdgpu-sched-strategy=iterative-ilp` mapping were
removed, and that is why `scheduleHint` was dropped from the perfConfig schema
in v3 (`RockDialect.cpp` still decodes and discards the v2 token). As of the
2026-11 bump upstream has reused the name for something else: `parse_options()`
now accepts exactly one value, `"mfma-schedule"`, and `make_llir()` ends with

```python
# Opt-in MFMA scheduler, see the schedule_hint option (validated in parse_options).
if options.schedule_hint == "mfma-schedule" and options.arch in ("gfx942", "gfx950"):
    amd.add_mfma_schedule_pass(kernel_fn, options.arch)
```

which runs `mlir::triton::AMD::runMFMASchedulePass` on the kernel function
after `optimize_module`. We do not expose it: the default is still `''`, so
not mirroring it matches upstream's default behaviour exactly. Treat it as an
open question rather than a settled omission -- it is an LLVM-level,
per-function, gfx942/gfx950-only scheduler, so if we ever want it, it is a
`BackendOptions` / `TritonToHsacoOptions` knob (like expert scheduling), not a
revival of the old perfConfig bitfield. Do not re-add the retired field name
to the schema; see section 8.2.

`knobs.amd.use_coexec_scheduler` (new in the 2026-11 bump) is likewise not
mirrored as a knob. `TritonToHsaco.cpp` mirrors only the arch half of
`is_coexec_scheduler_enabled()` plus the `num_warps <= 4` condition, i.e. the
upstream default with the environment override left out, which is how the other
debug-only knobs are handled.

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

### 12.1 Running the suites by hand

`ninja -C build check-rocmlir` is the whole suite. To iterate on a subset, run
`lit` directly against the configured test directories under `build/mlir/test`
(not `mlir/test`, which has no `lit.site.cfg.py`):

```bash
LD_LIBRARY_PATH=$ROCM_PATH/lib lit -v -j 16 build/mlir/test/Dialect build/mlir/test/Conversion
LD_LIBRARY_PATH=$ROCM_PATH/lib lit -v build/mlir/test/e2e --filter 'config_23_7'
```

Three things that cost time if you do not know them:

- `LD_LIBRARY_PATH` is not optional. `build/mlir/test/lit.site.cfg.py` calls
  `apply_arch_features()`, which imports the `hip` Python module to enumerate
  agents; without `libamdhip64.so` on the loader path every suite dies while
  parsing its config, before running a single test.
- `ninja -C build` does **not** rebuild everything the suite runs. The gtest
  binary and the CAPI test binaries are not in the default target, so after a
  change to a `.td` or to a dialect you can spend a while reading failures from
  a stale binary. Build them explicitly:
  `ninja -C build MLIRRockUnitTests mlir-mixr-capi-test mlir-mixr-fullc-test
  mlir-mixr-split-k-test mlir-reduce-fusible-test`.
- The GPU e2e suites (`build/mlir/test/e2e`, `build/mlir/test/fusion/e2e`) get
  flaky above roughly `-j 8` on a single-GPU host: contention makes individual
  configs time out. Before investigating an e2e failure from a full-suite run,
  re-run it alone with `--filter`.

### 12.2 Upstream MLIR/LLVM churn that only shows up in the lit suite

Most of the fallout from a bump is not in the Triton mirrors at all: it is
upstream MLIR changing how IR is *spelled*. The build never notices, and it
lands as hundreds of lit failures at once. Expect this to be the bulk of the
work, and triage by error text rather than test by test, since one upstream
change usually accounts for a whole block of the failure list:

```bash
LD_LIBRARY_PATH=$ROCM_PATH/lib lit -v -j 16 build/mlir/test/... > /tmp/lit.log 2>&1
grep -oE "error: [^\"]{0,90}" /tmp/lit.log | sed 's/[0-9]\+/N/g' | sort | uniq -c | sort -rn
```

The table below is what the 2026-11 bump hit, which doubles as a list of the
kinds of change to look for. Note how many are a *printer* change: the input IR
still parses, only the `CHECK` lines fail, and it reads like a behaviour
regression until you diff the actual tool output.

| Symptom in the log | Upstream change | What we did |
|--------------------|-----------------|-------------|
| `inherent attribute 'x' cannot be parsed from attr-dict when strict properties in assembly format is enabled` | `useStrictPropertiesInAssemblyFormat` now defaults to 1 in `external/llvm-project/mlir/include/mlir/IR/DialectBase.td`, and the old `usePropertiesForAttributes` opt-out is gone: every ODS-declared attribute is a property, and `attr-dict` no longer round-trips it | Set `useStrictPropertiesInAssemblyFormat = 0` on `Rock_Dialect` and `MIGraphX_Dialect`, keeping the legacy merged `{...}` spelling. That flag is deprecated upstream; migrating the formats to `prop-dict` (which prints `{discardable} <inherent>`) is a separate, test-wide change |
| `expected '<'` on `rock.store ... by set`, or `#rock<CacheModifier none>` no longer parsing | the default `EnumAttr` assembly format became `` `<` $value `>` ``, turning `#rock<Foo bar>` into `#rock.Foo<bar>` and a bound enum operand into `<bar>` | Pinned the old spelling: explicit `let assemblyFormat = "$value";` on the Rock `EnumAttr`s, plus the `enum($x)` op-format directive for enum operands |
| `custom op 'tosa.transpose' expected 'perms'` and friends | TOSA moved inherent attributes out of `attr-dict` into keyword syntax: `perms([...])`, `axis(N)`, `values(...)`, `pad(...) stride(...) dilation(...) acc_type(...)`, `block_size<...>`, `operator_name(...)` | Mechanical test rewrite; `scripts/migrate_tosa_attr_syntax.py` does it, including the `<{...}>` properties form needed by ops written in generic form |
| `CHECK` lines for `tosa.const` / `tosa.cast` / `tosa.conv2d` that used to match | `tosa.const` gained an assembly format (so it stopped printing in generic form), and the optional `input_unsigned(false)` / `local_bound(false)` clauses print whenever the attribute is present, which it is on anything our builders create | Test-only, but *not* uniform: a cast parsed from test input has no clause while one a pass created does, so blanket edits are wrong |
| `custom op 'bufferization.to_tensor' inherent attribute 'restrict' ...`, `rocdl.make.buffer.rsrc ... expected ','` | the same strict-properties rule applied to upstream dialects, plus `make.buffer.rsrc` now spelling the `numRecords` type in its format | Test-only syntax updates |
| `'llvm.mlir.constant' op attribute and type have different integer types` | new exact-match verifier in `LLVMDialect.cpp`; `llvm.mlir.constant(-1 : i32) : i64` is no longer representable | Decide per test: in `fold-oob-buffer-ops.mlir` it makes the sign/zero-extension cases unreachable, along with the `FoldOobBufferOps.cpp::matchConstantInt` logic they covered |
| `fastmathFlags = #llvm.fastmath<...>` not found | LLVM-dialect arithmetic ops print a `fastmath<...>` keyword clause now; `llvm.call` still uses the attr-dict form | Test-only, but the two spellings coexist, so rewrite per op rather than globally |
| `'False' is invalid value for boolean argument! Try 0 or 1` | `cl::opt<bool>` stopped matching case-insensitively. Lowercase `true`/`false` and `1`/`0` still work; only Python's capitalized `True`/`False` broke, so the message's "Try 0 or 1" undersells it | Lowercased `-transA=False` and friends on the RUN lines. Also fix the generators: anything that interpolates a Python `bool` straight into a flag (`f"-transA={self.trans_a}"` in perfRunner/tuningRunner) emits the capitalized form, so it needs `str(x).lower()`. Leave the `to_command_line()` tuning key alone -- it already spelled these in words, and the word form is what `RockTuningImpl.cpp`, the shipped `tier1-*`/`selected-*` configs and existing tuning DBs all use |
| `pipelines.mlir` / `runner-pipelines.mlir` option-string mismatches | upstream pass options changed (`arith-expand` gained `include-f8e4m3fn` / `include-f8e5m2`, `tosa-to-linalg` gained `allow-non-finites`, `one-shot-bufferize` renamed `check-parallel-regions` to `may-have-parallel-regions`) | Re-pin from the actual `-dump-pipelines` output, after checking that no rocMLIR pass vanished from the list |
| `.amdgcn_target "amdgcn-amd-amdhsa-..."` not found; the "codegen with no subarch in the target triple" warning gone | the AMDGPU triple now encodes the subarch (`amdgpu12.50-amd-amdhsa-unknown-gfx1250`) | Loosened the triple expectations; the vanished warning is an improvement, not a loss |
| `hidden_block_count_x` missing from the HSA kernarg metadata | the backend no longer lists unused hidden kernargs | Re-anchored `no-heap-ptr-func-check.mlir` on `kernarg_segment_size` |
| `tt.reshape ... allow_reorder` failing to parse, then `Expected result encoding ...` | `allow_reorder` is gone (reshape always reorders; the opt-in is now `efficient_layout`) and the verifier requires the inferred linear result layout | Dropped the keyword and gave the result the linear encoding |
| `'builtin.module' op requires "ttg.num-warps" to be a positive power of two` | new TritonGPU module verifier | The test used 17 warps to exceed the workgroup limit; switched to 32 and updated the expected block size |
| `Assertion `status == expectedStatus' failed` out of `createConstantFloatOp` | not upstream churn as such, but it surfaces through it: `f8E8M0FNU` (the MX scale type) has **no representable zero**, so converting `0.0` into it yields 2^-127 with `opInexact`. `APFloat::semanticsHasZero` is the predicate | The caller was building a zero-initialised vector purely as scaffolding and then overwriting every lane, so it was changed to materialise the pattern as one `arith.constant dense<[...]>`. Do not relax the `expectedStatus` assert: it is what catches genuine quantization mistakes |
| A `COUNT-n` on `@llvm.exp2` (or any elementwise intrinsic) finding fewer calls than pinned | LLVM's SLP vectorizer now widens them, per-arch: 8 scalar `exp2.f32` became 2 `exp2.v4f32` on CDNA and 8 `exp2.v2f32` on gfx1250, while gfx1100/gfx1200 stayed scalar | Re-pin to the vector form and say in the comment that `n` x width is the same work the scalar count used to express. Counting calls stopped being a proxy for counting lanes |

Two lessons worth repeating on the next bump:

- **Fix the parse errors first, then re-triage.** A single unparseable op fails
  a whole directory of e2e tests at the first command in the pipe, and those
  failures say nothing about the rest of the pipeline. The 2026-11 bump went
  from 736 failures to 33 on two one-line dialect changes plus the mechanical
  syntax rewrites; only then were the real regressions visible.
- **A printed-form change can hide a behaviour change.** Confirm against actual
  tool output (run `rocmlir-opt` / `rocmlir-driver` on the test input) instead
  of assuming an expectation is merely stale. That is how the fast-tanh
  regression in section 5.4.2 and the `oob-buffer-store-fold` store-count drop
  were caught, and conversely how the `tosa.cast` clause churn was confirmed
  harmless.

## Checklist Summary

Use this checklist to track progress:

- [ ] Record `OLD_REPO`
- [ ] Create/update external branches: `rmt-<date>`, `rmt-core-<date>`, `rmt-amd-<date>`
- [ ] Pull upstream commits into those branches and cherry-pick surviving downstream commits
- [ ] Update `ROCMLIR_TRITON_GIT_BRANCH`, `ROCMLIR_LLVM_GIT_BRANCH`, `ROCMLIR_AMD_CODEGEN_LLVM_GIT_BRANCH` in CMake files
- [ ] Build with `cmake.sh`
- [ ] Generate diff for `third_party/amd/backend/compiler.py`
- [ ] Generate diff for `python/src/llvm.cc`
- [ ] Generate diff for `third_party/amd/python/triton_amd.cc`
- [ ] Generate diff for `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`
- [ ] Generate diff for `third_party/amd/include/Dialect/TritonAMDGPU/IR/TargetFeatures.h` and `.../lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp`
- [ ] Generate diff for `CMakeLists.txt` and `python/build_helpers.py`, then update `cmake/triton.cmake` for any new build options, downloads, or generated cache variables
- [ ] Generate diff for `include/triton/Dialect/Triton/IR/TritonAttrEnums.td` (and `TritonAttrDefs.td`) and reconcile the mirrored `CacheModifier` enum (see section 5.4)
- [ ] Generate diff for `include/triton/Dialect/Triton/IR/Traits.h` and reconcile the mirrored `kTritonMaxTensorNumElements` constant (see section 5.4.1)
- [ ] Generate diff for `third_party/amd/language/hip/libdevice.py` and `BuiltinFuncToLLVM.cpp`, and reconcile the mirrored `__ocml_*` / `__triton_hip_fast_tanhf` symbol names in `LegalizeMathForTriton.cpp` and the arch gate in `tritonLowersTanhToNativeInst()`, keeping the fast and library spellings distinct (see section 5.4.2)
- [ ] Generate diff for `llvm/cmake/modules/HandleLLVMOptions.cmake` and reconcile the mirrored `LLVM_ENABLE_ASSERTIONS` flag block in `cmake/triton.cmake` (see section 5.4.3)
- [ ] Generate diff for `third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp` and re-read `classifyOobOffset()` against the masked-lane offset it builds; a moved store count in `oob-buffer-store-fold*.mlir` is the signal (see section 5.4.4)
- [ ] Check whether the pinned LLVM revision fixes the KV-cache raw-buffer bounds-checking bug and re-evaluate the N-loop clamp (see section 5.3.2)
- [ ] Check whether the pinned LLVM revision fixes the gfx950 packed-FP32 / MFMA hazard and re-evaluate the perfConfig pin in `mixr-attention-flash-decoding-3d-output-fusion.mlir` (see section 5.3.3)
- [ ] Check whether the pinned LLVM revision fixes llvm/llvm-project#225054 and re-evaluate `SKIPPED_PERF_CONFIGS` in `parameterSweeps.py` (see section 5.3.4)
- [ ] Check whether the pinned LLVM revision still needs the `convertInstTo3Addr` live-range fix for the gfx11 WMMA InlineSpiller assertion, and carry it on both `rmt-core-*` and `rmt-amd-*` if so (see section 5.3.5)
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
- [ ] Update `AmdArchDb.cpp` if new `ISAFamily` added, including the two switches that still fall through a `default:` (`getMaxNumChiplets`, `getMaxWavesPerEU`) and the pybind enum in `mlir/test/common_utils/amd_arch_db/binding.cpp` (see section 5.5)
- [ ] Diff `TargetFeatures.cpp` and re-check the `isCDNA` / `isRDNA` classification that picks the tuning space (see section 5.5)
- [ ] Add a representative chip to `DEFAULT_ARCHES` in `generateLDSBlacklist.py` if a new ISA family was added (see section 5.5)
- [ ] Mirror new or dropped `knobs.amd.*` switches, re-check `HIPOptions.schedule_hint` (upstream reused the name for the opt-in MFMA scheduler), and if the perfConfig gained or lost a field, keep configs saved against the old schema readable (see sections 8.1 and 8.2)
- [ ] Reconcile downstream-only commits on the new external branch tips (drop upstreamed commits, keep required ones)
- [ ] Keep architecture removals/renames/reclassification in sync with upstream (`AmdArchDb.cpp`, tests, perf scripts; see section 5.5), e.g. `GCN5_1`/`CDNA1` removal and `GFX1250 -> CDNA5`
- [ ] Build project with `cmake.sh`
- [ ] Regenerate `librockcompiler_deps.cmake` with `scripts/get_fat_library_deps.sh` (from the build directory; check that no `LLVMX86*` entries reappeared)
- [ ] Regenerate the LDS blacklist (`generateLDSBlacklist.py` from `build/bin`), rebuild, and commit the updated `.inc`
- [ ] Run tests with `cd build && ninja check-mlir && ninja check-triton-lit-tests && ninja check-rocmlir`
- [ ] Refresh IR-syntax expectations for upstream MLIR printer/parser changes, triaging by error text, and confirm each one against real tool output rather than assuming the expectation is stale (see section 12.2)
- [ ] Rebuild the unit-test and CAPI binaries before trusting their results, and re-run e2e failures individually (see section 12.1)
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
| Triton `CacheModifier` source | `external/triton/include/triton/Dialect/Triton/IR/TritonAttrEnums.td` |
| Mirrored `kTritonMaxTensorNumElements` constant | `mlir/lib/Dialect/Rock/Tuning/RockTuningImpl.cpp` |
| Triton `maxTensorNumElements` source | `external/triton/include/triton/Dialect/Triton/IR/Traits.h` |
| Mirrored `__ocml_*` / `__triton_hip_fast_tanhf` symbol names | `mlir/lib/Dialect/Rock/Transforms/LegalizeMathForTriton.cpp` |
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
| Fat library deps generator | `scripts/get_fat_library_deps.sh` |
| Fat library deps list | `mlir/tools/rocmlir-lib/librockcompiler_deps.cmake` |
| TOSA attribute-syntax migration helper | `scripts/migrate_tosa_attr_syntax.py` |
| Mirrored masked-lane offset shape | `mlir/lib/Dialect/Rock/Transforms/FoldOobBufferOps.cpp` (`classifyOobOffset`) |
| Triton buffer-op emitter source | `external/triton/third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp` |
| LDS blacklist generator | `mlir/utils/performance/generateLDSBlacklist.py` |
| LDS blacklist table | `mlir/include/mlir/Dialect/Rock/Tuning/LdsBlacklistPerfconfigs.inc` |
