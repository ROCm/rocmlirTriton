# Analysis of the GEMM Tuning Space in `RockTuningImpl.cpp`

This document is a deep analysis of the tuning parameter ranges used by the
`Exhaustive` / `Full` tuning space for `RockGemmWrapperInterface` ops, declared in
`getAccelRangeGemm` / `getAccelRangeGemmGemm` in
`mlir/lib/Dialect/Rock/Tuning/RockTuningImpl.cpp`.

The values were originally chosen as quick defaults based on rocMLIR experience and a short
look at AITer. The goal of this document is to:

1. Catalog **what we currently search**.
2. Compare to **what AITer and tritonBLAS actually use** in production / for autotuning.
3. Use the **LDS budget** of each architecture to validate or invalidate each choice
   (especially the `numStages={1,2,3}` MFMA / `{1,2}` WMMA question).
4. Recommend a **revised tuning space** per accelerator family.

Cross-references throughout assume the file paths in this repo:

- Tuning entry point: `mlir/lib/Dialect/Rock/Tuning/RockTuningImpl.cpp`
- Architecture DB: `mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp`
- LDS / wave size source of truth: `external/triton/third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp`
- Workgroup constant: `mlir/include/mlir/Dialect/Rock/IR/Rock.h` (`maxHardwareWorkgroupSize = 1024`)

---

## 1. Current tuning space in `getAccelRangeGemm`

```117:172:mlir/lib/Dialect/Rock/Tuning/RockTuningImpl.cpp
static std::vector<std::vector<uint32_t>>
getAccelRangeGemm(RockGemmWrapperInterface gemmOp, int64_t waveSize,
                  int64_t maxWavesPerEU, TuningParamSetKind kind) {
  auto dPerBlock = computeDPerBlock(gemmOp, kind);
  std::vector<uint32_t> numWavesRange = computeNumWaves(kind, waveSize);

  std::vector<uint32_t> wavesPerEUList = {0};
  std::vector<uint32_t> gridGroupSizeList = {0};
  // ...

  std::vector<uint32_t> kPerBlock = {16, 32, 64, 128};
  if (is8b && isMfma)
    kPerBlock = {32, 64, 128};

  // MFMA (CDNA) parameters
  // Note: kPack max is 2
  std::vector<std::vector<uint32_t>> validRangeMfmaParams = {
      dPerBlock,        // M/block
      dPerBlock,        // N/block
      kPerBlock,        // K/block
      {1, 2},           // kPack
      numWavesRange,    // numWaves
      {16, 32},         // matrixInstrNonkdim
      {1, 2, 3},        // numStages
      wavesPerEUList,   // wavesPerEU
      gridGroupSizeList // gridGroupSize
  };

  // WMMA (RDNA) parameters
  std::vector<std::vector<uint32_t>> validRangeWmmaParams = {
      dPerBlock,        // M/block
      dPerBlock,        // N/block
      {32, 64},         // K/block
      {1},              // kPack
      numWavesRange,    // numWaves
      {0},              // matrixInstrNonkdim
      {1, 2},           // numStages
      wavesPerEUList,   // wavesPerEU
      gridGroupSizeList // gridGroupSize
  };
```

Helper ranges:

- `computeDPerBlock` (accel) → `{16, 32, 64, 128, 256}`; non-accel → `{4, 8, 16, 32, 64}`.
- `computeNumWaves` (exhaustive) → powers of 2 up to `maxHardwareWorkgroupSize/waveSize`,
  i.e. `{1,2,4,8,16}` on CDNA (waveSize=64) and `{1,2,4,8,16,32}` on RDNA (waveSize=32).
- `wavesPerEUList = {0}` and `gridGroupSizeList = {0}` use a heuristic in later passes, so
  we are not actually exploring those two axes today.

`getAccelRangeGemmGemm` (attention / fused GEMMs) reuses the same `numStages` ranges
(`{1,2,3}` for MFMA, `{1,2}` for WMMA), but pins `kPerBlock` to the actual K dim
when not exhaustive, and uses `{16,32,64,128,512,1024,2048}` exhaustively.

### Summary of axes we currently search

| Axis (MFMA / CDNA) | Values                                  |
|--------------------|-----------------------------------------|
| `mPerBlock`        | `{16,32,64,128,256}`                    |
| `nPerBlock`        | `{16,32,64,128,256}`                    |
| `kPerBlock`        | `{16,32,64,128}` (8-bit MFMA: `{32,64,128}`) |
| `kPack`            | `{1, 2}`                                |
| `numWaves`         | `{1,2,4,8,16}` (powers of 2 ≤ 1024/64)  |
| `matrixInstrNonkdim` | `{16, 32}`                            |
| `numStages`        | `{1, 2, 3}`                             |
| `wavesPerEU`       | `{0}` (heuristic placeholder)           |
| `gridGroupSize`    | `{0}` (heuristic placeholder)           |

| Axis (WMMA / RDNA) | Values                                  |
|--------------------|-----------------------------------------|
| `mPerBlock`        | `{16,32,64,128,256}`                    |
| `nPerBlock`        | `{16,32,64,128,256}`                    |
| `kPerBlock`        | `{32, 64}`                              |
| `kPack`            | `{1}`                                   |
| `numWaves`         | `{1,2,4,8,16,32}` (powers of 2 ≤ 1024/32) |
| `matrixInstrNonkdim` | `{0}` (encoded as 0 / not used)        |
| `numStages`        | `{1, 2}`                                |
| `wavesPerEU`       | `{0}`                                   |
| `gridGroupSize`    | `{0}`                                   |

---

## 2. LDS budget per architecture (the source of the `numStages` constraint)

LDS capacity comes from
`external/triton/third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp::getSharedMemorySize`,
which is what `getLDSSize(arch)` in `AmdArchDb.cpp` returns:

| Arch family                 | Wave size | LDS capacity |
|-----------------------------|-----------|--------------|
| CDNA1 / CDNA2 / CDNA3       | 64        | **64 KB**    |
| CDNA4 (gfx950)              | 64        | **160 KB**   |
| RDNA1 / RDNA2 / RDNA3       | 32        | **64 KB**    |
| GFX1250 (RDNA4‑class new)   | 32        | **320 KB**   |

`getMaxWavesPerEU` returns 8 for CDNA, 16 for RDNA / GFX1250.

### LDS cost of `numStages` (Triton software-pipelining model)

tritonBLAS encodes this in `calculate_lds_usage` (from
`tools/triton_gemm_bench.py`), which mirrors what the AMD Triton backend actually
does:

```
LDSA = block_m * block_k * elemBytes_a
LDSB = block_n * block_k * elemBytes_b

if num_stages <= 1:
    LDS = max(LDSA, LDSB)                  # A & B can alias the same buffer
    if async_copy:                          # async loads need their own buffer
        LDS = LDSA + LDSB
else:
    base_buffers = max(1, num_stages - 1)
    if async_copy:
        LDS = (LDSA + LDSB) * (base_buffers + 1)
    else:
        LDS = (LDSA + LDSB) * base_buffers
```

In practice (no async copy) this gives:

- `num_stages=1` → `max(LDSA, LDSB)` (single buffer, no overlap)
- `num_stages=2` → `1 × (LDSA + LDSB)`
- `num_stages=3` → `2 × (LDSA + LDSB)` (double-buffered pipeline)

### Concrete LDS pressure of common tiles (BF16 / FP16, 2 bytes per element)

For `M=N`, both `LDSA` and `LDSB` are `block_mn * block_k * 2` bytes.

| Tile (mn × mn × k) | `LDSA + LDSB` | `ns=1` | `ns=2` | `ns=3` |
|--------------------|---------------|--------|--------|--------|
| 128 × 128 × 32     | 16 KB         | 8 KB   | 16 KB  | 32 KB  |
| 128 × 128 × 64     | 32 KB         | 16 KB  | 32 KB  | **64 KB** |
| 128 × 128 × 128    | 64 KB         | 32 KB  | **64 KB** | 128 KB |
| 256 × 256 × 32     | 32 KB         | 16 KB  | 32 KB  | **64 KB** |
| 256 × 256 × 64     | 64 KB         | 32 KB  | **64 KB** | 128 KB |
| 256 × 256 × 128    | 128 KB        | 64 KB  | 128 KB | 192 KB |

Boldface = right at the LDS cap of a 64 KB-LDS architecture.

For 8-bit element types (FP8 / I8), divide all numbers above by 2.
For FP4, divide by 4.

### What this implies for the existing `numStages` ranges

For **`gfx908 / gfx90a / gfx942` (64 KB LDS)**:

- `numStages = 3` is only feasible for **small tiles**: roughly
  `(LDSA + LDSB) * 2 ≤ 64 KB` → `mn² * k ≤ 16384` (in elements per matrix).
  For BF16 that means tiles like 128×128×32, 64×128×64, 32×64×128, etc.
  Larger tiles (the ones that actually win on big GEMMs, e.g. 256×256×64) **cannot**
  fit `numStages=3` and will be rejected by the downstream LDS check.
- `numStages = 2` is the sweet spot — every AITer 64 KB-LDS config we inspected uses 2.

For **`gfx950` (160 KB LDS)** and **`gfx1250` (320 KB LDS)**:

- `numStages = 3` is generally feasible, even for 256×256×64 BF16
  (128 KB ≤ 160 / 320 KB), and AITer’s tuned configs actually pick 3 for almost
  every shape on these two architectures.

This is the central finding for Daniel’s original question:

> *"`num_stages={1, 2, 3}` on MFMA, `{1, 2}` on WMMA. I didn't try if num_stages=3
> works on Navis or not, just assumed there isn't enough LDS."*

The assumption is **correct for old Navi (RDNA1‑3, 64 KB LDS)** but **wrong for
gfx1250 (320 KB LDS)**. See § 5 for what to actually do.

---

## 3. What AITer uses — empirical, post-tuning

AITer ships pre-tuned JSON configs under
`aiter/ops/triton/configs/gemm/<arch>-<config_name>.json`, keyed by `M_LEQ_<bound>`
buckets. These are the **winners** of a real tuning sweep on real hardware, so they
are an excellent ground truth for "what is actually picked". We pulled the
`a16w16` (BF16 weights/activations) and FP4 configs.

### `gfx942-a16w16.json` (CDNA3 / MI300X, 64 KB LDS)

Every bucket from `M_LEQ_64` through `M_LEQ_2048` and `any` selects:

- `num_stages = 2` (always)
- `num_warps = 4` for `M ≤ 64`, then `8`
- `BLOCK_SIZE_K = 128` for small M, dropping to `64` for large tiles
  (256×256×64 — exactly the largest config that fits with `ns=2`)
- `matrix_instr_nonkdim = 16` (always)
- `kpack = 1` (always)
- `waves_per_eu = 2` (always)

### `gfx950-a16w16.json` (CDNA4 / MI355X, 160 KB LDS)

Every bucket selects:

- `num_stages = 3` (always — this is the architecture where ns=3 wins on MFMA)
- `num_warps = 4` for small M, `8` from `M_LEQ_128` onward
- `BLOCK_SIZE_K = 256` for small M, 128 for medium, then 64 for the largest tiles
- `matrix_instr_nonkdim = 16`
- `waves_per_eu` ∈ `{4, 6, 8}` (actively used — not just 0)
- No `kpack` field (assumed 1)

### `gfx1250-a16w16.json` (320 KB LDS)

Buckets pick a mix:

- `num_stages = 3` for **5 of 6 buckets** (small, medium-large, large, and `any`)
- `num_stages = 2` for `M_LEQ_128` only (a single 64×64×128 tile)
- `num_warps` ∈ `{4, 8}`
- `BLOCK_SIZE_K` peaks at 128 for medium M and drops to 32 for large M
  (because the M-tile grows to 256×256, so K must shrink to keep LDS in budget)
- `matrix_instr_nonkdim = 16`
- `kpack = 1`
- `waves_per_eu` ∈ `{2, 3}` (note: not a power of 2, AITer picks 3)

### `gfx1250-fp4.json` (FP4)

For 4-bit data, LDS pressure drops by 4× so even bigger tiles fit:

- `BLOCK_SIZE_K = 256` consistently, `BLOCK_SIZE_N = 256` from `M_LEQ_64` up
- `num_stages` mixes `2` and `3` depending on bucket
- `matrix_instr_nonkdim` = 32 for medium/large M (different from the BF16 case!)
- `NUM_KSPLIT = 16` for the very smallest M (M ≤ 16) — heavy split-K reliance

### Pattern across all AITer configs we inspected

| Knob                 | Practical range observed      |
|----------------------|-------------------------------|
| `BLOCK_SIZE_M/N`     | `{16, 32, 64, 128, 256}`      |
| `BLOCK_SIZE_K`       | `{32, 64, 128, 256}` (256 only on FP8/FP4) |
| `GROUP_SIZE_M`       | `{1, 2, 4, 6, 8}`             |
| `num_warps`          | `{4, 8}`                      |
| `num_stages`         | `{2}` on 64 KB-LDS arches; `{2, 3}` on ≥ 160 KB |
| `waves_per_eu`       | `{1, 2, 3, 4, 6, 8}` — actively tuned |
| `matrix_instr_nonkdim` | `{16, 32}` (16 dominates BF16, 32 used for FP4/FP8) |
| `kpack`              | `1` only                      |
| `NUM_KSPLIT`         | `{1, 4, 8, 16}` (split-K)     |

`num_stages = 1` **never appears** in any AITer winning config we looked at.

---

## 4. What tritonBLAS searches — Triton-style autotuning

tritonBLAS exposes both an analytical selector (`OrigamiMatmulSelector`) and a
brute-force tuner (`tools/triton_gemm_bench.py`). The brute-force tuner is the
direct counterpart of our `getAccelRangeGemm`.

### `get_full_tuning_space` (tritonBLAS, `triton_gemm_bench.py`)

```
block_mn_range          = [16, 32, 64, 128, 256]
block_k_range           = [16, 32, 64, 128, 256, 512]
num_warps_range         = [4, 8]
group_m_range           = [1, 4, 6, 8, 16, 32]
num_stage_range         = [2, 3]                    # NB: 1 is intentionally omitted
waves_per_eu_range      = [0, 1, 2, 4]
matrix_instr_nonkdim_range = [16, 32]
kpack_range             = [1]                       # 2 is intentionally omitted
num_sms_range           = [num_cus]
chunk_size_range        = [32]
```

Inline comment in tritonBLAS:

> *"For now we see better perf with num_stages=2 for all gemm configs we care
> But keep this explicit so that we do not forget we may need to set it to other
> values in the future"*

### `OrigamiMatmulSelector` (analytical selector)

- `_block_mn_range = [16, 32, 64, 128, 256]`
- `_block_k_range  = [16, 32, 64, 128, 256, 512]`
- Adjusts `_block_mn_range` and `_block_k_range` per arch (e.g. expands
  `_block_mn_range` for FP8 on gfx950 / gfx942; lets `_block_k_range`
  reach 256 for those)
- Default `num_stages = 2`, default `num_warps = 8`
- Has an exact LDS check (`estimate_triton_lds_bytes`) using the same formula
  as above and `check_triton_lds_capacity` to reject infeasible configs.

### Pattern across tritonBLAS

| Knob                 | tritonBLAS exhaustive range       |
|----------------------|-----------------------------------|
| `BLOCK_SIZE_M/N`     | `{16, 32, 64, 128, 256}`          |
| `BLOCK_SIZE_K`       | `{16, 32, 64, 128, 256, 512}`     |
| `GROUP_SIZE_M`       | `{1, 4, 6, 8, 16, 32}`            |
| `num_warps`          | `{4, 8}` (selector also adds `16`) |
| `num_stages`         | `{2, 3}` (no `1`)                 |
| `waves_per_eu`       | `{0, 1, 2, 4}` (actively tuned!)  |
| `matrix_instr_nonkdim` | `{16, 32}`                      |
| `kpack`              | `{1}` (no `2`!)                   |

---

## 5. Side-by-side comparison vs. `RockTuningImpl.cpp`

Mapping (rocMLIR ↔ Triton-world):

| `RockTuningImpl.cpp` | Triton autotune key             |
|----------------------|---------------------------------|
| `mPerBlock`, `nPerBlock` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N` |
| `kPerBlock`          | `BLOCK_SIZE_K`                  |
| `kPack`              | `kpack`                         |
| `numWaves`           | `num_warps`                     |
| `matrixInstrNonkdim` | `matrix_instr_nonkdim`          |
| `numStages`          | `num_stages`                    |
| `wavesPerEU`         | `waves_per_eu`                  |
| `gridGroupSize`      | `GROUP_SIZE_M` (broadly)        |
| `splitKFactor`       | `NUM_KSPLIT`                    |

### Comparison table (MFMA / CDNA path)

| Knob                | rocMLIR (current)   | tritonBLAS exhaustive | AITer winners (gfx942 / gfx950) |
|---------------------|---------------------|------------------------|----------------------------------|
| `mPerBlock`         | `{16,32,64,128,256}`| `{16,32,64,128,256}`   | `{16,32,64,128,256}`             |
| `nPerBlock`         | `{16,32,64,128,256}`| `{16,32,64,128,256}`   | `{16,32,64,128,256}`             |
| `kPerBlock`         | `{16,32,64,128}` (8-bit: `{32,64,128}`) | `{16,32,64,128,256,512}` | `{32,64,128,256}` (256 only on gfx950 / FP8) |
| `kPack`             | `{1,2}`             | `{1}`                  | `{1}` (always)                   |
| `numWaves`          | `{1,2,4,8,16}`      | (= `num_warps`) `{4,8}` | `{4,8}`                          |
| `matrixInstrNonkdim`| `{16,32}`           | `{16,32}`              | `{16}` for BF16; `{16,32}` for FP4/FP8 |
| `numStages`         | `{1,2,3}`           | `{2,3}`                | `{2}` on gfx942; `{3}` on gfx950 |
| `wavesPerEU`        | `{0}`               | `{0,1,2,4}`            | `{1..8}` (actively tuned)        |
| `gridGroupSize`     | `{0}`               | `{1,4,6,8,16,32}`      | `{1,2,4,6,8}`                    |
| `splitKFactor`      | `{1}` (mostly)      | implicit               | `{1,4,8,16}` (heavy on small M)  |

### Comparison table (WMMA / RDNA path)

| Knob                | rocMLIR (current)   | AITer winners (gfx1250) |
|---------------------|---------------------|--------------------------|
| `mPerBlock`         | `{16,32,64,128,256}`| `{16,64,128,256}`        |
| `nPerBlock`         | `{16,32,64,128,256}`| `{64,128,256}`           |
| `kPerBlock`         | `{32,64}`           | `{32,64,128,256}` (256 used on FP4) |
| `kPack`             | `{1}`               | `{1}`                    |
| `numWaves`          | `{1,2,4,8,16,32}`   | `{4,8}`                  |
| `matrixInstrNonkdim`| `{0}`               | `{16}` for BF16; `{32}` mostly for FP4 |
| `numStages`         | **`{1, 2}`**        | **mostly `{3}`, sometimes `{2}`** |
| `wavesPerEU`        | `{0}`               | `{2,3}` (3 used a lot)   |
| `gridGroupSize`     | `{0}`               | `{1,2,4,6}`              |
| `splitKFactor`      | `{1}`               | `{1,16}` (16 only for tiny M) |

### Key gaps & opportunities

1. **`numStages` for WMMA is too narrow on gfx1250.** Our `{1, 2}` excludes the
   exact value (`3`) that wins for almost every BF16 shape on gfx1250 in AITer.
   We should make this **arch-conditional**: keep `{1, 2}` on RDNA1‑3 (64 KB LDS),
   but **add `3`** for gfx1250 (320 KB LDS) and CDNA4 (160 KB LDS).

2. **`numStages = 1` is essentially dead weight.** Neither tritonBLAS nor any
   AITer winner ever picks `1`. The only argument for keeping it is correctness
   on tiles where `num_stages=2` exceeds the LDS budget; but in those cases the
   downstream LDS check will already drop the config, and we could fall back to
   `1` on rejection rather than searching it for every tile. Suggestion: drop
   `1` from the default exhaustive range, or make it the rescue choice only.

3. **`kPack = 2` is searched but is rarely if ever a winner.** Every AITer
   winner uses `kpack=1`; tritonBLAS hard-codes `kpack=[1]`. The comment in
   `getAccelRangeGemm` already notes "kPack max is 2" as a hard upper bound,
   but it would be reasonable to default to `{1}` and only enable `{1,2}`
   when an explicit search request is made — or vice versa, keep `{1,2}`
   given the search is already cheap.

4. **`wavesPerEU` is currently `{0}` (heuristic only).** Both tritonBLAS and
   AITer treat this as a real tuning axis (`{0,1,2,4}` and `{1..8}` respectively).
   Especially on gfx950 we see odd values (3, 6) winning, which a heuristic
   will rarely hit. Adding even `{0, 2, 4}` would be a very cheap upgrade.

5. **`gridGroupSize` is currently `{0}` (heuristic only).** The Triton-world
   equivalent (`GROUP_SIZE_M`) is one of the most impactful axes for L2 reuse
   on big GEMMs; AITer winners frequently use `{4, 6, 8}`. Even adding
   `{0, 4, 8}` would surface configs we’re missing today.

6. **`numWaves` upper bound.** rocMLIR enumerates up to
   `maxHardwareWorkgroupSize / waveSize` = 16 (CDNA) or 32 (RDNA). AITer never
   picks more than 8 in our sample. The high end (16, 32) almost certainly
   produces configs that the downstream legality / LDS check will reject;
   capping at `{1, 2, 4, 8}` would shrink the space substantially without losing
   any winning configs. (Note `computeNumWaves` already returns `{2,4,8}` in the
   non-exhaustive path; we should consider whether the exhaustive expansion to
   16 / 32 is worth the search cost.)

7. **`kPerBlock = {32, 64}` for WMMA is too narrow.** AITer uses
   `BLOCK_SIZE_K = 128` (BF16, M_LEQ_128) and even `256` (FP4) on gfx1250.
   Adding 128 to the WMMA range is essentially free if we are exhaustive,
   and it unlocks one of the tile families that gfx1250 actually wins with.

---

## 6. The `numStages` question, focused

Daniel's original question:

> "*num_stages is `{1,2,3}` on MFMA, `{1,2}` on WMMA. I didn't try if num_stages=3
> works on Navis or not, just assumed there isn't enough LDS.*"

### Direct answer

- The assumption is **correct** for **old Navis (RDNA1‑3, gfx10xx/gfx11xx, 64 KB LDS)**.
  At those LDS sizes `numStages=3` only fits for tiles ≤ ~64×64×64 (BF16) or
  similarly small, which are not where WMMA wants to operate. Keeping
  `{1, 2}` for these arches is the right default.

- The assumption is **wrong for gfx1250** (the new RDNA4‑class part with
  **320 KB LDS**), and arguably wrong for **CDNA4 / gfx950** (160 KB LDS, MFMA
  path):

  - On gfx950, AITer picks `num_stages=3` for **every** BF16 bucket.
  - On gfx1250, AITer picks `num_stages=3` for **5 of 6** buckets in
    `gfx1250-a16w16.json`.

So the practical fix is to make `numStages` arch-dependent rather than
"MFMA vs WMMA":

| Arch family               | LDS  | Recommended `numStages` set |
|---------------------------|------|------------------------------|
| CDNA1 / CDNA2 / CDNA3     | 64K  | `{2}` (drop `1`, drop `3`)   |
| CDNA4 (gfx950)            | 160K | `{2, 3}`                     |
| RDNA1 / RDNA2 / RDNA3     | 64K  | `{2}`                        |
| GFX1250                   | 320K | `{2, 3}` (and consider `{2,3,4}` for FP4) |

The downstream LDS check still catches over-budget tiles, so adding `3` on the
larger-LDS arches is safe; the win is that we **also** add `3` to WMMA on
gfx1250, which the current code path forbids entirely.

`numStages = 1` is conservative; in our search no production reference selects
it. Removing it from the default exhaustive set would shrink the space by ~33 %
on the MFMA path with effectively zero risk.

### Caveat about async copies

If `triton.knobs.amd.use_async_copy = True` (or the equivalent backend setting),
each load needs an extra LDS buffer. The tritonBLAS formula above accounts
for this; any change we make to `numStages` should respect that the LDS
check downstream uses the same `async_copy`-aware accounting. (rocMLIR does
not currently feed an `async_copy` flag into the tuning space; if/when it
does, that will further constrain what `numStages=3` configs survive on
64 KB-LDS arches.)

---

## 7. Recommended changes (concrete)

These are minimal, low-risk edits to `getAccelRangeGemm` /
`getAccelRangeGemmGemm`. All can be done without touching downstream
legality checks.

### 7.1. Make `numStages` arch-aware

Today both ranges are hard-coded as `{1,2,3}` (MFMA) and `{1,2}` (WMMA).
Drive the range from `getLDSSize(arch)` (or directly from a small switch on
ISA family):

```cpp
auto archStr = rock::getArchValue(gemmOp);
int64_t lds  = rock::getLDSSize(archStr);
std::vector<uint32_t> numStagesRange;
if (lds >= 160 * 1024)
  numStagesRange = {2, 3};   // CDNA4 (gfx950), GFX1250 — ns=3 frequently wins
else
  numStagesRange = {2};      // 64 KB LDS — ns=3 essentially never fits
```

Apply this **uniformly** to both the MFMA and WMMA branches. Compared to the
current behavior:

- MFMA on gfx942 (64 KB LDS): `{1,2,3}` → `{2}`. Drops 2 values that AITer never
  picks, eliminates a large slice of search space.
- MFMA on gfx950 (160 KB LDS): `{1,2,3}` → `{2,3}`. Drops `1` (never picked).
- WMMA on RDNA1‑3 (64 KB LDS): `{1,2}` → `{2}`. Drops `1`.
- **WMMA on gfx1250 (320 KB LDS): `{1,2}` → `{2,3}`. Adds the value AITer
  actually picks for almost every BF16 shape on this part.**

If we want to keep `1` for safety / conservative fallback, the suggested set
becomes `{1, 2, 3}` on big-LDS arches and `{1, 2}` on small-LDS — i.e. only
**add** `3` to gfx1250 / gfx950. That is the smallest possible diff.

### 7.2. Open up `kPerBlock` for WMMA

Change `kPerBlock` for WMMA from `{32, 64}` to `{32, 64, 128}`. AITer’s
gfx1250 BF16 winners include `BLOCK_SIZE_K = 128`; we currently cannot find
that family at all.

(For consistency consider also `{32, 64, 128, 256}` for WMMA when input dtype
is 8 or 4 bits, mirroring what tritonBLAS / AITer do for FP8/FP4.)

### 7.3. Promote `wavesPerEU` and `gridGroupSize` to real axes (optional)

Today both are `{0}` (heuristic). Even adding small explicit lists is cheap
and matches what both AITer and tritonBLAS do:

```cpp
// minimal upgrade — still cheap
wavesPerEUList    = {0, 2, 4};
gridGroupSizeList = {0, 4, 8};
```

This is independent from the `numStages` change but addresses the same root
cause (we currently can’t find the configs that AITer wins with on gfx950 /
gfx1250).

### 7.4. Cap `numWaves` at 8

Change `computeNumWaves` (exhaustive path) to stop at 8 instead of going to
`maxHardwareWorkgroupSize / waveSize` (16 or 32). No AITer or tritonBLAS
config we observed uses more than 8 warps. The high end produces configs
that almost always exceed LDS / register budgets.

### 7.5. Optionally drop `kPack = 2`

Both tritonBLAS and every AITer winner we inspected use `kpack=1`. Keeping
`{1, 2}` doubles the MFMA search at low expected payoff. If exhaustive
runtime is a concern, default to `{1}` and treat `{1, 2}` as opt-in.

---

## 8. Open questions / TODOs

- Confirm whether `triton.knobs.amd.use_async_copy` is enabled by default in
  this rocMLIR-Triton build. If yes, the LDS budget per `numStages` is
  effectively `(LDSA + LDSB) * num_stages` (one extra buffer per load), which
  pushes `ns=3` configs out of the budget on gfx942 even more aggressively.

- We did not look at AITer's MoE / fused / streamk kernels, only basic
  `gemm_a16w16`, `gemm_a8w8`, `gemm_afp4wfp4`, and the gfx1250 specializations.
  Those could surface different `numStages` / `kpack` patterns (e.g. fused
  epilogues compete for LDS).

- `getAccelRangeGemmGemm` (attention path) inherits the same `numStages`
  ranges. The LDS pressure there is higher because attention pipelines two
  GEMMs back-to-back; the recommendation in §7.1 is even more conservative for
  64 KB-LDS arches. Worth checking that the attention exhaustive sweep doesn't
  rely on `numStages=3` on gfx942 today.

- The `is8b` branch for `kPerBlock` (`{32, 64, 128}`) is MFMA-only by
  construction. For 8-bit WMMA on gfx1250 we likely want the same expansion;
  AITer's FP8 / FP4 configs use `BLOCK_SIZE_K = 256`.

---

## 9. References

- Source: `mlir/lib/Dialect/Rock/Tuning/RockTuningImpl.cpp`
- LDS DB:  `mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp` →
  `external/triton/third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp`
- AITer configs: `https://github.com/ROCm/aiter/tree/main/aiter/ops/triton/configs/gemm`
  - `gfx942-a16w16.json`, `gfx950-a16w16.json`, `gfx1250-a16w16.json`,
    `gfx1250-fp4.json`, `gfx942-a8w8.json`, `gfx950-n128k4096.json`
- AITer config loader: `aiter/ops/triton/utils/gemm_config_utils.py`
- tritonBLAS exhaustive tuner: `tools/triton_gemm_bench.py` (`get_full_tuning_space`,
  `calculate_lds_usage`)
- tritonBLAS analytical selector: `aiter/origami.py` /
  `tritonblas/origami.py` (`OrigamiMatmulSelector`,
  `estimate_triton_lds_bytes`, `check_triton_lds_capacity`)
- Triton tutorial reference (HIP autotune configs):
  `ROCm/triton/python/tutorials/03-matrix-multiplication.py`
  (`get_hip_autotune_config`)
