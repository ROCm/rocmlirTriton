# Closing the K-Loop Gap on Odd-K Convolutions

On the DenseNet first-layer convolution (`32x3x224x224` input, `64x3x7x7`
filter) rocmlirTriton runs 14.5% slower than rocMLIR on gfx1100. The whole gap
is in the K-loop, and it takes **two independent fixes** to close it. Neither
one helps much alone:

1. **Tile-alignment pads block carry hoisting**, so `kPerBlock > 1` pays a
   `divui`/`remui` im2col address computation every iteration.
2. **Address divisibility shrinks the coalesced layout**, so `kPerBlock > 1`
   isn't profitable even once the address math is fixed.

With both applied the kernel goes from 385.4 µs to 339.8 µs against rocMLIR's
336.7 µs — from 14.5% behind to 0.9% behind.

Both are prototyped in the working tree behind clear markers. **Neither is
ready to merge**; see [status](#6-status-of-the-prototypes).

---

## 1. Background: the carry rewrite

`RockTransformsInvariantCodeMotionPass` incrementalizes the im2col coordinates
of a K-loop, replacing the per-iteration `divui`/`remui` chain with an
add/compare/select odometer. `analyzeCarryCandidate` looks for a single `Merge`
that the induction variable reaches with a nonzero diff — for a 7x7 convolution
that is the im2col decomposition of the GEMM's K coordinate:

```
<Merge{3, 7, 7} ["gemmK"] at [1] -> ["ci", "0", "1"] at [2, 3, 5]>
```

The chain is split at that merge: the part **above** is expanded once in the
preheader to seed the coordinates at `iv == lb`, and the part **below** is
re-expanded every iteration by `buildCarryMask` to rebuild the validity mask.
Because the mask comes only from below the merge, the analysis rejects any
validity-impacting map at or above it:

```392:396:mlir/lib/Dialect/Rock/Transforms/TransformsInvariantCodeMotion.cpp
    // We reconstruct the variant mask only from the sub-chain *below* the
    // merge, so no validity-impacting map may sit at or above it.
    if (mapImpactsValidity(map) &&
        (mergeIdx == -1 || static_cast<int>(mapIdx) <= mergeIdx))
      return bail("validity-impacting map at or above the iv-traversed merge");
```

## 2. Fix 1: admit tile-alignment pads

M=64, N=401408, K=147. With `kPerBlock = 2`, 147 is not divisible by 2, so the
GEMM lowering rounds K up to 148 with a tile-alignment pad. The chain feeding
`rock.transforms_to_ptr` becomes, top down:

| # | map | note |
|---|---|---|
| 0 | `Unmerge{74, 2} ["k_loop", "k_iter"] -> ["k"]` | `k = iv * 2 + k_iter` |
| 1 | `Pad{0, 1} tileAlignment ["gemmKPad"] -> ["gemmK"]` | **validity-impacting** |
| 2 | `Merge{3, 7, 7} ["gemmK"] -> ["ci", "0", "1"]` | the iv-traversed merge |
| 3 | `Embed{1, 2}` x2 | im2col |
| 4 | `Pad{3, 2, 3, 2}` | convolution halo |

Map 1 is validity-impacting and `1 <= mergeIdx`, so the analysis bails. Only
`kPerBlock = 1` escapes, because 147 is divisible by 1 and no pad is inserted.
`-debug-only=rock-transforms-invariant-code-motion` confirms this is the sole
blocker for 2 and 4:

```
kPerBlock=1: (no carry bail)
kPerBlock=2: 2x validity-impacting map at or above the iv-traversed merge
kPerBlock=4: 2x validity-impacting map at or above the iv-traversed merge
kPerBlock=7: 10x more than one iv-traversed merge (not yet supported)
```

(`kPerBlock = 7` fails for an unrelated reason: `rock-decompose-nonpow2-tiles`
leaves an `Unmerge{21, 4}` immediately above a `Merge{21, 4}`, which the
analysis counts as a second iv-traversed merge even though the pair composes to
the identity. Separate fix, out of scope here.)

### The relaxation

The pad's predicate is `gemmK <u 147`, emitted by `updateValidityAfter` as an
unsigned bound check on the pad's *lower* coordinate. That coordinate lives
**above** the merge, where everything is affine in the induction variable —
`analyzeCarryCandidate` has already proved this, since `applyDiffOneMap`
returned a constant diff for every map down to the merge. So the conjunct can be
rebuilt inside the loop from the induction variable, with a couple of adds and
one compare, and **no additional loop-carried state**.

1. **Loosen the guard.** Replace `mapImpactsValidity(map)` with
   `validityImpactingUpperDims(map, /*ignoreTileAlignmentPads=*/true)`, which
   already exists for this purpose and is already used this way by
   `RegularizeInput.cpp`. Keep rejecting a validity-impacting map at
   `mapIdx == mergeIdx`: its conjunct would have to be expanded *through* the
   merge, putting the `divui`/`remui` back in the loop. Only `mapIdx < mergeIdx`
   becomes legal.

2. **Record the guard prefix.** Track the index of the last validity-impacting
   map above the merge and store `transforms.take_front(lastGuardIdx + 1)` on
   the `Candidate`, then carry it to `Reduced` alongside `belowMaps`.

3. **Rebuild the conjunct in the loop.** In `buildCarryMask`, seed that prefix
   with the *live* induction variable (the same `initValues` construction
   `buildReducedCarries` uses for `iter0Coords`, but binding the iv slot to
   `newLoop.getInductionVar()` instead of `lb`) and call
   `expandCoordsToOffsetAndMask(..., computeOffset=false)`. That helper stops at
   the last validity-impacting map and returns the AND of every conjunct in the
   prefix, so the result just needs `andi`-ing with the below-chain mask.

Nothing below the merge changes, so `variantImpactsValidity` still reports `ci`
as non-impacting and it is still dropped as the carry prefix. The rewrite keeps
the same three carried tiles it uses at `kPerBlock = 1`, just wider.

### Correctness note

The conjunct is not vacuous. With `kPerBlock = 2` the last lane of the last
iteration has `gemmKPad = 147`, which the merge decomposes to
`(ci, y, x) = (3, 0, 0)` — one past the end of the 3-channel input. The
below-chain mask does not reject it (the halo pad only constrains the spatial
coordinates, and `y = 0` is in bounds for any `0o >= 2`), so dropping the guard
without rebuilding the conjunct silently reads into the next image.

## 3. Why fix 1 alone is not enough

Relaxing the guard makes the address math collapse as intended — `arith.remui`
disappears from the loop body and the odometer's `cmpi`/`select` pairs replace
it — and every blocked configuration gets faster:

| `kPerBlock` | today | fix 1 only | change |
|---|---|---|---|
| 1 | 383.9 | 382.8 | -0.3% (carry already active) |
| 2 | 396.6 | 387.2 | -2.4% |
| 4 | 471.0 | 432.3 | -8.2% |
| 8 | 496.5 | 481.0 | -3.1% |

But `kPerBlock = 1` is still the fastest configuration, so the tuner's pick
never changes and the end-to-end number does not move. On K=147 something else
makes a taller K tile unattractive in the first place — and it is specific to
this shape. Sweeping the input channel count C (so K = 49C) at `kPerBlock = 4`,
with fix 1 applied so carry hoisting works everywhere, gives a perfectly clean
split by the **parity of K**:

| C | K | issue slots / K-step | VOPD packing | VGPRs | µs / K-step |
|---|---|---|---|---|---|
| 3 | 147 (odd) | 132.8 | 60.0% | 240 | 2.979 |
| 5 | 245 (odd) | 132.8 | 60.0% | 240 | 2.878 |
| 7 | 343 (odd) | 132.8 | 60.0% | 240 | 2.811 |
| 2 | 98 | 118.0 | 95.4% | 228 | 2.737 |
| 6 | 294 | 118.0 | 95.4% | 228 | 2.165 |
| 10 | 490 | 118.0 | 95.4% | 228 | 2.102 |
| 4 | 196 | 117.0 | 93.2% | 223 | 2.244 |
| 8 | 392 | 117.0 | 93.2% | 223 | 2.070 |
| 12 | 588 | 117.0 | 93.2% | 223 | 2.208 |

Odd K collapses VOPD dual-issue packing from ~94% to 60%. Both cases issue
exactly 512 FMAs per loop body; the odd case just needs 320 issue slots to do it
instead of 265. Occupancy is 6 in every row, so that is not the mechanism.

The split is by parity, not by divisibility by 4, and it vanishes entirely at
`kPerBlock = 1` (every C gives 131.0 slots/K-step, 88.2% packing, 230 VGPRs) —
which is exactly what you expect if the cause is the width of a k-contiguous
filter access.

## 4. Fix 2: stop letting divisibility shrink the layout

The filter is row-major `64 x K` f32, so a `kPerBlock`-wide slice of one row is
contiguous but starts at element `m * K + k0`. When K is odd that is odd for odd
`m`, so Triton's `AxisInfo` computes divisibility 1. `getNumElementsPerThread`
then folds divisibility into the layout:

```178:199:external/triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp
  unsigned maxMultipleBytes = valInfo.getDivisibility(order[0]);
  unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
  unsigned maxContig =
      std::min(valInfo.getContiguity(order[0]), shapePerCTA[order[0]]);
  unsigned alignment = std::min(maxMultiple, maxContig);
```

and `buildCoalescedEncoding` turns the result into `sizePerThread`. Dumping the
layouts confirms it — for the `64 x 4` filter tile:

```
C=3 (K=147): sizePerThread = [1, 1], threadsPerWarp = [8, 4]    divisibility -> 1
C=4 (K=196): sizePerThread = [1, 2], threadsPerWarp = [16, 2]   divisibility -> 4
```

That single value drives everything downstream. This function feeds two
decisions at once, and divisibility is only relevant to one of them:

- **how wide one memory instruction may be** — divisibility genuinely
  constrains this;
- **how many elements of the tile each thread owns** — divisibility has no
  bearing on this at all. A thread can own two adjacent elements and load them
  with two separate dword instructions.

The prototype drops the divisibility term from the layout decision only, leaving
the instruction-width cap in `ModuleAxisInfoAnalysis::getAlignment` untouched.
On K=147 at `kPerBlock = 4`:

| | slots/K-step | VOPD | VGPRs | filter loads in loop |
|---|---|---|---|---|
| before | 132.8 | 60.0% | 240 | 12x `buffer_load_b32` |
| after | 120.0 | 94.7% | 224 | 12x `buffer_load_b32` |

The loads are **unchanged** — still twelve b32, because the addresses really are
only dword-aligned. The entire win comes from the thread-level distribution
allocating better and letting the scheduler pair the FMAs.

This was verified by splitting the patch: relaxing the layout alone reproduces
the full speedup, and additionally relaxing the instruction-width cap in
`getAlignment` adds nothing. That matters, because the layout half is safe by
construction (a distribution choice cannot make a load read the wrong address)
while the width half would emit under-aligned vector loads.

## 5. Result

Five interleaved rounds, each a median of 500 reps after 100 warmup
iterations, gfx1100, microseconds:

| | median | vs rocMLIR |
|---|---|---|
| rocMLIR, `v3:64,64,128,8,2,4,1,1,2` | 336.7 | — |
| rocmlirTriton today, best config (`kPerBlock=1`) | 385.4 | +14.5% |
| rocmlirTriton, both fixes (`kPerBlock=4`) | **339.8** | **+0.9%** |

Per `kPerBlock`, with both fixes:

| `kPerBlock` | today | both fixes |
|---|---|---|
| 1 | 391.2 | 391.2 |
| 2 | 395.3 | 363.1 |
| 4 | 435.9 | **345.5** |
| 8 | 485.3 | 381.1 |

The tuner will pick `kPerBlock = 4` on its own once both fixes are in, since it
becomes the fastest configuration by a wide margin.

## 6. Status of the prototypes

Both changes currently in the working tree are **experiments, not candidates**:

- `mlir/lib/Dialect/Rock/Transforms/TransformsInvariantCodeMotion.cpp` — step 1
  of fix 1 only (the guard is loosened; the mask conjunct is *not* rebuilt).
  This generates the intended code shape but is **numerically wrong**, per the
  correctness note above. Steps 2 and 3 are what make it real.
- `external/triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp` — fix 2,
  gated on the `ROCMLIR_RELAX_LOAD_ALIGN` environment variable so it can be
  A/B-tested. Landing it means deciding whether to make it unconditional or
  AMD-only, and recording it in `triton-patches/`.

Open questions before either can land:

- Fix 2 changes layout selection for *every* load and store in *every* kernel,
  not just odd-K filters. It needs a run across the E2E and performance suites
  to check nothing regresses, and a decision on whether it belongs upstream in
  Triton (the reasoning is not AMD-specific) or as a downstream patch.
- With fix 2 on, two `buffer_load_b64` appear in the K=147 prologue that were
  b32 before. They come from `getVectorSize`'s `std::max(vec, op.getContiguity())`
  in the AMD buffer-op lowering, which trusts the op's contiguity attribute over
  the axis-info alignment. Worth confirming those addresses are genuinely
  8-byte aligned.
- Neither fix has been checked for numerical correctness yet, which for fix 1 is
  expected (it is incomplete) and for fix 2 is simply not done.

## 7. Testing

- A lit test with `Pad{0, 1} tileAlignment` above `Merge{3, 7, 7}`, checking
  that the loop gains the carried coordinate `iter_args`, that the body has no
  `arith.remui`, and that the mask still contains the `147` bound check.
- A negative lit test keeping the bail when the validity-impacting map is in the
  same map as the merge.
- An E2E numerical check on a convolution whose K is not divisible by the tuned
  `kPerBlock` — that is what catches a dropped conjunct.
- For fix 2, a lit test on the coalesced encoding chosen for a load whose
  offsets are contiguous but not divisible.
