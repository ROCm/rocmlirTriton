# Standalone reduce kernel pipeline

Living plan for the dedicated lowering path for standalone `reduce_sum`
kernels. Supersedes the exploratory notes in
`reduce-kernel-plan-session-notes.md`.

## Scope

- `reduce_sum` only, last-dimension reduction only.
- A separate `rock-reduce-pipeline`, not mixed with the GEMM/conv/attention or
  elementwise pipelines.
- Direct reduction ownership (one writer per output), no atomic fallback.
- Pre-reduce elementwise fusion is allowed; post-reduce fusion is not.
- `rock.reduce` and `rock.elementwise` are mutually exclusive kernel markers.

## Why there is no algorithm selection

MIGraphX's reduce JIT picks between `lane`, `wave` and `block` kernels
(`src/targets/gpu/jit/reduce.cpp`). That taxonomy does not transfer: in the
Rock -> TTIR path the kernel body is just `tt.load` -> `tt.reduce` ->
`tt.store`, and the schedule is decided downstream by the tensor's blocked
layout, which is in turn decided by the *tile shape we load*.

Concretely, for a tile `R x C` reduced along the last axis:

1. `TritonGPUCoalesce` (`CoalesceUtils.cpp`, `CoalescedEncodingAnalysis::getEncoding`)
   derives `order` from the pointer's `AxisInfo` contiguity and sets
   `perThread = min(vectorBound, max(R*C / numThreads, 1))` on `order[0]`.
2. The `BlockedEncodingAttr::get` builder (`TritonGPUAttrDefs.td`) then spends
   lanes on the contiguous axis first:
   `threadsPerCTA = clamp(numThreads, 1, max(1, C / perThread))`, splits that
   into `threadsPerWarp[C]` and `warpsPerCTA[C]`, and gives the remainder to
   the row axis.
3. `ReduceOpConversion` costs the result as: within a thread -> free registers;
   across lanes -> DPP / shuffle; across warps -> LDS + barrier.

So, at f32 with `numWaves = 4` and a 64-wide wave (`numThreads = 256`,
`vectorBound = 4`), the same 1024-element tile gives:

| tile `R x C` | `threadsPerWarp[C]` | `warpsPerCTA[C]` | `tt.reduce` lowers to | MIGraphX calls it |
| --- | --- | --- | --- | --- |
| `1 x 2048` | 64 | 4 | registers + DPP + LDS + barrier | `block` |
| `4 x 256` | 64 | 1 | registers + DPP, no LDS | `wave` |
| `64 x 16` | 4 | 1 | registers + 2 DPP steps | `subwave<4>` |
| `256 x 4` | 1 | 1 | registers only | `lane` |

Picking the tile therefore *is* picking the algorithm, so the pass has no
`algo` option. Nothing downstream will correct a bad choice:
`TritonGPUOptimizeThreadLocality` in the pinned Triton carries only the reshape
and gather patterns (no reduce rewrite), and `rock-set-reduction-layout` only
fires on conv kernels.

## Invariant: `numWaves` must match `TritonOptions::numWarps`

`numWarps` sets both the real launch block size
(`compileUtils.cpp`: `info.blockSize = numWarps * warpSize`) and the layout
math above. It is never filled from a perf config for a kernel with no GEMM op,
so it keeps its `init(4)` default. The pass takes a `numWaves` option with the
same default and records it in the op's params. Running the reduce pipeline and
then `rock-triton-pipeline{numWarps=8}` silently mis-tunes the tile; threading
this through properly is deferred to Phase C.

## Schedule on the op, launch config on the func

Mirrors the GEMM precedent, where `params` rides on `rock.gemm`, is filled by
`rock-affix-params`, is consumed by `rock-gemm-to-gridwise`, and dies with the
op while the derived `rock.grid_size` lands on the function.

- `Rock_ReduceParamsAttr` (`rock.reduce_params`) with
  `{rowsPerBlock, colsPerBlock, numWaves}`, attached to `rock.reduce` as
  `OptionalAttr<...>:$params`. Its verifier checks only structural facts
  (positive, powers of two); anything arch-dependent is checked in the pass.
- The grid size is a launch parameter and must be a function attribute:
  `rock.grid_size`, which `RockTensorToTritonPtr` promotes to the module
  attribute that `compileUtils.cpp` requires. `rock.block_size` is *not* set:
  the launch block size comes from `numWarps * warpSize`, not from that
  attribute.

The pass fills `params` when absent and validates (never silently overrides)
when present, so a lit test can pin a schedule and the heuristic stays
separately testable. Selection lives in a free function
(`chooseReduceParams`) so promoting it to a `rock-affix-reduce-params` pass
later is a mechanical move.

## Phase A: `rock-lower-reduce-to-blockwise`, single tile

Runs on `func::FuncOp`, gated on `rock.kernel` + `rock.reduce`. Options:
`arch` (for the wave size), `numWaves` (default 4), `maxTileCols`
(default 2048). Everything below hard-fails with a diagnostic rather than
silently bailing out.

### Legality

- exactly one `rock.reduce` in the function
- `reduceMethod` is `sum`
- `axis == rank - 1`
- all shapes static
- the reduce result reaches only a `rock.store`, through `rock.transform` ops
  only; any fusion op on the way means post-reduce fusion and is rejected
- every leaf input's transform chain is contiguous along the last axis,
  otherwise `order[0]` flips in the Coalesce analysis and the tile reasoning
  above inverts

### Schedule

```
T    = numWaves * getWaveSize(arch)
vmax = 128 / elementBitWidth
C    = min(nextPow2(N), maxTileCols)
R    = clamp(nextPow2(T * vmax / C), 1, nextPow2(M))
```

The formula keeps `R*C / T` near `vmax`, so every thread issues full-width
loads, and it reproduces the taxonomy above as a side effect: short rows give a
large `R` (lane/wave-like), long rows drive `R` to 1 (block-like).

A pre-existing `params` is validated instead of recomputed; it is rejected when
it is not a power of two, when `rowsPerBlock > nextPow2(M)`, when `numWaves`
disagrees with the option, or when `colsPerBlock < nextPow2(N)` (which implies
the Phase B loop). Phase A also rejects `nextPow2(N) > maxTileCols`.

A small `M` that leaves `R*C < T` is accepted: it is correct but underutilizes
the device, and fixing it needs the split-reduce work deferred to Phase C.

### Emission

1. Merge leading dimensions to a 2-D `M x N`, pad `M` up to `gridSize * R` and
   `N` up to `C`.
2. `%bid = tt.get_program_id x`; `gridSize = ceil(M / R)` onto
   `rock.grid_size`. `rock.block_size` is deliberately not set: the launch
   block size comes from `numWarps * warpSize` (`compileUtils.cpp`), not from
   that attribute.
3. One `rock.load_marker ... [%bid]` per leaf input, through the view
   `{bid, r, c} -> {row = bid*R + r, col = c}`, giving `tensor<RxC>`;
   `propagateOutputType` retypes the pre-reduce fusion chain onto the tile.
4. `rock.reduce` becomes `rock.blockwise_reduce sum {axis = 1}` yielding
   `tensor<R>`.
5. A `rock.store_marker` over `{bid, r} -> row` becomes the store's source,
   and the store's destination is re-viewed as the same flat, padded row
   space.

`R >= 1` always, so the tile is always rank 2 and the
`tt.reduce` -> scalar -> `tt.splat` path in `RockToTTIR` is never taken.

Padding needs no extra work: `createMaskFillValue` in
`PreserveMaskedLoadSemantics` already fills masked lanes with zero for sum
consumers (and `-inf` for a max `rock.blockwise_reduce`).

### Notes from the implementation

Several details only surfaced once the pass ran end to end.

- **Kernel arguments are flat.** `rock-flatten-tosa-func-args` rewrites both
  the arguments and the return operands of a kernel to rank 1, and
  `rock-transforms-to-pointer-arith` relies on it: it requires the composed
  transform chain to produce a single offset. A kernel whose arguments are
  multi-dimensional fails there, in the elementwise pipeline just as much as
  in this one, so the tests here feed flat arguments viewed through an
  `Unmerge`.
- **The contiguity check works on the composed map.** Since the chain bottoms
  out in a flat buffer, "contiguous along the reduced axis" is exactly
  "stepping the last coordinate by one steps the offset by one", which
  `AffineMap::constantFold` answers directly. Pattern-matching the individual
  transforms, which the plan first called for, would have rejected the
  `Unmerge` that every real input carries.
- **The views after the reduction are dropped, not kept.** Because the kernel
  result is flattened, a reduction is normally followed by a reshape before
  its `rock.store`. `rock-lower-stores` expects the store's source to reach a
  `rock.store_marker` through fusion ops only, so those views cannot stay.
  Both sides of the store move into one flat padded row space instead (the
  shape the elementwise pipeline also settles on), and the dropped views are
  first checked to be order-preserving reshapes by comparing the composed map
  against row-major order at zero and at each power-of-two index, which is
  exact for affine maps.
- **Degenerate transforms are avoided.** A `Merge` of a single dimension does
  not survive the pointer-arith expansion, so a rank-2 input passes its
  dimensions through instead of merging one leading dimension.
- **Views of a block argument are anchored after `tt.get_program_id`.**
  Inserting them at the top of the block would place them before the id the
  tile markers take as an index.
- **`rock-insert-output-stores` needed the reduce marker.** It only gave a
  store to kernels with a `FusionRoot` or the `rock.elementwise` attribute,
  and a `rock.reduce` is neither; it now recognises the `rock.reduce` marker
  on the same path.
- **A follow-up for the output layout.** When the tile holds many rows, the
  reduction result is laid out across warps while the store wants it in a 1-D
  layout, so Triton inserts one `convert_layout` through LDS after the
  reduction. It is not on the reduction's critical path, but it is avoidable.

### Pipeline

```
rock-regularize-reduce
rock-lower-reduce-to-blockwise
rock-regularize-input
rock-lower-loads
rock-lower-stores
<buildKernelPipelineTail>
```

### Tests

- `lowering_rock_lower_reduce_to_blockwise.mlir`: long `N` (`R == 1`), short
  `N` (large `R`), f16 (`vecWidth == 8`), both axes padded, a rank-3 input,
  pre-reduce fusion, a pinned `params`, and a function without the marker.
- `lowering_rock_lower_reduce_to_blockwise_invalid.mlir`: non-last axis, a
  `max` reduction, post-reduce fusion, `N` over `maxTileCols`, a pinned
  `params` that is too narrow or disagrees on `numWaves`, and a transposed
  input.
- `rock-reduce-pipeline.mlir`: end to end, checking `tt.load` / `tt.reduce` /
  `tt.store` and the `rock.grid_size.<name>` module attribute.

## Phase B: loop over the reduction axis

Lifts `maxTileCols`. When `nextPow2(N) > maxTileCols`, set `C = maxTileCols`
and `K = ceil(N / C)`, then clone the leaf-to-reduce chain into an `scf.for`
with a `tensor<RxC>` accumulator `iter_arg` and `%k` as an extra index on each
`load_marker`. One `rock.blockwise_reduce` still runs after the loop, so there
is a single cross-lane reduction regardless of `N`, no atomics, and ownership
is preserved. This is the same structure as MIGraphX's `local_stride`
accumulate-then-combine, and as the Rock GEMM K loop, which already puts a
`rock.load_marker` with a loop-varying index inside an `scf.for`.

Phase A is the `K == 1` case, so this is strictly additive.

## Phase C: deferred

- Transposed / strided inputs, where the reduce axis is last logically but not
  in memory.
- Split-reduce plus atomics (or a two-kernel decomposition) for a small `M`
  with a huge `N`.
- Per-kernel `numWaves` plumbed into `TritonOptions`.
- Promote `chooseReduceParams` into a `rock-affix-reduce-params` pass with a
  `perf_config` string and tuning-driver support.
- Unify `rock-regularize-reduce` with `rock-regularize-elementwise`.
