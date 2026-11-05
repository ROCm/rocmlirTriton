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
disagrees with the option, or when `colsPerBlock` exceeds `maxTileCols`, which
is the register budget the loop exists to respect. A `colsPerBlock` below the
reduced extent is legal and asks for the loop explicitly.

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
  pre-reduce fusion, a pinned `params`, a function without the marker, and the
  looped forms: one with fusion, one whose reduced extent is not a multiple of
  the tile, and one where a pinned `params` forces the loop.
- `lowering_rock_lower_reduce_to_blockwise_invalid.mlir`: non-last axis, a
  `max` reduction, post-reduce fusion, a pinned `params` that is too wide or
  disagrees on `numWaves`, and a transposed input.
- `rock-reduce-pipeline.mlir`: end to end for both the single-tile and the
  looped form, checking `tt.load` / `tt.reduce` / `tt.store` and the
  `rock.grid_size.<name>` module attribute.

## Phase B: loop over the reduction axis

A reduction wider than `maxTileCols` is summed a tile at a time. The tile keeps
the shape Phase A chose, `colsPerBlock` stays at the limit, and the reduced axis
is cut into `numIters = ceil(N / colsPerBlock)` chunks:

```mlir
%acc = scf.for %iter = 0 to numIters step 1 iter_args(%a = %zero) -> tensor<RxC> {
  %tile = rock.load_marker %in views [#view] [%bid, %iter] : ... -> tensor<RxC>
  ...                                        // the elementwise ops, moved in
  %next = arith.addf %a, %fused : tensor<RxC>
  scf.yield %next
}
%sum = rock.blockwise_reduce sum %acc {axis = 1}   // once, after the loop
```

Each lane keeps a running sum in registers and the workgroup combines once, so
the register footprint is set by the tile rather than by `N`, there is still a
single cross-lane reduction, and each output still has one writer. This is the
same accumulate-then-combine shape as MIGraphX's `local_stride`, and as the
Rock GEMM K loop, which already puts a `rock.load_marker` with a loop-varying
index inside an `scf.for`.

Measured on gfx942: a reduction of 3000 elements and one of 1048576 compile to
the same 7 LDS operations and 8 cross-lane shuffles, and the loop body holds
nothing but the loads and the `addf` -- no LDS, no shuffles. Only the trip
count differs.

Phase A is the `numIters == 1` case and its output is unchanged.

Two details the loop form needs beyond Phase A:

- **The elementwise ops move into the loop.** They read a tile that only exists
  inside it. They are collected during the same backward walk that finds the
  leaves, sorted back into program order, and moved to the top of the loop body
  after the markers. Without a loop they stay where they are, and each marker
  is instead placed next to the view it reads so that it dominates them.
- **The reduced axis is padded to `numIters * colsPerBlock`**, which is a
  multiple of the tile but not generally a power of two. That is fine: only the
  tile becomes a Triton tensor, the padded extent is just a view, and the tail
  lanes of the last iteration read as zero.

The accumulation happens in the element type of the reduction, which is what a
single `tt.reduce` over the same values would use too. It is a sequential sum
over chunks rather than a tree, though, so a long `f16` reduction loses more
precision than the same reduction would in one tile.

## Phase C: deferred

### C1. A strided reduce axis, or: where the `lane` schedule comes from

`rock-lower-reduce-to-blockwise` rejects an input whose reduced axis is not
the contiguous one, because the tile it would pick is the wrong way round.
This is the case MIGraphX serves with its `lane` kernel, and it is the one
remaining schedule the tile formula cannot express.

The reason the tile has to change is that `tritongpu-coalesce` takes `order`
from the pointer's `AxisInfo`, not from our axis numbering. When the reduced
axis is strided the rows are what is contiguous, so `order` becomes `[0, 1]`
and the `BlockedEncodingAttr` builder spends its lanes on the *row* axis
first. Everything in the Phase A table flips.

Measured on gfx942 with 4 waves of 64 lanes (so 256 threads and a
`vecWidth` of 4 for f32), on an f32 tile whose element `(r, c)` lives at
offset `c * R + r`:

| tile `R x C` | `threadsPerWarp` | `warpsPerCTA` | LDS ops | DPP ops |
| --- | --- | --- | --- | --- |
| `64 x 16` | `[16, 4]` | `[1, 4]` | 16 | 0 |
| `128 x 16` | `[32, 2]` | `[1, 4]` | 28 | 8 |
| `256 x 16` | `[64, 1]` | `[1, 4]` | 28 | 8 |
| `1024 x 16` | `[64, 1]` | `[4, 1]` | **0** | **0** |
| `4096 x 16` | `[64, 1]` | `[4, 1]` | **0** | **0** |

Every row has `sizePerThread = [4, 1]` and `order = [0, 1]`: the vector is
spent on rows, and rows are the fast axis.

So the rule for a strided reduction is the mirror image of the contiguous
one. Rows have to absorb every thread before the reduced axis gets any,
which happens exactly when

```
rowsPerBlock >= numThreads * vecWidth
```

since the builder gives the row axis `clamp(numThreads, 1, R / sizePerThread)`
threads and `sizePerThread` along rows is `vecWidth`. Below that threshold the
leftover warps land on the reduced axis and the reduction pays for LDS; at or
above it each thread owns whole rows and the reduction is a handful of `addf`
instructions on registers. On the table above `1024 = 256 * 4` is where it
switches over, and the switch is abrupt: at `256 x 16` the reduction still
costs 28 LDS operations and 8 cross-lane shuffles, at `1024 x 16` it costs
neither.

Three consequences for the implementation:

- **Detection is already done.** `checkLastAxisIsContiguous` computes the
  stride of the reduced axis by folding the composed map at two points, and
  currently only compares it against one. Returning the number instead of
  failing is all the analysis this needs.
- **It leans on the Phase B loop.** A thread holds `vecWidth * colsPerBlock`
  elements, so with `rowsPerBlock` pinned at `numThreads * vecWidth` the
  register footprint is set entirely by the reduced extent. Anything but a
  short reduction has to be cut into chunks, which the loop already does; the
  strided path needs a `colsPerBlock` budget of its own, since the limit that
  suits a contiguous tile is far too generous when every thread holds whole
  rows.
- **Inputs can disagree.** With several leaf inputs, one may be contiguous
  along the reduced axis and another strided, and no single tile suits both.
  MIGraphX settles this by byte dominance: it only consults inputs whose size
  is within a factor of four of the largest, on the grounds that the small
  ones are served from cache either way. The same tie-break fits here, and
  the leaves are already collected.

### C2. Too few rows to fill the device

`gridSize` is `ceil(M / rowsPerBlock)`, so a reduction of a few very long rows
occupies a few workgroups on a device with hundreds of compute units. Phase A
accepts this and runs slowly; the fix costs the single-writer property that
the rest of the design leans on.

Two ways out:

- **Split the reduced axis across workgroups and accumulate atomically.** The
  zero-initialisation this needs already exists: the reduce converter in
  `TosaToRock` attaches `rock.prefill` with a zero to the kernel result, and
  the attribute is carried through `RockTensorToTritonPtr` into the module
  attribute `RockEmitGpuBinary` turns into a prefill request. `rock.store`
  already has an `atomic_add` method. The cost is that a float atomic
  accumulates in whatever order the workgroups retire, so results stop being
  reproducible run to run, which the CPU verifier comparisons would have to
  tolerate.
- **Two kernels, partials through a workspace.** Deterministic given a fixed
  split, but the pipeline has no notion of a scratch allocation to put the
  partials in, so this is the larger change.

MIGraphX draws the line at `outputs < cuCount * 1024`, and splits with a
threshold of 8192 elements and at most 16 ways
(`split_reduce` in `jit/reduce.cpp`).

### C3. `numWaves` reaching `TritonOptions`

`numWaves` is a pass option today, defaulting to 4 to match
`TritonOptions::numWarps`, and nothing enforces the agreement. The two must
match: `numWarps` sets the launch block size in `compileUtils.cpp`
(`info.blockSize = numWarps * warpSize`) as well as the lane and warp split
the tile is chosen against, so a mismatch silently mis-tunes the kernel and
mis-reports its block size.

The channel for this already exists but is GEMM-shaped: the driver calls
`fillCompilationConfigs`, which reads `numWaves` out of a perf config and
writes it into `TritonOptions`, for every op implementing
`RockGemmWrapperInterface`. A reduce kernel has no such op, which is why
`numWarps` keeps its default. Either the driver learns to look at
`rock.reduce_params`, or the attribute implements the tuning interface. The
knob is worth having mainly for long reductions, where more waves means more
parallelism per row and fewer trips through the Phase B loop.

### C4. The layout conversion after a many-row reduction

When the tile holds several rows, the reduction result is laid out across the
warps that produced it, while the store wants the plain 1-D layout
`tritongpu-coalesce` picks for its pointer tensor. Triton bridges the two with
a `ttg.convert_layout`, which costs one round trip through LDS. For a
`64 x 16` f32 tile that is 256 bytes against the 4 KiB the tile itself loads,
and it is off the reduction's critical path, so Phase A leaves it alone.
Making the store side agree with the layout the reduction produces would
remove it.

### C5. Smaller items

- Promote `chooseReduceParams` into a `rock-affix-reduce-params` pass, so the
  lowering can require `rock.reduce_params` rather than compute it, mirroring
  `rock-affix-params` and `rock-gemm-to-gridwise`. That is also where a
  `perf_config` string and tuning-driver support would go.
- Unify `rock-regularize-reduce` with `rock-regularize-elementwise`; the two
  patterns differ only in the marker they gate on.
- Reductions other than `sum`. `max` needs the `-inf` neutral fill, which
  `PreserveMaskedLoadSemantics` already emits for a max
  `rock.blockwise_reduce`, plus a decision about NaN semantics.
