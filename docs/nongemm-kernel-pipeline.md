# The nongemm kernel pipeline

A **nongemm** kernel is a graph of views (transpose, reshape, broadcast) and
elementwise ops with no GEMM, convolution or attention root, optionally ending
in a single `rock.reduce`. It is marked `rock.nongemm` on the function and
lowered by `rock-nongemm-pipeline`, which needs no tuning parameters.

This supersedes `elementwise-kernel-pipeline-status.md` and
`reduce-kernel-plan.md`, which described the two halves of this before they
were one thing.

## Scope

- One output: exactly one `rock.store`.
- At most one `rock.reduce`, and it must be last: only views may stand between
  it and the store.
- `reduce_sum` only, over the last axis, with static shapes.
- Direct ownership of the output, one writer per element, no atomic fallback.
- A kernel with no reduce is called a **map** below; one with a reduce, a
  **reduction**.

## The two kinds are one kernel over one view

Both are tiled over the same canonical `M x N` view:

| | `M` | `N` |
| --- | --- | --- |
| reduction | rows of the reduced tensor, leading axes merged | the reduced extent |
| map | `1` | total elements |

so one schedule covers both. A map has `M == 1`, which hands it
`rowsPerBlock == 1` through the ordinary clamp rather than through a special
case: with no cross-lane combine to pay for, rows buy a map nothing and only
cap the vector width each thread loads with.

The one structural difference is **where the index of a chunk of `N` comes
from**. A reduction has to accumulate its chunks in one workgroup to keep a
single writer per output row, so chunks become an `scf.for` carrying the
running sum in registers. A map's chunks are independent, so they go to the
grid and there is no loop.

## Why the tile is the entire schedule

MIGraphX's reduce JIT picks between `lane`, `wave` and `block` kernels
(`src/targets/gpu/jit/reduce.cpp`). That taxonomy does not transfer: in the
Rock -> TTIR path the kernel body is just `tt.load` -> `tt.reduce` ->
`tt.store`, and the schedule is decided downstream by the tensor's blocked
layout, which is in turn decided by *the tile shape we load*.

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
`vecWidth = 4`), the same 1024-element tile gives:

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

## Choosing the tile

With `budget = numWaves * waveSize * vecWidth`, the elements one workgroup
holds when every lane issues one full-width (128-bit) load:

```
vecWidth = 128 / elementBitWidth        // of the widest type the kernel touches
budget   = numWaves * waveSize * vecWidth
C        = min(nextPow2(N), hasReduce ? maxTileCols : budget)
R        = clamp(nextPow2(budget / C), 1, nextPow2(M))
numIters = ceil(N / C)
gridSize = ceil(M / R) * (hasReduce ? 1 : numIters)
```

`vecWidth` follows the **widest** element type in the kernel, not the
reduction input's: a kernel that extends f16 to f32 ends up holding f32, and
sizing to the narrower type would overspend registers.

Only a reduction may spend past `budget`, up to the register limit
`maxTileCols` (2048). Its row has to be summed by a single workgroup, so a long
row buys fewer, fatter blocks and there is no alternative. A map is under no
such obligation, and a tile past the budget would only mean fewer blocks each
doing more, so it stops at `budget`.

A small kernel shrinks rather than pads: 12 elements gives `C = 16`, not a full
tile with 1012 masked lanes.

### `C` first, then `R` — and why that order matters

Element `(r, c)` of block `b` sits at buffer offset `(b*R + r) * Npadded + c`,
so:

- **`C == Npadded`** (the tile holds whole rows): consecutive `r` are `C`
  apart, so the whole `R x C` tile is **one contiguous run** of `R*C` elements.
  A wave's lanes spread along the fast axis, run out, and continue into the
  next row — still consecutive addresses. This is the `64 x 16` short-rows
  case, and it is where `R > 1` is pure win: a short row cannot fill a
  workgroup on its own.
- **`C < Npadded`** (`C` capped by `maxTileCols`): each row is a contiguous run
  of `C` and consecutive rows are `Npadded` apart. `R > 1` then buys more
  strided bursts rather than contiguity — but `C` has already eaten the whole
  budget, so the clamp puts `R` at 1 anyway.

Hence the rule the formula implements: fill `C` up to `nextPow2(N)` first, then
spend what is left on `R`. `R > 1` only ever appears when `C` already covers
the whole row. A hand-pinned small `C` with a large `R` is the one combination
that is strictly bad, and it is the one the heuristic never produces.

## Schedule on the function, launch config too

- `rock.nongemm_params<rowsPerBlock, colsPerBlock, numWaves, numStages>` is a
  **function** attribute, not an op attribute: a map has no op to hang it on.
  The lowering fills it in when it is absent and validates — never silently
  overrides — one that is present, so a lit test or a future perf config can
  pin a schedule.
- `rock.grid_size` is a launch parameter and lands on the function, where
  `RockTensorToTritonPtr` promotes it to the module attribute `compileUtils.cpp`
  requires.
- `numWaves` and `numStages` are not consumed by rock at all. They are the
  configuration the Triton pipeline must be run with, and the tile is only
  valid against them, so they are recorded and validated against the
  `TritonOptions` defaults they currently take.

### `rock.block_size` is not the tile size

Two different quantities, and conflating them was a bug in the elementwise
pipeline before the merge (its e2e tests pinned `rock.block_size = 64` while
the kernel ran 256 threads):

```
threads per block  = waveSize(arch) * numWaves      // Triton's numWarps * warpSize
elements per block = rowsPerBlock * colsPerBlock
```

`rock.block_size` means the first everywhere else in rock
(`AffixTuningParameters`, `AddTritonMetadata`, `RockEmitGpuBinary`,
`CAPI/MIGraphX.cpp`). This pipeline does not set it: the launch block size
comes from `numWarps * warpSize`, not from that attribute.

### The thread count is not ours to pick

`numWarps` sets both the real launch block size
(`compileUtils.cpp`: `info.blockSize = numWarps * warpSize`) and the layout
math above. It is never filled from a perf config for a kernel with no GEMM op,
so it keeps the `init(4)` default of `TritonOptions::numWarps`; `numStages`
likewise keeps its `init(2)`.

The tile has to be chosen against the thread count that implies, so
`kNonGemmNumWaves` and `kNonGemmNumStages` in `NonGemmUtils.h` are constants
tracking those defaults rather than options: offering a knob the pass cannot
make the Triton pipeline honour would only invite a tile tuned for a workgroup
that never runs. A pinned `rock.nongemm_params` is checked against the same
constants for the same reason.

Nothing enforces the correspondence: running this pipeline and then
`rock-triton-pipeline{numWarps=8}` still silently mis-tunes the tile.
Connecting the two is [deferred item 3](#3-numwaves-and-numstages-reaching-tritonoptions).

## The pipeline

```
rock-regularize-nongemm
cse
rock-lower-nongemm-to-blockwise{arch, maxTileCols=2048}
rock-regularize-input
rock-lower-loads
rock-lower-stores
<buildKernelPipelineTail>
```

`rocmlir-driver -c --nongemm` selects it; `rocmlir-opt --rock-nongemm-pipeline`
runs it directly.

### `rock-regularize-nongemm`

Swaps each `rock.transform` with the fusion op feeding it, repeatedly, until
every view has reached a leaf and the elementwise ops are left stacked on one
shape. A view and an elementwise op commute — the view renumbers coordinates,
the op maps the value at each one — which is what makes this sound and what
makes a view legal *anywhere* in the graph. The `cse` after it merges the
copies of a view this leaves on an input that several ops read.

A view below the reduce stays, since a reduce is not a fusion op; see
[the views below a reduction](#the-views-below-a-reduction).

### `rock-lower-nongemm-to-blockwise`

1. **Legality**, all hard errors rather than silent bail-outs: one
   `rock.store`; at most one `rock.reduce`; for a reduction, `sum`, last axis,
   static shapes, nothing but views between it and the store, and every leaf
   contiguous or broadcast along the reduced axis; for a map, at least one
   elementwise op (a pure copy has no tile to choose and no op to retype).
2. **Leaves and chain**: one backward walk from the store source collects the
   top of each input's transform chain, which becomes a `rock.load_marker`,
   and the fusion ops, which the loop form has to move into the loop body.
3. **Schedule**: fill or validate `rock.nongemm_params`; set `rock.grid_size`.
4. **Canonicalize**: a reduction views each leaf as 2-D `M x N` and pads to
   `gridSize*R x numIters*C`; a map flattens to 1-D and pads to `gridSize*C`.
5. **Emit**: `tt.get_program_id`, the tile views, a `load_marker` per leaf with
   `propagateOutputType` retyping the fusion chain onto the tile, the optional
   `scf.for`, the optional `rock.blockwise_reduce`, and a `store_marker`.

Padding needs no extra work: `createMaskFillValue` in
`PreserveMaskedLoadSemantics` already fills masked lanes with zero for sum
consumers (and `-inf` for a max `rock.blockwise_reduce`).

### The loop over a long reduction

A reduction wider than `maxTileCols` is summed a tile at a time:

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
Rock GEMM K loop.

Measured on gfx942: a reduction of 3000 elements and one of 1048576 compile to
the same 7 LDS operations and 8 cross-lane shuffles, and the loop body holds
nothing but the loads and the `addf`. Only the trip count differs.

The accumulation happens in the element type of the reduction, which is what a
single `tt.reduce` over the same values would use too. It is a sequential sum
over chunks rather than a tree, though, so a long `f16` reduction loses more
precision than the same reduction would in one tile.

## Notes from the implementation

- **Kernel arguments are flat.** `rock-flatten-tosa-func-args` rewrites both
  the arguments and the return operands of a kernel to rank 1, and
  `rock-transforms-to-pointer-arith` relies on it: it requires the composed
  transform chain to produce a single offset. Tests therefore feed flat
  arguments viewed through an `Unmerge`.
- **Contiguity is checked on the composed map.** Since the chain bottoms out in
  a flat buffer, "contiguous along the reduced axis" is exactly "stepping the
  last coordinate by one steps the offset by one", which `AffineMap::constantFold`
  answers directly. Pattern-matching the individual transforms would reject the
  `Unmerge` that every real input carries.
- **A step of zero is a broadcast, not a stride.** Every lane of a row then
  reads the same element, which carries no layout hazard and says nothing about
  the tile; the inputs that do vary along the axis decide it. This is the shape
  `x - mean` takes, so rejecting it would rule out the most ordinary pre-reduce
  fusion there is.
- **Degenerate transforms are avoided.** A `Merge` of a single dimension does
  not survive the pointer-arith expansion, so a rank-2 input passes its
  dimensions through instead of merging one leading dimension.
- **Views of a block argument are anchored after `tt.get_program_id`.**
  Inserting them at the top of the block would place them before the id the
  tile markers take as an index.
- **A map needs no post-store view handling.** `rock-regularize-nongemm` pushes
  the boundary reshape above the fusion ops, so nothing is left between the
  last of them and the store.

### The views below a reduction

`rock.reduce` keeps the reduced axis at extent one (`4x512 -> 4x1`) while
`rock-flatten-tosa-func-args` makes the kernel result rank 1, so the frontend
leaves a view behind to join the two. A reduce is not a fusion op, so
`rock-regularize-nongemm` cannot push it up.

Those views are **dropped**, with both sides of the store re-expressed in one
flat padded row space, because `rock-lower-stores` wants the store source to be
the marker itself. Dropping is only sound when the view does nothing: element
`i` in row-major order has to stay element `i`. `traceReduceToStore` therefore
vets each view as it crosses it, structurally — `Merge` and `Unmerge` are
row-major by definition, so an order-preserving reshape is exactly one built
from those and `PassThrough` whose transforms, read in order of their upper
dimensions, walk the lower dimensions in order too.

What this rules out is a view that *moves* data, such as the `PassThrough` a
`tosa.transpose` after the reduction would become: the types would still line
up after dropping it, and every output would be written to the wrong address.

[Deferred item 6](#6-dropping-the-reduced-axis) would remove the need for all
of this.

## Tests

- `Dialect/Rock/rock_regularize_nongemm.mlir` — views hoisted to the leaves,
  with and without a reduction, through casts, and with a multi-use op.
- `Dialect/Rock/lowering_rock_lower_nongemm_to_blockwise.mlir` — the tile and
  grid for long rows, short rows, f16, both axes padded, rank 3, pre-reduce
  fusion, a broadcast along the reduced axis, a pinned `params`, the looped
  forms, and the map cases (grid-tiled, padded, flattened, widest-type sized).
- `Dialect/Rock/lowering_rock_lower_nongemm_to_blockwise_invalid.mlir` — every
  diagnostic: non-last axis, a `max` reduction, post-reduce fusion, a bad
  pinned `params`, a transposed input, two stores, a copy-only map, and a view
  below the reduce that reorders or pads.
- `Dialect/Rock/rock-nongemm-pipeline-{map,reduce,matrix}.mlir` — end to end to
  TTIR, including the mixed graph this kernel type exists for.
- `rocmlir-driver/pipelines.mlir` — the pass list `--nongemm` builds.

Numerical tests are `fusion/pr-e2e/nongemm-*.e2e.mlir`: `rocmlir-gen
--clone-harness` emits a CPU twin, the highlevel pipeline lowers the TOSA, and
`rocmlir-driver -c --nongemm` compiles the kernel before `rocm-run` compares the
two. They all run with `-rand 1`, which draws integers in `[-5, 5)`: every
element differs, so a tile that reads the wrong one is caught, and every sum is
exact in f32, so the GPU not adding in the verifier's order cannot make the
comparison flaky.

The schedules they take on gfx942 (they differ on a 32-lane architecture, where
a workgroup is 128 threads and the heuristic spends the difference on rows):

| test | grid | tile `R x C` | loop |
| --- | --- | --- | --- |
| `nongemm-add-map` | 1 | `1 x 64` | no |
| `nongemm-mul-relu-map` | 1 | `1 x 64` | no |
| `nongemm-tanh-map` | 1 | `1 x 64` | no |
| `nongemm-transpose-map` | 1 | `1 x 512` | no |
| `nongemm-add-sum` | 2 | `2 x 512` | no |
| `nongemm-broadcast-sub-sum` | 4 | `2 x 512` | no |
| `nongemm-reshape-add-sum` | 4 | `8 x 128` | no |
| `nongemm-sum-long-rows` | 8 | `1 x 1024` | no |
| `nongemm-sum-short-rows` | 8 | `64 x 16` | no |
| `nongemm-sum-padded` | 50 | `2 x 512` | no |
| `nongemm-sum-wide` | 4 | `1 x 2048` | yes |

## Deferred

### 1. A strided reduce axis, or: where the `lane` schedule comes from

The lowering rejects a reduction whose reduced axis is not the contiguous one,
because the tile it would pick is the wrong way round. This is the case
MIGraphX serves with its `lane` kernel, and the one remaining schedule the tile
formula cannot express. It is also what stops a `tosa.transpose` from feeding a
reduction today, though it may feed a map.

The reason the tile has to change is that `tritongpu-coalesce` takes `order`
from the pointer's `AxisInfo`, not from our axis numbering. When the reduced
axis is strided the rows are what is contiguous, so `order` becomes `[0, 1]`
and the `BlockedEncodingAttr` builder spends its lanes on the *row* axis first.
Everything in the table above flips.

Measured on gfx942 with 4 waves of 64 lanes, on an f32 tile whose element
`(r, c)` lives at offset `c * R + r`:

| tile `R x C` | `threadsPerWarp` | `warpsPerCTA` | LDS ops | DPP ops |
| --- | --- | --- | --- | --- |
| `64 x 16` | `[16, 4]` | `[1, 4]` | 16 | 0 |
| `128 x 16` | `[32, 2]` | `[1, 4]` | 28 | 8 |
| `256 x 16` | `[64, 1]` | `[1, 4]` | 28 | 8 |
| `1024 x 16` | `[64, 1]` | `[4, 1]` | **0** | **0** |
| `4096 x 16` | `[64, 1]` | `[4, 1]` | **0** | **0** |

So the rule for a strided reduction is the mirror image of the contiguous one.
Rows have to absorb every thread before the reduced axis gets any, which
happens exactly when `rowsPerBlock >= numThreads * vecWidth`, and the switch is
abrupt: at `256 x 16` the reduction still costs 28 LDS operations and 8
shuffles, at `1024 x 16` neither.

Three consequences for the implementation:

- **Detection is already done.** `checkLastAxisIsContiguous` computes the
  stride of the reduced axis by folding the composed map at two points.
  Returning the number instead of failing is all the analysis this needs.
- **It leans on the loop.** A thread holds `vecWidth * colsPerBlock` elements,
  so with `rowsPerBlock` pinned at `numThreads * vecWidth` the register
  footprint is set entirely by the reduced extent; the strided path needs a
  `colsPerBlock` budget of its own.
- **Inputs can disagree.** One leaf may be contiguous along the reduced axis
  and another strided, and no single tile suits both. MIGraphX settles this by
  byte dominance: it only consults inputs whose size is within a factor of four
  of the largest. The same tie-break fits here, and the leaves are already
  collected.

### 2. Too few rows to fill the device

`gridSize` is `ceil(M / rowsPerBlock)`, so a reduction of a few very long rows
occupies a few workgroups on a device with hundreds of compute units. The tile
formula accepts this and runs slowly; the fix costs the single-writer property
the rest of the design leans on.

- **Split the reduced axis across workgroups and accumulate atomically.** The
  zero-initialisation this needs already exists: the reduce converter in
  `TosaToRock` attaches `rock.prefill` with a zero to the kernel result, carried
  through `RockTensorToTritonPtr` into the module attribute `RockEmitGpuBinary`
  turns into a prefill request, and `rock.store` already has `atomic_add`. The
  cost is that a float atomic accumulates in whatever order the workgroups
  retire, so results stop being reproducible run to run, which the CPU verifier
  comparisons would have to tolerate.
- **Two kernels, partials through a workspace.** Deterministic given a fixed
  split, but the pipeline has no notion of a scratch allocation for the
  partials, so this is the larger change.

MIGraphX draws the line at `outputs < cuCount * 1024`, splitting with a
threshold of 8192 elements and at most 16 ways (`split_reduce` in
`jit/reduce.cpp`).

### 3. `numWaves` and `numStages` reaching `TritonOptions`

Both are recorded in `rock.nongemm_params` and validated against constants that
track the `TritonOptions` defaults, because nothing carries them to the Triton
pipeline. They must agree: `numWarps` sets the launch block size as well as the
lane and warp split the tile is chosen against, and `numStages` decides whether
the loop over a long reduction is software-pipelined by
`createTritonAMDGPUScheduleLoops`.

The channel exists but is GEMM-shaped: the driver calls
`fillCompilationConfigs`, which reads both out of a perf config implementing
`RockTuningParamAttrInterface` and writes them into `TritonOptions`, for every
op implementing `RockGemmWrapperInterface`. A nongemm kernel has no such op.
Either the driver learns to look at `rock.nongemm_params`, or the attribute
implements that interface — which would mean stubbing a dozen GEMM-shaped
accessors (`getMPerBlock`, `getKpack`, ...) to reuse four lines.

### 4. The layout conversion after a many-row reduction

When the tile holds several rows, the reduction result is laid out across the
warps that produced it, while the store wants the plain 1-D layout
`tritongpu-coalesce` picks for its pointer tensor. Triton bridges the two with a
`ttg.convert_layout`, one round trip through LDS. For a `64 x 16` f32 tile that
is 256 bytes against the 4 KiB the tile itself loads, and it is off the
reduction's critical path. Making the store side agree with the layout the
reduction produces would remove it.

### 5. `rock-affix-nongemm-params`, for tuning

Selection lives in `chooseNonGemmParams`, a free function, and
`rock.nongemm_params` is optional-and-validated, so promoting the choice to its
own pass — mirroring `rock-affix-params` and `rock-gemm-to-gridwise` — is a
mechanical move. That pass boundary is what tuning needs: somewhere for a perf
config string to be written before the lowering reads it. There is no reason to
add it before there is a tuning driver to use it.

### 6. Dropping the reduced axis

`rock.reduce` keeps the reduced axis at extent one, which is the only reason a
view sits between a reduction and its store, and hence the only reason
`traceReduceToStore` has to vet and drop views at all. If the op dropped the
axis instead, the store would take the result directly and that machinery would
go. It is a change to the op's verifier, to `TosaToRock`, and to the GEMM-fusion
reduce path in `LowerReduce.cpp`, which lowers a reduce to a `Broadcast` back
over that axis plus an atomic store — so it is a real cleanup but not a local
one.

### 7. Reductions other than `sum`

`max` needs the `-inf` neutral fill, which `PreserveMaskedLoadSemantics`
already emits for a max `rock.blockwise_reduce`, plus a decision about NaN
semantics.
