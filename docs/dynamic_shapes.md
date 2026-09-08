# Dynamic Shapes

## The rule

A `!migraphx.shaped` type may have **at most one dynamic length, and it must sit
on the slowest-moving (largest-stride) dimension**. Strides themselves must
always be static.

This is forced, not a policy choice. A stride depends only on the extents
*inside* it, so a dynamic length anywhere but the outermost position would make
the strides outside it dynamic too, and a dynamic stride leaves the tensor with
no computable memory layout. The one-dynamic-length limit comes from the other
end: kernel arguments are flattened to 1-D, and the reshapes that restore the
logical shape can only ask TOSA to infer a single extent.

The rule is about **stride order, not position in the logical shape**. A
transpose may move the unknown extent to the middle of the shape and it stays
legal as long as it keeps the largest stride.

Enforcement lives in `MIXRShapedType::asMemoryLayoutTensor()`, so every consumer
of the type gets the same answer.

## Per-operation support

### GEMM (`dot`, `quant_dot`)

Batch and M may be dynamic. **K and N must be static**.

Block-scaled `quant_dot` supports a dynamic M. The lowering undoes MIGraphX's
broadcast scales with a `tosa.slice` that narrows only the block lane axis, so a
dynamic M passes through at full extent.

When B is unbatched, A's batch folds into M. Either extent may be unknown, since
their product is still the single inferable extent of that reshape.

Rejected: a dynamic batch against a *batched* B (e.g. `?` vs `2`). Only B may be
broadcast, so the sole applicable lowering is a plain batched matmul, and that is
correct only if the two batches are equal at runtime, which cannot be proven.
Note that two dynamic batches (`?` vs `?`) *are* accepted, and are assumed equal,
unless the other non-dynamic dimensions differ.

### Convolution (`convolution`, `backwards_data_convolution`)

**Only the batch may be dynamic.** N is the slowest-moving axis in both the NCHW
logical shape and the NHWC memory layout. Channel and spatial extents feed
padding and window arithmetic, and the filter is a weight, so all must be static.

Both the `tosa.conv2d` path and the `conv_bwd_data` custom-op path carry the
dynamic batch through. 1-D convolutions, which are expanded to 2-D, work too.

### Attention

Attention is not a distinct op at this level -- it is `dot` + `softmax` + `dot`
-- so it inherits the GEMM rules. **A dynamic batch is supported**, for both 3-D
(batch, seq, head\_dim) and 4-D multi-head shapes.

In the 4-D case the batch dimensions flatten into a single batch dimension
(`[?, 4, M, K]` becomes `[?*4, M, K]`), which is legal because both operands get
the identical reshape, preserving which slice of Q meets which slice of K. This
is distinct from the batch-into-M fold used for an unbatched B.

### Elementwise and reductions

Ops that map onto TOSA elementwise ops and reductions (`add`, `mul`, `softmax`)
propagate a dynamic extent without special handling; `softmax` keeps it across
its `reduce_max` / `reduce_sum` decomposition.

Rejected: lowerings that materialise a dense constant sized to the result, such
as `relu`, since a dense attribute cannot have a dynamic type.

## Other rejections

| Case | Why |
|----|----|
| Dynamic length off the slowest-moving axis | No computable memory layout |
| More than one dynamic length | A reshape can only infer one extent |
| Any dynamic stride | Nothing supplies a stride at runtime |
| Dynamic length on a broadcast axis (stride 0) | The layout is fine, but rebuilding the logical shape would have to broadcast out to an unknown extent |