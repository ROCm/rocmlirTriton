# Elementwise kernel pipeline bring-up status

## Goal

Introduce a dedicated lowering path for standalone elementwise GPU kernels with:

- a function marker attribute: `rock.elementwise` (unit attr)
- no perf-config/tuning-param dependency for elementwise kernels
- a separate kernel pipeline from GEMM/conv/attention lowering
- no reduce support in this bring-up (explicitly out of scope)

## Current baseline (before changes)

- The current `buildKernelPipeline` is GEMM/conv/attention centric.
- Many passes are guarded only by `rock.kernel` and assume GEMM-style IR invariants.
- `rock-insert-output-fusion-loads` expects `rock.store_marker` chains produced by GEMM lowering and fails when missing.
- `rock-affix-params` validates known function attributes; new `rock.*` attrs must be added there.
- No `rock.elementwise` marker exists today.
- Existing elementwise conversion pass (`rock-tosa-to-elementwise`) only rewrites TOSA ops to arith/math; it does not establish a standalone elementwise kernel lowering pipeline.

## Design direction

Hybrid approach:

1. Reuse robust late-stage lowering (`lower-loads/stores`, pointer-arith path, `rock-to-ttir`, `rock-tensor-to-triton-ptr`, Triton/backend).
2. Reintroduce elementwise-specific mid-stage passes (adapted from `elementwise-fusions`) to build required marker/tiled structure:
   - `rock-regularize-elementwise`
   - `rock-elementwise-to-gridwise`
   - `rock-gridwise-elementwise-to-blockwise`
3. Remove perf-config dependency for standalone elementwise kernels; derive tile/grid policy from defaults/function attrs.
4. Keep reduce unsupported for this path.

## Implementation phases

1. **IR marker + attr plumbing**
   - add `rock.elementwise` attr
   - whitelist in rock attribute validators/classifiers

2. **Pass bring-up**
   - add and register three elementwise passes listed above
   - gate on `rock.elementwise`
   - remove perf_config requirements from those passes

3. **Pipeline split**
   - add `buildElementwiseKernelPipeline`
   - route kernel funcs by marker attr
   - keep GEMM pipeline unchanged for non-elementwise kernels

4. **Driver integration**
   - run dedicated elementwise pipeline for marked kernels
   - preserve existing kernel/host separation behavior

5. **Validation and tests**
   - add lit tests per pass
   - add pipeline-level tests to ensure no GEMM-only pass failures on elementwise kernels

## Out of scope for this bring-up

- reduction support in elementwise kernels (`rock.reduce` / reduce lowering)
- tuning/perf database integration for standalone elementwise kernels

## Progress log

- [x] Created this status/design document.
- [ ] Add `rock.elementwise` attr + whitelist updates.
- [ ] Add three elementwise passes and unit tests.
- [ ] Wire separate elementwise kernel pipeline.
- [ ] Driver integration and end-to-end tests.
