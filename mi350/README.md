# MI350 (gfx950) reproducer scripts

These scripts reproduce the **bugs that still fire on MI350 after the
2026-05-07 sweep + recent PR fixes**. The original analysis produced
nine candidate scripts; after running on MI350, seven of them (attention
groups 1-6 and standalone GEMM) had all of their cases PASS and have
been deleted. Only two scripts remain:

- `group7_conv.sh` -- 1 conv numerical-result bug (the [0 0 0] case)
- `group9_gemm_gemm.sh` -- 3 gemm_gemm bugs: 1 numerical-result FAIL,
  1 compiler OOM, 1 compiler crash (`std::bad_array_new_length`)

## What was filtered out

A failure in the raw log is kept only if running it on the **current
MI350 build** still reproduces a real bug. We removed:

- Cases that now PASS (presumably fixed by recent PRs).
- The already-reported `rock::TransformMapAttr::getUpperBounds()`
  SIGSEGV (8 conv cases hit it on this run).
- Structural-rejection failures the sweep harness would have classified
  as `NOT_APPLICABLE` but which surface as crashes when the scripts run
  the pipeline directly:
  - "Fusion with SplitK perfConfig is not legal" (3 conv cases).
  - "ttg.shared exceeds LDS limit" (2 gemm_gemm cases).

## Usage

```bash
# from the rocmlirTriton repo root, with a built tree in ./build
./mi350/group7_conv.sh
./mi350/group9_gemm_gemm.sh
# or
BUILD_DIR=/path/to/build ./mi350/group7_conv.sh
```

The pipelines mirror `mlir/utils/performance/parameterSweeps.py::test_config`:

- conv: `rocmlir-gen <args> | rocmlir-driver -c | rocm-run`
- gemm_gemm: `rocmlir-gen <args> | rocmlir-driver --host-pipeline=highlevel - | rocmlir-driver -c | rocm-run`

`[1 1 1]` on the last line is PASS; any zero, missing output, crash,
OOM, hang, or timeout is a FAIL.
