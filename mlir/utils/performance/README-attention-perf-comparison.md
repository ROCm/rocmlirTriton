# Attention branch performance comparison

These scripts compare quick-tuned attention performance between two pinned
rocmlirTriton revisions. They are intended to run unchanged on any supported
AMD GPU server.

## 1. Prepare isolated source and build trees

Use separate worktrees and build directories. Never reuse a build after
switching its source tree.

```bash
REPO=/path/to/rocmlirTriton
BASE=/tmp/rocmlir-pr347-base
CANDIDATE="$REPO"

git -C "$REPO" worktree add --detach "$BASE" \
  0409d83a7f151269baad29592a4deac79bd39ae7

(cd "$BASE" && \
  bash ./cmake.sh && \
  cmake --build build --target check-rocmlir-build-only && \
  cmake --build build --target ci-performance-scripts)
(cd "$CANDIDATE" && \
  bash ./cmake.sh && \
  cmake --build build --target check-rocmlir-build-only && \
  cmake --build build --target ci-performance-scripts)
```

The runner rejects dirty trees and records both exact revisions, source paths,
build paths, and consumed-artifact fingerprints in its manifest. The
`ci-performance-scripts` target writes a source-revision stamp only after the
preceding build succeeds. Python 3.9 or newer is required because the existing
tuning runner uses `argparse.BooleanOptionalAction`.

## 2. Generate causal and KV-cache performance configs

```bash
OUT=/tmp/rocmlir-pr347-perf
mkdir -p "$OUT"

python3 "$CANDIDATE/mlir/utils/performance/filterAttentionConfigs.py" \
  --input "$CANDIDATE/mlir/utils/performance/configs/tier1-attention-configs" \
  --output "$OUT/causal-kvcache-attention-configs" \
  --mode performance
```

Performance mode emits 122 unique problems: the two causal-only tier-1 cases
and 120 pure KV-cache variants. It expands omitted datatypes, preserves every
original query shape, forces the KV-cache variants to non-causal, and supplies
`current_seq_len=SeqLenK-1` for every group. This exercises both PR #347 paths
without changing the valid K/V tile count. The adjacent JSON manifest records
the source line or lines for every generated problem.

Conservative mode remains available for the smaller four-config smoke
population, while `--mode exact` selects only the two existing causal cases.

## 3. Quick-tune and benchmark both branches

Launch detached from the SSH session. The command can be rerun unchanged after
an interruption: tuningRunner resumes from its TSV/state files and completed
benchmark samples are skipped.

```bash
python3 "$CANDIDATE/mlir/utils/performance/runAttentionBranchBenchmark.py" \
  --base-source "$BASE" \
  --base-build "$BASE/build" \
  --candidate-source "$CANDIDATE" \
  --candidate-build "$CANDIDATE/build" \
  --configs "$OUT/causal-kvcache-attention-configs" \
  --output-dir "$OUT/results" \
  --samples 3 \
  --detach
```

The runner:

- proves a GPU is idle using live process, activity, and VRAM information;
- reserves and exposes exactly one physical GPU;
- quick-tunes both branches, then alternates paired base/candidate samples to
  reduce thermal and clock-order bias;
- quick-tunes with winning-config CPU verification disabled;
- retries failed subprocesses with backoff and atomically checkpoints results;
- writes a PID, detached log, per-stage logs, and a reproducibility manifest.

Pass `--gpu N` to require a specific idle physical GPU. The command fails
closed if it cannot prove that device is idle. Do not override
`ROCR_VISIBLE_DEVICES`; the runner manages it.

## 4. Compare one architecture

Replace `gfx90a` with the architecture recorded in `run-manifest.json`.

```bash
ARCH=gfx90a
python3 "$CANDIDATE/mlir/utils/performance/compareAttentionPerformance.py" \
  --base "$OUT/results/$ARCH/base/performance.csv" \
  --candidate "$OUT/results/$ARCH/candidate/performance.csv" \
  --expected-samples 3 \
  --output "$OUT/results/$ARCH/comparison.csv"
```

The comparison uses the median of process-level samples and computes:

```text
% Diff = 100 * (Candidate TFlops - Base TFlops) / Base TFlops
```

Duplicate samples, mismatched config sets, inconsistent revisions/perf configs,
and invalid measurements are errors rather than silently dropped rows.

## 5. Combine architecture results into a workbook

Copy each server's `comparison.csv` to one machine, then pass all files:

```bash
python3 \
  "$CANDIDATE/mlir/utils/performance/generateAttentionPerformanceWorkbook.py" \
  --input /tmp/gfx90a-comparison.csv \
          /tmp/gfx950-comparison.csv \
          /tmp/gfx1201-comparison.csv \
  --output /tmp/pr347-attention-performance.xlsx
```

The workbook has one sheet per detected `gfx*` architecture. Base TFlops,
candidate TFlops, and percentage difference are colored by row-wise
regression/speedup, with yellow indicating a result within the default
0.5-percentage-point tie threshold.
