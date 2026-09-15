# Per-problem quick tuning

## Implementation plan

### Problem identity

The compiler serializes the operation's problem arguments and removes the
hardware prefix, operation/data-type tokens, and every field that
`ParamLookupTable::findFallback` may substitute. Transpose, layout, shape,
convolution, and attention semantic fields remain. The spelling is independent
of architecture, CU count, chiplet count, kernel type, and data type.

Convolution identities use Rock's internal lower-case layout spelling. The
generator maps external filter `k` to internal GEMM `n` and external output `k`
to internal GEMM `c`. The debug schema has no group column, so generated
convolution identities use group 1. Recorded grouped problems therefore miss
and safely use the unchanged cover.

### Data format

`QuickTuningProblemPerfconfigs.inc` has one independent block per lookup key:

- sorted exact problem names for collision-free binary search;
- a dictionary of config strings shared by problems under that key;
- 32-bit offsets delimiting each problem's list; and
- 16-bit indices into the config dictionary.

Nothing is shared across lookup keys. This costs some duplicated strings, but
keeps regeneration local and avoids coupling architecture updates. Compiler and
linker string merging reduces the shipped cost.

The budget is 100 KiB average and 200 KiB maximum per populated key. The
generated 24-key snapshot is 1,822,263 bytes of source. Its conservative
payload ranges from 7,029 to 185,602 bytes per key and averages about 72 KiB,
which extrapolates to about 7 MiB for 100 populated keys before linker merging.

### Generator

`quickTuningProblemGen.py` is additive and does not call or modify the set-cover
generator. It auto-detects each debug TSV's operation, groups measured rows by
stable problem identity, and writes the complete generated include in one run:

```bash
python3 mlir/utils/performance/analysis/quickTuningProblemGen.py \
  /tmp/qt-gfx1200/*.tsv.debug \
  /tmp/qt-gfx1101/*.tsv.debug \
  /tmp/qt-gfx1151/*.tsv.debug
```

The generation-time `--top-n` option defaults to 3. Lists are ordered by
descending measured TFlops with the config string as a deterministic tie
breaker. If the leaders contain no `splitKFactor=1` config, the last slot is
replaced by the best measured non-split-K config. If none was measured, the
leaders are retained and a warning is emitted; the existing conservative
default insertion still protects `front()`.

### Lookup

The new `ParamLookupTable::lookupProblem` entry point resolves the ordinary
cover key first, including all existing fallback substitutions, and then
searches the per-problem data under that resolved key. An absent name, absent
key, unnamed operation, or disabled path returns an empty vector. Existing
callers then invoke the unchanged `lookup` entry point and receive the original
cover.

`ROCMLIR_DISABLE_PROBLEM_QUICK_TUNING` is an operational and A/B escape hatch;
it disables only the new path.

### Tests

Tests use synthetic fixtures and never assert checked-in tuning contents:

- Python tests cover measured ordering, non-split-K slot reservation, the
  no-non-split case, sorted names, and per-key config interning.
- C++ tests construct a synthetic `ProblemTuningData` and cover exact lookup,
  ordering, and misses.
- Existing fallback tests continue to cover key resolution independently.

Regenerating tuning data cannot change these expectations.

## Acceptance and validation

- `QuickTuningPerfconfigs.inc` is unchanged from `develop`; set-cover generation
  and lookup are unchanged.
- Unknown and grouped-convolution problems fall back to the original cover.
- gfx1201 GEMM and convolution preflights resolve the substituted gfx1200 key
  and return three configs.
- All 24 generated keys have complete problem rows, and generation requires no
  manual edits.
- Rock unit tests and the performance-script lit tests pass.

The gfx1201 A/B validates narrowing but disproves the plan's no-regression
assumption for cross-architecture data:

- GEMM: 18.07 to 3.02 candidates/problem, 3007.31 s to 253.39 s; 70/315 winner
  identities unchanged.
- Convolution: 57.71 to 6.34 candidates/problem, 10097.45 s to 1253.54 s;
  240/938 winner identities unchanged. The extra candidates are safe cover
  fallback for grouped problems whose group is absent from the snapshot.
- Attention: 8.92 to 3.06 candidates/problem, 179.30 s to 49.98 s; 18/50 winner
  identities unchanged.

Different winners are not uniformly no worse: their geometric TFlops ratios
are 0.9991 for GEMM, 0.9657 for convolution, and 1.0203 for attention, with
minimum ratios 0.6853, 0.5032, and 0.7980 respectively. A gfx1200-measured
leader is not guaranteed to remain a leader on gfx1201. Meeting a strict
no-regression criterion on substituted hardware would require measurements for
that hardware or retaining enough of the cover, which is a different
quality/compile-time tradeoff from top-N-only lookup.
