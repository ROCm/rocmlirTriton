# MI350 (gfx950) attention/conv/gemm/gemm_gemm reproducer scripts

These scripts reproduce the **real numerical bugs** found in the MI350
performance sweeps logged in
`mi350/{attn,conv,gemm,gemm_gemm}_errors.log` (sweep run 2026-05-07).

They were classified from a host on Navi3 hardware (no MI350 available
locally), so the scripts have *not* been runtime-verified on gfx950.
Run them on an MI350 box to confirm which cases still reproduce.

## Filtering rules

A failure in the raw log is treated as a **real bug** if it is *all*
of the following:

- A `FAIL: Runner returned incorrect result` block (i.e. the kernel
  built and ran, but produced wrong values).
- Not pure threshold noise. We exclude cases where `maxAbsDiff < 1e-2`
  *and* `RMS < 5e-2` *and* no metric is `nan`. Everything with a
  detectable kernel divergence or any NaN is kept.
- Not the already-reported `rock::TransformMapAttr::getUpperBounds()`
  SIGSEGV crash (13 conv cases). Those are tracked separately.
- Not a timeout (3 cases across the four logs).

After filtering: **201 real bugs** distributed across 9 scripts.

## Scripts

| Script | Op | Cases | Pattern |
|---|---|---|---|
| `group1_attn_decode_kvcache.sh` | attention | 18 | KV-cache decode (`seq_len_q=1`, `--current_seq_len`), basic split |
| `group2_attn_decode_kvcache_lse_split.sh` | attention | 12 | KV-cache decode + `return_lse` + `split_kv>1` (online-softmax merge) |
| `group3_attn_prefill_plain.sh` | attention | 11 | Prefill, no mask, no KV-cache |
| `group4_attn_prefill_causal.sh` | attention | 9 | Prefill + `causal=True` |
| `group5_attn_prefill_split_lse.sh` | attention | 7 | Prefill + `return_lse` + `split_kv>1` |
| `group6_attn_nan.sh` | attention | 3 | Per-element NaN injection / zero-output saturation |
| `group7_conv.sh` | conv | 45 | Conv fwd/bwd_data/bwd_weight wrong results (excl. SIGSEGV) |
| `group8_gemm.sh` | gemm | 51 | Standalone GEMM wrong results |
| `group9_gemm_gemm.sh` | gemm_gemm | 45 | Back-to-back fused GEMM+GEMM wrong results |

## Usage

```bash
# from the rocmlirTriton repo root, with a built tree in ./build
./mi350/group1_attn_decode_kvcache.sh
# or
BUILD_DIR=/path/to/build ./mi350/group1_attn_decode_kvcache.sh
```

Each script invokes the same pipeline the sweep harness uses
(per `mlir/utils/performance/parameterSweeps.py::test_config`):

- Attention & GEMM+GEMM:
  `rocmlir-gen <args> | rocmlir-driver --host-pipeline=highlevel - | rocmlir-driver -c | rocm-run`
- GEMM & Conv:
  `rocmlir-gen <args> | rocmlir-driver -c | rocm-run`

Each kernel prints the verifier flag triple `[RMS_pass absDiff_pass
relDiff_pass]` on the last line. `[1 1 1]` is PASS; any zero (or a
crash/hang/timeout, or missing output) is a FAIL.

## Caveats

- Inputs are deterministic (`rocmlir-gen -rand "fixed"` is the
  default), so individual cases reproduce bit-for-bit unless the
  GPU kernel itself is non-deterministic (atomics, reduction order).
- Intermittent failures (a case that passes most runs but fails some)
  are still bugs. Re-run cases multiple times if a single PASS is
  observed before declaring it fixed.
- Some cases may now PASS due to fixes landed after the sweep was
  run on 2026-05-07. For attention this is especially likely for
  `split_kv > 1` cases, because commit `eec615b55cd1` (2026-05-08)
  fixed a buggy CPU split-KV reference path that produced false
  positives on Navi3.
