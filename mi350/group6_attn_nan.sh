#!/usr/bin/env bash
# group6_attn_nan.sh -- Attention NaN bugs (kernel or CPU reference injects NaN) (3 cases)
#
# Cases where RMS/avgAbsDiff/avgRelDiff print as nan -- a NaN escapes somewhere.
#
# Logs source: mi350/attn_errors.log (sweep run 2026-05-07).
# Filtering rule: excluded purely threshold-only misses (maxAbsDiff < 1e-2
# and RMS < 5e-2 and no NaN) and the already-reported
# rock::TransformMapAttr::getUpperBounds() SIGSEGV in conv (13 cases).
#
# Pipeline (per parameterSweeps.py::test_config and README.md):
#   rocmlir-gen <args> | rocmlir-driver --host-pipeline=highlevel - |
#       rocmlir-driver -c | rocm-run
#
# Each kernel prints the verifier flag triple `[RMS_pass absDiff_pass
# relDiff_pass]` on the last line of output. `[1 1 1]` is a clean
# PASS; any zero (or missing output) is a FAIL. Crashes/hangs/timeouts
# are also FAILs.
#
# Hardware: gfx950 (MI350); the harness was sampled with
# --num_cu 256 --num_chiplets 8.
#
# Usage (from the rocmlirTriton repo root):
#   ./mi350/group6_attn_nan.sh
# or with a non-default build directory:
#   BUILD_DIR=/path/to/build ./mi350/group6_attn_nan.sh

set -u -o pipefail
# Note: -e is intentionally NOT set -- failing pipelines are the point.

BUILD_DIR="${BUILD_DIR:-build}"
GEN="${BUILD_DIR}/bin/rocmlir-gen"
DRV="${BUILD_DIR}/bin/rocmlir-driver"
RUN="${BUILD_DIR}/bin/rocm-run"

for bin in "$GEN" "$DRV" "$RUN"; do
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: required binary not found or not executable: $bin" >&2
        echo "Build the project first (see README.md), or set BUILD_DIR." >&2
        exit 1
    fi
done

run_case() {
    local label="$1"; shift
    printf "================================================================\n"
    printf "  %s\n" "$label"
    printf "================================================================\n"
    "$GEN" "$@" \
        | "$DRV" --host-pipeline=highlevel - \
        | "$DRV" -c \
        | "$RUN"
    printf "\n"
}

run_case "[1/3] dt=bf16 rls split_kv=4 decode kv-cache | RMS=nan, mad=0.000488, mrd=8.6e-04" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 8 -seq_len_q 1 -seq_len_k 502 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 18 -head_dim_v 55 -with-attn-scale=False -with-attn-bias=False -transQ=True -transK=False -transV=False -transO=False -causal=False -return_lse=True -split_kv=4 --perf_config=attn:v1:32,256,128,2,1,4,32,1,3,0,2 --current_seq_len=252,241,251,442,447,279,72,141 -pv -RMS_threshold 1e-2

run_case "[2/3] dt=i8 causal rls split_kv=32 | RMS=nan, mad=0.000000, mrd=0.0e+00" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 7 -seq_len_q 80 -seq_len_k 26 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 153 -head_dim_v 177 -with-attn-scale=True -with-attn-bias=True -transQ=False -transK=False -transV=False -transO=False -causal=True -return_lse=True -split_kv=32 --perf_config=attn:v1:16,128,128,2,1,16,16,1,3,4,4 -pv

run_case "[3/3] dt=i8 rls split_kv=16 | RMS=nan, mad=0.000000, mrd=0.0e+00" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 4 -seq_len_q 459 -seq_len_k 213 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 54 -head_dim_v 96 -with-attn-scale=False -with-attn-bias=False -transQ=True -transK=False -transV=True -transO=False -causal=False -return_lse=True -split_kv=16 --perf_config=attn:v1:16,256,16,1,1,4,32,1,1,0,0 -pv
