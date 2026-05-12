#!/usr/bin/env bash
# group5_attn_prefill_split_lse.sh -- Attention prefill + return_lse + split_kv>1 (7 cases)
#
# Multi-split prefill with online-softmax merge across K splits (no KV-cache).
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
#   ./mi350/group5_attn_prefill_split_lse.sh
# or with a non-default build directory:
#   BUILD_DIR=/path/to/build ./mi350/group5_attn_prefill_split_lse.sh

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

run_case "[1/7] dt=f16 rls split_kv=8 decode | RMS=8.2e-02, mad=0.064453, mrd=9.0e-02" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 7 -seq_len_q 1 -seq_len_k 117 -num_heads_q 64 -num_heads_kv 4 -head_dim_qk 91 -head_dim_v 61 -with-attn-scale=False -with-attn-bias=True -transQ=False -transK=False -transV=True -transO=True -causal=False -return_lse=True -split_kv=8 --perf_config=attn:v1:64,32,16,2,1,8,32,1,1,0,1 -pv

run_case "[2/7] dt=bf16 rls split_kv=4 | RMS=4.1e-01, mad=0.644531, mrd=7.9e-01" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 5 -seq_len_q 708 -seq_len_k 185 -num_heads_q 8 -num_heads_kv 4 -head_dim_qk 218 -head_dim_v 152 -with-attn-scale=True -with-attn-bias=True -transQ=False -transK=True -transV=False -transO=False -causal=False -return_lse=True -split_kv=4 --perf_config=attn:v1:16,16,16,2,1,2,0,1,3,0,0 -pv -RMS_threshold 1e-2

run_case "[3/7] dt=i8 rls split_kv=4 | RMS=3.7e-03, mad=0.013184, mrd=2.3e-02" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 7 -seq_len_q 213 -seq_len_k 524 -num_heads_q 64 -num_heads_kv 8 -head_dim_qk 205 -head_dim_v 97 -with-attn-scale=True -with-attn-bias=True -transQ=True -transK=False -transV=True -transO=False -causal=False -return_lse=True -split_kv=4 --perf_config=attn:v1:256,256,64,2,1,4,16,1,1,8,2 -pv

run_case "[4/7] dt=bf16 rls split_kv=4 | RMS=5.8e-01, mad=0.415771, mrd=9.3e-01" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -seq_len_q 68 -seq_len_k 1798 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 81 -head_dim_v 206 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=True -transV=False -transO=False -causal=False -return_lse=True -split_kv=4 --perf_config=attn:v1:16,32,32,2,1,8,16,1,2,8,0 -pv -RMS_threshold 1e-2

run_case "[5/7] dt=f32 rls split_kv=2 | RMS=2.9e-01, mad=0.596436, mrd=1.2e+00" \
    -operation attention -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 7 -seq_len_q 78 -seq_len_k 290 -num_heads_q 128 -num_heads_kv 16 -head_dim_qk 190 -head_dim_v 122 -with-attn-scale=True -with-attn-bias=True -transQ=False -transK=False -transV=True -transO=False -causal=False -return_lse=True -split_kv=2 --perf_config=attn:v1:256,64,64,1,1,16,16,1,3,4,2 -pv -relDiff_threshold 1e-4

run_case "[6/7] dt=f32 rls split_kv=32 | RMS=6.9e-03, mad=0.033203, mrd=3.4e+00" \
    -operation attention -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -seq_len_q 136 -seq_len_k 231 -num_heads_q 8 -num_heads_kv 2 -head_dim_qk 115 -head_dim_v 18 -with-attn-scale=True -with-attn-bias=True -transQ=True -transK=False -transV=False -transO=False -causal=False -return_lse=True -split_kv=32 --perf_config=attn:v1:256,256,128,1,1,8,32,1,2,0,4 -pv -relDiff_threshold 1e-4

run_case "[7/7] dt=bf16 rls split_kv=4 | RMS=1.2e-01, mad=0.105469, mrd=1.4e-01" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 8 -seq_len_q 281 -seq_len_k 6 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 167 -head_dim_v 199 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=False -transV=True -transO=True -causal=False -return_lse=True -split_kv=4 --perf_config=attn:v1:256,64,16,2,1,4,0,1,3,4,1 -pv -RMS_threshold 1e-2
