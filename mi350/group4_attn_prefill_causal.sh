#!/usr/bin/env bash
# group4_attn_prefill_causal.sh -- Attention prefill + causal=True (causal-mask path) (9 cases)
#
# Causal-masked prefill. Failures here implicate the causal-mask + softmax merge.
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
#   ./mi350/group4_attn_prefill_causal.sh
# or with a non-default build directory:
#   BUILD_DIR=/path/to/build ./mi350/group4_attn_prefill_causal.sh

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

run_case "[1/9] dt=f16 causal rls split_kv=2 | RMS=1.2e-01, mad=0.298828, mrd=5.6e-01" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 4 -seq_len_q 34 -seq_len_k 343 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 226 -head_dim_v 6 -with-attn-scale=True -with-attn-bias=True -transQ=False -transK=True -transV=True -transO=False -causal=True -return_lse=True -split_kv=2 --perf_config=attn:v1:16,64,32,2,1,16,0,1,2,8,2 -pv

run_case "[2/9] dt=i8 causal split_kv=1 | RMS=6.5e-02, mad=0.160156, mrd=1.6e-01" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 3 -seq_len_q 935 -seq_len_k 152 -num_heads_q 128 -num_heads_kv 8 -head_dim_qk 216 -head_dim_v 172 -with-attn-scale=True -with-attn-bias=False -transQ=True -transK=False -transV=False -transO=False -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:64,128,128,1,1,16,16,1,2,1,4 -pv

run_case "[3/9] dt=bf16 causal split_kv=1 | RMS=2.0e-01, mad=0.444824, mrd=1.0e+02" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 5 -seq_len_q 402 -seq_len_k 449 -num_heads_q 32 -num_heads_kv 2 -head_dim_qk 173 -head_dim_v 73 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=False -transV=True -transO=True -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:16,128,16,2,1,8,32,1,2,2,4 -pv -RMS_threshold 1e-2

run_case "[4/9] dt=i8 causal split_kv=1 | RMS=7.7e-01, mad=0.999998, mrd=1.0e+00" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 7 -seq_len_q 415 -seq_len_k 581 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 149 -head_dim_v 207 -with-attn-scale=True -with-attn-bias=True -transQ=True -transK=False -transV=True -transO=True -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:256,16,32,2,1,8,0,1,1,4,1 -pv

run_case "[5/9] dt=i8 causal split_kv=1 | RMS=6.5e-01, mad=0.966492, mrd=9.7e-01" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 8 -seq_len_q 116 -seq_len_k 458 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 157 -head_dim_v 23 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=True -transV=False -transO=True -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:64,16,16,2,1,1,16,1,3,4,2 -pv

run_case "[6/9] dt=bf16 causal split_kv=1 | RMS=4.6e-01, mad=0.772461, mrd=7.7e-01" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -seq_len_q 201 -seq_len_k 1263 -num_heads_q 8 -num_heads_kv 4 -head_dim_qk 180 -head_dim_v 113 -with-attn-scale=True -with-attn-bias=False -transQ=False -transK=False -transV=False -transO=True -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:16,32,16,2,1,8,32,1,1,2,4 -pv -RMS_threshold 1e-2

run_case "[7/9] dt=bf16 causal rls split_kv=4 | RMS=2.4e-01, mad=0.509766, mrd=1.5e+00" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -seq_len_q 30 -seq_len_k 3481 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 205 -head_dim_v 69 -with-attn-scale=True -with-attn-bias=False -transQ=False -transK=True -transV=True -transO=False -causal=True -return_lse=True -split_kv=4 --perf_config=attn:v1:16,32,16,2,1,2,16,1,1,2,0 -pv -RMS_threshold 1e-2

run_case "[8/9] dt=i8 causal split_kv=1 | RMS=5.6e-01, mad=0.588013, mrd=7.8e-01" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 7 -seq_len_q 59 -seq_len_k 192 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 205 -head_dim_v 78 -with-attn-scale=False -with-attn-bias=True -transQ=False -transK=False -transV=True -transO=False -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:32,16,128,2,1,16,0,1,3,2,4 -pv

run_case "[9/9] dt=f16 causal split_kv=1 | RMS=5.3e-01, mad=0.977264, mrd=9.8e-01" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -seq_len_q 17 -seq_len_k 1900 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 243 -head_dim_v 175 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=False -transV=False -transO=True -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:16,256,128,1,1,1,16,1,1,2,8 -pv
