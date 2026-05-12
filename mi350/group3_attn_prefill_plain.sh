#!/usr/bin/env bash
# group3_attn_prefill_plain.sh -- Attention prefill (no mask, no KV-cache) -- baseline kernel (11 cases)
#
# Simplest attention path; narrow reproducer for fundamental kernel correctness.
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
#   ./mi350/group3_attn_prefill_plain.sh
# or with a non-default build directory:
#   BUILD_DIR=/path/to/build ./mi350/group3_attn_prefill_plain.sh

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

run_case "[1/11] dt=f32 split_kv=1 | RMS=6.1e-01, mad=0.654480, mrd=1.4e+01" \
    -operation attention -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 7 -seq_len_q 348 -seq_len_k 122 -num_heads_q 16 -num_heads_kv 4 -head_dim_qk 251 -head_dim_v 4 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=False -transV=True -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:128,128,64,2,1,16,0,1,1,4,8 -pv -relDiff_threshold 1e-4

run_case "[2/11] dt=f16 split_kv=1 | RMS=1.4e-01, mad=0.299805, mrd=3.0e-01" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 8 -seq_len_q 394 -seq_len_k 182 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 136 -head_dim_v 9 -with-attn-scale=True -with-attn-bias=True -transQ=True -transK=True -transV=True -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:128,64,64,2,1,1,0,1,3,0,8 -pv

run_case "[3/11] dt=f16 split_kv=1 | RMS=1.0e-01, mad=0.238281, mrd=2.4e-01" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 6 -seq_len_q 149 -seq_len_k 125 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 187 -head_dim_v 212 -with-attn-scale=True -with-attn-bias=True -transQ=True -transK=False -transV=True -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:16,32,32,2,1,16,32,1,2,2,2 -pv

run_case "[4/11] dt=f16 split_kv=1 | RMS=8.4e-01, mad=0.414490, mrd=8.4e-01" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 8 -seq_len_q 87 -seq_len_k 267 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 129 -head_dim_v 91 -with-attn-scale=False -with-attn-bias=True -transQ=False -transK=True -transV=True -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:16,32,16,2,1,8,16,1,3,0,2 -pv

run_case "[5/11] dt=i8 rls split_kv=1 | RMS=9.7e-01, mad=0.728958, mrd=9.7e-01" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -seq_len_q 154 -seq_len_k 657 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 251 -head_dim_v 122 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=False -transV=True -transO=False -causal=False -return_lse=True -split_kv=1 --perf_config=attn:v1:32,16,32,2,1,8,32,1,3,8,0 -pv

run_case "[6/11] dt=bf16 rls split_kv=1 | RMS=5.7e-01, mad=0.764648, mrd=7.7e-01" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 6 -seq_len_q 225 -seq_len_k 253 -num_heads_q 64 -num_heads_kv 2 -head_dim_qk 254 -head_dim_v 83 -with-attn-scale=True -with-attn-bias=False -transQ=False -transK=True -transV=False -transO=False -causal=False -return_lse=True -split_kv=1 --perf_config=attn:v1:16,32,128,2,1,1,0,1,2,2,4 -pv -RMS_threshold 1e-2

run_case "[7/11] dt=bf16 split_kv=1 | RMS=7.5e-01, mad=0.964844, mrd=9.7e-01" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 3 -seq_len_q 52 -seq_len_k 697 -num_heads_q 64 -num_heads_kv 8 -head_dim_qk 255 -head_dim_v 84 -with-attn-scale=True -with-attn-bias=False -transQ=True -transK=False -transV=False -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:64,16,64,2,1,16,0,1,2,1,8 -pv -RMS_threshold 1e-2

run_case "[8/11] dt=f32 split_kv=1 | RMS=7.3e-02, mad=0.077637, mrd=8.6e-01" \
    -operation attention -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 5 -seq_len_q 21 -seq_len_k 784 -num_heads_q 128 -num_heads_kv 8 -head_dim_qk 94 -head_dim_v 206 -with-attn-scale=True -with-attn-bias=False -transQ=True -transK=False -transV=False -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:32,32,16,1,1,16,32,1,2,2,8 -pv -relDiff_threshold 1e-4

run_case "[9/11] dt=i8 split_kv=1 | RMS=7.3e-01, mad=0.942688, mrd=9.4e-01" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 3 -seq_len_q 89 -seq_len_k 1019 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 150 -head_dim_v 209 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=False -transV=True -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:32,32,32,2,1,1,16,1,3,1,1 -pv

run_case "[10/11] dt=f16 split_kv=1 | RMS=1.2e-02, mad=0.035156, mrd=3.5e-02" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 8 -seq_len_q 289 -seq_len_k 8 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 190 -head_dim_v 213 -with-attn-scale=True -with-attn-bias=False -transQ=True -transK=False -transV=True -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:32,128,64,1,1,2,0,1,3,8,2 -pv

run_case "[11/11] dt=bf16 split_kv=1 | RMS=4.5e-01, mad=0.806274, mrd=8.1e-01" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -seq_len_q 3 -seq_len_k 3977 -num_heads_q 32 -num_heads_kv 4 -head_dim_qk 26 -head_dim_v 215 -with-attn-scale=False -with-attn-bias=True -transQ=False -transK=True -transV=True -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:32,128,32,2,1,16,32,1,2,4,4 -pv -RMS_threshold 1e-2
