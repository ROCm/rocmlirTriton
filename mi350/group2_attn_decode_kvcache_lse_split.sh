#!/usr/bin/env bash
# group2_attn_decode_kvcache_lse_split.sh -- Attention KV-cache decode + return_lse + split_kv>1 (online-softmax merge) (12 cases)
#
# Multi-split decode that merges per-split partials via the online-softmax + LSE protocol.
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
#   ./mi350/group2_attn_decode_kvcache_lse_split.sh
# or with a non-default build directory:
#   BUILD_DIR=/path/to/build ./mi350/group2_attn_decode_kvcache_lse_split.sh

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

run_case "[1/12] dt=i8 rls split_kv=4 decode kv-cache | RMS=3.6e-01, mad=0.602051, mrd=6.0e-01" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 7 -seq_len_q 1 -seq_len_k 311 -num_heads_q 64 -num_heads_kv 32 -head_dim_qk 117 -head_dim_v 195 -with-attn-scale=True -with-attn-bias=False -transQ=False -transK=True -transV=True -transO=False -causal=False -return_lse=True -split_kv=4 --perf_config=attn:v1:16,16,16,2,1,16,0,1,1,2,1 --current_seq_len=265,131,48,85,278,131,195 -pv

run_case "[2/12] dt=f16 rls split_kv=2 decode kv-cache | RMS=1.2e-02, mad=0.028809, mrd=3.8e-02" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -seq_len_q 1 -seq_len_k 851 -num_heads_q 128 -num_heads_kv 16 -head_dim_qk 175 -head_dim_v 211 -with-attn-scale=True -with-attn-bias=True -transQ=False -transK=False -transV=True -transO=False -causal=False -return_lse=True -split_kv=2 --perf_config=attn:v1:64,64,32,2,1,16,16,1,1,0,1 --current_seq_len=297 -pv

run_case "[3/12] dt=f16 rls split_kv=8 decode kv-cache | RMS=1.6e-01, mad=0.503052, mrd=2.6e+00" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 7 -seq_len_q 1 -seq_len_k 151 -num_heads_q 64 -num_heads_kv 2 -head_dim_qk 154 -head_dim_v 34 -with-attn-scale=True -with-attn-bias=False -transQ=False -transK=True -transV=False -transO=False -causal=False -return_lse=True -split_kv=8 --perf_config=attn:v1:32,16,32,2,1,4,0,1,3,2,8 --current_seq_len=13,102,30,68,85,139,8 -pv

run_case "[4/12] dt=i8 rls split_kv=8 decode kv-cache | RMS=3.2e-02, mad=0.039062, mrd=7.8e-02" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 7 -seq_len_q 1 -seq_len_k 405 -num_heads_q 128 -num_heads_kv 8 -head_dim_qk 97 -head_dim_v 144 -with-attn-scale=True -with-attn-bias=False -transQ=True -transK=True -transV=True -transO=True -causal=False -return_lse=True -split_kv=8 --perf_config=attn:v1:128,64,16,2,1,4,0,1,1,1,2 --current_seq_len=359,378,290,348,229,130,233 -pv

run_case "[5/12] dt=i8 rls split_kv=8 decode kv-cache | RMS=4.9e-02, mad=0.126465, mrd=1.3e-01" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 4 -seq_len_q 1 -seq_len_k 55 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 195 -head_dim_v 39 -with-attn-scale=True -with-attn-bias=False -transQ=False -transK=False -transV=True -transO=False -causal=False -return_lse=True -split_kv=8 --perf_config=attn:v1:256,256,64,1,1,1,32,1,1,2,1 --current_seq_len=48,12,48,12 -pv

run_case "[6/12] dt=f16 rls split_kv=4 decode kv-cache | RMS=1.7e-01, mad=0.437500, mrd=4.4e-01" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 8 -seq_len_q 1 -seq_len_k 169 -num_heads_q 8 -num_heads_kv 4 -head_dim_qk 189 -head_dim_v 200 -with-attn-scale=True -with-attn-bias=True -transQ=True -transK=False -transV=True -transO=False -causal=False -return_lse=True -split_kv=4 --perf_config=attn:v1:32,16,64,2,1,16,0,1,1,4,0 --current_seq_len=13,159,45,40,78,76,65,74 -pv

run_case "[7/12] dt=bf16 rls split_kv=32 decode kv-cache | RMS=2.3e-01, mad=0.410156, mrd=5.2e-01" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -seq_len_q 1 -seq_len_k 595 -num_heads_q 16 -num_heads_kv 2 -head_dim_qk 253 -head_dim_v 62 -with-attn-scale=True -with-attn-bias=True -transQ=True -transK=False -transV=False -transO=False -causal=False -return_lse=True -split_kv=32 --perf_config=attn:v1:64,16,32,2,1,16,16,1,2,1,2 --current_seq_len=539,526 -pv -RMS_threshold 1e-2

run_case "[8/12] dt=f16 rls split_kv=4 decode kv-cache | RMS=3.3e-01, mad=0.641113, mrd=7.2e-01" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 7 -seq_len_q 1 -seq_len_k 419 -num_heads_q 8 -num_heads_kv 2 -head_dim_qk 239 -head_dim_v 16 -with-attn-scale=True -with-attn-bias=True -transQ=True -transK=True -transV=True -transO=True -causal=False -return_lse=True -split_kv=4 --perf_config=attn:v1:128,16,32,2,1,2,16,1,3,2,8 --current_seq_len=221,163,262,167,60,113,24 -pv

run_case "[9/12] dt=f16 rls split_kv=2 decode kv-cache | RMS=4.3e-03, mad=0.013672, mrd=3.4e-02" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 6 -seq_len_q 1 -seq_len_k 349 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 235 -head_dim_v 43 -with-attn-scale=True -with-attn-bias=False -transQ=True -transK=False -transV=False -transO=True -causal=False -return_lse=True -split_kv=2 --perf_config=attn:v1:64,16,16,1,1,16,32,1,2,8,1 --current_seq_len=36,121,169,303,287,183 -pv

run_case "[10/12] dt=i8 rls split_kv=2 decode kv-cache | RMS=6.6e-02, mad=0.122070, mrd=2.1e-01" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -seq_len_q 1 -seq_len_k 1216 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 65 -head_dim_v 134 -with-attn-scale=True -with-attn-bias=False -transQ=False -transK=False -transV=True -transO=False -causal=False -return_lse=True -split_kv=2 --perf_config=attn:v1:32,256,32,2,1,1,0,1,1,0,2 --current_seq_len=268,22 -pv

run_case "[11/12] dt=f16 rls split_kv=8 decode kv-cache | RMS=7.6e-01, mad=0.741028, mrd=1.0e+00" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -seq_len_q 1 -seq_len_k 510 -num_heads_q 32 -num_heads_kv 8 -head_dim_qk 22 -head_dim_v 211 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=False -transV=False -transO=True -causal=False -return_lse=True -split_kv=8 --perf_config=attn:v1:64,256,32,2,1,1,16,1,1,2,0 --current_seq_len=289 -pv

run_case "[12/12] dt=f32 rls split_kv=4 decode kv-cache | RMS=2.4e-01, mad=0.206665, mrd=8.4e-01" \
    -operation attention -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 7 -seq_len_q 1 -seq_len_k 53 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 208 -head_dim_v 24 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=True -transV=False -transO=True -causal=False -return_lse=True -split_kv=4 --perf_config=attn:v1:128,128,128,2,1,16,32,1,2,0,0 --current_seq_len=19,48,32,21,32,39,14 -pv -relDiff_threshold 1e-4
