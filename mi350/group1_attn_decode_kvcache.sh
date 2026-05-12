#!/usr/bin/env bash
# group1_attn_decode_kvcache.sh -- Attention KV-cache decode (seq_len_q=1, --current_seq_len) -- basic path (18 cases)
#
# Decode-time attention with KV-cache truncation. split_kv=1 or basic non-split paths.
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
#   ./mi350/group1_attn_decode_kvcache.sh
# or with a non-default build directory:
#   BUILD_DIR=/path/to/build ./mi350/group1_attn_decode_kvcache.sh

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

run_case "[1/18] dt=i8 split_kv=1 decode kv-cache | RMS=2.6e-02, mad=0.034180, mrd=6.4e-02" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 3 -seq_len_q 1 -seq_len_k 552 -num_heads_q 8 -num_heads_kv 2 -head_dim_qk 38 -head_dim_v 25 -with-attn-scale=True -with-attn-bias=False -transQ=True -transK=True -transV=True -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:32,128,32,2,1,8,16,1,1,1,0 --current_seq_len=502,341,256 -pv

run_case "[2/18] dt=f16 rls split_kv=1 decode kv-cache | RMS=2.2e-02, mad=0.028320, mrd=4.9e-02" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 4 -seq_len_q 1 -seq_len_k 798 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 123 -head_dim_v 166 -with-attn-scale=True -with-attn-bias=False -transQ=True -transK=True -transV=False -transO=False -causal=False -return_lse=True -split_kv=1 --perf_config=attn:v1:64,64,16,2,1,4,16,1,3,8,1 --current_seq_len=493,675,319,17 -pv

run_case "[3/18] dt=f32 split_kv=1 decode kv-cache | RMS=2.2e-01, mad=0.429596, mrd=6.4e+01" \
    -operation attention -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 3 -seq_len_q 1 -seq_len_k 366 -num_heads_q 128 -num_heads_kv 2 -head_dim_qk 61 -head_dim_v 226 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=True -transV=True -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:256,128,64,1,1,4,0,1,3,0,1 --current_seq_len=320,253,145 -pv -relDiff_threshold 1e-4

run_case "[4/18] dt=i8 split_kv=1 decode kv-cache | RMS=8.2e-01, mad=0.719116, mrd=9.6e-01" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 4 -seq_len_q 1 -seq_len_k 672 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 148 -head_dim_v 185 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=False -transV=True -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:64,16,32,2,1,8,0,1,3,4,2 --current_seq_len=665,478,217,24 -pv

run_case "[5/18] dt=f16 split_kv=1 decode kv-cache | RMS=5.4e-01, mad=0.898621, mrd=9.0e-01" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 6 -seq_len_q 1 -seq_len_k 374 -num_heads_q 64 -num_heads_kv 8 -head_dim_qk 46 -head_dim_v 110 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=False -transV=False -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:64,32,16,2,1,16,16,1,1,2,8 --current_seq_len=117,321,144,326,36,69 -pv

run_case "[6/18] dt=bf16 split_kv=1 decode kv-cache | RMS=4.5e-02, mad=0.028809, mrd=6.5e-01" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 5 -seq_len_q 1 -seq_len_k 46 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 229 -head_dim_v 175 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=True -transV=True -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:256,64,128,1,1,1,16,1,2,8,1 --current_seq_len=12,5,20,17,16 -pv -RMS_threshold 1e-2

run_case "[7/18] dt=f16 split_kv=1 decode kv-cache | RMS=2.7e-01, mad=0.515869, mrd=8.8e-01" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 5 -seq_len_q 1 -seq_len_k 515 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 164 -head_dim_v 197 -with-attn-scale=True -with-attn-bias=True -transQ=False -transK=True -transV=False -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:128,32,16,2,1,2,32,1,1,0,0 --current_seq_len=280,354,54,398,139 -pv

run_case "[8/18] dt=f32 split_kv=1 decode kv-cache | RMS=4.9e-01, mad=24.125000, mrd=5.1e-01" \
    -operation attention -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 6 -seq_len_q 1 -seq_len_k 650 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 207 -head_dim_v 139 -with-attn-scale=True -with-attn-bias=False -transQ=False -transK=False -transV=True -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:64,256,32,1,1,1,16,1,1,8,4 --current_seq_len=156,471,52,212,43,224 -pv -relDiff_threshold 1e-4

run_case "[9/18] dt=f16 split_kv=1 decode kv-cache | RMS=6.3e-02, mad=0.165527, mrd=2.2e-01" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 4 -seq_len_q 1 -seq_len_k 845 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 203 -head_dim_v 42 -with-attn-scale=True -with-attn-bias=True -transQ=True -transK=False -transV=True -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:256,32,16,2,1,4,0,1,2,1,0 --current_seq_len=677,122,32,373 -pv

run_case "[10/18] dt=bf16 split_kv=1 decode kv-cache | RMS=6.3e-01, mad=0.713867, mrd=9.5e-01" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 6 -seq_len_q 1 -seq_len_k 516 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 62 -head_dim_v 38 -with-attn-scale=True -with-attn-bias=True -transQ=True -transK=True -transV=True -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:64,16,16,2,1,1,16,1,1,2,8 --current_seq_len=131,280,61,94,438,454 -pv -RMS_threshold 1e-2

run_case "[11/18] dt=bf16 split_kv=1 decode kv-cache | RMS=6.2e-01, mad=0.075317, mrd=9.0e-01" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 5 -seq_len_q 1 -seq_len_k 203 -num_heads_q 64 -num_heads_kv 8 -head_dim_qk 195 -head_dim_v 193 -with-attn-scale=False -with-attn-bias=False -transQ=True -transK=True -transV=True -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:128,16,64,2,1,8,32,1,1,8,1 --current_seq_len=72,93,44,68,134 -pv -RMS_threshold 1e-2

run_case "[12/18] dt=bf16 split_kv=1 decode kv-cache | RMS=8.5e-01, mad=1.136719, mrd=1.3e+00" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 5 -seq_len_q 1 -seq_len_k 678 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 135 -head_dim_v 5 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=False -transV=False -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:64,256,16,2,1,2,0,1,1,1,2 --current_seq_len=251,627,37,571,46 -pv -RMS_threshold 1e-2

run_case "[13/18] dt=bf16 split_kv=1 decode kv-cache | RMS=6.5e-01, mad=0.895020, mrd=1.0e+00" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 4 -seq_len_q 1 -seq_len_k 426 -num_heads_q 32 -num_heads_kv 16 -head_dim_qk 174 -head_dim_v 139 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=False -transV=False -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:16,32,16,2,1,16,16,1,1,2,1 --current_seq_len=93,229,255,86 -pv -RMS_threshold 1e-2

run_case "[14/18] dt=f16 rls split_kv=1 decode kv-cache | RMS=6.3e-01, mad=0.918884, mrd=9.2e-01" \
    -operation attention -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 6 -seq_len_q 1 -seq_len_k 344 -num_heads_q 64 -num_heads_kv 2 -head_dim_qk 206 -head_dim_v 149 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=True -transV=True -transO=True -causal=False -return_lse=True -split_kv=1 --perf_config=attn:v1:32,16,32,2,1,8,32,1,3,2,1 --current_seq_len=143,244,58,195,150,96 -pv

run_case "[15/18] dt=bf16 split_kv=1 decode kv-cache | RMS=3.4e-01, mad=0.734375, mrd=1.4e+00" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -seq_len_q 1 -seq_len_k 1372 -num_heads_q 128 -num_heads_kv 8 -head_dim_qk 241 -head_dim_v 254 -with-attn-scale=True -with-attn-bias=False -transQ=True -transK=False -transV=False -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:64,256,16,2,1,2,0,1,1,1,2 --current_seq_len=600 -pv -RMS_threshold 1e-2

run_case "[16/18] dt=f32 split_kv=1 decode kv-cache | RMS=1.9e-01, mad=0.489868, mrd=6.8e-01" \
    -operation attention -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -seq_len_q 1 -seq_len_k 1589 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 232 -head_dim_v 201 -with-attn-scale=True -with-attn-bias=True -transQ=False -transK=True -transV=False -transO=True -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:256,32,16,1,1,1,16,1,1,4,8 --current_seq_len=805 -pv -relDiff_threshold 1e-4

run_case "[17/18] dt=i8 split_kv=1 decode kv-cache | RMS=6.2e-01, mad=0.964081, mrd=9.6e-01" \
    -operation attention -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -seq_len_q 1 -seq_len_k 948 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 245 -head_dim_v 98 -with-attn-scale=False -with-attn-bias=True -transQ=False -transK=False -transV=False -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:128,16,64,2,1,16,16,1,3,1,0 --current_seq_len=513,58 -pv

run_case "[18/18] dt=bf16 rls split_kv=1 decode kv-cache | RMS=1.2e-01, mad=0.096680, mrd=1.5e+00" \
    -operation attention -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -seq_len_q 1 -seq_len_k 2862 -num_heads_q 16 -num_heads_kv 2 -head_dim_qk 8 -head_dim_v 220 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=True -transV=True -transO=True -causal=False -return_lse=True -split_kv=1 --perf_config=attn:v1:256,128,128,2,1,2,16,1,2,1,2 --current_seq_len=2821 -pv -RMS_threshold 1e-2
