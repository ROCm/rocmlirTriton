#!/usr/bin/env bash

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

run_case "[1/6] dtype=i8 causal=True split_kv=1 | RMS=6.5e-02, maxAbsDiff=0.1602, maxRelDiff=0.16" \
    -operation attention -t i8 --arch gfx1100 --num_cu 70 --num_chiplets 1 -g 3 -seq_len_q 935 -seq_len_k 152 -num_heads_q 128 -num_heads_kv 8 -head_dim_qk 216 -head_dim_v 172 -with-attn-scale=True -with-attn-bias=False -transQ=True -transK=False -transV=False -transO=False -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:64,128,128,1,1,16,16,1,2,1,4 -pv

run_case "[2/6] dtype=f32 causal=True split_kv=1 | RMS=6.3e-08, maxAbsDiff=0, maxRelDiff=1" \
    -operation attention -t f32 --arch gfx1100 --num_cu 70 --num_chiplets 1 -g 7 -seq_len_q 397 -seq_len_k 410 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 129 -head_dim_v 146 -with-attn-scale=False -with-attn-bias=False -transQ=True -transK=True -transV=False -transO=False -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:128,16,128,2,1,8,32,1,1,1,4 -pv -relDiff_threshold 1e-4

run_case "[3/6] dtype=i8 causal=True split_kv=1 | RMS=2.9e-01, maxAbsDiff=0.5, maxRelDiff=1" \
    -operation attention -t i8 --arch gfx1100 --num_cu 70 --num_chiplets 1 -g 3 -seq_len_q 1216 -seq_len_k 162 -num_heads_q 32 -num_heads_kv 2 -head_dim_qk 217 -head_dim_v 128 -with-attn-scale=True -with-attn-bias=False -transQ=True -transK=True -transV=True -transO=False -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:16,256,64,2,1,8,16,1,1,2,2 -pv

run_case "[4/6] dtype=i8 causal=True split_kv=1 | RMS=6.9e-03, maxAbsDiff=0.0332, maxRelDiff=3.4" \
    -operation attention -t i8 --arch gfx1100 --num_cu 70 --num_chiplets 1 -g 6 -seq_len_q 567 -seq_len_k 408 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 131 -head_dim_v 91 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=True -transV=False -transO=True -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:64,128,32,1,1,16,32,1,1,4,4 -pv

run_case "[5/6] dtype=bf16 causal=True split_kv=1 return_lse | RMS=1.0e-02, maxAbsDiff=0.5, maxRelDiff=0.012" \
    -operation attention -t bf16 --arch gfx1100 --num_cu 70 --num_chiplets 1 -g 6 -seq_len_q 618 -seq_len_k 415 -num_heads_q 64 -num_heads_kv 16 -head_dim_qk 87 -head_dim_v 114 -with-attn-scale=True -with-attn-bias=True -transQ=False -transK=True -transV=True -transO=True -causal=True -return_lse=True -split_kv=1 --perf_config=attn:v1:32,64,32,1,1,8,16,1,2,1,1 -pv -RMS_threshold 1e-2

run_case "[6/6] dtype=f32 causal=True split_kv=1 | RMS=6.6e-08, maxAbsDiff=0, maxRelDiff=0.75" \
    -operation attention -t f32 --arch gfx1100 --num_cu 70 --num_chiplets 1 -g 5 -seq_len_q 406 -seq_len_k 641 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 186 -head_dim_v 34 -with-attn-scale=False -with-attn-bias=False -transQ=True -transK=True -transV=True -transO=True -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:64,16,32,1,1,8,0,1,2,4,0 -pv -relDiff_threshold 1e-4

