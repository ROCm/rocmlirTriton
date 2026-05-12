#!/usr/bin/env bash
# group3_reproducer.sh -- KV-cache decode (seq_len_q=1, --current_seq_len, split_kv>=1) (2 cases)
#
# Original log (2026-05-07) flagged 8 cases. After commit eec615b55cd1
# ("Fix SplitKV CPU lowering path bug", 2026-05-08) corrected the CPU
# verifier's split-KV reference path, 6 of those 8 cases now PASS
# stably on the current build (verified 10/10 in isolated re-runs).
# Those were false positives from a buggy CPU reference.
#
# Kept (still buggy on current build):
#   [1/2] (was 3/8): dtype=f16 split_kv=2 kv-cache return_lse
#         -> [0 1 1] (10/10 stable FAIL in isolated re-runs)
#   [2/2] (was 8/8): dtype=i8 split_kv=16 kv-cache return_lse
#         -> PASSes in single end-to-end runs but HANGS in repeated
#            re-runs (kernel timeout >60s after stress). Treated as
#            a real bug (intermittent kernel stall / GPU lockup).
#
# Removed (10/10 PASS stable on current build):
#   was 1/8: bf16 split_kv=8  kv-cache return_lse -> [1 1 1]
#   was 2/8: bf16 split_kv=32 kv-cache return_lse -> [1 1 1]
#   was 4/8: f16  split_kv=8  kv-cache return_lse -> [1 1 1]
#   was 5/8: f16  split_kv=1  kv-cache return_lse -> [1 1 1] [1 1 1]
#   was 6/8: f16  split_kv=4  kv-cache return_lse -> [1 1 1]
#   was 7/8: f16  split_kv=8  kv-cache return_lse -> [1 1 1]
#
# Pipeline (per parameterSweeps.py::test_config and README.md):
#   rocmlir-gen <args> | rocmlir-driver --host-pipeline=highlevel - |
#       rocmlir-driver -c | rocm-run
#
# Each kernel prints the verifier flag triple `[RMS_pass absDiff_pass
# relDiff_pass]` on the last line of output. `[1 1 1]` is a clean
# PASS; any zero (or missing output) is a FAIL. A hang/timeout is
# also a FAIL.
#
# Usage (from the rocmlirTriton repo root):
#   ./navi3/group3_reproducer.sh
# or with a non-default build directory:
#   BUILD_DIR=/path/to/build ./navi3/group3_reproducer.sh

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

run_case "[1/2] dtype=f16 causal=False split_kv=2 kv-cache return_lse | observed -> [0 1 1] (stable)" \
    -operation attention -t f16 --arch gfx1100 --num_cu 70 --num_chiplets 1 -g 6 -seq_len_q 1 -seq_len_k 622 -num_heads_q 8 -num_heads_kv 4 -head_dim_qk 105 -head_dim_v 52 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=True -transV=True -transO=False -causal=False -return_lse=True -split_kv=2 --perf_config=attn:v1:256,128,32,1,1,4,16,1,2,2,4 --current_seq_len=535,118,433,540,149,409 -pv

run_case "[2/2] dtype=i8 causal=False split_kv=16 kv-cache return_lse | observed -> intermittent hang/timeout (>60s)" \
    -operation attention -t i8 --arch gfx1100 --num_cu 70 --num_chiplets 1 -g 2 -seq_len_q 1 -seq_len_k 1079 -num_heads_q 64 -num_heads_kv 2 -head_dim_qk 156 -head_dim_v 227 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=True -transV=False -transO=True -causal=False -return_lse=True -split_kv=16 --perf_config=attn:v1:256,32,16,2,1,1,0,1,2,1,0 --current_seq_len=395,674 -pv
