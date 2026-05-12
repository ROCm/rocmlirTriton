#!/usr/bin/env bash
# group9_gemm_gemm.sh -- GEMM+GEMM remaining bugs after MI350 sweep + PR fixes (3 cases)
#
# Originally generated with 45 candidate cases from mi350/gemm_gemm_errors.log
# (sweep 2026-05-07). Re-run on MI350 reduced this to 3 persistent bugs;
# the rest are either now-PASS or structurally not-applicable.
#
# Status from MI350 re-run:
#   PASS                          : 40/45  (removed -- presumably fixed by recent PRs)
#   FAIL [0 1 1]                  :  1/45  KEPT  (was 2/45)
#   LLVM ERROR: out of memory     :  1/45  KEPT  (was 12/45) -- compiler OOM
#   std::bad_array_new_length crash: 1/45  KEPT  (was 14/45) -- compiler crash
#   "ttg.shared exceeds LDS limit" (structural NOT_APPLICABLE; not a real bug):
#                                    2/45  removed (was 16, 44)
#
# Pipeline (per parameterSweeps.py::test_config and README.md):
#   rocmlir-gen <args> | rocmlir-driver --host-pipeline=highlevel - |
#       rocmlir-driver -c | rocm-run
#
# Each kernel prints the verifier flag triple `[RMS_pass absDiff_pass
# relDiff_pass]` on the last line of output. `[1 1 1]` is a clean
# PASS; any zero (or missing output, crash, OOM, hang, timeout) is a FAIL.
#
# Hardware: gfx950 (MI350); the harness was sampled with
# --num_cu 256 --num_chiplets 8.
#
# Usage (from the rocmlirTriton repo root):
#   ./mi350/group9_gemm_gemm.sh
# or with a non-default build directory:
#   BUILD_DIR=/path/to/build ./mi350/group9_gemm_gemm.sh

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

run_case "[1/3] dt=f16 m9 k70 n3 gO96 trans[tB] (was 2/45) | observed -> [0 1 1]  RMS=3.2e-03, mad=0.0156, mrd=1.8e-02" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 9 -k 70 -n 3 -gemmO 96 -transA=False -transB=True -transC=False -transO=False --perf_config=attn:v1:64,64,128,2,1,2,16,4,2,8,2 -pv

run_case "[2/3] dt=bf16 m61 k39 n92 gO5 trans[tA,tB,tC] (was 12/45) | observed -> LLVM ERROR: out of memory (compiler OOM)" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 61 -k 39 -n 92 -gemmO 5 -transA=True -transB=True -transC=True -transO=False --perf_config=attn:v1:128,256,16,1,1,1,32,1,3,4,8 -pv -RMS_threshold 1e-2

run_case "[3/3] dt=bf16 m12 k79 n124 gO7 trans[tC,tO] (was 14/45) | observed -> std::bad_array_new_length crash (compiler)" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 12 -k 79 -n 124 -gemmO 7 -transA=False -transB=False -transC=True -transO=True --perf_config=attn:v1:32,64,32,2,1,4,32,1,2,0,8 -pv -RMS_threshold 1e-2
