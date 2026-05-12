#!/usr/bin/env bash
# group7_conv.sh -- Conv numerical bug remaining after MI350 sweep + PR fixes (1 case)
#
# Originally generated with 45 candidate cases from mi350/conv_errors.log
# (sweep 2026-05-07). Re-run on MI350 reduced this to one persistent
# numerical bug; the rest are either now-PASS or already-known.
#
# Status from MI350 re-run:
#   PASS                                 : 33/45  (removed -- presumably fixed by recent PRs)
#   FAIL [0 0 0]                         :  1/45  KEPT below
#   getUpperBounds() SIGSEGV (already fil. as known bug):
#                                          8/45  removed (was 4, 7, 8, 11, 21, 27, 30, 33)
#   "Fusion with SplitK perfConfig is not legal" (structural NOT_APPLICABLE; not a real bug):
#                                          3/45  removed (was 3, 40, 44)
#
# Pipeline (per parameterSweeps.py::test_config and README.md):
#   rocmlir-gen <args> | rocmlir-driver -c | rocm-run
#
# Each kernel prints the verifier flag triple `[RMS_pass absDiff_pass
# relDiff_pass]` on the last line of output. `[1 1 1]` is a clean
# PASS; any zero (or missing output) is a FAIL.
#
# Hardware: gfx950 (MI350); the harness was sampled with
# --num_cu 256 --num_chiplets 8.
#
# Usage (from the rocmlirTriton repo root):
#   ./mi350/group7_conv.sh
# or with a non-default build directory:
#   BUILD_DIR=/path/to/build ./mi350/group7_conv.sh

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
        | "$DRV" -c \
        | "$RUN"
    printf "\n"
}

run_case "[1/1] conv dt=i8 (was 24/45) | observed -> [0 0 0]  RMS=5.2e-01, mad=3.65, mrd=1.6e+01" \
    --operation conv -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 8 --in_channels 128 --in_h 224 --in_w 64 --out_channels 256 --fil_h 3 --fil_w 2 --dilation_h 1 --dilation_w 2 --conv_stride_h 1 --conv_stride_w 2 --padding_h 1 --padding_w_l 3 --padding_w_r 0 --groupsize 1 --perf_config=gemm:v1:128,256,128,2,1,8,16,1,2,2,1 -pv
