#!/usr/bin/env bash
# group8_gemm.sh -- GEMM numerical bugs (51 cases)
#
# rocmlir-gen -operation gemm failures with wrong numerical results (real bugs, not threshold misses).
#
# Logs source: mi350/gemm_errors.log (sweep run 2026-05-07).
# Filtering rule: excluded purely threshold-only misses (maxAbsDiff < 1e-2
# and RMS < 5e-2 and no NaN) and the already-reported
# rock::TransformMapAttr::getUpperBounds() SIGSEGV in conv (13 cases).
#
# Pipeline (per parameterSweeps.py::test_config and README.md):
#   rocmlir-gen <args> | rocmlir-driver -c | rocm-run
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
#   ./mi350/group8_gemm.sh
# or with a non-default build directory:
#   BUILD_DIR=/path/to/build ./mi350/group8_gemm.sh

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

run_case "[1/51] gemm dt=bf16 | RMS=4.8e-01, mad=159.000000, mrd=6.7e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 106 -k 410 -n 428 -transA=False -transB=False --perf_config=gemm:v1:32,128,32,2,1,4,16,4,3,0,8 -pv -RMS_threshold 1e-2

run_case "[2/51] gemm dt=f16 | RMS=2.8e-01, mad=19.531250, mrd=4.1e-01" \
    -operation gemm -t f16 -out_datatype f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 451 -k 80 -n 457 -transA=True -transB=True --perf_config=gemm:v1:256,256,32,2,1,1,16,2,3,8,1 -pv

run_case "[3/51] gemm dt=bf16 | RMS=6.0e-01, mad=186.750000, mrd=8.6e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 399 -k 427 -n 277 -transA=False -transB=True --perf_config=gemm:v1:256,256,64,1,1,1,0,1,3,8,4 -pv -RMS_threshold 1e-2

run_case "[4/51] gemm dt=fp8 | RMS=2.0e-01, mad=14.500000, mrd=2.9e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 325 -k 83 -n 456 -transA=False -transB=True --perf_config=gemm:v1:32,32,16,2,1,2,0,3,2,0,0 -pv -relDiff_threshold 1e-5

run_case "[5/51] gemm dt=f16 | RMS=6.3e-01, mad=21.613281, mrd=8.5e-01" \
    -operation gemm -t f16 -out_datatype f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 174 -k 310 -n 159 -transA=False -transB=False --perf_config=gemm:v1:128,128,16,2,1,16,16,2,2,1,0 -pv

run_case "[6/51] gemm dt=fp8 | RMS=6.3e-01, mad=83.919800, mrd=8.9e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 379 -k 165 -n 191 -transA=False -transB=False --perf_config=gemm:v1:64,16,16,2,1,4,32,1,2,8,1 -pv -relDiff_threshold 1e-5

run_case "[7/51] gemm dt=bf16 | RMS=2.0e-01, mad=39.000000, mrd=3.0e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 502 -k 218 -n 255 -transA=False -transB=True --perf_config=gemm:v1:32,16,32,2,1,16,16,4,2,8,1 -pv -RMS_threshold 1e-2

run_case "[8/51] gemm dt=fp8 | RMS=6.4e-01, mad=31.398438, mrd=8.3e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 342 -k 456 -n 348 -transA=True -transB=True --perf_config=gemm:v1:32,64,32,2,1,1,16,2,1,8,2 -pv -relDiff_threshold 1e-5

run_case "[9/51] gemm dt=f16 | RMS=5.6e-01, mad=175.125000, mrd=8.1e-01" \
    -operation gemm -t f16 -out_datatype f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 436 -k 358 -n 23 -transA=False -transB=True --perf_config=gemm:v1:32,256,16,2,1,8,32,3,1,4,0 -pv

run_case "[10/51] gemm dt=f16 | RMS=6.7e-01, mad=290.273438, mrd=9.6e-01" \
    -operation gemm -t f16 -out_datatype f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 216 -k 503 -n 240 -transA=False -transB=True --perf_config=gemm:v1:256,256,16,2,1,16,16,1,2,0,2 -pv

run_case "[11/51] gemm dt=bf16 | RMS=6.6e-01, mad=149.062500, mrd=9.2e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 256 -k 282 -n 431 -transA=True -transB=False --perf_config=gemm:v1:128,64,16,2,1,2,0,1,2,2,1 -pv -RMS_threshold 1e-2

run_case "[12/51] gemm dt=bf16 | RMS=3.6e-01, mad=71.500000, mrd=5.2e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 85 -k 234 -n 398 -transA=True -transB=True --perf_config=gemm:v1:32,16,32,2,1,2,16,3,1,2,1 -pv -RMS_threshold 1e-2

run_case "[13/51] gemm dt=fp8 | RMS=6.5e-01, mad=27.358276, mrd=8.6e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 173 -k 381 -n 183 -transA=True -transB=False --perf_config=gemm:v1:128,64,16,2,1,16,32,2,1,2,0 -pv -relDiff_threshold 1e-5

run_case "[14/51] gemm dt=f16 | RMS=7.2e-01, mad=26.071289, mrd=9.6e-01" \
    -operation gemm -t f16 -out_datatype f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 249 -k 322 -n 494 -transA=True -transB=False --perf_config=gemm:v1:64,32,16,2,1,4,0,1,2,1,2 -pv

run_case "[15/51] gemm dt=bf16 | RMS=7.3e-01, mad=26.070312, mrd=9.7e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 491 -k 324 -n 21 -transA=True -transB=False --perf_config=gemm:v1:16,16,16,2,1,2,32,1,2,1,2 -pv -RMS_threshold 1e-2

run_case "[16/51] gemm dt=bf16 | RMS=5.1e-01, mad=85.750000, mrd=7.4e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 484 -k 207 -n 230 -transA=False -transB=False --perf_config=gemm:v1:64,256,16,2,1,16,0,3,2,0,0 -pv -RMS_threshold 1e-2

run_case "[17/51] gemm dt=bf16 | RMS=3.4e-01, mad=72.000000, mrd=4.8e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 22 -k 249 -n 330 -transA=True -transB=True --perf_config=gemm:v1:128,16,32,2,1,16,0,3,1,1,2 -pv -RMS_threshold 1e-2

run_case "[18/51] gemm dt=fp8 | RMS=5.5e-01, mad=58.351074, mrd=8.0e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 451 -k 121 -n 256 -transA=False -transB=False --perf_config=gemm:v1:32,16,16,2,1,2,32,1,1,8,1 -pv -relDiff_threshold 1e-5

run_case "[19/51] gemm dt=f16 | RMS=4.0e-01, mad=11.656250, mrd=5.4e-01" \
    -operation gemm -t f16 -out_datatype f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 469 -k 256 -n 360 -transA=False -transB=False --perf_config=gemm:v1:128,32,32,2,1,2,16,2,1,4,8 -pv

run_case "[20/51] gemm dt=i8 | RMS=5.9e-01, mad=154.750000, mrd=8.5e-01" \
    -operation gemm -t i8 -out_datatype i32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 423 -k 421 -n 138 -transA=True -transB=False --perf_config=gemm:v1:128,16,32,2,1,2,0,1,1,0,1 -pv

run_case "[21/51] gemm dt=bf16 | RMS=5.4e-01, mad=48.500000, mrd=7.7e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 49 -k 105 -n 461 -transA=True -transB=True --perf_config=gemm:v1:256,256,16,2,1,16,0,1,3,8,8 -pv -RMS_threshold 1e-2

run_case "[22/51] gemm dt=bf16 | RMS=6.6e-01, mad=24.031250, mrd=8.5e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 257 -k 339 -n 270 -transA=True -transB=False --perf_config=gemm:v1:256,16,16,2,1,1,0,3,2,1,1 -pv -RMS_threshold 1e-2

run_case "[23/51] gemm dt=fp8 | RMS=4.6e-01, mad=164.000000, mrd=6.5e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 19 -k 420 -n 484 -transA=True -transB=True --perf_config=gemm:v1:32,16,32,2,1,4,0,4,2,1,1 -pv -relDiff_threshold 1e-5

run_case "[24/51] gemm dt=f32 | RMS=4.3e-01, mad=123.250000, mrd=6.2e-01" \
    -operation gemm -t f32 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 298 -k 484 -n 334 -transA=True -transB=True --perf_config=gemm:v1:64,256,64,2,1,1,32,4,3,0,0 -pv

run_case "[25/51] gemm dt=fp8 | RMS=2.4e-01, mad=29.000000, mrd=3.5e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 396 -k 140 -n 70 -transA=False -transB=False --perf_config=gemm:v1:128,64,32,2,1,8,16,3,1,4,4 -pv -relDiff_threshold 1e-5

run_case "[26/51] gemm dt=bf16 | RMS=6.2e-01, mad=270.000000, mrd=8.9e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 475 -k 504 -n 185 -transA=True -transB=True --perf_config=gemm:v1:128,32,16,2,1,16,32,2,3,2,0 -pv -RMS_threshold 1e-2

run_case "[27/51] gemm dt=bf16 | RMS=4.7e-01, mad=241.000000, mrd=7.7e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 110 -k 436 -n 125 -transA=True -transB=False --perf_config=gemm:v1:256,256,128,1,1,4,0,1,3,4,0 -pv -RMS_threshold 1e-2

run_case "[28/51] gemm dt=f16 | RMS=2.6e-01, mad=28.000000, mrd=3.6e-01" \
    -operation gemm -t f16 -out_datatype f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 293 -k 132 -n 405 -transA=True -transB=True --perf_config=gemm:v1:256,128,32,2,1,2,16,3,2,4,4 -pv

run_case "[29/51] gemm dt=fp8 | RMS=2.3e-01, mad=3.000000, mrd=3.8e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 204 -k 98 -n 160 -transA=True -transB=True --perf_config=gemm:v1:16,256,16,2,1,2,32,4,3,1,0 -pv -relDiff_threshold 1e-5

run_case "[30/51] gemm dt=i8 | RMS=3.7e-01, mad=36.250000, mrd=5.5e-01" \
    -operation gemm -t i8 -out_datatype i32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 67 -k 139 -n 173 -transA=True -transB=False --perf_config=gemm:v1:64,256,16,2,1,4,0,4,3,4,8 -pv

run_case "[31/51] gemm dt=bf16 | RMS=3.6e-01, mad=7.375000, mrd=5.2e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 332 -k 158 -n 450 -transA=False -transB=False --perf_config=gemm:v1:128,128,16,2,1,2,16,4,2,0,1 -pv -RMS_threshold 1e-2

run_case "[32/51] gemm dt=f16 | RMS=6.3e-01, mad=232.437500, mrd=9.1e-01" \
    -operation gemm -t f16 -out_datatype f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 34 -k 425 -n 60 -transA=False -transB=True --perf_config=gemm:v1:32,256,16,2,1,8,0,2,3,8,1 -pv

run_case "[33/51] gemm dt=i8 | RMS=4.6e-01, mad=79.437500, mrd=6.5e-01" \
    -operation gemm -t i8 -out_datatype i32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 227 -k 289 -n 7 -transA=False -transB=True --perf_config=gemm:v1:32,256,16,2,1,2,0,1,1,2,4 -pv

run_case "[34/51] gemm dt=i8 | RMS=3.2e-01, mad=13.937500, mrd=4.4e-01" \
    -operation gemm -t i8 -out_datatype i32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 124 -k 505 -n 142 -transA=True -transB=False --perf_config=gemm:v1:128,32,64,2,1,1,32,4,3,2,8 -pv

run_case "[35/51] gemm dt=fp8 | RMS=5.4e-01, mad=8.542969, mrd=8.4e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 229 -k 125 -n 465 -transA=True -transB=False --perf_config=gemm:v1:64,256,16,2,1,2,0,1,3,1,2 -pv -relDiff_threshold 1e-5

run_case "[36/51] gemm dt=bf16 | RMS=4.3e-01, mad=49.500000, mrd=6.4e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 84 -k 131 -n 148 -transA=False -transB=True --perf_config=gemm:v1:256,32,16,2,1,16,32,4,3,8,1 -pv -RMS_threshold 1e-2

run_case "[37/51] gemm dt=f16 | RMS=5.1e-01, mad=194.250000, mrd=7.3e-01" \
    -operation gemm -t f16 -out_datatype f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 377 -k 443 -n 37 -transA=False -transB=True --perf_config=gemm:v1:16,64,32,2,1,16,16,2,3,8,2 -pv

run_case "[38/51] gemm dt=f32 | RMS=5.3e-01, mad=96.875000, mrd=7.4e-01" \
    -operation gemm -t f32 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 97 -k 339 -n 28 -transA=False -transB=False --perf_config=gemm:v1:32,256,64,2,1,4,32,3,3,8,0 -pv

run_case "[39/51] gemm dt=i8 | RMS=5.5e-01, mad=112.235352, mrd=7.8e-01" \
    -operation gemm -t i8 -out_datatype i32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 174 -k 57 -n 68 -transA=True -transB=False --perf_config=gemm:v1:128,64,16,2,1,1,0,1,2,2,1 -pv

run_case "[40/51] gemm dt=fp8 | RMS=4.8e-01, mad=12.750000, mrd=6.9e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 102 -k 214 -n 105 -transA=True -transB=True --perf_config=gemm:v1:64,64,16,2,1,1,32,4,3,1,1 -pv -relDiff_threshold 1e-5

run_case "[41/51] gemm dt=f16 | RMS=8.5e-01, mad=116.187500, mrd=8.5e-01" \
    -operation gemm -t f16 -out_datatype f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 103 -k 225 -n 79 -transA=False -transB=True --perf_config=gemm:v1:256,16,32,2,1,1,16,1,2,1,4 -pv

run_case "[42/51] gemm dt=bf16 | RMS=5.1e-01, mad=156.500000, mrd=7.3e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 51 -k 353 -n 349 -transA=False -transB=False --perf_config=gemm:v1:256,256,16,2,1,8,16,4,3,1,0 -pv -RMS_threshold 1e-2

run_case "[43/51] gemm dt=bf16 | RMS=6.3e-01, mad=243.250000, mrd=8.8e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 160 -k 476 -n 353 -transA=True -transB=False --perf_config=gemm:v1:16,32,32,2,1,2,0,1,2,0,0 -pv -RMS_threshold 1e-2

run_case "[44/51] gemm dt=fp8 | RMS=2.8e-01, mad=28.000000, mrd=3.9e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 470 -k 123 -n 229 -transA=True -transB=False --perf_config=gemm:v1:256,256,32,2,1,8,16,3,2,8,0 -pv -relDiff_threshold 1e-5

run_case "[45/51] gemm dt=i8 | RMS=6.4e-01, mad=177.625000, mrd=6.4e-01" \
    -operation gemm -t i8 -out_datatype i32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 389 -k 261 -n 314 -transA=False -transB=True --perf_config=gemm:v1:64,16,32,1,1,8,0,2,3,2,1 -pv

run_case "[46/51] gemm dt=fp8 | RMS=1.9e-01, mad=19.375000, mrd=2.9e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 49 -k 120 -n 442 -transA=False -transB=False --perf_config=gemm:v1:256,16,32,2,1,8,0,2,2,1,4 -pv -relDiff_threshold 1e-5

run_case "[47/51] gemm dt=fp8 | RMS=4.8e-01, mad=24.500000, mrd=6.1e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 470 -k 480 -n 495 -transA=True -transB=False --perf_config=gemm:v1:128,64,32,2,1,8,16,3,1,1,8 -pv -relDiff_threshold 1e-5

run_case "[48/51] gemm dt=bf16 | RMS=5.4e-01, mad=11.656250, mrd=7.2e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 182 -k 195 -n 45 -transA=True -transB=False --perf_config=gemm:v1:128,16,16,2,1,2,32,4,1,8,8 -pv -RMS_threshold 1e-2

run_case "[49/51] gemm dt=bf16 | RMS=6.8e-01, mad=177.375000, mrd=9.5e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 86 -k 324 -n 64 -transA=True -transB=True --perf_config=gemm:v1:16,128,16,2,1,4,16,1,2,0,8 -pv -RMS_threshold 1e-2

run_case "[50/51] gemm dt=fp8 | RMS=5.3e-01, mad=97.129883, mrd=7.6e-01" \
    -operation gemm -t fp8 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 151 -k 211 -n 16 -transA=True -transB=True --perf_config=gemm:v1:64,128,32,2,1,4,16,1,1,1,0 -pv -relDiff_threshold 1e-5

run_case "[51/51] gemm dt=bf16 | RMS=3.2e-01, mad=47.000000, mrd=4.5e-01" \
    -operation gemm -t bf16 -out_datatype bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 329 -k 181 -n 278 -transA=True -transB=True --perf_config=gemm:v1:128,32,16,2,1,4,16,4,2,2,1 -pv -RMS_threshold 1e-2
