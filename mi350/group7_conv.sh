#!/usr/bin/env bash
# group7_conv.sh -- Conv numerical bugs (excludes already-reported getUpperBounds() SIGSEGV) (45 cases)
#
# Direct conv forward / backward-data / backward-weight failures with numerically wrong results.
# The 13 ConvConfiguration cases that SIGSEGV in rock::TransformMapAttr::getUpperBounds() are
# intentionally NOT included here -- that crash has already been filed as a separate bug.
#
# Logs source: mi350/conv_errors.log (sweep run 2026-05-07).
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

run_case "[1/45] conv dt=f16 | RMS=2.6e-01, mad=9.614258, mrd=2.2e+01" \
    --operation conv_bwd_data -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 8 --in_channels 32 --in_h 56 --in_w 14 --out_channels 32 --fil_h 7 --fil_w 5 --dilation_h 1 --dilation_w 2 --conv_stride_h 1 --conv_stride_w 2 --padding_h_l 1 --padding_h_r 0 --padding_w_l 3 --padding_w_r 2 --groupsize 2 --perf_config=gemm:v1:16,16,32,2,1,1,16,1,2,8,4 -pv

run_case "[2/45] conv dt=f16 | RMS=2.0e-01, mad=1.312500, mrd=2.0e+00" \
    --operation conv_bwd_data -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 2 --in_channels 32 --in_h 8 --in_w 16 --out_channels 32 --fil_h 1 --fil_w 3 --dilation_h 2 --dilation_w 2 --conv_stride_h 3 --conv_stride_w 1 --padding_h_l 2 --padding_h_r 1 --padding_w_l 2 --padding_w_r 3 --groupsize 2 --perf_config=gemm:v1:256,256,16,2,1,4,32,1,2,0,2 -pv

run_case "[3/45] conv dt=i8 | RMS=6.2e-01, mad=169.447815, mrd=2.6e+00" \
    --operation conv -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 1 --in_channels 8 --in_h 14 --in_w 16 --out_channels 32 --fil_h 2 --fil_w 5 --dilation_h 1 --dilation_w 1 --conv_stride_h 3 --conv_stride_w 1 --padding_h_l 1 --padding_h_r 2 --padding_w 3 --groupsize 2 --perf_config=gemm:v1:256,32,128,1,1,1,16,3,2,1,2 -pv

run_case "[4/45] conv dt=f16 | RMS=2.0e-01, mad=75.250000, mrd=6.8e-01" \
    --operation conv_bwd_data -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 1 --in_channels 64 --in_h 32 --in_w 14 --out_channels 256 --fil_h 2 --fil_w 1 --dilation_h 1 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 3 --padding_h 0 --padding_w 3 --groupsize 4 --perf_config=gemm:v1:32,64,128,1,1,1,0,2,2,2,0 -pv

run_case "[5/45] conv dt=bf16 | RMS=4.0e-01, mad=37.562500, mrd=1.6e+01" \
    --operation conv_bwd_data -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 1 --in_channels 16 --in_h 8 --in_w 64 --out_channels 64 --fil_h 2 --fil_w 5 --dilation_h 1 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h_l 0 --padding_h_r 1 --padding_w_l 2 --padding_w_r 3 --groupsize 1 --perf_config=gemm:v1:64,16,16,2,1,16,0,1,2,0,1 -pv -RMS_threshold 1e-2

run_case "[6/45] conv dt=f16 | RMS=2.9e-01, mad=237.000000, mrd=6.2e-01" \
    --operation conv -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 4 --in_channels 64 --in_h 8 --in_w 8 --out_channels 128 --fil_h 2 --fil_w 5 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 3 --padding_h_l 3 --padding_h_r 0 --padding_w_l 2 --padding_w_r 0 --groupsize 1 --perf_config=gemm:v1:32,32,32,2,1,2,16,4,1,8,0 -pv

run_case "[7/45] conv dt=f32 | RMS=4.2e-01, mad=56.343750, mrd=7.7e-01" \
    --operation conv_bwd_data -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 1 --in_channels 512 --in_h 56 --in_w 28 --out_channels 256 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 3 --conv_stride_w 2 --padding_h_l 0 --padding_h_r 1 --padding_w_l 0 --padding_w_r 3 --groupsize 1 --perf_config=gemm:v1:128,64,64,2,1,2,0,1,1,1,8 -pv

run_case "[8/45] conv dt=f16 | RMS=2.7e-01, mad=12.000000, mrd=4.5e-01" \
    --operation conv_bwd_data -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 16 --in_channels 1 --in_h 14 --in_w 16 --out_channels 16 --fil_h 7 --fil_w 1 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 3 --padding_h_l 1 --padding_h_r 3 --padding_w_l 3 --padding_w_r 1 --groupsize 1 --perf_config=gemm:v1:128,128,64,1,1,1,16,1,2,4,8 -pv

run_case "[9/45] conv dt=bf16 | RMS=4.6e-01, mad=26.875000, mrd=1.4e+00" \
    --operation conv -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 2 --in_channels 32 --in_h 28 --in_w 8 --out_channels 128 --fil_h 2 --fil_w 3 --dilation_h 2 --dilation_w 1 --conv_stride_h 3 --conv_stride_w 2 --padding_h_l 1 --padding_h_r 2 --padding_w_l 3 --padding_w_r 0 --groupsize 2 --perf_config=gemm:v1:64,16,16,2,1,8,32,1,2,8,1 -pv -RMS_threshold 1e-2

run_case "[10/45] conv dt=bf16 | RMS=6.4e-01, mad=29.873207, mrd=5.4e+01" \
    --operation conv -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 1 --in_channels 64 --in_h 28 --in_w 112 --out_channels 32 --fil_h 3 --fil_w 5 --dilation_h 1 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 3 --padding_h_l 0 --padding_h_r 2 --padding_w_l 0 --padding_w_r 2 --groupsize 1 --perf_config=gemm:v1:64,16,32,2,1,4,16,1,1,4,4 -pv -RMS_threshold 1e-2

run_case "[11/45] conv dt=f16 | RMS=6.2e-01, mad=8.539062, mrd=3.2e+00" \
    --operation conv_bwd_data -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 16 --in_channels 128 --in_h 16 --in_w 8 --out_channels 32 --fil_h 7 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 3 --padding_h_l 1 --padding_h_r 2 --padding_w_l 1 --padding_w_r 0 --groupsize 4 --perf_config=gemm:v1:128,256,128,2,1,8,0,1,1,1,2 -pv

run_case "[12/45] conv dt=bf16 | RMS=2.2e-01, mad=11.750000, mrd=6.4e-01" \
    --operation conv_bwd_data -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 1 --in_channels 64 --in_h 56 --in_w 4 --out_channels 32 --fil_h 1 --fil_w 5 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h_l 0 --padding_h_r 1 --padding_w 3 --groupsize 1 --perf_config=gemm:v1:128,16,16,2,1,1,0,2,2,0,2 -pv -RMS_threshold 1e-2

run_case "[13/45] conv dt=f16 | RMS=4.5e-01, mad=144.789062, mrd=1.1e+00" \
    --operation conv_bwd_data -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 16 --in_channels 128 --in_h 28 --in_w 8 --out_channels 512 --fil_h 7 --fil_w 5 --dilation_h 2 --dilation_w 2 --conv_stride_h 3 --conv_stride_w 2 --padding_h_l 1 --padding_h_r 2 --padding_w_l 0 --padding_w_r 1 --groupsize 2 --perf_config=gemm:v1:16,16,16,2,1,4,16,4,3,0,4 -pv

run_case "[14/45] conv dt=fp8 | RMS=3.5e-01, mad=8.039017, mrd=1.8e+00" \
    --operation conv -t fp8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 2 --in_channels 256 --in_h 224 --in_w 32 --out_channels 128 --fil_h 2 --fil_w 3 --dilation_h 2 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 1 --padding_h_l 1 --padding_h_r 0 --padding_w 3 --groupsize 4 --perf_config=gemm:v1:128,32,16,2,1,16,32,1,2,8,1 -pv -relDiff_threshold 1e-5

run_case "[15/45] conv dt=bf16 | RMS=6.5e-01, mad=566.625000, mrd=9.5e-01" \
    --operation conv -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 16 --in_channels 512 --in_h 32 --in_w 8 --out_channels 64 --fil_h 1 --fil_w 5 --dilation_h 2 --dilation_w 2 --conv_stride_h 1 --conv_stride_w 3 --padding_h 2 --padding_w_l 0 --padding_w_r 2 --groupsize 2 --perf_config=gemm:v1:128,16,32,2,1,8,16,1,2,8,4 -pv -RMS_threshold 1e-2

run_case "[16/45] conv dt=fp8 | RMS=5.4e-01, mad=725.000000, mrd=9.4e-01" \
    --operation conv -t fp8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 4 --in_channels 256 --in_h 4 --in_w 32 --out_channels 16 --fil_h 1 --fil_w 5 --dilation_h 2 --dilation_w 1 --conv_stride_h 3 --conv_stride_w 3 --padding_h_l 3 --padding_h_r 1 --padding_w_l 0 --padding_w_r 2 --groupsize 1 --perf_config=gemm:v1:256,64,16,2,1,2,32,3,1,0,8 -pv -relDiff_threshold 1e-5

run_case "[17/45] conv dt=bf16 | RMS=2.7e-01, mad=9.234375, mrd=6.1e+00" \
    --operation conv_bwd_data -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 4 --in_channels 128 --in_h 56 --in_w 112 --out_channels 16 --fil_h 5 --fil_w 5 --dilation_h 2 --dilation_w 2 --conv_stride_h 1 --conv_stride_w 2 --padding_h_l 1 --padding_h_r 3 --padding_w_l 2 --padding_w_r 0 --groupsize 4 --perf_config=gemm:v1:32,64,16,2,1,2,0,2,3,4,1 -pv -RMS_threshold 1e-2

run_case "[18/45] conv dt=f16 | RMS=5.2e-01, mad=15.500977, mrd=1.1e+00" \
    --operation conv -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 4 --in_channels 512 --in_h 32 --in_w 16 --out_channels 256 --fil_h 3 --fil_w 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 3 --padding_h_l 0 --padding_h_r 3 --padding_w 2 --groupsize 2 --perf_config=gemm:v1:256,64,16,2,1,16,32,2,2,2,1 -pv

run_case "[19/45] conv dt=bf16 | RMS=3.1e-01, mad=80.914062, mrd=2.5e+00" \
    --operation conv_bwd_data -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 8 --in_channels 256 --in_h 32 --in_w 8 --out_channels 512 --fil_h 5 --fil_w 5 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w_l 0 --padding_w_r 1 --groupsize 4 --perf_config=gemm:v1:256,64,16,2,1,4,0,1,2,4,2 -pv -RMS_threshold 1e-2

run_case "[20/45] conv dt=bf16 | RMS=2.3e-01, mad=3.710938, mrd=2.5e+00" \
    --operation conv_bwd_data -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 4 --in_channels 8 --in_h 16 --in_w 224 --out_channels 64 --fil_h 7 --fil_w 5 --dilation_h 2 --dilation_w 2 --conv_stride_h 3 --conv_stride_w 3 --padding_h_l 3 --padding_h_r 0 --padding_w_l 2 --padding_w_r 3 --groupsize 2 --perf_config=gemm:v1:128,32,16,2,1,2,0,3,1,4,1 -pv -RMS_threshold 1e-2

run_case "[21/45] conv dt=f32 | RMS=2.3e-01, mad=229.000000, mrd=7.6e-01" \
    --operation conv_bwd_data -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 8 --in_channels 512 --in_h 8 --in_w 16 --out_channels 16 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 3 --padding_h_l 2 --padding_h_r 0 --padding_w 1 --groupsize 1 --perf_config=gemm:v1:16,128,32,2,1,2,32,1,1,0,4 -pv

run_case "[22/45] conv dt=bf16 | RMS=6.7e-01, mad=199.140625, mrd=1.0e+00" \
    --operation conv -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 2 --in_channels 64 --in_h 32 --in_w 224 --out_channels 256 --fil_h 7 --fil_w 5 --dilation_h 1 --dilation_w 2 --conv_stride_h 3 --conv_stride_w 1 --padding_h_l 3 --padding_h_r 0 --padding_w 0 --groupsize 1 --perf_config=gemm:v1:256,128,16,2,1,2,16,1,2,1,4 -pv -RMS_threshold 1e-2

run_case "[23/45] conv dt=bf16 | RMS=8.8e-02, mad=1.156250, mrd=1.0e+00" \
    --operation conv_bwd_data -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 2 --in_channels 256 --in_h 16 --in_w 14 --out_channels 32 --fil_h 1 --fil_w 3 --dilation_h 2 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h_l 1 --padding_h_r 2 --padding_w_l 2 --padding_w_r 0 --groupsize 1 --perf_config=gemm:v1:128,16,32,2,1,1,0,1,2,2,0 -pv -RMS_threshold 1e-2

run_case "[24/45] conv dt=i8 | RMS=5.2e-01, mad=3.648438, mrd=1.6e+01" \
    --operation conv -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 8 --in_channels 128 --in_h 224 --in_w 64 --out_channels 256 --fil_h 3 --fil_w 2 --dilation_h 1 --dilation_w 2 --conv_stride_h 1 --conv_stride_w 2 --padding_h 1 --padding_w_l 3 --padding_w_r 0 --groupsize 1 --perf_config=gemm:v1:128,256,128,2,1,8,16,1,2,2,1 -pv

run_case "[25/45] conv dt=f16 | RMS=3.8e-01, mad=38.101562, mrd=1.2e+02" \
    --operation conv -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 8 --in_channels 256 --in_h 32 --in_w 14 --out_channels 32 --fil_h 5 --fil_w 1 --dilation_h 1 --dilation_w 2 --conv_stride_h 3 --conv_stride_w 2 --padding_h 2 --padding_w_l 3 --padding_w_r 2 --groupsize 4 --perf_config=gemm:v1:16,32,16,2,1,2,0,3,1,2,2 -pv

run_case "[26/45] conv dt=bf16 | RMS=6.7e-01, mad=444.625000, mrd=9.6e-01" \
    --operation conv -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 2 --in_channels 256 --in_h 16 --in_w 32 --out_channels 256 --fil_h 2 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 3 --conv_stride_w 1 --padding_h_l 0 --padding_h_r 1 --padding_w 0 --groupsize 1 --perf_config=gemm:v1:64,128,16,2,1,2,16,2,1,2,8 -pv -RMS_threshold 1e-2

run_case "[27/45] conv dt=f32 | RMS=3.1e-01, mad=24.500000, mrd=5.2e-01" \
    --operation conv_bwd_data -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 16 --in_channels 32 --in_h 4 --in_w 64 --out_channels 256 --fil_h 1 --fil_w 2 --dilation_h 2 --dilation_w 1 --conv_stride_h 3 --conv_stride_w 2 --padding_h_l 3 --padding_h_r 0 --padding_w_l 0 --padding_w_r 2 --groupsize 2 --perf_config=gemm:v1:256,16,32,2,1,4,32,2,3,8,4 -pv

run_case "[28/45] conv dt=bf16 | RMS=2.4e-01, mad=18.250000, mrd=5.6e-01" \
    --operation conv_bwd_data -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 16 --in_channels 32 --in_h 224 --in_w 112 --out_channels 64 --fil_h 2 --fil_w 2 --dilation_h 1 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h_l 2 --padding_h_r 3 --padding_w_l 0 --padding_w_r 1 --groupsize 1 --perf_config=gemm:v1:16,64,16,2,1,1,16,3,3,2,4 -pv -RMS_threshold 1e-2

run_case "[29/45] conv dt=f16 | RMS=3.3e-01, mad=4.863281, mrd=9.1e+00" \
    --operation conv -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 1 --in_channels 128 --in_h 56 --in_w 64 --out_channels 32 --fil_h 3 --fil_w 2 --dilation_h 1 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 3 --padding_h_l 2 --padding_h_r 3 --padding_w_l 0 --padding_w_r 3 --groupsize 2 --perf_config=gemm:v1:256,128,16,2,1,4,0,4,2,1,0 -pv

run_case "[30/45] conv dt=bf16 | RMS=4.0e-01, mad=16.125000, mrd=8.0e-01" \
    --operation conv_bwd_data -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 16 --in_channels 8 --in_h 28 --in_w 16 --out_channels 128 --fil_h 1 --fil_w 1 --dilation_h 2 --dilation_w 2 --conv_stride_h 3 --conv_stride_w 2 --padding_h_l 2 --padding_h_r 0 --padding_w_l 1 --padding_w_r 2 --groupsize 4 --perf_config=gemm:v1:16,256,32,2,1,16,0,3,2,4,2 -pv -RMS_threshold 1e-2

run_case "[31/45] conv dt=bf16 | RMS=6.3e-01, mad=5.718750, mrd=1.2e+00" \
    --operation conv -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 2 --in_channels 128 --in_h 16 --in_w 64 --out_channels 16 --fil_h 5 --fil_w 3 --dilation_h 1 --dilation_w 2 --conv_stride_h 3 --conv_stride_w 1 --padding_h_l 1 --padding_h_r 0 --padding_w_l 2 --padding_w_r 0 --groupsize 2 --perf_config=gemm:v1:16,16,32,2,1,1,0,4,2,1,2 -pv -RMS_threshold 1e-2

run_case "[32/45] conv dt=bf16 | RMS=6.1e-01, mad=5.500000, mrd=1.1e+00" \
    --operation conv -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 8 --in_channels 256 --in_h 14 --in_w 14 --out_channels 32 --fil_h 3 --fil_w 1 --dilation_h 1 --dilation_w 2 --conv_stride_h 3 --conv_stride_w 1 --padding_h 1 --padding_w_l 1 --padding_w_r 3 --groupsize 1 --perf_config=gemm:v1:128,256,16,2,1,1,16,2,3,8,2 -pv -RMS_threshold 1e-2

run_case "[33/45] conv dt=bf16 | RMS=2.7e-01, mad=8.917969, mrd=3.1e+00" \
    --operation conv_bwd_data -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 2 --in_channels 64 --in_h 16 --in_w 16 --out_channels 512 --fil_h 7 --fil_w 1 --dilation_h 2 --dilation_w 2 --conv_stride_h 1 --conv_stride_w 3 --padding_h_l 0 --padding_h_r 3 --padding_w_l 0 --padding_w_r 3 --groupsize 4 --perf_config=gemm:v1:16,64,16,2,1,8,32,2,1,0,8 -pv -RMS_threshold 1e-2

run_case "[34/45] conv dt=bf16 | RMS=3.0e-01, mad=5.031250, mrd=1.1e+00" \
    --operation conv -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 2 --in_channels 64 --in_h 8 --in_w 64 --out_channels 512 --fil_h 2 --fil_w 1 --dilation_h 2 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 1 --padding_h_l 2 --padding_h_r 1 --padding_w 0 --groupsize 4 --perf_config=gemm:v1:64,16,16,2,1,4,0,1,3,4,0 -pv -RMS_threshold 1e-2

run_case "[35/45] conv dt=f16 | RMS=2.1e-01, mad=5.609375, mrd=6.8e-01" \
    --operation conv_bwd_data -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 8 --in_channels 8 --in_h 56 --in_w 32 --out_channels 32 --fil_h 2 --fil_w 5 --dilation_h 2 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 3 --padding_h_l 0 --padding_h_r 3 --padding_w_l 3 --padding_w_r 2 --groupsize 2 --perf_config=gemm:v1:16,16,16,2,1,16,32,1,3,0,8 -pv

run_case "[36/45] conv dt=bf16 | RMS=6.2e-01, mad=78.968750, mrd=1.1e+00" \
    --operation conv_bwd_data -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 4 --in_channels 32 --in_h 28 --in_w 8 --out_channels 512 --fil_h 5 --fil_w 5 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 2 --padding_h_l 1 --padding_h_r 2 --padding_w 1 --groupsize 2 --perf_config=gemm:v1:16,256,32,2,1,2,16,1,2,0,2 -pv -RMS_threshold 1e-2

run_case "[37/45] conv dt=bf16 | RMS=3.6e-01, mad=5.531250, mrd=1.4e+01" \
    --operation conv -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 2 --in_channels 32 --in_h 224 --in_w 14 --out_channels 256 --fil_h 7 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 3 --conv_stride_w 3 --padding_h_l 3 --padding_h_r 2 --padding_w_l 2 --padding_w_r 0 --groupsize 4 --perf_config=gemm:v1:256,256,16,2,1,1,32,3,1,1,1 -pv -RMS_threshold 1e-2

run_case "[38/45] conv dt=f16 | RMS=7.0e-02, mad=3.000000, mrd=3.5e-01" \
    --operation conv_bwd_data -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 16 --in_channels 8 --in_h 8 --in_w 14 --out_channels 32 --fil_h 2 --fil_w 1 --dilation_h 2 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 3 --padding_h_l 0 --padding_h_r 3 --padding_w_l 2 --padding_w_r 3 --groupsize 2 --perf_config=gemm:v1:64,16,16,2,1,2,16,1,1,4,2 -pv

run_case "[39/45] conv dt=f16 | RMS=1.4e-01, mad=0.875000, mrd=2.6e+00" \
    --operation conv_bwd_data -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 4 --in_channels 16 --in_h 4 --in_w 28 --out_channels 64 --fil_h 3 --fil_w 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 3 --padding_h 2 --padding_w_l 0 --padding_w_r 1 --groupsize 4 --perf_config=gemm:v1:64,64,16,2,1,4,32,1,1,0,1 -pv

run_case "[40/45] conv dt=i8 | RMS=5.9e-01, mad=89.390625, mrd=1.6e+00" \
    --operation conv -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 1 --in_channels 16 --in_h 64 --in_w 28 --out_channels 512 --fil_h 1 --fil_w 5 --dilation_h 1 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 1 --padding_h 2 --padding_w_l 1 --padding_w_r 0 --groupsize 2 --perf_config=gemm:v1:32,128,32,2,1,2,16,3,3,2,0 -pv

run_case "[41/45] conv dt=f16 | RMS=5.2e-01, mad=261.625000, mrd=8.8e-01" \
    --operation conv_bwd_data -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 2 --in_channels 16 --in_h 112 --in_w 224 --out_channels 512 --fil_h 1 --fil_w 2 --dilation_h 2 --dilation_w 2 --conv_stride_h 1 --conv_stride_w 3 --padding_h_l 2 --padding_h_r 0 --padding_w_l 0 --padding_w_r 3 --groupsize 1 --perf_config=gemm:v1:128,256,16,2,1,1,0,2,2,2,4 -pv

run_case "[42/45] conv dt=f16 | RMS=3.8e-01, mad=39.578125, mrd=3.3e+00" \
    --operation conv -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 4 --in_channels 64 --in_h 14 --in_w 8 --out_channels 64 --fil_h 7 --fil_w 5 --dilation_h 2 --dilation_w 2 --conv_stride_h 1 --conv_stride_w 2 --padding_h_l 0 --padding_h_r 2 --padding_w_l 0 --padding_w_r 3 --groupsize 2 --perf_config=gemm:v1:64,128,16,2,1,8,0,2,3,4,4 -pv

run_case "[43/45] conv dt=fp8 | RMS=6.0e-01, mad=119.373016, mrd=8.3e+00" \
    --operation conv -t fp8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 8 --in_channels 256 --in_h 56 --in_w 14 --out_channels 256 --fil_h 7 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 1 --padding_h_l 1 --padding_h_r 0 --padding_w_l 1 --padding_w_r 2 --groupsize 4 --perf_config=gemm:v1:128,32,32,2,1,2,16,3,1,8,8 -pv -relDiff_threshold 1e-5

run_case "[44/45] conv dt=i8 | RMS=5.3e-01, mad=232.625000, mrd=7.5e-01" \
    --operation conv -t i8 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 2 --in_channels 32 --in_h 16 --in_w 56 --out_channels 256 --fil_h 3 --fil_w 1 --dilation_h 2 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 3 --padding_h_l 3 --padding_h_r 0 --padding_w 1 --groupsize 1 --perf_config=gemm:v1:16,32,32,1,1,8,16,2,3,2,1 -pv

run_case "[45/45] conv dt=f16 | RMS=4.2e-01, mad=51.265625, mrd=1.3e+01" \
    --operation conv -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 1 --in_channels 256 --in_h 16 --in_w 56 --out_channels 128 --fil_h 5 --fil_w 5 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 1 --padding_h_l 1 --padding_h_r 2 --padding_w 3 --groupsize 4 --perf_config=gemm:v1:16,32,16,2,1,16,0,3,2,2,4 -pv
