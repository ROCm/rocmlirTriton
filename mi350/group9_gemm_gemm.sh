#!/usr/bin/env bash
# group9_gemm_gemm.sh -- GEMM+GEMM (back-to-back) numerical bugs (45 cases)
#
# rocmlir-gen -operation gemm_gemm failures (the 2-GEMM fused kernel used in attention-like flows).
#
# Logs source: mi350/gemm_gemm_errors.log (sweep run 2026-05-07).
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

run_case "[1/45] dt=bf16 m35 k83 n67 gO79 trans[tA,tB,tC] | RMS=5.9e-01, mad=1312.000000, mrd=8.1e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 35 -k 83 -n 67 -gemmO 79 -transA=True -transB=True -transC=True -transO=False --perf_config=attn:v1:32,32,16,2,1,16,32,1,2,4,4 -pv -RMS_threshold 1e-2

run_case "[2/45] dt=f16 m9 k70 n3 gO96 trans[tB] | RMS=3.2e-03, mad=0.015625, mrd=1.8e-02" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 9 -k 70 -n 3 -gemmO 96 -transA=False -transB=True -transC=False -transO=False --perf_config=attn:v1:64,64,128,2,1,2,16,4,2,8,2 -pv

run_case "[3/45] dt=bf16 m81 k85 n33 gO122 trans[tB,tC] | RMS=5.4e-01, mad=694.000000, mrd=7.7e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 81 -k 85 -n 33 -gemmO 122 -transA=False -transB=True -transC=True -transO=False --perf_config=attn:v1:16,64,16,2,1,1,16,1,2,4,2 -pv -RMS_threshold 1e-2

run_case "[4/45] dt=f16 m56 k127 n23 gO33 trans[tB,tO] | RMS=5.7e-01, mad=43.843750, mrd=7.6e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 56 -k 127 -n 23 -gemmO 33 -transA=False -transB=True -transC=False -transO=True --perf_config=attn:v1:32,32,16,2,1,16,0,3,1,8,2 -pv

run_case "[5/45] dt=bf16 m120 k56 n30 gO74 trans[tA,tO] | RMS=3.4e-01, mad=596.000000, mrd=6.1e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 120 -k 56 -n 30 -gemmO 74 -transA=True -transB=False -transC=False -transO=True --perf_config=attn:v1:32,32,16,2,1,4,0,4,1,4,0 -pv -RMS_threshold 1e-2

run_case "[6/45] dt=f16 m104 k109 n85 gO42 trans[tA,tC] | RMS=5.2e-01, mad=2230.000000, mrd=7.5e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 104 -k 109 -n 85 -gemmO 42 -transA=True -transB=False -transC=True -transO=False --perf_config=attn:v1:16,128,16,2,1,1,32,3,2,8,0 -pv

run_case "[7/45] dt=f16 m62 k34 n58 gO27 trans[tA,tB,tC] | RMS=4.0e-01, mad=387.000000, mrd=6.1e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 62 -k 34 -n 58 -gemmO 27 -transA=True -transB=True -transC=True -transO=False --perf_config=attn:v1:128,32,16,2,1,8,32,3,3,2,2 -pv

run_case "[8/45] dt=f16 m113 k22 n96 gO20 trans[tA,tO] | RMS=2.2e-01, mad=56.000000, mrd=4.0e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 113 -k 22 -n 96 -gemmO 20 -transA=True -transB=False -transC=False -transO=True --perf_config=attn:v1:64,256,16,2,1,2,16,1,3,0,2 -pv

run_case "[9/45] dt=bf16 m32 k32 n122 gO78 trans[tB] | RMS=1.5e-01, mad=15.250000, mrd=3.6e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 32 -k 32 -n 122 -gemmO 78 -transA=False -transB=True -transC=False -transO=False --perf_config=attn:v1:32,256,16,2,1,8,16,4,2,2,2 -pv -RMS_threshold 1e-2

run_case "[10/45] dt=f16 m39 k115 n49 gO113 trans[tC,tO] | RMS=5.8e-01, mad=1557.500000, mrd=8.4e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 39 -k 115 -n 49 -gemmO 113 -transA=False -transB=False -transC=True -transO=True --perf_config=attn:v1:128,128,16,2,1,1,0,3,1,1,1 -pv

run_case "[11/45] dt=bf16 m72 k113 n115 gO76 trans[tA,tB,tC,tO] | RMS=3.0e-01, mad=76.750000, mrd=2.1e+00" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 72 -k 113 -n 115 -gemmO 76 -transA=True -transB=True -transC=True -transO=True --perf_config=attn:v1:16,256,32,2,1,16,0,2,2,0,4 -pv -RMS_threshold 1e-2

run_case "[12/45] dt=bf16 m61 k39 n92 gO5 trans[tA,tB,tC] | RMS=3.2e-01, mad=67.500000, mrd=3.2e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 61 -k 39 -n 92 -gemmO 5 -transA=True -transB=True -transC=True -transO=False --perf_config=attn:v1:128,256,16,1,1,1,32,1,3,4,8 -pv -RMS_threshold 1e-2

run_case "[13/45] dt=bf16 m33 k115 n125 gO28 trans[tA,tC] | RMS=2.7e-01, mad=88.000000, mrd=8.9e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 33 -k 115 -n 125 -gemmO 28 -transA=True -transB=False -transC=True -transO=False --perf_config=attn:v1:64,256,32,2,1,4,16,1,3,1,0 -pv -RMS_threshold 1e-2

run_case "[14/45] dt=bf16 m12 k79 n124 gO7 trans[tC,tO] | RMS=2.3e-01, mad=41.250000, mrd=5.8e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 12 -k 79 -n 124 -gemmO 7 -transA=False -transB=False -transC=True -transO=True --perf_config=attn:v1:32,64,32,2,1,4,32,1,2,0,8 -pv -RMS_threshold 1e-2

run_case "[15/45] dt=f16 m2 k41 n42 gO95 trans[tO] | RMS=4.5e-01, mad=83.093750, mrd=7.9e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 2 -k 41 -n 42 -gemmO 95 -transA=False -transB=False -transC=False -transO=True --perf_config=attn:v1:16,16,16,2,1,8,32,1,3,4,1 -pv

run_case "[16/45] dt=f32 m35 k74 n63 gO13 trans[tC,tO] | RMS=3.9e-01, mad=36.703125, mrd=2.5e+00" \
    -operation gemm_gemm -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 35 -k 74 -n 63 -gemmO 13 -transA=False -transB=False -transC=True -transO=True --perf_config=attn:v1:256,64,64,1,1,4,0,4,3,0,4 -pv -relDiff_threshold 1e-4

run_case "[17/45] dt=bf16 m111 k63 n54 gO50 trans[tB,tO] | RMS=5.4e-01, mad=92.000000, mrd=5.4e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 111 -k 63 -n 54 -gemmO 50 -transA=False -transB=True -transC=False -transO=True --perf_config=attn:v1:64,64,16,2,1,2,16,1,2,1,1 -pv -RMS_threshold 1e-2

run_case "[18/45] dt=bf16 m84 k119 n107 gO19 trans[tB,tC,tO] | RMS=5.7e-01, mad=3328.000000, mrd=8.2e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 84 -k 119 -n 107 -gemmO 19 -transA=False -transB=True -transC=True -transO=True --perf_config=attn:v1:64,256,16,2,1,4,32,2,2,8,8 -pv -RMS_threshold 1e-2

run_case "[19/45] dt=bf16 m8 k46 n66 gO4 trans[tB] | RMS=7.4e-01, mad=1046.250000, mrd=1.0e+00" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 8 -k 46 -n 66 -gemmO 4 -transA=False -transB=True -transC=False -transO=False --perf_config=attn:v1:64,64,32,2,1,2,0,3,3,0,4 -pv -RMS_threshold 1e-2

run_case "[20/45] dt=f16 m69 k40 n44 gO96 trans[tA,tC,tO] | RMS=2.6e-01, mad=23.484375, mrd=3.1e+00" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 69 -k 40 -n 44 -gemmO 96 -transA=True -transB=False -transC=True -transO=True --perf_config=attn:v1:64,16,32,2,1,4,0,2,1,8,0 -pv

run_case "[21/45] dt=bf16 m48 k47 n76 gO49 trans[tA,tB,tC,tO] | RMS=3.4e-01, mad=28.125000, mrd=1.1e+00" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 48 -k 47 -n 76 -gemmO 49 -transA=True -transB=True -transC=True -transO=True --perf_config=attn:v1:32,32,16,2,1,1,32,2,1,4,8 -pv -RMS_threshold 1e-2

run_case "[22/45] dt=bf16 m29 k54 n37 gO22 trans[tA,tC] | RMS=4.4e-01, mad=417.000000, mrd=6.5e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 29 -k 54 -n 37 -gemmO 22 -transA=True -transB=False -transC=True -transO=False --perf_config=attn:v1:64,128,16,2,1,8,0,4,3,4,8 -pv -RMS_threshold 1e-2

run_case "[23/45] dt=bf16 m82 k65 n51 gO11 trans[tC,tO] | RMS=2.6e-01, mad=62.000000, mrd=3.1e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 82 -k 65 -n 51 -gemmO 11 -transA=False -transB=False -transC=True -transO=True --perf_config=attn:v1:128,16,32,2,1,1,32,2,3,8,4 -pv -RMS_threshold 1e-2

run_case "[24/45] dt=bf16 m39 k91 n2 gO109 trans[tA,tB,tC,tO] | RMS=3.3e-01, mad=6.265625, mrd=9.1e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 39 -k 91 -n 2 -gemmO 109 -transA=True -transB=True -transC=True -transO=True --perf_config=attn:v1:32,32,16,2,1,8,32,1,3,1,8 -pv -RMS_threshold 1e-2

run_case "[25/45] dt=bf16 m45 k62 n104 gO112 trans[tA,tB,tC,tO] | RMS=4.2e-01, mad=64.750000, mrd=8.4e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 45 -k 62 -n 104 -gemmO 112 -transA=True -transB=True -transC=True -transO=True --perf_config=attn:v1:32,16,16,2,1,2,16,3,3,4,1 -pv -RMS_threshold 1e-2

run_case "[26/45] dt=f16 m60 k28 n128 gO63 trans[tB,tO] | RMS=1.6e-01, mad=11.406250, mrd=3.9e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 60 -k 28 -n 128 -gemmO 63 -transA=False -transB=True -transC=False -transO=True --perf_config=attn:v1:256,32,16,2,1,8,0,2,3,0,1 -pv

run_case "[27/45] dt=bf16 m78 k31 n50 gO74 trans[tO] | RMS=1.9e-01, mad=128.000000, mrd=3.0e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 78 -k 31 -n 50 -gemmO 74 -transA=False -transB=False -transC=False -transO=True --perf_config=attn:v1:256,32,16,2,1,8,0,1,1,2,8 -pv -RMS_threshold 1e-2

run_case "[28/45] dt=f16 m96 k100 n37 gO52 trans[-] | RMS=4.8e-01, mad=830.000000, mrd=6.9e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 96 -k 100 -n 37 -gemmO 52 -transA=False -transB=False -transC=False -transO=False --perf_config=attn:v1:256,128,32,2,1,1,16,2,2,8,8 -pv

run_case "[29/45] dt=f16 m8 k107 n70 gO64 trans[tC,tO] | RMS=5.3e-01, mad=1837.000000, mrd=7.6e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 8 -k 107 -n 70 -gemmO 64 -transA=False -transB=False -transC=True -transO=True --perf_config=attn:v1:16,128,16,2,1,8,0,4,3,2,2 -pv

run_case "[30/45] dt=f16 m26 k119 n111 gO76 trans[tA,tB] | RMS=4.4e-01, mad=2277.000000, mrd=5.9e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 26 -k 119 -n 111 -gemmO 76 -transA=True -transB=True -transC=False -transO=False --perf_config=attn:v1:128,256,32,2,1,16,16,1,2,2,8 -pv

run_case "[31/45] dt=f16 m65 k93 n38 gO127 trans[tA,tB] | RMS=3.8e-01, mad=160.125000, mrd=7.3e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 65 -k 93 -n 38 -gemmO 127 -transA=True -transB=True -transC=False -transO=False --perf_config=attn:v1:128,32,16,2,1,2,0,4,1,8,8 -pv

run_case "[32/45] dt=f16 m126 k105 n107 gO97 trans[tA,tB,tC] | RMS=4.5e-01, mad=56.875000, mrd=6.5e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 126 -k 105 -n 107 -gemmO 97 -transA=True -transB=True -transC=True -transO=False --perf_config=attn:v1:64,64,32,2,1,4,16,4,2,2,8 -pv

run_case "[33/45] dt=f16 m53 k99 n48 gO89 trans[tA,tC] | RMS=8.9e-01, mad=218.875000, mrd=9.1e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 53 -k 99 -n 48 -gemmO 89 -transA=True -transB=False -transC=True -transO=False --perf_config=attn:v1:64,16,16,2,1,4,16,1,3,8,4 -pv

run_case "[34/45] dt=bf16 m84 k34 n29 gO114 trans[tO] | RMS=3.3e-01, mad=9.812500, mrd=6.0e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 84 -k 34 -n 29 -gemmO 114 -transA=False -transB=False -transC=False -transO=True --perf_config=attn:v1:256,256,32,2,1,8,16,1,1,2,8 -pv -RMS_threshold 1e-2

run_case "[35/45] dt=bf16 m121 k73 n31 gO7 trans[tB,tC,tO] | RMS=4.6e-01, mad=510.000000, mrd=6.7e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 121 -k 73 -n 31 -gemmO 7 -transA=False -transB=True -transC=True -transO=True --perf_config=attn:v1:128,16,16,2,1,1,0,1,1,4,4 -pv -RMS_threshold 1e-2

run_case "[36/45] dt=bf16 m64 k108 n28 gO118 trans[tC,tO] | RMS=5.3e-01, mad=746.000000, mrd=7.5e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 64 -k 108 -n 28 -gemmO 118 -transA=False -transB=False -transC=True -transO=True --perf_config=attn:v1:128,256,16,2,1,16,32,2,2,1,2 -pv -RMS_threshold 1e-2

run_case "[37/45] dt=f16 m82 k115 n56 gO79 trans[tB,tC] | RMS=5.9e-01, mad=1759.500000, mrd=8.4e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 82 -k 115 -n 56 -gemmO 79 -transA=False -transB=True -transC=True -transO=False --perf_config=attn:v1:32,32,16,2,1,8,0,3,2,4,2 -pv

run_case "[38/45] dt=f16 m5 k112 n33 gO114 trans[tO] | RMS=4.4e-01, mad=16.414062, mrd=6.5e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 5 -k 112 -n 33 -gemmO 114 -transA=False -transB=False -transC=False -transO=True --perf_config=attn:v1:64,256,32,2,1,16,16,3,2,0,2 -pv

run_case "[39/45] dt=bf16 m104 k116 n117 gO16 trans[tB] | RMS=5.9e-01, mad=3656.000000, mrd=8.4e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 104 -k 116 -n 117 -gemmO 16 -transA=False -transB=True -transC=False -transO=False --perf_config=attn:v1:32,256,16,2,1,4,32,2,1,0,1 -pv -RMS_threshold 1e-2

run_case "[40/45] dt=f16 m108 k76 n9 gO26 trans[tA,tC,tO] | RMS=4.1e-01, mad=217.500000, mrd=5.3e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 108 -k 76 -n 9 -gemmO 26 -transA=True -transB=False -transC=True -transO=True --perf_config=attn:v1:32,128,32,2,1,2,16,1,2,2,2 -pv

run_case "[41/45] dt=bf16 m94 k81 n80 gO78 trans[tB,tC,tO] | RMS=6.9e-01, mad=302.000000, mrd=7.9e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 94 -k 81 -n 80 -gemmO 78 -transA=False -transB=True -transC=True -transO=True --perf_config=attn:v1:128,32,16,2,1,1,16,3,2,4,4 -pv -RMS_threshold 1e-2

run_case "[42/45] dt=bf16 m50 k69 n81 gO67 trans[tA,tB,tC,tO] | RMS=6.1e-01, mad=232.250000, mrd=8.7e-01" \
    -operation gemm_gemm -t bf16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 50 -k 69 -n 81 -gemmO 67 -transA=True -transB=True -transC=True -transO=True --perf_config=attn:v1:16,32,16,2,1,16,16,1,2,1,1 -pv -RMS_threshold 1e-2

run_case "[43/45] dt=f16 m69 k36 n82 gO66 trans[tB] | RMS=4.3e-01, mad=986.000000, mrd=5.5e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 69 -k 36 -n 82 -gemmO 66 -transA=False -transB=True -transC=False -transO=False --perf_config=attn:v1:256,32,16,2,1,8,0,1,1,0,8 -pv

run_case "[44/45] dt=f32 m55 k71 n34 gO116 trans[tB,tC] | RMS=5.4e-01, mad=1101.500000, mrd=7.9e-01" \
    -operation gemm_gemm -t f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 55 -k 71 -n 34 -gemmO 116 -transA=False -transB=True -transC=True -transO=False --perf_config=attn:v1:256,256,64,1,1,1,16,3,3,0,4 -pv -relDiff_threshold 1e-4

run_case "[45/45] dt=f16 m43 k83 n119 gO56 trans[tB] | RMS=5.5e-01, mad=2507.000000, mrd=7.9e-01" \
    -operation gemm_gemm -t f16 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 2 -m 43 -k 83 -n 119 -gemmO 56 -transA=False -transB=True -transC=False -transO=False --perf_config=attn:v1:256,256,16,2,1,1,32,3,1,4,8 -pv
