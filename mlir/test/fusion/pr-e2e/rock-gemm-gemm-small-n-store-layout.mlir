// This regression covers a bf16 `gemm_gemm` shape whose final Triton store
// exposes an MFMA-like layout with a small N dimension. The optimized epilogue
// store layout swaps N-dim basis bits to enable wide stores; when N is too
// small, the target basis bit does not exist.

// RUN: rocmlir-gen -pv --operation gemm_gemm \
// RUN:   -t bf16 --arch %arch -g 2 -m 61 -k 39 -n 92 -gemmO 5 \
// RUN:   -transA=True -transB=True -transC=True -transO=False \
// RUN:   --perf_config="attn:v1:128,256,16,1,1,1,32,1,3,4,8" \
// RUN: | rocmlir-driver --host-pipeline=highlevel - \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
