// RUN: rocmlir-gen -operation gemm_gemm -t f32 --arch %arch -g 1 -m 115 -k 27 -n 104 -gemmO 2 -transA=False -transB=False -transC=True -transO=False --perf_config=attn:v1:256,16,128,1,1,4,0,4,2,0,4 -pv | rocmlir-driver -host-pipeline=highlevel | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
