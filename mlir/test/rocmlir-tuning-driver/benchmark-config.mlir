// RUN: rocmlir-gen --operation gemm --arch %arch -t f16 -out_datatype f16 -transA=false -transB=false -g 1 -m 384 -n 768 -k 3072 --perf_config= | rocmlir-tuning-driver --benchmark-config="gemm:v1:64,64,64,1,1,4,16,1,2,0,0" | FileCheck %s

// Two-stage tuning should also generate the same output.
// RUN: rocmlir-gen --operation gemm --arch %arch -t f16 -out_datatype f16 -transA=false -transB=false -g 1 -m 384 -n 768 -k 3072 --perf_config= | rocmlir-tuning-driver --benchmark-config="gemm:v1:64,64,64,1,1,4,16,1,2,0,0" --two-stage-topk=1 | FileCheck %s

// CHECK: {{gemm:v1:64,64,64,1,1,4,16,1,2,0,0[\t ]+[0-9].*}}
// CHECK-EMPTY:
