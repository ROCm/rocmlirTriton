// RUN: rocmlir-gen -pv --operation conv -t i8 --arch %arch --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 8 --in_channels 128 --in_h 28 --in_w 28 --out_channels 128 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   --perf_config="gemm:v4:128,128,16,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,1" \
// RUN:   | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
