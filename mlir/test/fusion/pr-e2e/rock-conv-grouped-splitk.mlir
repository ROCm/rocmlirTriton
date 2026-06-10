// RUN: rocmlir-gen -pv --operation conv -t f16 --arch %arch --num_cu 64 --num_chiplets 1 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 2 --in_channels 128 --in_h 112 --in_w 28 --out_channels 32 --fil_h 2 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 3 --conv_stride_w 2 --padding_h_l 1 --padding_h_r 3 --padding_w_l 0 --padding_w_r 1 --groupsize 4 --perf_config="gemm:v1:128,64,16,1,1,8,16,4,2,1,2" | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
