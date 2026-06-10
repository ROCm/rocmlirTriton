// RUN: rocmlir-gen --arch %arch --operation conv_bwd_weight -t bf16 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 1 --in_channels 3 --in_h 56 --in_w 32 --out_channels 1 --fil_h 2 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h_l 0 --padding_h_r 3 --padding_w 0 --groupsize 1 --perf_config=gemm:v1:256,256,32,1,1,8,16,1,2,2,2 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
