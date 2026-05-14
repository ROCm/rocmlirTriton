// fwd conv f16 -> f32, NCHW layout
// RUN: rocmlir-gen --arch %arch --operation conv -t f16 -out_datatype f32 --fil_layout k01c --in_layout nc01 --out_layout nk01 --batchsize 1 --in_channels 64 --in_h 32 --in_w 32 --out_channels 32 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 --kernel-repeats 1 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// bwd-weight conv f32 (asymmetric padding)
// RUN: rocmlir-gen --arch=%arch --operation conv_bwd_weight -t f32 -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -groupsize=1 -batchsize=64 -in_channels=64 -out_channels=64 -in_h=4 -in_w=4 -fil_h=2 -fil_w=2 -dilation_h=1 -dilation_w=1 -conv_stride_h=2 -conv_stride_w=2 -padding_h_l=2 -padding_h_r=1 -padding_w_l=2 -padding_w_r=0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// bwd-data conv f32 -- exercises multiple gemms within one kernel
// RUN: rocmlir-gen --arch=%arch --operation conv_bwd_data -t f32 -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -groupsize=1 -batchsize=64 -in_channels=64 -out_channels=64 -in_h=4 -in_w=4 -fil_h=2 -fil_w=2 -dilation_h=1 -dilation_w=1 -conv_stride_h=2 -conv_stride_w=2 -padding_h_l=2 -padding_h_r=1 -padding_w_l=2 -padding_w_r=0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
