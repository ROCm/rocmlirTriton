// RUN: rocmlir-gen --operation conv_bwd_data -t f32 --arch %arch --num_cu 64 --num_chiplets 1 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 192 --in_h 64 --in_w 64 --out_channels 384 \
// RUN:   --fil_h 4 --fil_w 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   --perf_config=gemm:v4:128,128,4,1,1,2,0,1,3,0,0,-1,-1,-1,-1,-1,-1 \
// RUN:   -ph -pv --kernel-repeats=200 | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
