// Regression for a runtime memory crash in the generated i8
// convolution. Without llvm-patches/patch208045.patch, executing the kernel
// produced for this pinned shape and gemm:v4 perf config with rocm-run crashes.
// The upstream fix replaces true16 i16-to-i32 zero-extension REG_SEQUENCE
// patterns with V_CVT_U32_U16.

// RUN: rocmlir-gen -pv --operation conv -t i8 --arch %arch --num_cu 64 --num_chiplets 1 \
// RUN:   --fil_layout kcyx --in_layout nchw --out_layout nkhw \
// RUN:   --batchsize 64 --in_channels 1536 --in_h 7 --in_w 7 \
// RUN:   --out_channels 2048 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 \
// RUN:   --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 \
// RUN:   --groupsize 1 \
// RUN:   --perf_config=gemm:v4:128,256,32,1,1,4,0,1,3,0,0,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver -c | rocm-run | FileCheck %s
//
// CHECK: [1 1 1]
