// gemm+gemm f32, 64x64x64 with gemmO=64
// RUN: rocmlir-gen --arch %arch --operation gemm_gemm -t f32 -g 1 -m 64 -n 64 -k 64 -gemmO 64 -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// conv+gemm f16, NCHW layout
// RUN: rocmlir-gen --arch %arch --operation conv_gemm -t f16 --fil_layout k01c --in_layout nc01 --out_layout nk01 --batchsize 2 --in_channels 256 --in_h 32 --in_w 32 --out_channels 128 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 -gemmO 16 -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
