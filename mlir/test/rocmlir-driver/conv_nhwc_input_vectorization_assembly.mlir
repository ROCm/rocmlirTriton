// Checks that the image (input) operand of an NHWC forward convolution is
// loaded with wide vector loads rather than one element at a time.
//
// The way in which we emit the Merge affine map helps Triton's AxisInfo analyzer
// to properly understand the constancy of the index and vectorize the loads, so
// we expect to see vector loads for all the cases.
// 
// See https://github.com/ROCm/rocmlirTriton/pull/448 for more details.

// RUN: rocmlir-gen --operation conv -t f16 --arch gfx942 \
// RUN:   --fil_layout k01gc --in_layout 01ngc --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 112 --in_h 96 --in_w 96 --out_channels 112 \
// RUN:   --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 \
// RUN:   --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 \
// RUN:   --groupsize 1 -p \
// RUN:   | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CDNA --implicit-check-not=buffer_load_ushort

// RUN: rocmlir-gen --operation conv -t f16 --arch gfx950 \
// RUN:   --fil_layout k01gc --in_layout 01ngc --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 112 --in_h 96 --in_w 96 --out_channels 112 \
// RUN:   --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 \
// RUN:   --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 \
// RUN:   --groupsize 1 -p \
// RUN:   | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CDNA --implicit-check-not=buffer_load_ushort

// RUN: rocmlir-gen --operation conv -t f16 --arch gfx1100 \
// RUN:   --fil_layout k01gc --in_layout 01ngc --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 112 --in_h 96 --in_w 96 --out_channels 112 \
// RUN:   --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 \
// RUN:   --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 \
// RUN:   --groupsize 1 -p \
// RUN:   | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RDNA --implicit-check-not=buffer_load_u16

// CDNA: buffer_load_dwordx4

// RDNA: buffer_load_b128
