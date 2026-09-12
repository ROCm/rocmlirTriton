// Verifies the im2col vectorization hint widens the convolution input global
// load in the emitted AMDGCN. The hint is attached in TransformsToPointerArith,
// propagated through TensorToTritonPtr, then re-attached by patch11068.patch.
// Without it AxisInfoAnalysis cannot prove the fast axis contiguous through the
// im2col divui/remui address math and scalarizes the load to buffer_load_dword.

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx950 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 1 --in_channels 4 --in_h 70 --in_w 70 --out_channels 64 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --groupsize 1 --perf_config=gemm:v2:64,256,16,1,1,16,32,1,2,0,0,-1,-1,-1,-1,-1,-1 \
// RUN:   | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX950
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 1 --in_channels 4 --in_h 70 --in_w 70 --out_channels 64 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --groupsize 1 --perf_config=gemm:v2:64,256,16,1,1,16,32,1,2,0,0,-1,-1,-1,-1,-1,-1 \
// RUN:   | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX942

// gfx950 (CDNA4) lost direct-to-LDS eligibility to 9e52dc8c11's dot-operand
// hoist cost guard, so the widened input load is a register load here too.
// GFX950: buffer_load_dwordx4

// gfx942 (CDNA3) direct-to-LDS is 32-bit-only, so the widened input load stays
// a 128-bit register load.
// GFX942: buffer_load_dwordx4
