// Verifies the im2col vectorization hint (attached in TransformsToPointerArith,
// propagated through TensorToTritonPtr, and preserved across
// canonicalize-pointers by rock-bridge-vectorization-hints) actually widens the
// convolution input global load in the emitted AMDGCN.
//
// This is a forward conv whose im2col input address is built from divui/remui
// coordinate math. Without the hint, Triton's AxisInfoAnalysis cannot prove the
// fast axis contiguous and scalarizes the input to per-lane `buffer_load_dword`.
// With the hint (getMaxVectorization proves contiguity 4 for this 4-divisible
// output width on f32) the load widens to a 128-bit `buffer_load_dwordx4`; on
// gfx950 it additionally becomes a direct-to-LDS load (`... offen lds`).

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx950 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 1 --in_channels 4 --in_h 70 --in_w 70 --out_channels 64 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --groupsize 1 --perf_config=gemm:v2:64,256,16,1,1,16,32,1,2,0,0,-1,-1,-1,-1,-1,-1 \
// RUN:   | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX950
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 1 --in_channels 4 --in_h 70 --in_w 70 --out_channels 64 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --groupsize 1 --perf_config=gemm:v2:64,256,16,1,1,16,32,1,2,0,0,-1,-1,-1,-1,-1,-1 \
// RUN:   | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX942

// gfx950 (CDNA4) supports 128-bit direct-to-LDS, so the widened load is a
// buffer_load_dwordx4 straight into LDS.
// GFX950: buffer_load_dwordx4 {{.*}} offen lds

// gfx942 (CDNA3) direct-to-LDS is 32-bit-only, so the widened input load stays
// a 128-bit register load (still proving the hint defeated scalarization).
// GFX942: buffer_load_dwordx4
