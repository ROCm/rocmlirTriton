// Tests that the `-operation` flag path in `computeReductionK` computes the
// correct K_eff per convolution direction. The key fix: conv_bwd_weight uses
// K = batchsize * product(output_spatial), NOT the forward im2col K.
//
// All three tests use the same convolution shape so the only variable is
// the operation type:
//   batchsize=2, Cin=8, Cout=4, in=6x6, fil=3x3, stride=1, pad=0
//   => outH = (6-3)/1+1 = 4, outW = 4
//
// Expected K_eff per direction (f32, sumErrorTolerance = 1e-4, matching rocBLAS):
//   conv (forward):     K = Cin*filH*filW  = 8*3*3 = 72  => atol = 1e-5 + 72*1e-4  = 7.21e-3
//   conv_bwd_data:      K = Cin*filH*filW  = 72           => atol = 7.21e-3 (same as fwd)
//   conv_bwd_weight:    K = N*outH*outW    = 2*4*4 = 32   => atol = 1e-5 + 32*1e-4  = 3.21e-3

// ============================================================================
// (1) Forward convolution. K_eff = Cin * filH * filW = 72.
// ============================================================================

// RUN: rocmlir-gen -operation conv \
// RUN:   -batchsize=2 -in_channels=8 -out_channels=4 \
// RUN:   -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 \
// RUN:   -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 \
// RUN:   -padding_h=0 -padding_w=0 \
// RUN:   -t f32 --arch %arch -pv -rand 1 --comparator=allclose \
// RUN:   | FileCheck %s --check-prefix=CONV_FWD --enable-var-scope

// atol = 1e-5 + 72*1e-4 = 7.21e-3.
// CONV_FWD:      arith.constant 7.21{{[0-9]*}}e-03 : f32
// CONV_FWD-NEXT: arith.constant 1.300000e-06 : f32
// CONV_FWD:      call @mcpuVerifyFloatAllclose

// ============================================================================
// (2) Backward data. Same K_eff as forward: Cin * filH * filW = 72.
// ============================================================================

// RUN: rocmlir-gen -operation conv_bwd_data \
// RUN:   -batchsize=2 -in_channels=8 -out_channels=4 \
// RUN:   -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 \
// RUN:   -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 \
// RUN:   -padding_h=0 -padding_w=0 \
// RUN:   -t f32 --arch %arch -pv -rand 1 --comparator=allclose \
// RUN:   | FileCheck %s --check-prefix=CONV_BWD_DATA --enable-var-scope

// atol = 1e-5 + 72*1e-4 = 7.21e-3 (same as forward).
// CONV_BWD_DATA:      arith.constant 7.21{{[0-9]*}}e-03 : f32
// CONV_BWD_DATA-NEXT: arith.constant 1.300000e-06 : f32
// CONV_BWD_DATA:      call @mcpuVerifyFloatAllclose

// ============================================================================
// (3) Backward weight. K_eff = batchsize * outH * outW = 2*4*4 = 32.
// This is the critical test: before the fix, this would have used the forward
// K = 72 (giving atol = 7.21e-3). After the fix it correctly uses K = 32
// (giving atol = 3.21e-3).
// ============================================================================

// RUN: rocmlir-gen -operation conv_bwd_weight \
// RUN:   -batchsize=2 -in_channels=8 -out_channels=4 \
// RUN:   -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 \
// RUN:   -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 \
// RUN:   -padding_h=0 -padding_w=0 \
// RUN:   -t f32 --arch %arch -pv -rand 1 --comparator=allclose \
// RUN:   | FileCheck %s --check-prefix=CONV_BWD_WEIGHT --enable-var-scope

// The forward K atol (7.21e-3) must NOT appear — proves we don't use forward K.
// CONV_BWD_WEIGHT-NOT: arith.constant 7.21{{[0-9]*}}e-03 : f32
// atol = 1e-5 + 32*1e-4 = 3.21e-3.
// CONV_BWD_WEIGHT:      arith.constant 3.21{{[0-9]*}}e-03 : f32
// CONV_BWD_WEIGHT-NEXT: arith.constant 1.300000e-06 : f32
// CONV_BWD_WEIGHT:      call @mcpuVerifyFloatAllclose
