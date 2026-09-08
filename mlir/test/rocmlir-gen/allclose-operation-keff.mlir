// Tests that the `-operation` flag path in `computeReductionK` computes the
// correct K_eff for forward and backward-data convolutions.
//
// The 2D tests use the same convolution shape so the only variable is the
// operation type:
//   batchsize=2, Cin=8, Cout=4, in=6x6, fil=3x3, stride=1, pad=0
//   => outH = (6-3)/1+1 = 4, outW = 4
//
// Expected K_eff (f32, sumErrorTolerance = 1e-5 for random data):
//   conv (forward): K = Cin*filH*filW = 8*3*3 = 72 => atol = 7.3e-4
//   conv_bwd_data:  K = Cin*filH*filW = 72         => atol = 7.3e-4
//
// The 3D forward test verifies that the depth dimension is included in K_eff.

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

// atol = 1e-5 + 72*1e-5 = 7.3e-4.
// CONV_FWD:      arith.constant 7.3{{[0-9]*}}e-04 : f32
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

// atol = 1e-5 + 72*1e-5 = 7.3e-4 (same as forward).
// CONV_BWD_DATA:      arith.constant 7.3{{[0-9]*}}e-04 : f32
// CONV_BWD_DATA-NEXT: arith.constant 1.300000e-06 : f32
// CONV_BWD_DATA:      call @mcpuVerifyFloatAllclose

// ============================================================================
// (3) 3D forward convolution. K_eff = Cin * filD * filH * filW = 4*3*3*3 = 108.
// Verifies that the depth dimension is included in the K_eff product.
// ============================================================================

// RUN: rocmlir-gen -operation conv \
// RUN:   -fil_layout=gkc012 -in_layout=ngc012 -out_layout=ngk012 \
// RUN:   -batchsize=2 -groupsize=1 -in_channels=4 -out_channels=4 \
// RUN:   -in_d=4 -in_h=4 -in_w=4 -fil_d=3 -fil_h=3 -fil_w=3 \
// RUN:   --conv_stride_d=1 --conv_stride_h=1 --conv_stride_w=1 \
// RUN:   --dilation_d=1 --dilation_h=1 --dilation_w=1 \
// RUN:   --padding_d=0 --padding_h=0 --padding_w=0 \
// RUN:   -t f32 --arch %arch -pv -rand 1 --comparator=allclose \
// RUN:   | FileCheck %s --check-prefix=CONV3D_FWD --enable-var-scope

// The 2D forward atol (7.3e-4) must NOT appear — proves depth is included.
// CONV3D_FWD-NOT: arith.constant 7.3{{[0-9]*}}e-04 : f32
// atol = 1e-5 + 108*1e-5 = 1.09e-3 (decimal: 0.00108999992).
// CONV3D_FWD:      arith.constant 0.00108{{[0-9]*}} : f32
// CONV3D_FWD-NEXT: arith.constant 1.300000e-06 : f32
// CONV3D_FWD:      call @mcpuVerifyFloatAllclose
