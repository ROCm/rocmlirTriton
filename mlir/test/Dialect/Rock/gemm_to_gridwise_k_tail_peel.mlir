// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Checks which shapes gemm-to-gridwise leaves K unpadded for, so that
// gridwise-gemm-to-blockwise peels the leftover K tail instead.
//
// Two things decide this. `useKTailPeeling` asks for peeling at all -- its
// default pads, as the pass did before the knob existed -- and `canPeelKTail`
// says whether peeling is legal for the shape. kPerBlock is 8 below, so every
// K that is not a multiple of 8 leaves a tail; what changes between the cases
// is the knob and the element type.

// RUN: rocmlir-opt -rock-gemm-to-gridwise -mlir-print-local-scope %s | FileCheck %s

#peel = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, useKTailPeeling = 1>
#peel_off = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, useKTailPeeling = 0>
#peel_unset = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#peel_splitk = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 2, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, useKTailPeeling = 1>

// The knob is set, the shape is peelable, so K stays 71.
// CHECK-LABEL: func.func @peel_f16_odd_k
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x71xf16>, tensor<1x71x64xf16>
func.func @peel_f16_odd_k(%a: tensor<1x71x64xf16>, %b: tensor<1x71x64xf16>, %c: tensor<1x64x64xf16>) -> tensor<1x64x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #peel
  } : tensor<1x71x64xf16> * tensor<1x71x64xf16> -> tensor<1x64x64xf16>
  %out = rock.store %result to %c by set : tensor<1x64x64xf16> -> tensor<1x64x64xf16> to tensor<1x64x64xf16>
  func.return %out : tensor<1x64x64xf16>
}

// Same for bf16.
// CHECK-LABEL: func.func @peel_bf16_odd_k
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x71xbf16>, tensor<1x71x64xbf16>
func.func @peel_bf16_odd_k(%a: tensor<1x71x64xbf16>, %b: tensor<1x71x64xbf16>, %c: tensor<1x64x64xbf16>) -> tensor<1x64x64xbf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #peel
  } : tensor<1x71x64xbf16> * tensor<1x71x64xbf16> -> tensor<1x64x64xbf16>
  %out = rock.store %result to %c by set : tensor<1x64x64xbf16> -> tensor<1x64x64xbf16> to tensor<1x64x64xbf16>
  func.return %out : tensor<1x64x64xbf16>
}

// Peelability is not enough: with the knob left at its default, the same shape
// takes the padded path, so K is rounded up to 72.
// CHECK-LABEL: func.func @no_peel_knob_unset
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x72xf16>, tensor<1x72x64xf16>
func.func @no_peel_knob_unset(%a: tensor<1x71x64xf16>, %b: tensor<1x71x64xf16>, %c: tensor<1x64x64xf16>) -> tensor<1x64x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #peel_unset
  } : tensor<1x71x64xf16> * tensor<1x71x64xf16> -> tensor<1x64x64xf16>
  %out = rock.store %result to %c by set : tensor<1x64x64xf16> -> tensor<1x64x64xf16> to tensor<1x64x64xf16>
  func.return %out : tensor<1x64x64xf16>
}

// An explicit 0 pads too.
// CHECK-LABEL: func.func @no_peel_knob_off
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x72xf16>, tensor<1x72x64xf16>
func.func @no_peel_knob_off(%a: tensor<1x71x64xf16>, %b: tensor<1x71x64xf16>, %c: tensor<1x64x64xf16>) -> tensor<1x64x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #peel_off
  } : tensor<1x71x64xf16> * tensor<1x71x64xf16> -> tensor<1x64x64xf16>
  %out = rock.store %result to %c by set : tensor<1x64x64xf16> -> tensor<1x64x64xf16> to tensor<1x64x64xf16>
  func.return %out : tensor<1x64x64xf16>
}

// Legality does not depend on the row alignment the tuner screens for, so an
// even K peels as readily as an odd one when asked. Only the tuning space
// knows that this one is unlikely to be worth picking.
// CHECK-LABEL: func.func @peel_f16_even_k
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x70xf16>, tensor<1x70x64xf16>
func.func @peel_f16_even_k(%a: tensor<1x70x64xf16>, %b: tensor<1x70x64xf16>, %c: tensor<1x64x64xf16>) -> tensor<1x64x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #peel
  } : tensor<1x70x64xf16> * tensor<1x70x64xf16> -> tensor<1x64x64xf16>
  %out = rock.store %result to %c by set : tensor<1x64x64xf16> -> tensor<1x64x64xf16> to tensor<1x64x64xf16>
  func.return %out : tensor<1x64x64xf16>
}

// Integer operands peel only when every tail segment is at least
// kMinIntegerKSegment wide. kRem = 68 % 8 = 4 is a single 4-wide segment, so
// this one is fine and K stays 68.
// CHECK-LABEL: func.func @peel_i8_wide_enough_tail
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x68xi8>, tensor<1x68x64xi8>
func.func @peel_i8_wide_enough_tail(%a: tensor<1x68x64xi8>, %b: tensor<1x68x64xi8>, %c: tensor<1x64x64xi32>) -> tensor<1x64x64xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #peel
  } : tensor<1x68x64xi8> * tensor<1x68x64xi8> -> tensor<1x64x64xi32>
  %out = rock.store %result to %c by set : tensor<1x64x64xi32> -> tensor<1x64x64xi32> to tensor<1x64x64xi32>
  func.return %out : tensor<1x64x64xi32>
}

// kRem = 70 % 8 = 6 decomposes into 4 + 2, and that 2-wide integer dot would
// round-trip the i32 accumulator through f32. Pad instead, even though 2-byte
// rows are exactly where peeling would otherwise pay for an 8-bit operand.
// CHECK-LABEL: func.func @no_peel_i8_narrow_segment
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x72xi8>, tensor<1x72x64xi8>
func.func @no_peel_i8_narrow_segment(%a: tensor<1x70x64xi8>, %b: tensor<1x70x64xi8>, %c: tensor<1x64x64xi32>) -> tensor<1x64x64xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #peel
  } : tensor<1x70x64xi8> * tensor<1x70x64xi8> -> tensor<1x64x64xi32>
  %out = rock.store %result to %c by set : tensor<1x64x64xi32> -> tensor<1x64x64xi32> to tensor<1x64x64xi32>
  func.return %out : tensor<1x64x64xi32>
}

// An odd K leaves a 1-wide segment, same problem.
// CHECK-LABEL: func.func @no_peel_i8_odd_k
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x72xi8>, tensor<1x72x64xi8>
func.func @no_peel_i8_odd_k(%a: tensor<1x71x64xi8>, %b: tensor<1x71x64xi8>, %c: tensor<1x64x64xi32>) -> tensor<1x64x64xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #peel
  } : tensor<1x71x64xi8> * tensor<1x71x64xi8> -> tensor<1x64x64xi32>
  %out = rock.store %result to %c by set : tensor<1x64x64xi32> -> tensor<1x64x64xi32> to tensor<1x64x64xi32>
  func.return %out : tensor<1x64x64xi32>
}

// A K that divides kPerBlock exactly has no tail to peel in the first place.
// CHECK-LABEL: func.func @no_peel_exact_k
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x72xf16>, tensor<1x72x64xf16>
func.func @no_peel_exact_k(%a: tensor<1x72x64xf16>, %b: tensor<1x72x64xf16>, %c: tensor<1x64x64xf16>) -> tensor<1x64x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #peel
  } : tensor<1x72x64xf16> * tensor<1x72x64xf16> -> tensor<1x64x64xf16>
  %out = rock.store %result to %c by set : tensor<1x64x64xf16> -> tensor<1x64x64xf16> to tensor<1x64x64xf16>
  func.return %out : tensor<1x64x64xf16>
}

// Split-K reshapes K across the split dimension, so it needs K to divide
// exactly and keeps the padding whatever the knob says: K is padded to 72,
// split in two, then each half padded again to 40.
// CHECK-LABEL: func.func @no_peel_splitk
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<2x64x40xf16>, tensor<2x40x64xf16>
func.func @no_peel_splitk(%a: tensor<1x71x64xf16>, %b: tensor<1x71x64xf16>, %c: tensor<1x64x64xf16>) -> tensor<1x64x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #peel_splitk
  } : tensor<1x71x64xf16> * tensor<1x71x64xf16> -> tensor<1x64x64xf16>
  %out = rock.store %result to %c by set : tensor<1x64x64xf16> -> tensor<1x64x64xf16> to tensor<1x64x64xf16>
  func.return %out : tensor<1x64x64xf16>
}
