// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Checks which shapes gemm-to-gridwise leaves K unpadded for, so that
// gridwise-gemm-to-blockwise peels the leftover K tail instead.
//
// The gate (kTailPeelingPays) fires only when the operand rows land on exactly
// 2-byte alignment, i.e. when gemmK * sizeof(element) is divisible by 2 but
// not by 4. kPerBlock is 8 below, so K = 71 leaves a tail either way; what
// changes between the cases is the element type.

// RUN: rocmlir-opt -rock-gemm-to-gridwise -mlir-print-local-scope %s | FileCheck %s

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#gemm_params_splitk = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 2, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// 16-bit operands with an odd K: rows are 2-byte aligned, so peel. K stays 71.
// CHECK-LABEL: func.func @peel_f16_odd_k
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x71xf16>, tensor<1x71x64xf16>
func.func @peel_f16_odd_k(%a: tensor<1x71x64xf16>, %b: tensor<1x71x64xf16>, %c: tensor<1x64x64xf16>) -> tensor<1x64x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #gemm_params
  } : tensor<1x71x64xf16> * tensor<1x71x64xf16> -> tensor<1x64x64xf16>
  %out = rock.store %result to %c by set : tensor<1x64x64xf16> -> tensor<1x64x64xf16> to tensor<1x64x64xf16>
  func.return %out : tensor<1x64x64xf16>
}

// Same for bf16.
// CHECK-LABEL: func.func @peel_bf16_odd_k
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x71xbf16>, tensor<1x71x64xbf16>
func.func @peel_bf16_odd_k(%a: tensor<1x71x64xbf16>, %b: tensor<1x71x64xbf16>, %c: tensor<1x64x64xbf16>) -> tensor<1x64x64xbf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #gemm_params
  } : tensor<1x71x64xbf16> * tensor<1x71x64xbf16> -> tensor<1x64x64xbf16>
  %out = rock.store %result to %c by set : tensor<1x64x64xbf16> -> tensor<1x64x64xbf16> to tensor<1x64x64xbf16>
  func.return %out : tensor<1x64x64xbf16>
}

// 16-bit operands with an even K: rows are at least 4-byte aligned, so the
// mask costs nothing and peeling would only add the tail. K is padded to 72.
// CHECK-LABEL: func.func @no_peel_f16_even_k
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x72xf16>, tensor<1x72x64xf16>
func.func @no_peel_f16_even_k(%a: tensor<1x70x64xf16>, %b: tensor<1x70x64xf16>, %c: tensor<1x64x64xf16>) -> tensor<1x64x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #gemm_params
  } : tensor<1x70x64xf16> * tensor<1x70x64xf16> -> tensor<1x64x64xf16>
  %out = rock.store %result to %c by set : tensor<1x64x64xf16> -> tensor<1x64x64xf16> to tensor<1x64x64xf16>
  func.return %out : tensor<1x64x64xf16>
}

// 32-bit operands are always at least 4-byte aligned, whatever K is.
// CHECK-LABEL: func.func @no_peel_f32_odd_k
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x72xf32>, tensor<1x72x64xf32>
func.func @no_peel_f32_odd_k(%a: tensor<1x71x64xf32>, %b: tensor<1x71x64xf32>, %c: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #gemm_params
  } : tensor<1x71x64xf32> * tensor<1x71x64xf32> -> tensor<1x64x64xf32>
  %out = rock.store %result to %c by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  func.return %out : tensor<1x64x64xf32>
}

// 8-bit operands reach 2-byte alignment at K = 2 (mod 4), which is even.
// CHECK-LABEL: func.func @peel_i8_k_two_mod_four
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x70xi8>, tensor<1x70x64xi8>
func.func @peel_i8_k_two_mod_four(%a: tensor<1x70x64xi8>, %b: tensor<1x70x64xi8>, %c: tensor<1x64x64xi32>) -> tensor<1x64x64xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #gemm_params
  } : tensor<1x70x64xi8> * tensor<1x70x64xi8> -> tensor<1x64x64xi32>
  %out = rock.store %result to %c by set : tensor<1x64x64xi32> -> tensor<1x64x64xi32> to tensor<1x64x64xi32>
  func.return %out : tensor<1x64x64xi32>
}

// An odd K puts 8-bit operands at 1-byte alignment, where the loop is so
// load-bound that removing the mask buys nothing. Keep the padding.
// CHECK-LABEL: func.func @no_peel_i8_odd_k
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x72xi8>, tensor<1x72x64xi8>
func.func @no_peel_i8_odd_k(%a: tensor<1x71x64xi8>, %b: tensor<1x71x64xi8>, %c: tensor<1x64x64xi32>) -> tensor<1x64x64xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #gemm_params
  } : tensor<1x71x64xi8> * tensor<1x71x64xi8> -> tensor<1x64x64xi32>
  %out = rock.store %result to %c by set : tensor<1x64x64xi32> -> tensor<1x64x64xi32> to tensor<1x64x64xi32>
  func.return %out : tensor<1x64x64xi32>
}

// A K that divides kPerBlock exactly has no tail to peel in the first place.
// CHECK-LABEL: func.func @no_peel_exact_k
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<1x64x72xf16>, tensor<1x72x64xf16>
func.func @no_peel_exact_k(%a: tensor<1x72x64xf16>, %b: tensor<1x72x64xf16>, %c: tensor<1x64x64xf16>) -> tensor<1x64x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #gemm_params
  } : tensor<1x72x64xf16> * tensor<1x72x64xf16> -> tensor<1x64x64xf16>
  %out = rock.store %result to %c by set : tensor<1x64x64xf16> -> tensor<1x64x64xf16> to tensor<1x64x64xf16>
  func.return %out : tensor<1x64x64xf16>
}

// Split-K reshapes K across the split dimension, so it needs K to divide
// exactly and keeps the padding even where the gate would otherwise fire:
// K is padded to 72, split in two, then each half padded again to 40.
// CHECK-LABEL: func.func @no_peel_splitk
// CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) {{.*}} : tensor<2x64x40xf16>, tensor<2x40x64xf16>
func.func @no_peel_splitk(%a: tensor<1x71x64xf16>, %b: tensor<1x71x64xf16>, %c: tensor<1x64x64xf16>) -> tensor<1x64x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %result = rock.gemm tr %a * %b {
    params = #gemm_params_splitk
  } : tensor<1x71x64xf16> * tensor<1x71x64xf16> -> tensor<1x64x64xf16>
  %out = rock.store %result to %c by set : tensor<1x64x64xf16> -> tensor<1x64x64xf16> to tensor<1x64x64xf16>
  func.return %out : tensor<1x64x64xf16>
}
