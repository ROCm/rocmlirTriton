// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file -rock-gridwise-attn-to-blockwise -canonicalize -verify-diagnostics | FileCheck %s --check-prefixes=CHECK,RISK,F16,SCALE,EXTSCALE,CAUSAL,SPLIT
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file -rock-gridwise-attn-to-blockwise -canonicalize -verify-diagnostics | FileCheck %s --check-prefix=NO-CMP

// CHECK-LABEL: func @gridwise_attn_simple
// CHECK-SAME: (%[[Q:.+]]: tensor<1x384x64xf32>, %[[K:.+]]: tensor<1x64x384xf32>, %[[V:.+]]: tensor<1x384x64xf32>)

// Constants: ln2 = log2(e) softmax scale; zero32/zero64/zeroRow/negInf are
// the flash-attention accumulator inits; c12 = gemmN/nPerBlock outer count,
// c4 = gemmK/kPerBlock inner count, c24 = grid_size (block-coord divisor).
// CHECK-DAG: %[[ln2:.+]] = arith.constant dense<1.44269502> : tensor<16x32xf32>
// CHECK-DAG: %[[zero32:.+]] = arith.constant dense<0.000000e+00> : tensor<16x32xf32>
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[zero64:.+]] = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
// CHECK-DAG: %[[c12:.+]] = arith.constant 12 : i32
// CHECK-DAG: %[[zeroRow:.+]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK-DAG: %[[negInf:.+]] = arith.constant dense<0xFF800000> : tensor<16xf32>
// CHECK-DAG: %[[c24:.+]] = arith.constant 24 : i32
// CHECK-DAG: %[[c4:.+]] = arith.constant 4 : i32
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : i32

// Triton-style block id.
// CHECK: %[[PID:.+]] = tt.get_program_id x : i32

// Outer N-tile loop, iter_args = (attn_output, running_max, running_sum).
// CHECK: %[[OUT:.+]]:3 = scf.for %{{.*}} = %[[c0]] to %[[c12]] step %[[c1]] iter_args(%[[acc:.+]] = %[[zero64]], %[[rmax:.+]] = %[[negInf]], %[[rsum:.+]] = %[[zeroRow]])

// Per-iteration block-M / block-N from PID.
// CHECK: arith.divui %{{.*}}, %[[c24]] : i32
// CHECK: arith.remui %{{.*}}, %[[c24]] : i32

// gemm0 = Q @ K^T : inner KpacksPerBlock loop accumulating into zero32.
// CHECK: %[[G0:.+]] = scf.for %{{.*}} = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[g0acc:.+]] = %[[zero32]])
// CHECK: %[[qTile:.+]] = rock.load_marker %[[Q]] views [#{{.*}}]
// CHECK: %[[kTile:.+]] = rock.load_marker %[[K]] views [#{{.*}}]
// CHECK: %[[g0gemm:.+]] = rock.blockwise_gemm(%[[qTile]], %[[kTile]], %[[g0acc]])
// CHECK: scf.yield %[[g0gemm]]

// Softmax: scale by ln2, online row-max, exp2(scores - newMax), row-sum.
// CHECK: %[[scaled:.+]] = arith.mulf %[[G0]], %[[ln2]]
// CHECK: %[[tileMax:.+]] = rock.blockwise_reduce max %[[scaled]] {axis = 1 : index}
// CHECK: %[[newMax:.+]] = arith.maximumf %[[rmax]], %[[tileMax]]
// CHECK: %[[maxExp:.+]] = tt.expand_dims %[[newMax]]
// CHECK: %[[maxBcast:.+]] = tt.broadcast %[[maxExp]]
// CHECK: %[[sub:.+]] = arith.subf %[[scaled]], %[[maxBcast]]
// CHECK: %[[P:.+]] = math.exp2 %[[sub]]
// CHECK: %[[tileSum:.+]] = rock.blockwise_reduce sum %[[P]] {axis = 1 : index}

// Row-sum rescale: l = exp2(m_{j-1} - m_j) * l_{j-1} + rowsum(P).
// CHECK: %[[newMax2:.+]] = arith.maximumf %[[rmax]], %[[tileMax]]
// CHECK: %[[maxDiff:.+]] = arith.subf %[[rmax]], %[[newMax2]]
// CHECK: %[[corr:.+]] = math.exp2 %[[maxDiff]]
// CHECK: %[[sumScaled:.+]] = arith.mulf %[[corr]], %[[rsum]]
// CHECK: %[[newSum:.+]] = arith.addf %[[sumScaled]], %[[tileSum]]

// gemm1 = P @ V.
// CHECK: arith.divui %{{.*}}, %[[c24]] : i32
// CHECK: arith.remui %{{.*}}, %[[c24]] : i32
// CHECK: %[[vTile:.+]] = rock.load_marker %[[V]] views [#{{.*}}]
// CHECK: %[[g1gemm:.+]] = rock.blockwise_gemm(%[[P]], %[[vTile]], %[[zero64]])

// Output rescale: acc = acc * corr + gemm1.
// CHECK: %[[corrExp:.+]] = tt.expand_dims %[[corr]]
// CHECK: %[[corrBcast:.+]] = tt.broadcast %[[corrExp]]
// CHECK: %[[accScaled:.+]] = arith.mulf %[[acc]], %[[corrBcast]]
// CHECK: %[[newAcc:.+]] = arith.addf %[[accScaled]], %[[g1gemm]]
// CHECK: scf.yield %[[newAcc]], %[[newMax2]], %[[newSum]]

// Final normalize by row-sum and store.
// CHECK-NOT: arith.maxnumf
// CHECK: %[[sumExp:.+]] = tt.expand_dims %[[OUT]]#2
// CHECK: %[[sumBcast:.+]] = tt.broadcast %[[sumExp]]
// CHECK: %[[norm:.+]] = arith.divf %[[OUT]]#0, %[[sumBcast]]
// CHECK: arith.divui %{{.*}}, %[[c24]] : i32
// CHECK: arith.remui %{{.*}}, %[[c24]] : i32
// CHECK: rock.store_marker %[[norm]] views [#{{.*}}]
func.func @gridwise_attn_simple(
    %q: tensor<1x384x64xf32>,
    %k: tensor<1x64x384xf32>,
    %v: tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
    attributes {
      rock.block_size = 64 : i32,
      rock.grid_size = 24 : i32,
      rock.kernel,
      rock.arch = "##TOKEN_ARCH##"
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}

// -----

// Causal plus sliding-window masking can produce a fully masked row. It uses a
// finite row-max sentinel and one row-vector denominator clamp, without the
// score-tile floating compare/select guards from PR 416.
// NO-CMP-LABEL: func @gridwise_attn_causal_sliding
// NO-CMP-NOT: arith.cmpf
// NO-CMP: return
// RISK-LABEL: func @gridwise_attn_causal_sliding
// RISK-DAG: %[[LOWEST:.+]] = arith.constant dense<-3.40282347E+38> : tensor<16xf32>
// RISK-DAG: %[[ZERO_ROW:.+]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// RISK-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<16xf32>
// RISK: %[[OUT:.+]]:3 = scf.for
// RISK-SAME: %[[RMAX:.+]] = %[[LOWEST]]
// RISK-SAME: %[[RSUM:.+]] = %[[ZERO_ROW]]
// RISK: %[[SUB:.+]] = arith.subf
// RISK-NEXT: %[[P:.+]] = math.exp2 %[[SUB]]
// RISK: %[[SAFE_SUM:.+]] = arith.maxnumf %[[OUT]]#2, %[[ONE]]
// RISK: %[[SUM_EXP:.+]] = tt.expand_dims %[[SAFE_SUM]]
// RISK: arith.divf
func.func @gridwise_attn_causal_sliding(
    %q: tensor<1x384x64xf32>,
    %k: tensor<1x64x384xf32>,
    %v: tensor<1x384x64xf32>,
    %lastValidKVIndex: tensor<1xi32>) -> tensor<1x384x64xf32>
    attributes {
      rock.block_size = 64 : i32,
      rock.grid_size = 24 : i32,
      rock.kernel,
      rock.arch = "##TOKEN_ARCH##"
    } {
  %result = rock.gridwise_attention(
      %q, %k, %v, %lastValidKVIndex) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    causal,
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    slidingWindowLookBack = 1 : i32,
    splitKV = 1 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32>, tensor<1xi32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}

// -----

// The finite sentinel follows the actual softmax type rather than using an
// f32 literal that would round to -inf in f16.
// F16-LABEL: func @gridwise_attn_causal_sliding_f16
// F16: arith.constant dense<{{(-6.550400e\+04|0xFBFF)}}> : tensor<16xf16>
// F16: arith.maxnumf
func.func @gridwise_attn_causal_sliding_f16(
    %q: tensor<1x384x64xf16>,
    %k: tensor<1x64x384xf16>,
    %v: tensor<1x384x64xf16>,
    %lastValidKVIndex: tensor<1xi32>) -> tensor<1x384x64xf16>
    attributes {
      rock.block_size = 64 : i32,
      rock.grid_size = 24 : i32,
      rock.kernel,
      rock.arch = "##TOKEN_ARCH##"
    } {
  %result = rock.gridwise_attention(
      %q, %k, %v, %lastValidKVIndex) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf16>):
    rock.yield %arg_qk : tensor<1x16x32xf16>
  } {
    causal,
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    slidingWindowLookBack = 1 : i32,
    softmaxType = f16,
    splitKV = 1 : i32
  } : tensor<1x384x64xf16>, tensor<1x64x384xf16>, tensor<1x384x64xf16>, tensor<1xi32> -> tensor<1x384x64xf16>
  return %result : tensor<1x384x64xf16>
}

// -----

// Identity plus finite compile-time scaling is statically safe: retain the
// original -inf running maximum and add no denominator clamp.
// SCALE-LABEL: func @gridwise_attn_finite_scale
// SCALE: arith.constant dense<0xFF800000> : tensor<16xf32>
// SCALE-NOT: arith.maxnumf
// SCALE: return
func.func @gridwise_attn_finite_scale(
    %q: tensor<1x384x64xf32>,
    %k: tensor<1x64x384xf32>,
    %v: tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
    attributes {
      rock.block_size = 64 : i32,
      rock.grid_size = 24 : i32,
      rock.kernel,
      rock.arch = "##TOKEN_ARCH##"
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    %scale = arith.constant dense<1.250000e-01> : tensor<1x16x32xf32>
    %scaled = arith.mulf %arg_qk, %scale : tensor<1x16x32xf32>
    rock.yield %scaled : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}

// -----

// Widening the QK value before finite constant scaling is also statically safe.
// This is the shape produced by f16 MIGraphX attention with f32 softmax.
// EXTSCALE-LABEL: func @gridwise_attn_finite_ext_scale
// EXTSCALE: arith.constant dense<0xFF800000> : tensor<16xf32>
// EXTSCALE-NOT: arith.maxnumf
// EXTSCALE: return
func.func @gridwise_attn_finite_ext_scale(
    %q: tensor<1x384x64xf16>,
    %k: tensor<1x64x384xf16>,
    %v: tensor<1x384x64xf16>) -> tensor<1x384x64xf16>
    attributes {
      rock.block_size = 64 : i32,
      rock.grid_size = 24 : i32,
      rock.kernel,
      rock.arch = "##TOKEN_ARCH##"
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf16>):
    %extended = arith.extf %arg_qk : tensor<1x16x32xf16> to tensor<1x16x32xf32>
    %scale = arith.constant dense<1.250000e-01> : tensor<1x16x32xf32>
    %scaled = arith.mulf %extended, %scale : tensor<1x16x32xf32>
    rock.yield %scaled : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    softmaxType = f32,
    splitKV = 1 : i32
  } : tensor<1x384x64xf16>, tensor<1x64x384xf16>, tensor<1x384x64xf16> -> tensor<1x384x64xf16>
  return %result : tensor<1x384x64xf16>
}

// -----

// Causal-only attention always retains key zero and needs no empty-row guard.
// CAUSAL-LABEL: func @gridwise_attn_causal_only
// CAUSAL: arith.constant dense<0xFF800000> : tensor<16xf32>
// CAUSAL-NOT: arith.maxnumf
// CAUSAL: return
func.func @gridwise_attn_causal_only(
    %q: tensor<1x384x64xf32>,
    %k: tensor<1x64x384xf32>,
    %v: tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
    attributes {
      rock.block_size = 64 : i32,
      rock.grid_size = 24 : i32,
      rock.kernel,
      rock.arch = "##TOKEN_ARCH##"
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    causal,
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}

// -----

// A split-KV partial can contain no keys and therefore needs the guard.
// SPLIT-LABEL: func @gridwise_attn_split_kv
// SPLIT: arith.constant dense<-3.40282347E+38> : tensor<64xf32>
// SPLIT: arith.maxnumf
// SPLIT: return
func.func @gridwise_attn_split_kv(
    %q: tensor<1x64x32xf32>,
    %k: tensor<1x32x128xf32>,
    %v: tensor<1x128x32xf32>) -> tensor<2x64x32xf32>
    attributes {
      rock.block_size = 256 : i32,
      rock.grid_size = 2 : i32,
      rock.kernel,
      rock.arch = "##TOKEN_ARCH##"
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  } {
    causal,
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 3, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 32, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 3, wavesPerEU = 0, gridGroupSize = 0>,
    softmaxType = f32,
    splitKV = 2 : i32
  } : tensor<1x64x32xf32>, tensor<1x32x128xf32>, tensor<1x128x32xf32> -> tensor<2x64x32xf32>
  return %result : tensor<2x64x32xf32>
}

