// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file -rock-gridwise-attn-to-blockwise -verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: func @mlir_attention
  // Causal look-back keeps causal's upper bound and adds a per-row lower bound,
  // so the valid region is the band max(0, query - L) <= key <= query.

  // The loop start is derived from the block's first query row, so the block's
  // last rows can find that first tile entirely below their own lower bound.
  // Such a row is fully masked on its first processed tile, which needs the
  // empty-row guard: the running row max starts at the lowest finite value
  // rather than -inf, so every exp2 term of an empty row is 0 instead of NaN.
  // CHECK: %[[MAX_INIT:.*]] = arith.constant dense<-3.40282347E+38> : tensor<32xf32>

  // The causal upper bound is unchanged; it clamps the last N-block.
  // CHECK: %[[END:.*]] = arith.minui

  // N-loop start: the band's lower edge for this block's first query row,
  // rounded down to an N-block. Every key below it is outside the band for
  // every query in the block, so those N-blocks are skipped entirely. L = 16,
  // mPerBlock = nPerBlock = 32.
  // CHECK: %[[MIN_ROW:.*]] = arith.muli %{{.*}}, %c32{{.*}} : i32
  // CHECK: %[[RAW_LB:.*]] = arith.subi %[[MIN_ROW]], %c16{{.*}} : i32
  // CHECK: %[[LB:.*]] = arith.maxsi %[[RAW_LB]], %c0{{.*}} : i32
  // CHECK: %[[LB_START:.*]] = arith.divui %[[LB]], %c32{{.*}} : i32
  // CHECK: %[[START:.*]] = arith.maxui %{{.*}}, %[[LB_START]] : i32

  // The sentinel above is this loop's running-max accumulator.
  // CHECK: scf.for %{{.*}} = %[[START]] to %[[END]] step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %[[MAX_INIT]],

  // Per-element mask. The lower bound is a tensor, not a splatted scalar,
  // because it moves with the query row: max(0, mIndex - L).
  // CHECK: %[[L_SPLAT:.*]] = tt.splat %c16{{.*}} : i32 -> tensor<32x32xi32>
  // CHECK: %[[ZERO_SPLAT:.*]] = tt.splat %c0{{.*}} : i32 -> tensor<32x32xi32>
  // CHECK: %[[RAW_ROW_LB:.*]] = arith.subi %{{.*}}, %[[L_SPLAT]] : tensor<32x32xi32>
  // CHECK: %[[ROW_LB:.*]] = arith.maxsi %[[RAW_ROW_LB]], %[[ZERO_SPLAT]] : tensor<32x32xi32>

  // Both band edges are folded into one predicate, so the standalone causal
  // mask is not emitted a second time.
  // CHECK: %[[TOO_OLD:.*]] = arith.cmpi ult, %{{.*}}, %[[ROW_LB]] : tensor<32x32xi32>
  // CHECK: %[[IN_FUTURE:.*]] = arith.cmpi ugt, %{{.*}}, %{{.*}} : tensor<32x32xi32>
  // CHECK: %[[INVALID:.*]] = arith.ori %[[TOO_OLD]], %[[IN_FUTURE]] : tensor<32x32xi1>
  // CHECK: arith.select %[[INVALID]], %{{.*}}, %{{.*}} : tensor<32x32xi1>, tensor<32x32xf32>

  // An empty row ends with sumRow == 0, so the denominator is clamped to 1 and
  // the row is written as zeros instead of 0/0.
  // CHECK: %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
  // CHECK: arith.maxnumf %{{.*}}, %[[ONE]] : tensor<32xf32>

  func.func @mlir_attention(
      %q: tensor<1x64x32xf16>,
      %k: tensor<1x32x64xf16>,
      %v: tensor<1x64x32xf16>) -> tensor<1x64x32xf16>
      attributes {
        rock.block_size = 256 : i32,
        rock.grid_size = 2 : i32,
        rock.kernel,
        rock.arch = "##TOKEN_ARCH##"
      } {
    %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
    ^bb0(%arg_qk: tensor<1x32x32xf16>):
      %cst = arith.constant dense<1.250000e-01> : tensor<1x32x32xf16>
      %scaled = arith.mulf %arg_qk, %cst : tensor<1x32x32xf16>
      rock.yield %scaled : tensor<1x32x32xf16>
    } {
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
      causal,
      causalLookBack = 16 : i32,
      softmaxType = f32,
      splitKV = 1 : i32,
      params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
      params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
    } : tensor<1x64x32xf16>, tensor<1x32x64xf16>, tensor<1x64x32xf16> -> tensor<1x64x32xf16>
    return %result : tensor<1x64x32xf16>
  }
}

// -----

module {
  // Under GQA the M axis packs numRepeatsGQA query heads, so an M index maps to
  // query row M / numRepeatsGQA. The N-loop lower bound must convert before
  // subtracting L, or it skips key blocks that are still inside the band.
  // CHECK-LABEL: func @attn_causal_look_back_gqa
  // CHECK: %[[END:.*]] = arith.minui
  // CHECK: %[[MIN_ROW:.*]] = arith.muli %{{.*}}, %c32{{.*}} : i32
  // CHECK: %[[ROW:.*]] = arith.divui %[[MIN_ROW]], %c2{{.*}} : i32
  // CHECK: %[[RAW_LB:.*]] = arith.subi %[[ROW]], %c16{{.*}} : i32
  // CHECK: %[[LB:.*]] = arith.maxsi %[[RAW_LB]], %c0{{.*}} : i32
  // CHECK: %[[LB_START:.*]] = arith.divui %[[LB]], %c32{{.*}} : i32
  // CHECK: %[[START:.*]] = arith.maxui %{{.*}}, %[[LB_START]] : i32
  // CHECK: scf.for %{{.*}} = %[[START]] to %[[END]] step
  func.func @attn_causal_look_back_gqa(
      %q: tensor<1x64x32xf16>,
      %k: tensor<1x32x64xf16>,
      %v: tensor<1x64x32xf16>) -> tensor<1x64x32xf16>
      attributes {
        rock.block_size = 256 : i32,
        rock.grid_size = 2 : i32,
        rock.kernel,
        rock.arch = "##TOKEN_ARCH##"
      } {
    %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
    ^bb0(%arg_qk: tensor<1x32x32xf16>):
      %cst = arith.constant dense<1.250000e-01> : tensor<1x32x32xf16>
      %scaled = arith.mulf %arg_qk, %cst : tensor<1x32x32xf16>
      rock.yield %scaled : tensor<1x32x32xf16>
    } {
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
      causal,
      causalLookBack = 16 : i32,
      numRepeatsGQA = 2 : index,
      softmaxType = f32,
      splitKV = 1 : i32,
      params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
      params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
    } : tensor<1x64x32xf16>, tensor<1x32x64xf16>, tensor<1x64x32xf16> -> tensor<1x64x32xf16>
    return %result : tensor<1x64x32xf16>
  }
}

// -----

module {
  // Plain causal, no look-back: the N-loop still starts at block 0, no per-row
  // lower bound is computed, and no fully masked row is possible, so the
  // empty-row guard stays off.
  // CHECK-LABEL: func @attn_causal_no_look_back
  // CHECK: %[[MAX_INIT:.*]] = arith.constant dense<0xFF800000> : tensor<32xf32>
  // CHECK: scf.for %{{.*}} = %c0{{.*}} to %{{.*}} step %c1
  // CHECK-NOT: arith.ori
  // CHECK-NOT: arith.maxnumf
  // CHECK: return
  func.func @attn_causal_no_look_back(
      %q: tensor<1x64x32xf16>,
      %k: tensor<1x32x64xf16>,
      %v: tensor<1x64x32xf16>) -> tensor<1x64x32xf16>
      attributes {
        rock.block_size = 256 : i32,
        rock.grid_size = 2 : i32,
        rock.kernel,
        rock.arch = "##TOKEN_ARCH##"
      } {
    %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
    ^bb0(%arg_qk: tensor<1x32x32xf16>):
      %cst = arith.constant dense<1.250000e-01> : tensor<1x32x32xf16>
      %scaled = arith.mulf %arg_qk, %cst : tensor<1x32x32xf16>
      rock.yield %scaled : tensor<1x32x32xf16>
    } {
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
      causal,
      softmaxType = f32,
      splitKV = 1 : i32,
      params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
      params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
    } : tensor<1x64x32xf16>, tensor<1x32x64xf16>, tensor<1x64x32xf16> -> tensor<1x64x32xf16>
    return %result : tensor<1x64x32xf16>
  }
}
