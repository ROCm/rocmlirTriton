// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-gridwise-attn-to-blockwise -verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: func @mlir_attention
  // Runtime sliding-window masking on the KV-cache N-loop. The lower bound is
  // slidingWindowLowerBound = max(0, currentSeqLen - slidingWindowSize); key
  // positions (nIndex) below it are masked with -inf.

  // currentSeqLen is loaded from the tensor as a scalar i32.
  // CHECK: %[[SEQLEN:.*]] = tt.unsplat %{{.*}} : tensor<1xi32>
  // slidingWindowLowerBound = max(0, currentSeqLen - slidingWindowSize)
  // CHECK: %[[SEQ_MINUS_WINDOW:.*]] = arith.subi %[[SEQLEN]], %c3{{.*}} : i32
  // CHECK: %[[LOWER_BOUND:.*]] = arith.maxsi %[[SEQ_MINUS_WINDOW]], %c0{{.*}} : i32

  // The N-loop start iteration skips blocks fully below the lower bound:
  // start = max(0, floor(slidingWindowLowerBound / nPerBlock)).
  // CHECK: %[[SW_START:.*]] = arith.divui %[[LOWER_BOUND]], %c32{{.*}} : i32
  // CHECK: %[[START:.*]] = arith.maxsi %c0{{.*}}, %[[SW_START]] : i32
  // CHECK: scf.for %{{.*}} = %[[START]] to %{{.*}} step %c1

  // Sliding-window masking: nIndex < slidingWindowLowerBound is set to -inf.
  // CHECK: %[[LB_SPLAT:.*]] = tt.splat %[[LOWER_BOUND]] : i32 -> tensor<32x32xi32>
  // CHECK: %[[SW_MASK:.*]] = arith.cmpi ult, %{{.*}}, %[[LB_SPLAT]] : tensor<32x32xi32>
  // CHECK: arith.select %[[SW_MASK]], %{{.*}}, %{{.*}} : tensor<32x32xi1>, tensor<32x32xf32>

  func.func @mlir_attention(
      %currentSeqLen: tensor<1xi32>,
      %q: tensor<1x64x32xf16>,
      %k: tensor<1x32x64xf16>,
      %v: tensor<1x64x32xf16>) -> tensor<1x64x32xf16>
      attributes {
        rock.block_size = 256 : i32,
        rock.grid_size = 2 : i32,
        rock.kernel,
        rock.arch = "##TOKEN_ARCH##"
      } {
    %result = rock.gridwise_attention(%q, %k, %v, %currentSeqLen) preSoftmaxOps = {
    ^bb0(%arg_qk: tensor<1x32x32xf16>):
      %cst = arith.constant dense<1.250000e-01> : tensor<1x32x32xf16>
      %scaled = arith.mulf %arg_qk, %cst : tensor<1x32x32xf16>
      rock.yield %scaled : tensor<1x32x32xf16>
    } {
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>,
      prePadG0M = 16 : index,
      prePadG0N = 4 : index,
      softmaxType = f32,
      splitKV = 1 : i32,
      slidingWindowSize = 3 : i32,
      params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
      params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
    } : tensor<1x64x32xf16>, tensor<1x32x64xf16>, tensor<1x64x32xf16>, tensor<1xi32> -> tensor<1x64x32xf16>
    return %result : tensor<1x64x32xf16>
  }
}
