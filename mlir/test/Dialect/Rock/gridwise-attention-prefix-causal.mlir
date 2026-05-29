// RUN: rocmlir-opt -rock-gridwise-attn-to-blockwise -split-input-file -verify-diagnostics %s | FileCheck %s

module {
  // CHECK-LABEL: func @mlir_attention
  // Verify prefix causal loop bound calculation: effectiveSeqLen = min(maxRowOfBlock + prefixOffset, gemm0M - 1)
  // The muli computes n_block * blockSize, then subi computes maxRowOfBlock = nextBlockStart - 1
  // CHECK: arith.muli %{{.*}}, %c32{{.*}} : i32
  // CHECK: %[[MAX_ROW:.*]] = arith.subi %{{.*}}, %c1{{.*}} : i32
  // The prefixOffset is loaded from tensor as a scalar i32
  // CHECK: %[[OFFSET:.*]] = tt.unsplat %{{.*}} : tensor<1xi32>
  // effectiveSeqLen = maxRowOfBlock + prefixOffset
  // CHECK: %[[EFFECTIVE_SEQ_UNBOUND:.*]] = arith.addi %[[MAX_ROW]], %[[OFFSET]] : i32
  // Bound by gemm0M - 1 (key sequence length - 1) to prevent out-of-bounds access
  // CHECK: %[[EFFECTIVE_SEQ:.*]] = arith.minui %[[EFFECTIVE_SEQ_UNBOUND]], %c{{.*}} : i32
  // Ceiling division: (effectiveSeqLen + blockSize) / blockSize
  // CHECK: arith.addi %[[EFFECTIVE_SEQ]], %c32{{.*}} : i32
  // CHECK: %[[END:.*]] = arith.divui %{{.*}}, %c32{{.*}} : i32

  // Verify the main loop uses the computed end bound
  // CHECK: scf.for %{{.*}} = %c0{{.*}} to %[[END]] step %c1{{.*}}

  // Verify prefix causal masking: key_pos > query_pos + prefixOffset
  // CHECK: %[[COL_PLUS_OFFSET:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor<32x32xi32>
  // CHECK: %[[MASK_COND:.*]] = arith.cmpi ugt, %{{.*}}, %[[COL_PLUS_OFFSET]] : tensor<32x32xi32>
  // CHECK: arith.select %[[MASK_COND]], %{{.*}}, %{{.*}} : tensor<32x32xi1>, tensor<32x32xf32>

  func.func @mlir_attention(
      %prefixOffset: tensor<1xi32>,
      %q: tensor<1x64x32xf16>,
      %k: tensor<1x32x64xf16>,
      %v: tensor<1x64x32xf16>) -> tensor<1x64x32xf16>
      attributes {
        rock.block_size = 256 : i32,
        rock.grid_size = 2 : i32,
        rock.kernel,
        rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"
      } {
    %result = rock.gridwise_attention(%q, %k, %v, %prefixOffset) preSoftmaxOps = {
    ^bb0(%arg_qk: tensor<1x32x32xf16>):
      %cst = arith.constant dense<1.250000e-01> : tensor<1x32x32xf16>
      %scaled = arith.mulf %arg_qk, %cst : tensor<1x32x32xf16>
      rock.yield %scaled : tensor<1x32x32xf16>
    } {
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1>,
      causal,
      prePadG0M = 16 : index,
      prePadG0N = 4 : index,
      softmaxType = f32,
      splitKV = 1 : i32,
      params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
      params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
    } : tensor<1x64x32xf16>, tensor<1x32x64xf16>, tensor<1x64x32xf16>, tensor<1xi32> -> tensor<1x64x32xf16>
    return %result : tensor<1x64x32xf16>
  }
}
