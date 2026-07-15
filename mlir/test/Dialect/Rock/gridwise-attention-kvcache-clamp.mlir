// Verify the KV-cache N-loop trip count is clamped to the static K/V block
// count, so a runtime currentSeqLen larger than the K/V allocation cannot
// drive the loop out of bounds.

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-gridwise-attn-to-blockwise -verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: func @attn_kvcache_clamps_nloop
  // currentSeqLen is loaded as a scalar.
  // CHECK: %[[CSL:.*]] = tt.unsplat %{{.*}} : tensor<1xi32>
  // end = (currentSeqLen + nPerBlock) / nPerBlock.
  // CHECK: arith.addi %[[CSL]], %c32{{.*}} : i32
  // CHECK: %[[END_UNBOUND:.*]] = arith.divui %{{.*}}, %c32{{.*}} : i32
  // The trip count is clamped to the static N-block count (gemm0N / nPerBlock =
  // 32768 / 32 = 1024) so a runtime currentSeqLen cannot drive the loop past
  // the K/V allocation.
  // CHECK: %[[END:.*]] = arith.minui %[[END_UNBOUND]], %c1024{{.*}} : i32
  // The main loop uses the clamped bound.
  // CHECK: scf.for %{{.*}} = %c0{{.*}} to %[[END]] step %c1{{.*}}
  func.func @attn_kvcache_clamps_nloop(
      %q: tensor<1x16x64xf32>, %k: tensor<1x64x32768xf32>, %v: tensor<1x32768x64xf32>,
      %currentSeqLen: tensor<1xi32>) -> tensor<1x16x64xf32>
      attributes {rock.block_size = 64 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
    %result = rock.gridwise_attention(%q, %k, %v, %currentSeqLen) preSoftmaxOps = {
    ^bb0(%arg_qk: tensor<1x16x32xf32>):
      rock.yield %arg_qk : tensor<1x16x32xf32>
    } {
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>,
      params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
      params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
      splitKV = 1 : i32
    } : tensor<1x16x64xf32>, tensor<1x64x32768xf32>, tensor<1x32768x64xf32>, tensor<1xi32> -> tensor<1x16x64xf32>
    return %result : tensor<1x16x64xf32>
  }
}
