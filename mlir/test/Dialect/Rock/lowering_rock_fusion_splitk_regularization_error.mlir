// Error tests for rock-fusion-splitk-regularization pass.

// RUN: rocmlir-opt -rock-fusion-splitk-regularization -verify-diagnostics --split-input-file %s

// ============================================================
// Error: gemm op without params attribute.
// ============================================================

module {
  // expected-error @below {{rewriteFusionForSplitK: found gemm op without params}}
  func.func @error_no_params(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %fused = arith.addf %gemm, %ext : tensor<1x4x4xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }
}

// -----

// ============================================================
// Error: mulf with both operands from gemm (gemmOut^2).
// ============================================================

module {
  func.func @error_mulf_gemm_squared(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    // expected-error @below {{'rock.gemm' op has invalid output fusion for a K reduction}}
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %sq = arith.mulf %gemm, %gemm : tensor<1x4x4xf16>
    %r = rock.store %sq to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }
}

// -----

// ============================================================
// Error: divf with both operands from gemm (gemmOut/gemmOut).
// ============================================================

module {
  func.func @error_divf_gemm_squared(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    // expected-error @below {{'rock.gemm' op has invalid output fusion for a K reduction}}
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %d = arith.divf %gemm, %gemm : tensor<1x4x4xf16>
    %r = rock.store %d to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }
}
