// Error tests for the rock-decompose-nonpow2-tiles pass.
//
// The pass runs at the gridwise layer, so each case is a non-power-of-two
// rock.gridwise_gemm that hits one of the pass's diagnostics.

// RUN: rocmlir-opt -rock-decompose-nonpow2-tiles -verify-diagnostics -split-input-file -mlir-print-local-scope %s

#pk48 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 80, kPerBlock = 48, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Error: non-power-of-two kPerBlock. M/N are non-power-of-two
// (so the gemm is selected for splitting), but this pass only
// peels M/N and cannot split the contraction dimension.
// ============================================================

func.func @test_nonpow2_k(%a: tensor<1x160x96xf16>, %b: tensor<1x96x160xf16>, %c: tensor<1x160x160xf32>) -> tensor<1x160x160xf32> attributes {rock.kernel} {
  // expected-error @+1 {{non-power-of-two kPerBlock is not supported}}
  %r = rock.gridwise_gemm(%a, %b) {params = #pk48} : tensor<1x160x96xf16>, tensor<1x96x160xf16> -> tensor<1x160x160xf32>
  %out = rock.store %r to %c by set : tensor<1x160x160xf32> -> tensor<1x160x160xf32> to tensor<1x160x160xf32>
  return %out : tensor<1x160x160xf32>
}

// -----

#p80x80 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 80, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Error: scaled (block-scaled) gridwise_gemms are not handled
// yet, because the scales would need slicing along M/N too.
// ============================================================

func.func @test_scaled_gemm(%a: tensor<1x160x64xf8E4M3FN>, %b: tensor<1x64x160xf8E4M3FN>,
    %sa: tensor<1x160x4xf8E8M0FNU>, %sb: tensor<1x160x4xf8E8M0FNU>,
    %c: tensor<1x160x160xf32>) -> tensor<1x160x160xf32> attributes {rock.kernel} {
  // expected-error @+1 {{scaled gridwise_gemm not supported}}
  %r = rock.gridwise_gemm(%a, %b, %sa, %sb) {quantBlockSize = 16 : i64, params = #p80x80} : tensor<1x160x64xf8E4M3FN>, tensor<1x64x160xf8E4M3FN>, tensor<1x160x4xf8E8M0FNU>, tensor<1x160x4xf8E8M0FNU> -> tensor<1x160x160xf32>
  %out = rock.store %r to %c by set : tensor<1x160x160xf32> -> tensor<1x160x160xf32> to tensor<1x160x160xf32>
  return %out : tensor<1x160x160xf32>
}

// -----

#p80x80b = #rock.gemm_params<mPerBlock = 80, nPerBlock = 80, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Error: the gridwise_gemm result has no rock.store sink (the
// result is unused), so the output side cannot be traced.
// ============================================================

func.func @test_no_store(%a: tensor<1x160x64xf16>, %b: tensor<1x64x160xf16>) attributes {rock.kernel} {
  // expected-error @+1 {{cannot trace gridwise_gemm output to rock.store}}
  %r = rock.gridwise_gemm(%a, %b) {params = #p80x80b} : tensor<1x160x64xf16>, tensor<1x64x160xf16> -> tensor<1x160x160xf32>
  return
}

// -----

#p80x80c = #rock.gemm_params<mPerBlock = 80, nPerBlock = 80, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Error: the gridwise_gemm result reaches an op that is neither
// a fusion op nor a rock.store (here, func.return), so no store
// sink is reachable. Shares the trace diagnostic with @test_no_store.
// ============================================================

func.func @test_unsupported_use(%a: tensor<1x160x64xf16>, %b: tensor<1x64x160xf16>) -> tensor<1x160x160xf32> attributes {rock.kernel} {
  // expected-error @+1 {{cannot trace gridwise_gemm output to rock.store}}
  %r = rock.gridwise_gemm(%a, %b) {params = #p80x80c} : tensor<1x160x64xf16>, tensor<1x64x160xf16> -> tensor<1x160x160xf32>
  return %r : tensor<1x160x160xf32>
}
