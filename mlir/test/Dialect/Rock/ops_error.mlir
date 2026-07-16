// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -verify-diagnostics

// -----------------------------------------------------------------------------
// gridwise_attention tests
// -----------------------------------------------------------------------------

// prefixOffset requires causal to be enabled.
func.func @gridwise_attn_prefix_offset_requires_causal(
    %q: tensor<1x384x64xf32>, %k: tensor<1x64x384xf32>, %v: tensor<1x384x64xf32>,
    %prefixOffset: tensor<1xi32>) -> tensor<1x384x64xf32>
    attributes {rock.block_size = 64 : i32, rock.grid_size = 24 : i32, rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{prefixOffset requires causal to be enabled}}
  %r = rock.gridwise_attention(%q, %k, %v, %prefixOffset) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x384x384xf32>):
    rock.yield %arg_qk : tensor<1x384x384xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1>,
    params0 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    params1 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    splitKV = 1 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32>, tensor<1xi32> -> tensor<1x384x64xf32>
  return %r : tensor<1x384x64xf32>
}

// -----------------------------------------------------------------------------
// attention tests
// -----------------------------------------------------------------------------

func.func @attention_numheadskv_negative(%q: tensor<1x384x64xf16>, %k: tensor<1x384x64xf16>, %v: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{numHeadsKV must be positive}}
  %r = rock.attention{
   qk = %q * tr %k : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   softmax(qk) * %v : tensor<1x384x64xf16>
  } {splitKV = 1 : i32, numHeadsKV = -1 : i32, numHeadsQ = 1 : i32} -> tensor<1x384x64xf16>
  return %r : tensor<1x384x64xf16>
}

func.func @attention_numheadsq_negative(%q: tensor<1x384x64xf16>, %k: tensor<1x384x64xf16>, %v: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{numHeadsQ must be positive}}
  %r = rock.attention{
   qk = %q * tr %k : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   softmax(qk) * %v : tensor<1x384x64xf16>
  } {splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = -1 : i32} -> tensor<1x384x64xf16>
  return %r : tensor<1x384x64xf16>
}

func.func @attention_numheadsq_not_divisible(%q: tensor<1x384x64xf16>, %k: tensor<1x384x64xf16>, %v: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{numHeadsQ is not divisible by numHeadsKV}}
  %r = rock.attention{
   qk = %q * tr %k : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   softmax(qk) * %v : tensor<1x384x64xf16>
  } {splitKV = 1 : i32, numHeadsKV = 3 : i32, numHeadsQ = 4 : i32} -> tensor<1x384x64xf16>
  return %r : tensor<1x384x64xf16>
}

func.func @attention_numheadsq_smaller_than_numheadskv(%q: tensor<1x384x64xf16>, %k: tensor<1x384x64xf16>, %v: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{numHeadsQ is not divisible by numHeadsKV}}
  %r = rock.attention{
   qk = %q * tr %k : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   softmax(qk) * %v : tensor<1x384x64xf16>
  } {splitKV = 1 : i32, numHeadsKV = 4 : i32, numHeadsQ = 2 : i32} -> tensor<1x384x64xf16>
  return %r : tensor<1x384x64xf16>
}

func.func @attention_prefix_offset_requires_causal(%q: tensor<1x384x64xf16>, %k: tensor<1x384x64xf16>, %v: tensor<1x384x64xf16>, %prefixOffset: tensor<1xi32>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{prefixOffset requires causal to be enabled}}
  %r = rock.attention{
   qk = %q * tr %k : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   prefixOffset = (%prefixOffset : tensor<1xi32>)
   softmax(qk) * %v : tensor<1x384x64xf16>
  } {splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32} -> tensor<1x384x64xf16>
  return %r : tensor<1x384x64xf16>
}

func.func @attention_sliding_window_not_positive(%q: tensor<1x384x64xf16>, %k: tensor<1x384x64xf16>, %v: tensor<1x384x64xf16>, %csl: tensor<1xi32>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{slidingWindowSize must be positive}}
  %r = rock.attention{
   qk = %q * tr %k : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   currentSeqLen = (%csl : tensor<1xi32>)
   softmax(qk) * %v : tensor<1x384x64xf16>
  } {splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, slidingWindowSize = -5 : i32} -> tensor<1x384x64xf16>
  return %r : tensor<1x384x64xf16>
}

func.func @attention_sliding_window_requires_current_seq_len(%q: tensor<1x384x64xf16>, %k: tensor<1x384x64xf16>, %v: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{slidingWindowSize requires currentSeqLen to be set}}
  %r = rock.attention{
   qk = %q * tr %k : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   softmax(qk) * %v : tensor<1x384x64xf16>
  } {splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, slidingWindowSize = 128 : i32} -> tensor<1x384x64xf16>
  return %r : tensor<1x384x64xf16>
}

func.func @attention_sliding_window_exceeds_max_seq_len(%q: tensor<1x384x64xf16>, %k: tensor<1x384x64xf16>, %v: tensor<1x384x64xf16>, %csl: tensor<1xi32>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{slidingWindowSize must not exceed max sequence length}}
  %r = rock.attention{
   qk = %q * tr %k : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   currentSeqLen = (%csl : tensor<1xi32>)
   softmax(qk) * %v : tensor<1x384x64xf16>
  } {splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, slidingWindowSize = 600 : i32} -> tensor<1x384x64xf16>
  return %r : tensor<1x384x64xf16>
}

// slidingWindowSize on a gemm-gemm-like op with softmax disabled is not attention.
func.func @gridwise_attention_sliding_window_only_for_attention(%q: tensor<1x384x64xf32>, %k: tensor<1x64x384xf32>, %v: tensor<1x384x64xf32>) -> tensor<1x384x64xf32> attributes {rock.block_size = 64 : i32, rock.grid_size = 24 : i32, rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{slidingWindowSize only works for attention}}
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x384x384xf32>):
    rock.yield %arg_qk : tensor<1x384x384xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    enableSoftmax = false,
    params0 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    params1 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    splitKV = 1 : i32,
    slidingWindowSize = 128 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}

func.func @gridwise_attention_sliding_window_not_positive(%q: tensor<1x384x64xf32>, %k: tensor<1x64x384xf32>, %v: tensor<1x384x64xf32>, %csl: tensor<1xi32>) -> tensor<1x384x64xf32> attributes {rock.block_size = 64 : i32, rock.grid_size = 24 : i32, rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{slidingWindowSize must be positive}}
  %result = rock.gridwise_attention(%q, %k, %v, %csl) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x384x384xf32>):
    rock.yield %arg_qk : tensor<1x384x384xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>,
    params0 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    params1 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    splitKV = 1 : i32,
    slidingWindowSize = -5 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32>, tensor<1xi32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}

func.func @gridwise_attention_sliding_window_requires_current_seq_len(%q: tensor<1x384x64xf32>, %k: tensor<1x64x384xf32>, %v: tensor<1x384x64xf32>) -> tensor<1x384x64xf32> attributes {rock.block_size = 64 : i32, rock.grid_size = 24 : i32, rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{slidingWindowSize requires currentSeqLen to be set}}
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x384x384xf32>):
    rock.yield %arg_qk : tensor<1x384x384xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    params1 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    splitKV = 1 : i32,
    slidingWindowSize = 128 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}

func.func @gridwise_attention_sliding_window_exceeds_max_seq_len(%q: tensor<1x384x64xf32>, %k: tensor<1x64x384xf32>, %v: tensor<1x384x64xf32>, %csl: tensor<1xi32>) -> tensor<1x384x64xf32> attributes {rock.block_size = 64 : i32, rock.grid_size = 24 : i32, rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{slidingWindowSize must not exceed max sequence length}}
  %result = rock.gridwise_attention(%q, %k, %v, %csl) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x384x384xf32>):
    rock.yield %arg_qk : tensor<1x384x384xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>,
    params0 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    params1 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    splitKV = 1 : i32,
    slidingWindowSize = 600 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32>, tensor<1xi32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}

// -----------------------------------------------------------------------------
// gemm tests 
// -----------------------------------------------------------------------------

// Float input must produce a floating-point output type
func.func @gemm_float_input_int_output(%a: tensor<64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{float-valued inputs must have a floating-point output type}}
  rock.gemm %a * %b
    : tensor<64x128xf32> * tensor<128x32xf32> -> tensor<64x32xi32>
  func.return
}

// Test case: Matrix A with invalid rank (rank 1)
func.func @gemm_matrixA_wrong_rank(%a: tensor<64xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Matrix A must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a * %b
    : tensor<64xf32> * tensor<128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Matrix A with invalid rank (rank 4)
func.func @gemm_matrixA_rank4(%a: tensor<1x2x64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Matrix A must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a * %b
    : tensor<1x2x64x128xf32> * tensor<128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Matrix B with invalid rank (rank 1)
func.func @gemm_matrixB_wrong_rank(%a: tensor<64x128xf32>, %b: tensor<32xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Matrix B must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a * %b
    : tensor<64x128xf32> * tensor<32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Matrix B with invalid rank (rank 4)
func.func @gemm_matrixB_rank4(%a: tensor<64x128xf32>, %b: tensor<1x2x128x32xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Matrix B must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a * %b
    : tensor<64x128xf32> * tensor<1x2x128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Matrix C with invalid rank (rank 1)
func.func @gemm_matrixC_wrong_rank(%a: tensor<64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{op Result must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a * %b
    : tensor<64x128xf32> * tensor<128x32xf32> -> tensor<64xf32>
  func.return
}

// Test case: Matrix C with invalid rank (rank 4)
func.func @gemm_matrixC_rank4(%a: tensor<64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{op Result must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a * %b
    : tensor<64x128xf32> * tensor<128x32xf32> -> tensor<1x2x64x32xf32>
  func.return
}

// Test case: Mixed ranks - A is rank 3, B and C are rank 2
func.func @gemm_mixed_ranks1(%a: tensor<2x64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{group dimensions don't match g_a = 2 g_b = 1 g_result = 1}}
  rock.gemm %a * %b
    : tensor<2x64x128xf32> * tensor<128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Mixed ranks - B is rank 3, A and C are rank 2
func.func @gemm_mixed_ranks2(%a: tensor<64x128xf32>, %b: tensor<2x128x32xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{group dimensions don't match g_a = 1 g_b = 2 g_result = 1}}
  rock.gemm %a * %b
    : tensor<64x128xf32> * tensor<2x128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: oTransposed but result is in non-transposed order (G x M x N instead of G x N x M)
func.func @gemm_cTransposed_wrong_result(%a: tensor<2x64x128xf32>, %b: tensor<2x128x32xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{M dimensions don't match m_a = 64 m_result = 32}}
  rock.gemm %a * %b {oTransposed}
    : tensor<2x64x128xf32> * tensor<2x128x32xf32> -> tensor<2x64x32xf32>
  func.return
}

// N mismatch: result's N differs from B's N
func.func @gemm_n_mismatch(%a: tensor<2x64x128xf32>, %b: tensor<2x128x32xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{N dimensions don't match n_b = 32 n_result = 16}}
  rock.gemm %a * %b
    : tensor<2x64x128xf32> * tensor<2x128x32xf32> -> tensor<2x64x16xf32>
  func.return
}

// K mismatch: A's K differs from B's K
func.func @gemm_k_mismatch(%a: tensor<2x64x128xf32>, %b: tensor<2x64x32xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{K dimensions don't match k_a = 128 k_b = 64}}
  rock.gemm %a * %b
    : tensor<2x64x128xf32> * tensor<2x64x32xf32> -> tensor<2x64x32xf32>
  func.return
}

// Test case: missing quantBlockSize
func.func @gemm_scaled_missing_quantblocksize(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>,
                                  %scaleA: tensor<64x4xf8E8M0FNU>,
                                  %scaleB: tensor<32x4xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{quantBlockSize not defined}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB
    : tensor<64x128xf4E2M1FN> scaled by tensor<64x4xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// Test case: ScaleA with invalid rank (rank 1)
func.func @gemm_scaleA_wrong_rank(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                                  %scaleA: tensor<64xf8E8M0FNU>,
                                  %scaleB: tensor<32x4xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleA must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB {quantBlockSize = 32 : i64}
    : tensor<64x128xf4E2M1FN> scaled by tensor<64xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// Test case: ScaleA with invalid rank (rank 4)
func.func @gemm_scaleA_rank4(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                             %scaleA: tensor<1x2x64x128xf8E8M0FNU>,
                             %scaleB: tensor<32x4xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleA must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB {quantBlockSize = 32 : i64}
    : tensor<64x128xf4E2M1FN> scaled by tensor<1x2x64x128xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// Test case: ScaleB with invalid rank (rank 1)
func.func @gemm_scaleB_wrong_rank(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                                  %scaleA: tensor<64x4xf8E8M0FNU>,
                                  %scaleB: tensor<32xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleB must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB {quantBlockSize = 32 : i64}
    : tensor<64x128xf4E2M1FN> scaled by tensor<64x4xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// Test case: ScaleB with invalid rank (rank 4)
func.func @gemm_scaleB_rank4(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                             %scaleA: tensor<64x4xf8E8M0FNU>,
                             %scaleB: tensor<1x2x128x32xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleB must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB {quantBlockSize = 32 : i64}
    : tensor<64x128xf4E2M1FN> scaled by tensor<64x4xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<1x2x128x32xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

func.func @gemm_scale_presence_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x4xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  rock.gemm %a scaled by %scaleA * %b {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_k_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA_kbad: tensor<2x64x3xf8E8M0FNU>, %scaleB: tensor<2x32x4xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleA's K dimension must match matrix A's K dimension}}
  rock.gemm %a scaled by %scaleA_kbad * %b scaled by %scaleB {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x3xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x32x4xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_m_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA_mbad: tensor<2x63x4xf8E8M0FNU>, %scaleB: tensor<2x32x4xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleA's M dimension must match matrix A's M dimension}}
  rock.gemm %a scaled by %scaleA_mbad * %b scaled by %scaleB {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x63x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x32x4xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_g_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA_gbad: tensor<3x64x4xf8E8M0FNU>, %scaleB: tensor<2x32x4xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleA's G dimension must match matrix A's G dimension}}
  rock.gemm %a scaled by %scaleA_gbad * %b scaled by %scaleB {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<3x64x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x32x4xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleB_k_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x4xf8E8M0FNU>, %scaleB_kbad: tensor<2x32x3xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleB's K dimension must match matrix B's K dimension}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB_kbad {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x32x3xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleB_n_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x4xf8E8M0FNU>, %scaleB_nbad: tensor<2x31x4xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleB's N dimension must match matrix B's N dimension}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB_nbad {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x31x4xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleB_g_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x4xf8E8M0FNU>, %scaleB_gbad: tensor<3x32x4xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleB's G dimension must match matrix B's G dimension}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB_gbad {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<3x32x4xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_transposed_k_mismatch(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>,
  %scaleA_tbad: tensor<3x64xf8E8M0FNU>, %scaleB: tensor<32x4xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleA's K dimension must match matrix A's K dimension}}
  rock.gemm %a scaled by tr %scaleA_tbad * %b scaled by %scaleB {quantBlockSize = 32 : i64}
  : tensor<64x128xf4E2M1FN> scaled by tensor<3x64xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

func.func @gemm_scaleB_transposed_k_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x4xf8E8M0FNU>, %scaleB_kbad: tensor<2x3x32xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleB's K dimension must match matrix B's K dimension}}
  rock.gemm %a scaled by %scaleA * %b scaled by tr %scaleB_kbad {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x3x32xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

// scaleA type must be f8E8M0FNU
func.func @gemm_scaleA_type_invalid(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>,
  %scaleA_bad: tensor<64x4xf8E4M3FN>, %scaleB: tensor<32x4xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{'rock.gemm' op operand #2 must be tensor of f8E8M0FNU type values, but got 'tensor<64x4xf8E4M3FN>'}}
  rock.gemm %a scaled by %scaleA_bad * %b scaled by %scaleB {quantBlockSize = 32 : i64}
  : tensor<64x128xf4E2M1FN> scaled by tensor<64x4xf8E4M3FN> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// scaleB type must be f8E8M0FNU
func.func @gemm_scaleB_type_invalid(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>,
  %scaleA: tensor<64x4xf8E8M0FNU>, %scaleB_bad: tensor<32x4xf8E4M3FN>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{'rock.gemm' op operand #3 must be tensor of f8E8M0FNU type values, but got 'tensor<32x4xf8E4M3FN>'}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB_bad {quantBlockSize = 32 : i64}
  : tensor<64x128xf4E2M1FN> scaled by tensor<64x4xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E4M3FN> -> tensor<64x32xf32>
  func.return
}

// kPerBlock in tuning params must be divisible by quantBlockSize
func.func @gemm_kperblock_not_divisible_by_quantblocksize(
    %a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>,
    %scaleA: tensor<64x4xf8E8M0FNU>, %scaleB: tensor<32x4xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{kPerBlock must be divisible by quantBlockSize}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB {
    quantBlockSize = 32 : i64,
    params = #rock.gemm_params<
      kPerBlock = 4, kpack = 1, mPerBlock = 64, nPerBlock = 64, numWaves = 4,
      matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0,
      gridGroupSize = 0, numCTAs = 1>
  } : tensor<64x128xf4E2M1FN> scaled by tensor<64x4xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// -----------------------------------------------------------------------------
// Gridwise gemm tests 
// -----------------------------------------------------------------------------
#common_params = #rock.gemm_params<
  kPerBlock = 4,
  kpack = 1,
  mPerBlock = 64,
  nPerBlock = 64,
  numWaves = 4,
  matrixInstrNonkdim = 0,
  splitKFactor = 1,
  numStages = 2,
  wavesPerEU = 0, gridGroupSize = 0,
  numCTAs = 1>

func.func @gridwise_gemm_scale_presence_a_only(%A: tensor<1x4x8xf4E2M1FN>, %B: tensor<1x4x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleA: tensor<1x4x8xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA) {
    params = #common_params
  } : tensor<1x4x8xf4E2M1FN>, tensor<1x4x16xf4E2M1FN>, tensor<1x4x8xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// Scale presence B only
func.func @gridwise_gemm_scale_presence_b_only(%A: tensor<1x4x8xf4E2M1FN>, %B: tensor<1x4x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleB: tensor<1x4x16xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  %result = rock.gridwise_gemm(%A, %B, %scaleB) {
    params = #common_params
  } : tensor<1x4x8xf4E2M1FN>, tensor<1x4x16xf4E2M1FN>, tensor<1x4x16xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// Scales provided but quantBlockSize attribute missing
func.func @gridwise_gemm_scale_missing_quantblocksize(%A: tensor<1x8x32xf4E2M1FN>, %B: tensor<1x32x16xf4E2M1FN>, %C: tensor<1x8x16xf32>,
    %scaleA: tensor<1x8x1xf8E8M0FNU>, %scaleB: tensor<1x16x1xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{quantBlockSize is not defined}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB) {
    params = #common_params
  } : tensor<1x8x32xf4E2M1FN>, tensor<1x32x16xf4E2M1FN>, tensor<1x8x1xf8E8M0FNU>, tensor<1x16x1xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleA dims mismatch
func.func @gridwise_gemm_scaleA_dims_mismatch(%A: tensor<1x8x32xf4E2M1FN>, %B: tensor<1x32x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleA_bad_dims: tensor<1x8x7xf8E8M0FNU>, %scaleB: tensor<1x16x1xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{ScaleA shape must match matrixA shape.}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA_bad_dims, %scaleB) {
    quantBlockSize = 32 : i64,
    params = #common_params
  } : tensor<1x8x32xf4E2M1FN>, tensor<1x32x16xf4E2M1FN>, tensor<1x8x7xf8E8M0FNU>, tensor<1x16x1xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleB dims mismatch
func.func @gridwise_gemm_scaleB_dims_mismatch(%A: tensor<1x8x32xf4E2M1FN>, %B: tensor<1x32x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleA_bad_dims: tensor<1x8x1xf8E8M0FNU>, %scaleB: tensor<1x16x2xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{ScaleB shape must match matrixB shape.}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA_bad_dims, %scaleB) {
    quantBlockSize = 32 : i64,
    params = #common_params
  } : tensor<1x8x32xf4E2M1FN>, tensor<1x32x16xf4E2M1FN>, tensor<1x8x1xf8E8M0FNU>, tensor<1x16x2xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleA type must be f8E8M0FNU
func.func @gridwise_gemm_scaleA_type_invalid(%A: tensor<1x8x32xf4E2M1FN>, %B: tensor<1x32x16xf4E2M1FN>, %C: tensor<1x8x16xf32>,
    %scaleA_bad: tensor<1x8x1xf8E4M3FN>, %scaleB: tensor<1x16x1xf8E8M0FNU>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{'rock.gridwise_gemm' op operand #2 must be 3D tensor of f8E8M0FNU type values, but got 'tensor<1x8x1xf8E4M3FN>'}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA_bad, %scaleB) {
    quantBlockSize = 32 : i64,
    params = #common_params
  } : tensor<1x8x32xf4E2M1FN>, tensor<1x32x16xf4E2M1FN>, tensor<1x8x1xf8E4M3FN>, tensor<1x16x1xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleB type must be f8E8M0FNU
func.func @gridwise_gemm_scaleB_type_invalid(%A: tensor<1x8x32xf4E2M1FN>, %B: tensor<1x32x16xf4E2M1FN>, %C: tensor<1x8x16xf32>,
    %scaleA: tensor<1x8x1xf8E8M0FNU>, %scaleB_bad: tensor<1x16x1xf8E4M3FN>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{'rock.gridwise_gemm' op operand #3 must be 3D tensor of f8E8M0FNU type values, but got 'tensor<1x16x1xf8E4M3FN>'}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB_bad) {
    quantBlockSize = 32 : i64,
    params = #common_params
  } : tensor<1x8x32xf4E2M1FN>, tensor<1x32x16xf4E2M1FN>, tensor<1x8x1xf8E8M0FNU>, tensor<1x16x1xf8E4M3FN> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// G mismatch: B[0] differs from A[0]
func.func @gridwise_gemm_g_mismatch(%A: tensor<1x8x32xf32>, %B_bad: tensor<2x32x16xf32>, %C: tensor<1x8x16xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Mismatched G dimensions in matrix multiply}}
  %result = rock.gridwise_gemm(%A, %B_bad) {
    params = #common_params
  } : tensor<1x8x32xf32>, tensor<2x32x16xf32> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// M mismatch: result[1] differs from A[1]
func.func @gridwise_gemm_m_mismatch(%A: tensor<1x8x32xf32>, %B: tensor<1x32x16xf32>, %C: tensor<1x4x16xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Mismatched M dimensions in matrix multiply}}
  %result = rock.gridwise_gemm(%A, %B) {
    params = #common_params
  } : tensor<1x8x32xf32>, tensor<1x32x16xf32> -> tensor<1x4x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x4x16xf32> -> tensor<1x4x16xf32> to tensor<1x4x16xf32>
  func.return
}

// K mismatch: B[1] differs from A[2]
func.func @gridwise_gemm_k_mismatch(%A: tensor<1x8x32xf32>, %B_bad: tensor<1x16x16xf32>, %C: tensor<1x8x16xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Mismatched K dimensions in matrix multiply}}
  %result = rock.gridwise_gemm(%A, %B_bad) {
    params = #common_params
  } : tensor<1x8x32xf32>, tensor<1x16x16xf32> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// N mismatch: result[2] differs from B[2]
func.func @gridwise_gemm_n_mismatch(%A: tensor<1x8x32xf32>, %B: tensor<1x32x16xf32>, %C: tensor<1x8x8xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Mismatched N dimensions in matrix multiply}}
  %result = rock.gridwise_gemm(%A, %B) {
    params = #common_params
  } : tensor<1x8x32xf32>, tensor<1x32x16xf32> -> tensor<1x8x8xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x8xf32> -> tensor<1x8x8xf32> to tensor<1x8x8xf32>
  func.return
}

// -----------------------------------------------------------------------------
// Blockwise gemm tests (scaled gemm, active)
// -----------------------------------------------------------------------------

// scaleA present but scaleB missing
func.func @blockwise_gemm_scaleA_only(
    %a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>,
    %scaleA: tensor<64x2xi8>, %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleA and scaleB must both be present or both be null.}}
  %0 = rock.blockwise_gemm(%a scaled by %scaleA, %b, %c)
    {quantBlockSize = 32 : i64}
    : tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>,
      tensor<64x64xf8E4M3FN>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// scaleB present but scaleA missing
func.func @blockwise_gemm_scaleB_only(
    %a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>,
    %scaleB: tensor<64x2xi8>, %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{scaleA and scaleB must both be present or both be null.}}
  %0 = rock.blockwise_gemm(%a, %b scaled by %scaleB, %c)
    {quantBlockSize = 32 : i64}
    : tensor<64x64xf8E4M3FN>,
      tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// quantBlockSize is required when scales are present
func.func @blockwise_gemm_no_quantblocksize(
    %a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>,
    %scaleA: tensor<64x2xi8>, %scaleB: tensor<64x2xi8>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{quantBlockSize is not set but we found scale}}
  %0 = rock.blockwise_gemm(%a scaled by %scaleA, %b scaled by %scaleB, %c)
    : tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>,
      tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// scaleA shape must match matrixA shape (non-packed case)
func.func @blockwise_gemm_scaleA_shape_mismatch(
    %a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>,
    %scaleA: tensor<64x3xi8>, %scaleB: tensor<64x2xi8>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{If scaleA is non-null, its shape must match A's shape.}}
  %0 = rock.blockwise_gemm(%a scaled by %scaleA, %b scaled by %scaleB, %c)
    {quantBlockSize = 32 : i64}
    : tensor<64x64xf8E4M3FN> scaled by tensor<64x3xi8>,
      tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// scaleB shape must match matrixB shape (non-packed case)
func.func @blockwise_gemm_scaleB_shape_mismatch(
    %a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>,
    %scaleA: tensor<64x2xi8>, %scaleB: tensor<3x64xi8>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{If scaleB is non-null, its shape must match B's shape.}}
  %0 = rock.blockwise_gemm(%a scaled by %scaleA, %b scaled by %scaleB, %c)
    {quantBlockSize = 32 : i64}
    : tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>,
      tensor<64x64xf8E4M3FN> scaled by tensor<3x64xi8>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// Packed matrix shape: after packing f4->i8, the K dim is halved.
// Doubling the packed dim should recover the scale shape.
// Here: matrixA = 64x32xi8 (K halved from 64 to 32), scale = 64x2 (K=64/32=2)
// Expected: 64x64 (doubled K=32*2=64), normalized scale = 64x64. Match.
// Test the mismatch case: scaleA has wrong shape.
func.func @blockwise_gemm_packed_scaleA_mismatch(
    %a: tensor<64x32xi8>, %b: tensor<32x64xi8>,
    %scaleA: tensor<64x3xi8>, %scaleB: tensor<64x2xi8>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Packed matrixA shape (with dim 1 2x) must match normalized scaleA shape.}}
  %0 = rock.blockwise_gemm(%a scaled by %scaleA, %b scaled by %scaleB, %c)
    {quantBlockSize = 32 : i64,
     matrixAOrigElemType = f4E2M1FN,
     matrixBOrigElemType = f4E2M1FN,
     matrixAKPack = true,
     matrixBKPack = true}
    : tensor<64x32xi8> scaled by tensor<64x3xi8>,
      tensor<32x64xi8> scaled by tensor<64x2xi8>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// Packed shape mismatch on B: matrixB = 32x64xi8 (K halved from 64 to 32 in dim 0).
// scaleB with bad shape triggers the packed-B mismatch.
func.func @blockwise_gemm_packed_scaleB_mismatch(
    %a: tensor<64x32xi8>, %b: tensor<32x64xi8>,
    %scaleA: tensor<64x2xi8>, %scaleB: tensor<3x64xi8>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Packed matrixB shape (with dim 0 2x) must match normalized scaleB shape.}}
  %0 = rock.blockwise_gemm(%a scaled by %scaleA, %b scaled by %scaleB, %c)
    {quantBlockSize = 32 : i64,
     matrixAOrigElemType = f4E2M1FN,
     matrixBOrigElemType = f4E2M1FN,
     matrixAKPack = true,
     matrixBKPack = true}
    : tensor<64x32xi8> scaled by tensor<64x2xi8>,
      tensor<32x64xi8> scaled by tensor<3x64xi8>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// Matrix shape must be 2D
func.func @blockwise_gemm_3d_matrix(
    %a: tensor<1x64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{matrix shape must be 2D}}
  %0 = rock.blockwise_gemm(%a, %b, %c)
    : tensor<1x64x64xf8E4M3FN>,
      tensor<64x64xf8E4M3FN>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// =============================================================================
// rock.store tests
// =============================================================================

// Element type mismatch between source, dest, and result
func.func @store_elem_type_mismatch(
    %source: tensor<4x4xf32>, %dest: tensor<4x4xf32>) -> tensor<4x4xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{failed to verify that all of {source, dest, result} have same element type}}
  %out = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf16> to tensor<4x4xf32>
  return %out : tensor<4x4xf16>
}

// Shape mismatch between source and dest
func.func @store_shape_mismatch(
    %source: tensor<4x4xf32>, %dest: tensor<8x8xf32>) -> tensor<4x4xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{source and dest shapes must match}}
  %out = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<8x8xf32>
  return %out : tensor<4x4xf32>
}

// Result used by a non-return op
func.func @store_result_not_returned(
    %source: tensor<4x4xf32>, %dest: tensor<4x4xf32>) -> tensor<4x4xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{result must be used directly by a func.return, view-like op, or resultAlias operand of another same-kind store}}
  %out = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  %neg = arith.negf %out : tensor<4x4xf32>
  return %neg : tensor<4x4xf32>
}

// Result used as another store's source
func.func @store_result_used_as_store_source(
    %source: tensor<4x4xf32>, %dest: tensor<4x4xf32>) -> tensor<4x4xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{result must be used directly by a func.return, view-like op, or resultAlias operand of another same-kind store}}
  %out = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  %out2 = rock.store %out to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  return %out2 : tensor<4x4xf32>
}

// Result used directly as another store's dest
func.func @store_result_used_as_store_dest(
    %source: tensor<4x4xf32>, %dest: tensor<4x4xf32>) -> tensor<4x4xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{result must be used directly by a func.return, view-like op, or resultAlias operand of another same-kind store}}
  %out = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  %out2 = rock.store %source to %out by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  return %out2 : tensor<4x4xf32>
}

// Result used through a view as another store's dest
func.func @store_result_view_used_as_store_dest(
    %source: tensor<4x4xf32>, %dest: tensor<4x4xf32>) -> tensor<4x4xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %out = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  %view = rock.transform %out by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["m", "n"] at [0, 1] -> ["m", "n"] at [0, 1]>] bounds = [4, 4] -> [4, 4]> : tensor<4x4xf32> to tensor<4x4xf32>
  // expected-error @+1 {{dest transform chain root must be a function entry block argument}}
  %out2 = rock.store %source to %view by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  return %out2 : tensor<4x4xf32>
}

// Dest rooted at a non-entry block argument
func.func @store_dest_rooted_at_loop_block_arg(
    %source: tensor<4x4xf32>, %dest: tensor<4x4xf32>) -> tensor<4x4xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter_dest = %dest) -> (tensor<4x4xf32>) {
    %view = rock.transform %iter_dest by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["m", "n"] at [0, 1] -> ["m", "n"] at [0, 1]>] bounds = [4, 4] -> [4, 4]> : tensor<4x4xf32> to tensor<4x4xf32>
    // expected-error @+1 {{dest transform chain root must be a function entry block argument}}
    %stored = rock.store %source to %view alias %iter_dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32> alias tensor<4x4xf32>
    scf.yield %stored : tensor<4x4xf32>
  }
  return %out : tensor<4x4xf32>
}

// Result has multiple uses
func.func @store_result_multiple_uses(
    %source: tensor<4x4xf32>, %dest: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{result may be returned at most once}}
  %out = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  return %out, %out : tensor<4x4xf32>, tensor<4x4xf32>
}

// =============================================================================
// rock.cast_to_ptr tests
// =============================================================================

// Source is not i32
func.func @cast_to_ptr_src_not_i32(%src: tensor<64x64xf32>) -> tensor<64x64x!tt.ptr<f16>> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{operand #0 must be ranked tensor of 32-bit signless integer or 64-bit signless integer values}}
  %0 = rock.cast_to_ptr %src : tensor<64x64xf32> -> tensor<64x64x!tt.ptr<f16>>
  return %0 : tensor<64x64x!tt.ptr<f16>>
}

// Result is not a pointer tensor
func.func @cast_to_ptr_result_not_ptr(%src: tensor<64x64xi32>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{result must be a tensor of !tt.ptr}}
  %0 = rock.cast_to_ptr %src : tensor<64x64xi32> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// Shape mismatch between src and result
func.func @cast_to_ptr_shape_mismatch(%src: tensor<32x64xi32>) -> tensor<64x64x!tt.ptr<f16>> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{failed to verify that all of {src, result} have same shape}}
  %0 = rock.cast_to_ptr %src : tensor<32x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  return %0 : tensor<64x64x!tt.ptr<f16>>
}

// =============================================================================
// rock.extract_ptr tests
// =============================================================================

// Source is not a block argument
func.func @extract_ptr_not_block_arg(%src: tensor<64x64xf32>) -> i32 attributes {rock.arch = "##TOKEN_ARCH##"} {
  %cst = arith.constant dense<0.0> : tensor<64x64xf32>
  // expected-error @+1 {{source must be a block argument}}
  %0 = rock.extract_ptr %cst : tensor<64x64xf32> -> i32
  return %0 : i32
}

// -----

// Result must be an i32 or i64 placeholder
func.func @extract_ptr_bad_result_type(%src: tensor<64x64xf32>) -> i16 attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{result #0 must be 32-bit signless integer or 64-bit signless integer}}
  %0 = rock.extract_ptr %src : tensor<64x64xf32> -> i16
  return %0 : i16
}

// =============================================================================
// rock.blockwise_reduce tests
// =============================================================================

// Axis out of range
func.func @blockwise_reduce_axis_oob(%input: tensor<64x64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{axis is out of range}}
  %0 = rock.blockwise_reduce sum %input {axis = 3 : index} : tensor<64x64xf32> -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// Rank mismatch (output rank must be input rank - 1)
func.func @blockwise_reduce_rank_mismatch(%input: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{output rank must be input rank - 1}}
  %0 = rock.blockwise_reduce sum %input {axis = 0 : index} : tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// Non-reduction dimension mismatch
func.func @blockwise_reduce_non_red_dim_mismatch(%input: tensor<64x64xf32>) -> tensor<32xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{non-reduction dimension size mismatch at output dim 0}}
  %0 = rock.blockwise_reduce sum %input {axis = 1 : index} : tensor<64x64xf32> -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// Element type mismatch
func.func @blockwise_reduce_elem_type_mismatch(%input: tensor<64x64xf32>) -> tensor<64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{failed to verify that all of {input, result} have same element type}}
  %0 = rock.blockwise_reduce sum %input {axis = 1 : index} : tensor<64x64xf32> -> tensor<64xf16>
  return %0 : tensor<64xf16>
}

// =============================================================================
// rock.transforms_to_ptr tests
// =============================================================================

// Pointers element type not i32
func.func @transforms_to_ptr_ptr_not_i32(%src: tensor<64x64xf32>) -> (tensor<64x64xf16>, tensor<64x64xi1>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{result #0 must be ranked tensor of 32-bit signless integer or 64-bit signless integer values}}
  %ptrs, %mask = rock.transforms_to_ptr %src : tensor<64x64xf32> -> tensor<64x64xf16>, tensor<64x64xi1>
  return %ptrs, %mask : tensor<64x64xf16>, tensor<64x64xi1>
}

// Mask element type not i1
func.func @transforms_to_ptr_mask_not_i1(%src: tensor<64x64xf32>) -> (tensor<64x64xi32>, tensor<64x64xi32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{result #1 must be ranked tensor of 1-bit signless integer values}}
  %ptrs, %mask = rock.transforms_to_ptr %src : tensor<64x64xf32> -> tensor<64x64xi32>, tensor<64x64xi32>
  return %ptrs, %mask : tensor<64x64xi32>, tensor<64x64xi32>
}

// Shape mismatch between pointers and mask
func.func @transforms_to_ptr_shape_mismatch(%src: tensor<64x64xf32>) -> (tensor<32x32xi32>, tensor<64x64xi1>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{failed to verify that all of {pointers, mask} have same shape}}
  %ptrs, %mask = rock.transforms_to_ptr %src : tensor<64x64xf32> -> tensor<32x32xi32>, tensor<64x64xi1>
  return %ptrs, %mask : tensor<32x32xi32>, tensor<64x64xi1>
}

// Rank mismatch: extraIndices + pointers.rank != source.rank
func.func @transforms_to_ptr_rank_mismatch(
    %src: tensor<4x1x1x2x64x64xf16>, %i0: i32) -> (tensor<64x64xi32>, tensor<64x64xi1>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{extraIndices.size() + pointers rank must equal source rank}}
  %ptrs, %mask = rock.transforms_to_ptr %src[%i0] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  return %ptrs, %mask : tensor<64x64xi32>, tensor<64x64xi1>
}

// Source last dimensions mismatch with pointers shape
func.func @transforms_to_ptr_last_dims_mismatch(
    %src: tensor<4x1x1x2x64x64xf16>, %i0: i32, %i1: i32, %i2: i32, %i3: i32) -> (tensor<32x64xi32>, tensor<32x64xi1>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Source last dimensions must match with pointers shape}}
  %ptrs, %mask = rock.transforms_to_ptr %src[%i0, %i1, %i2, %i3] : tensor<4x1x1x2x64x64xf16> -> tensor<32x64xi32>, tensor<32x64xi1>
  return %ptrs, %mask : tensor<32x64xi32>, tensor<32x64xi1>
}

// =============================================================================
// rock.blockwise_store tests
// =============================================================================

// Element type mismatch (source vs result)
func.func @blockwise_store_elem_type_mismatch(
    %src: tensor<16x16xf32>, %dest: tensor<3x4x32x16x16xf32>,
    %i0: i32, %i1: i32, %i2: i32) -> tensor<32768xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{failed to verify that all of {source, dest, result} have same element type}}
  %0 = rock.blockwise_store %src -> %dest[%i0, %i1, %i2] by set
    : tensor<16x16xf32> -> tensor<3x4x32x16x16xf32> -> tensor<32768xf16>
  return %0 : tensor<32768xf16>
}

// Rank mismatch: extraIndices + source.rank != dest.rank
func.func @blockwise_store_rank_mismatch(
    %src: tensor<16x16xf32>, %dest: tensor<3x4x32x16x16xf32>,
    %i0: i32) -> tensor<32768xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{extraIndices.size() + source rank must equal dest rank}}
  %0 = rock.blockwise_store %src -> %dest[%i0] by set
    : tensor<16x16xf32> -> tensor<3x4x32x16x16xf32> -> tensor<32768xf32>
  return %0 : tensor<32768xf32>
}

// Result not used by return
func.func @blockwise_store_not_returned(
    %src: tensor<16x16xf32>, %dest: tensor<3x16x16xf32>,
    %alias: tensor<768xf32>, %i0: i32) -> tensor<768xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{result must be used directly by a func.return, view-like op, or resultAlias operand of another same-kind store}}
  %0 = rock.blockwise_store %src -> %dest alias %alias [%i0] by set
    : tensor<16x16xf32> -> tensor<3x16x16xf32> alias tensor<768xf32> -> tensor<768xf32>
  %neg = arith.negf %0 : tensor<768xf32>
  return %neg : tensor<768xf32>
}

// Result used as another blockwise_store's source
func.func @blockwise_store_result_used_as_store_source(
    %src: tensor<768xf32>, %dest0: tensor<3x768xf32>,
    %dest1: tensor<3x2304xf32>, %alias0: tensor<2304xf32>,
    %alias1: tensor<6912xf32>, %i0: i32) -> tensor<6912xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{result must be used directly by a func.return, view-like op, or resultAlias operand of another same-kind store}}
  %0 = rock.blockwise_store %src -> %dest0 alias %alias0 [%i0] by set
    : tensor<768xf32> -> tensor<3x768xf32> alias tensor<2304xf32> -> tensor<2304xf32>
  %1 = rock.blockwise_store %0 -> %dest1 alias %alias1 [%i0] by set
    : tensor<2304xf32> -> tensor<3x2304xf32> alias tensor<6912xf32> -> tensor<6912xf32>
  return %1 : tensor<6912xf32>
}

// Result used directly as another blockwise_store's dest
func.func @blockwise_store_result_used_as_store_dest(
    %src: tensor<16x16xf32>, %dest: tensor<16x16xf32>) -> tensor<16x16xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{result must be used directly by a func.return, view-like op, or resultAlias operand of another same-kind store}}
  %0 = rock.blockwise_store %src -> %dest by set
    : tensor<16x16xf32> -> tensor<16x16xf32> -> tensor<16x16xf32>
  %1 = rock.blockwise_store %src -> %0 by set
    : tensor<16x16xf32> -> tensor<16x16xf32> -> tensor<16x16xf32>
  return %1 : tensor<16x16xf32>
}

// Result used through a view as another blockwise_store's dest
func.func @blockwise_store_result_view_used_as_store_dest(
    %src: tensor<16x16xf32>, %dest: tensor<16x16xf32>) -> tensor<16x16xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.blockwise_store %src -> %dest by set
    : tensor<16x16xf32> -> tensor<16x16xf32> -> tensor<16x16xf32>
  %view = rock.transform %0 by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["m", "n"] at [0, 1] -> ["m", "n"] at [0, 1]>] bounds = [16, 16] -> [16, 16]> : tensor<16x16xf32> to tensor<16x16xf32>
  // expected-error @+1 {{dest transform chain root must be a function entry block argument}}
  %1 = rock.blockwise_store %src -> %view by set
    : tensor<16x16xf32> -> tensor<16x16xf32> -> tensor<16x16xf32>
  return %1 : tensor<16x16xf32>
}

// Dest rooted at a non-entry block argument
func.func @blockwise_store_dest_rooted_at_loop_block_arg(
    %src: tensor<16x16xf32>, %dest: tensor<16x16xf32>) -> tensor<16x16xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter_dest = %dest) -> (tensor<16x16xf32>) {
    %view = rock.transform %iter_dest by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["m", "n"] at [0, 1] -> ["m", "n"] at [0, 1]>] bounds = [16, 16] -> [16, 16]> : tensor<16x16xf32> to tensor<16x16xf32>
    // expected-error @+1 {{dest transform chain root must be a function entry block argument}}
    %stored = rock.blockwise_store %src -> %view alias %iter_dest by set
      : tensor<16x16xf32> -> tensor<16x16xf32> alias tensor<16x16xf32> -> tensor<16x16xf32>
    scf.yield %stored : tensor<16x16xf32>
  }
  return %out : tensor<16x16xf32>
}

// Result has multiple uses
func.func @blockwise_store_multiple_uses(
    %src: tensor<16x16xf32>, %dest: tensor<3x16x16xf32>,
    %alias: tensor<768xf32>, %i0: i32) -> (tensor<768xf32>, tensor<768xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{result may be returned at most once}}
  %0 = rock.blockwise_store %src -> %dest alias %alias [%i0] by set
    : tensor<16x16xf32> -> tensor<3x16x16xf32> alias tensor<768xf32> -> tensor<768xf32>
  return %0, %0 : tensor<768xf32>, tensor<768xf32>
}

// Dest last dimensions shape mismatch
func.func @blockwise_store_shape_mismatch(
    %src: tensor<16x32xf32>, %dest: tensor<3x4x32x16x16xf32>,
    %i0: i32, %i1: i32, %i2: i32) -> tensor<32768xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Dest last dimensions must match with input shape}}
  %0 = rock.blockwise_store %src -> %dest[%i0, %i1, %i2] by set
    : tensor<16x32xf32> -> tensor<3x4x32x16x16xf32> -> tensor<32768xf32>
  return %0 : tensor<32768xf32>
}

// =============================================================================
// rock.blockwise_load tests
// =============================================================================

// Element type mismatch between source and result
func.func @blockwise_load_elem_type_mismatch(
    %src: tensor<4x1x1x2x64x64xf16>, %i0: i32, %i1: i32, %i2: i32, %i3: i32) -> tensor<64x64xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{failed to verify that all of {source, result} have same element type}}
  %0 = rock.blockwise_load %src[%i0, %i1, %i2, %i3] {cacheModifier = #rock<CacheModifier none>} : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// Rank mismatch: sourceIndices.size() + result.rank != source.rank
func.func @blockwise_load_rank_mismatch(
    %src: tensor<4x1x1x2x64x64xf16>, %i0: i32) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{sourceIndices.size() + result rank must equal source rank}}
  %0 = rock.blockwise_load %src[%i0] {cacheModifier = #rock<CacheModifier none>} : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// Input last dimensions shape mismatch
func.func @blockwise_load_shape_mismatch(
    %src: tensor<4x1x1x2x64x64xf16>, %i0: i32, %i1: i32, %i2: i32, %i3: i32) -> tensor<64x32xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Input last dimensions must match with result shape}}
  %0 = rock.blockwise_load %src[%i0, %i1, %i2, %i3] {cacheModifier = #rock<CacheModifier none>} : tensor<4x1x1x2x64x64xf16> -> tensor<64x32xf16>
  return %0 : tensor<64x32xf16>
}

// Store-only cache modifier 'wb' is not allowed on a load
func.func @blockwise_load_cache_modifier_wb(
    %src: tensor<4x1x1x2x64x64xf16>, %i0: i32, %i1: i32, %i2: i32, %i3: i32) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{cache modifier 'wb' is a store-only modifier and cannot be used on a load}}
  %0 = rock.blockwise_load %src[%i0, %i1, %i2, %i3] {cacheModifier = #rock<CacheModifier wb>} : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// Store-only cache modifier 'wt' is not allowed on a load
func.func @blockwise_load_cache_modifier_wt(
    %src: tensor<4x1x1x2x64x64xf16>, %i0: i32, %i1: i32, %i2: i32, %i3: i32) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{cache modifier 'wt' is a store-only modifier and cannot be used on a load}}
  %0 = rock.blockwise_load %src[%i0, %i1, %i2, %i3] {cacheModifier = #rock<CacheModifier wt>} : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// =============================================================================
// rock.blockwise_load_ptr tests
// =============================================================================

// Pointer tensor element type not i32
func.func @blockwise_load_ptr_ptr_not_i32(
    %ptrs: tensor<64x64xf32>, %mask: tensor<64x64xi1>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{operand #0 must be ranked tensor of 32-bit signless integer or 64-bit signless integer values}}
  %0 = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xf32>, tensor<64x64xi1> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// Mask tensor element type not i1
func.func @blockwise_load_ptr_mask_not_i1(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi32>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{operand #1 must be ranked tensor of 1-bit signless integer values}}
  %0 = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi32> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// Shape mismatch between pointers and result
func.func @blockwise_load_ptr_shape_mismatch(
    %ptrs: tensor<32x32xi32>, %mask: tensor<64x64xi1>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{failed to verify that all of {pointerTensor, maskTensor, result} have same shape}}
  %0 = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// Store-only cache modifier 'wb' is not allowed on a load
func.func @blockwise_load_ptr_cache_modifier_wb(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{cache modifier 'wb' is a store-only modifier and cannot be used on a load}}
  %0 = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier wb>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// Store-only cache modifier 'wt' is not allowed on a load
func.func @blockwise_load_ptr_cache_modifier_wt(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{cache modifier 'wt' is a store-only modifier and cannot be used on a load}}
  %0 = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier wt>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// =============================================================================
// rock.blockwise_store_ptr tests
// =============================================================================

// Pointer tensor element type not i32
func.func @blockwise_store_ptr_ptr_not_i32(
    %src: tensor<64x64xf32>, %ptrs: tensor<64x64xf16>, %mask: tensor<64x64xi1>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{operand #0 must be ranked tensor of 32-bit signless integer or 64-bit signless integer values}}
  rock.blockwise_store_ptr %src -> %ptrs(%mask) by set
    : tensor<64x64xf32> -> tensor<64x64xf16>(tensor<64x64xi1>)
  return
}

// Mask tensor element type not i1
func.func @blockwise_store_ptr_mask_not_i1(
    %src: tensor<64x64xf32>, %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{operand #1 must be ranked tensor of 1-bit signless integer values}}
  rock.blockwise_store_ptr %src -> %ptrs(%mask) by set
    : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi32>)
  return
}

// Shape mismatch between pointers, mask, and source
func.func @blockwise_store_ptr_shape_mismatch(
    %src: tensor<64x64xf32>, %ptrs: tensor<32x32xi32>, %mask: tensor<64x64xi1>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{failed to verify that all of {pointerTensor, maskTensor, source} have same shape}}
  rock.blockwise_store_ptr %src -> %ptrs(%mask) by set
    : tensor<64x64xf32> -> tensor<32x32xi32>(tensor<64x64xi1>)
  return
}

// =============================================================================
// rock.load_marker tests
// =============================================================================

#load_marker_tmap = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 64 + d1, d2)> by [<Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 1] -> ["k"] at [0]>, <PassThrough ["n"] at [2] -> ["n"] at [1]>] bounds = [4, 64, 128] -> [256, 128]>

// Element type mismatch between source and result
func.func @load_marker_elem_type_mismatch(%src: tensor<256x128xf16>, %i0: i32) -> tensor<64x128xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{failed to verify that all of {source, result} have same element type}}
  %0 = rock.load_marker %src views [#load_marker_tmap] [%i0] {cacheModifier = #rock<CacheModifier none>} : tensor<256x128xf16> -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// Upper dims != result rank + extraIndices count
func.func @load_marker_rank_mismatch(%src: tensor<256x128xf16>, %i0: i32, %i1: i32) -> tensor<64x128xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{upper bounds must equal tensor rank + extraIndices count}}
  %0 = rock.load_marker %src views [#load_marker_tmap] [%i0, %i1] {cacheModifier = #rock<CacheModifier none>} : tensor<256x128xf16> -> tensor<64x128xf16>
  return %0 : tensor<64x128xf16>
}

// Upper bounds last dimensions mismatch with result shape
// #load_marker_tmap upper bounds = [4, 64, 128], result rank 2 → take_back(2) = [64, 128]
func.func @load_marker_upper_shape_mismatch(%src: tensor<256x128xf16>, %i0: i32) -> tensor<32x128xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Upper bounds last dimensions must match with result shape}}
  %0 = rock.load_marker %src views [#load_marker_tmap] [%i0] {cacheModifier = #rock<CacheModifier none>} : tensor<256x128xf16> -> tensor<32x128xf16>
  return %0 : tensor<32x128xf16>
}

// Lower bounds mismatch with source shape
// #load_marker_tmap lower bounds = [256, 128], source is [128, 128]
func.func @load_marker_lower_shape_mismatch(%src: tensor<128x128xf16>, %i0: i32) -> tensor<64x128xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Lower bounds must match with input shape}}
  %0 = rock.load_marker %src views [#load_marker_tmap] [%i0] {cacheModifier = #rock<CacheModifier none>} : tensor<128x128xf16> -> tensor<64x128xf16>
  return %0 : tensor<64x128xf16>
}

// Store-only cache modifier 'wb' is not allowed on a load
func.func @load_marker_cache_modifier_wb(%src: tensor<256x128xf16>, %i0: i32) -> tensor<64x128xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{cache modifier 'wb' is a store-only modifier and cannot be used on a load}}
  %0 = rock.load_marker %src views [#load_marker_tmap] [%i0] {cacheModifier = #rock<CacheModifier wb>} : tensor<256x128xf16> -> tensor<64x128xf16>
  return %0 : tensor<64x128xf16>
}

// Store-only cache modifier 'wt' is not allowed on a load
func.func @load_marker_cache_modifier_wt(%src: tensor<256x128xf16>, %i0: i32) -> tensor<64x128xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{cache modifier 'wt' is a store-only modifier and cannot be used on a load}}
  %0 = rock.load_marker %src views [#load_marker_tmap] [%i0] {cacheModifier = #rock<CacheModifier wt>} : tensor<256x128xf16> -> tensor<64x128xf16>
  return %0 : tensor<64x128xf16>
}

// Reduction tile axis must refer to an axis of the rank-2 loaded tile
func.func @load_marker_reduction_tile_axis_out_of_bounds(%src: tensor<256x128xf16>, %i0: i32) -> tensor<64x128xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{reduction tile axis 2 is not an axis of the loaded tile}}
  %0 = rock.load_marker %src views [#load_marker_tmap] [%i0] {cacheModifier = #rock<CacheModifier none>, reductionTileAxes = array<i64: 2>} : tensor<256x128xf16> -> tensor<64x128xf16>
  return %0 : tensor<64x128xf16>
}

// =============================================================================
// rock.store_marker tests
// =============================================================================

#store_marker_tmap = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 2, 64, 64] -> [1, 64, 128]>

// Element type mismatch between source and result
func.func @store_marker_elem_type_mismatch(%src: tensor<64x64xf32>, %i0: i32, %i1: i32, %i2: i32) -> tensor<1x64x128xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{failed to verify that all of {source, result} have same element type}}
  %0 = rock.store_marker %src views [#store_marker_tmap] [%i0, %i1, %i2] : tensor<64x64xf32> -> tensor<1x64x128xf16>
  return %0 : tensor<1x64x128xf16>
}

// Upper dims != source rank + extraIndices count
func.func @store_marker_rank_mismatch(%src: tensor<64x64xf32>, %i0: i32) -> tensor<1x64x128xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{upper bounds must equal tensor rank + extraIndices count}}
  %0 = rock.store_marker %src views [#store_marker_tmap] [%i0] : tensor<64x64xf32> -> tensor<1x64x128xf32>
  return %0 : tensor<1x64x128xf32>
}

// Upper bounds last dimensions mismatch with source shape
// #store_marker_tmap upper bounds = [1, 1, 2, 64, 64], source rank 2 → take_back(2) = [64, 64]
func.func @store_marker_upper_shape_mismatch(%src: tensor<32x32xf32>, %i0: i32, %i1: i32, %i2: i32) -> tensor<1x64x128xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Upper bounds last dimensions must match with result shape}}
  %0 = rock.store_marker %src views [#store_marker_tmap] [%i0, %i1, %i2] : tensor<32x32xf32> -> tensor<1x64x128xf32>
  return %0 : tensor<1x64x128xf32>
}

// Lower bounds mismatch with result shape
// #store_marker_tmap lower bounds = [1, 64, 128], result is [2, 64, 128]
func.func @store_marker_lower_shape_mismatch(%src: tensor<64x64xf32>, %i0: i32, %i1: i32, %i2: i32) -> tensor<2x64x128xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{Lower bounds must match with input shape}}
  %0 = rock.store_marker %src views [#store_marker_tmap] [%i0, %i1, %i2] : tensor<64x64xf32> -> tensor<2x64x128xf32>
  return %0 : tensor<2x64x128xf32>
}

// =============================================================================
// rock.untile tests
// =============================================================================

// Source rank greater than result rank
func.func @untile_source_rank_greater(%src: tensor<4x64x128xf16>) -> tensor<64x128xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{source rank is greater than result rank}}
  %0 = rock.untile %src : tensor<4x64x128xf16> -> tensor<64x128xf16>
  return %0 : tensor<64x128xf16>
}

// =============================================================================
// rock.transform tests
// =============================================================================

#xform = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 64 + d1, d2)> by [<Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 1] -> ["k"] at [0]>, <PassThrough ["n"] at [2] -> ["n"] at [1]>] bounds = [4, 64, 128] -> [256, 128]>

// Element type mismatch
func.func @transform_elem_type_mismatch(%arg0: tensor<256x128xf16>) -> tensor<4x64x128xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{failed to verify that all of {input, output} have same element type}}
  %0 = rock.transform %arg0 by #xform : tensor<256x128xf16> to tensor<4x64x128xf32>
  return %0 : tensor<4x64x128xf32>
}

// Input shape doesn't match lower bounds
func.func @transform_input_shape_mismatch(%arg0: tensor<128x128xf16>) -> tensor<4x64x128xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{input shape must match transform lower bounds}}
  %0 = rock.transform %arg0 by #xform : tensor<128x128xf16> to tensor<4x64x128xf16>
  return %0 : tensor<4x64x128xf16>
}

// Output shape doesn't match upper bounds
func.func @transform_output_shape_mismatch(%arg0: tensor<256x128xf16>) -> tensor<8x32x128xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+1 {{output shape must match transform upper bounds}}
  %0 = rock.transform %arg0 by #xform : tensor<256x128xf16> to tensor<8x32x128xf16>
  return %0 : tensor<8x32x128xf16>
}

// =============================================================================
// Pre-second-GEMM body verification tests
//
// `verifyGemmPlusGemmLikeOp` requires that, when the pre-second-GEMM region is
// non-empty, it contains a single block whose arguments are the first-GEMM
// result followed by one argument per elementwise input and whose terminator is
// a `rock.yield` that yields exactly one value. The same verifier is shared by
// `rock.gemm_elementwise_gemm`, `rock.conv_elementwise_gemm` and
// `rock.attention`.
// =============================================================================

// An empty pre-second-GEMM region is legal: the assembly format makes the
// elementwise clause optional, and the verifier should accept the op as-is.
func.func @gemm_elementwise_gemm_empty_body_is_legal(
    %a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %c: tensor<1x4x2xf32>)
    -> tensor<1x4x2xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %r = rock.gemm_elementwise_gemm{
   ab = %a * %b : tensor<1x4x4xf32>, tensor<1x4x4xf32>
   out = ab * %c : tensor<1x4x2xf32>
  } -> tensor<1x4x2xf32>
  return %r : tensor<1x4x2xf32>
}

// Multi-block region: the body must be a single block.
func.func @gemm_elementwise_gemm_body_multi_block(
    %a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %c: tensor<1x4x2xf32>)
    -> tensor<1x4x2xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+1 {{pre-second-GEMM region must contain a single block}}
  %r = rock.gemm_elementwise_gemm{
   ab = %a * %b : tensor<1x4x4xf32>, tensor<1x4x4xf32>
   ab = elementwise {
   ^bb0(%qk: tensor<1x4x4xf32>):
     cf.br ^bb1(%qk : tensor<1x4x4xf32>)
   ^bb1(%qk2: tensor<1x4x4xf32>):
     rock.yield %qk2 : tensor<1x4x4xf32>
   }
   out = ab * %c : tensor<1x4x2xf32>
  } -> tensor<1x4x2xf32>
  return %r : tensor<1x4x2xf32>
}

// Zero block arguments: the body's entry block must accept the running
// first-GEMM result as a block argument.
func.func @gemm_elementwise_gemm_body_no_block_args(
    %a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %c: tensor<1x4x2xf32>)
    -> tensor<1x4x2xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+1 {{pre-second-GEMM body argument count must be 1 (the first-GEMM result plus one per elementwise input), but is 0}}
  %r = rock.gemm_elementwise_gemm{
   ab = %a * %b : tensor<1x4x4xf32>, tensor<1x4x4xf32>
   ab = elementwise {
   ^bb0:
     %cst = arith.constant dense<0.0> : tensor<1x4x4xf32>
     rock.yield %cst : tensor<1x4x4xf32>
   }
   out = ab * %c : tensor<1x4x2xf32>
  } -> tensor<1x4x2xf32>
  return %r : tensor<1x4x2xf32>
}

// Non-zero but still wrong block argument count: when there are elementwise
// inputs, the entry block must accept one block argument for each of them.
func.func @gemm_elementwise_gemm_body_missing_elemwise_arg(
    %a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %c: tensor<1x4x2xf32>,
    %bias: tensor<1x4x4xf32>)
    -> tensor<1x4x2xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+1 {{pre-second-GEMM body argument count must be 2 (the first-GEMM result plus one per elementwise input), but is 1}}
  %r = rock.gemm_elementwise_gemm{
   ab = %a * %b : tensor<1x4x4xf32>, tensor<1x4x4xf32>
   ab = elementwise otherIns(%bias : tensor<1x4x4xf32>) {
   ^bb0(%qk: tensor<1x4x4xf32>):
     rock.yield %qk : tensor<1x4x4xf32>
   }
   out = ab * %c : tensor<1x4x2xf32>
  } -> tensor<1x4x2xf32>
  return %r : tensor<1x4x2xf32>
}

// Wrong terminator: the body must be terminated by a `rock.yield`. Use a
// self-branch to keep the region single-block while replacing the terminator.
func.func @gemm_elementwise_gemm_body_wrong_terminator(
    %a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %c: tensor<1x4x2xf32>)
    -> tensor<1x4x2xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+1 {{pre-second-GEMM body must be terminated by a rock.yield}}
  %r = rock.gemm_elementwise_gemm{
   ab = %a * %b : tensor<1x4x4xf32>, tensor<1x4x4xf32>
   ab = elementwise {
   ^bb0(%qk: tensor<1x4x4xf32>):
     cf.br ^bb0(%qk : tensor<1x4x4xf32>)
   }
   out = ab * %c : tensor<1x4x2xf32>
  } -> tensor<1x4x2xf32>
  return %r : tensor<1x4x2xf32>
}

// Yield with zero operands: must yield exactly one value.
func.func @gemm_elementwise_gemm_body_yield_zero_operands(
    %a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %c: tensor<1x4x2xf32>)
    -> tensor<1x4x2xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+1 {{pre-second-GEMM body must yield exactly one value}}
  %r = rock.gemm_elementwise_gemm{
   ab = %a * %b : tensor<1x4x4xf32>, tensor<1x4x4xf32>
   ab = elementwise {
   ^bb0(%qk: tensor<1x4x4xf32>):
     rock.yield
   }
   out = ab * %c : tensor<1x4x2xf32>
  } -> tensor<1x4x2xf32>
  return %r : tensor<1x4x2xf32>
}

// Yield with too many operands: must yield exactly one value.
func.func @gemm_elementwise_gemm_body_yield_two_operands(
    %a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %c: tensor<1x4x2xf32>,
    %bias: tensor<1x4x4xf32>)
    -> tensor<1x4x2xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+1 {{pre-second-GEMM body must yield exactly one value}}
  %r = rock.gemm_elementwise_gemm{
   ab = %a * %b : tensor<1x4x4xf32>, tensor<1x4x4xf32>
   ab = elementwise otherIns(%bias : tensor<1x4x4xf32>) {
   ^bb0(%qk: tensor<1x4x4xf32>, %b_in: tensor<1x4x4xf32>):
     %sum = arith.addf %qk, %b_in : tensor<1x4x4xf32>
     rock.yield %qk, %sum : tensor<1x4x4xf32>, tensor<1x4x4xf32>
   }
   out = ab * %c : tensor<1x4x2xf32>
  } -> tensor<1x4x2xf32>
  return %r : tensor<1x4x2xf32>
}

// The same body verifier is invoked for `rock.attention`. Sanity check that a
// malformed pre-softmax region is rejected on attention too.
func.func @attention_pre_softmax_yield_zero_operands(
    %q: tensor<1x4x4xf16>, %k: tensor<1x4x4xf16>,
    %v: tensor<1x4x2xf16>) -> tensor<1x4x2xf16>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+1 {{pre-second-GEMM body must yield exactly one value}}
  %r = rock.attention{
   qk = %q * %k : tensor<1x4x4xf16>, tensor<1x4x4xf16>
   qk = elementwise {
   ^bb0(%qk_in: tensor<1x4x4xf16>):
     rock.yield
   }
   softmax(qk) * %v : tensor<1x4x2xf16>
  } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 1 : i32} -> tensor<1x4x2xf16>
  return %r : tensor<1x4x2xf16>
}

// Sanity check that the body verifier also runs for `rock.attention`'s
// single-block requirement.
func.func @attention_pre_softmax_multi_block(
    %q: tensor<1x4x4xf16>, %k: tensor<1x4x4xf16>,
    %v: tensor<1x4x2xf16>) -> tensor<1x4x2xf16>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+1 {{pre-second-GEMM region must contain a single block}}
  %r = rock.attention{
   qk = %q * %k : tensor<1x4x4xf16>, tensor<1x4x4xf16>
   qk = elementwise {
   ^bb0(%qk_in: tensor<1x4x4xf16>):
     cf.br ^bb1(%qk_in : tensor<1x4x4xf16>)
   ^bb1(%qk2: tensor<1x4x4xf16>):
     rock.yield %qk2 : tensor<1x4x4xf16>
   }
   softmax(qk) * %v : tensor<1x4x2xf16>
  } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 1 : i32} -> tensor<1x4x2xf16>
  return %r : tensor<1x4x2xf16>
}

// Sanity check that the body verifier also runs for
// `rock.conv_elementwise_gemm`'s single-block requirement.
func.func @conv_elementwise_gemm_pre_second_gemm_multi_block(
    %filter: tensor<1x4x1x1x2xf32>, %input: tensor<2x2x2x1x2xf32>,
    %c: tensor<1x4x3xf32>) -> tensor<1x8x3xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+1 {{pre-second-GEMM region must contain a single block}}
  %r = rock.conv_elementwise_gemm{
   ab = conv(%filter, %input) : tensor<1x4x1x1x2xf32>, tensor<2x2x2x1x2xf32>
   ab = elementwise {
   ^bb0(%ab_in: tensor<1x4x8xf32>):
     cf.br ^bb1(%ab_in : tensor<1x4x8xf32>)
   ^bb1(%ab2: tensor<1x4x8xf32>):
     rock.yield %ab2 : tensor<1x4x8xf32>
   }
   out = ab * %c : tensor<1x4x3xf32>
  } {dilations = [1 : index, 1 : index],
     filter_layout = ["g", "k", "0", "1", "c"],
     input_layout = ["ni", "0i", "1i", "gi", "ci"],
     padding = [0 : index, 0 : index, 0 : index, 0 : index],
     strides = [1 : index, 1 : index]} -> tensor<1x8x3xf32>
  return %r : tensor<1x8x3xf32>
}

