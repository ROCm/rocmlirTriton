// RUN: rocmlir-opt -verify-diagnostics %s

// // TODO(roctriton): We need to unbufferize attention
// func.func @gridwise_attn_atomic_add_fail(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x64x384xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x64xf32>) attributes {block_size = 64 : i32, grid_size = 24 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
//   %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : tensor<1x384x64xf32> to tensor<1x64x384xf32>
//   
//   // expected-disabled-error @below {{Only set store method is supported for attention.}}
//   rock.gridwise_attention(%0, %arg1, %arg2, %arg3) preSoftmaxOps = {} {
//     blockSize = 64 : i32,
//     gridSize = 24 : i32,
//     params0 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
//     params1 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
//     firstGemmIndices = array<i64: 0>,
//     storeMethod = #rock<StoreMethod atomic_add>,
//     splitKV = 1 : i32,
//     enableSoftmax = true,
//     numHeadsKV = 1 : i32, 
//     numHeadsQ = 1 : i32,
//     operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 0>
//   } : tensor<1x64x384xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32>, tensor<1x384x64xf32>
//   return
// }
// 
// func.func @gridwise_attn_prefix_offset_requires_causal(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x64x384xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x64xf32>, %arg4: tensor<1xi32>) attributes {block_size = 64 : i32, grid_size = 24 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
//   %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : tensor<1x384x64xf32> to tensor<1x64x384xf32>
//   
//   // expected-disabled-error @below {{prefixOffset requires causal to be enabled}}
//   rock.gridwise_attention(%0, %arg1, %arg2, %arg4, %arg3) preSoftmaxOps = {} {
//     blockSize = 64 : i32,
//     gridSize = 24 : i32,
//     params0 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
//     params1 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
//     firstGemmIndices = array<i64: 0>,
//     storeMethod = #rock<StoreMethod set>,
//     splitKV = 1 : i32,
//     enableSoftmax = true,
//     numHeadsKV = 1 : i32, 
//     numHeadsQ = 1 : i32,
//     operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 1, 1, 0>
//   } : tensor<1x64x384xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32>, tensor<1xi32>, tensor<1x384x64xf32>
//   return
// }
// 
// func.func @attention_nonset(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @below {{Only set store method is supported for attention.}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod atomic_add>}
//   return
// }
// 
// func.func @attention_numheadskv_negative(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @below {{numHeadsKV must be positive}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = -1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }
// 
// func.func @attention_numheadsq_negative(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @below {{numHeadsQ must be positive}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = -1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }
// 
// func.func @attention_numheadsq_not_divisible(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @below {{numHeadsQ is not divisible by numHeadsKV}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 3 : i32, numHeadsQ = 4 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }
// 
// func.func @attention_numheadsq_smaller_than_numheadskv(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @below {{numHeadsQ is not divisible by numHeadsKV}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 4 : i32, numHeadsQ = 2 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }
// 
// func.func @attention_prefix_offset_requires_causal(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>, %arg4: tensor<1xi32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @below {{prefixOffset requires causal to be enabled}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    prefixOffset = (%arg4 : tensor<1xi32>)
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }

// -----------------------------------------------------------------------------
// gemm tests 
// -----------------------------------------------------------------------------

// Test case: Matrix A with invalid rank (rank 1)
func.func @gemm_matrixA_wrong_rank(%a: tensor<64xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{Matrix A must be a rank 2 or rank 3 tensor representing [G,] M, K}}
  rock.gemm %a * %b features = dot
    : tensor<64xf32> * tensor<128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Matrix A with invalid rank (rank 4)
func.func @gemm_matrixA_rank4(%a: tensor<1x2x64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{Matrix A must be a rank 2 or rank 3 tensor representing [G,] M, K}}
  rock.gemm %a * %b features = dot
    : tensor<1x2x64x128xf32> * tensor<128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Matrix B with invalid rank (rank 1)
func.func @gemm_matrixB_wrong_rank(%a: tensor<64x128xf32>, %b: tensor<32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{Matrix B must be a rank 2 or rank 3 tensor representing [G,] M, K}}
  rock.gemm %a * %b features = dot
    : tensor<64x128xf32> * tensor<32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Matrix B with invalid rank (rank 4)
func.func @gemm_matrixB_rank4(%a: tensor<64x128xf32>, %b: tensor<1x2x128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{Matrix B must be a rank 2 or rank 3 tensor representing [G,] M, K}}
  rock.gemm %a * %b features = dot
    : tensor<64x128xf32> * tensor<1x2x128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Matrix C with invalid rank (rank 1)
func.func @gemm_matrixC_wrong_rank(%a: tensor<64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{op Result must be a rank 2 or rank 3 tensor representing [G,] M, K}}
  rock.gemm %a * %b features = dot
    : tensor<64x128xf32> * tensor<128x32xf32> -> tensor<64xf32>
  func.return
}

// Test case: Matrix C with invalid rank (rank 4)
func.func @gemm_matrixC_rank4(%a: tensor<64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{op Result must be a rank 2 or rank 3 tensor representing [G,] M, K}}
  rock.gemm %a * %b features = dot
    : tensor<64x128xf32> * tensor<128x32xf32> -> tensor<1x2x64x32xf32>
  func.return
}

// Test case: Mixed ranks - A is rank 3, B and C are rank 2
func.func @gemm_mixed_ranks1(%a: tensor<2x64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{group dimensions don't match g_a = 2 g_b = 1 g_result = 1}}
  rock.gemm %a * %b features = dot
    : tensor<2x64x128xf32> * tensor<128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Mixed ranks - B is rank 3, A and C are rank 2
func.func @gemm_mixed_ranks2(%a: tensor<64x128xf32>, %b: tensor<2x128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{group dimensions don't match g_a = 1 g_b = 2 g_result = 1}}
  rock.gemm %a * %b features = dot
    : tensor<64x128xf32> * tensor<2x128x32xf32> -> tensor<64x32xf32>
  func.return
}

// TODO(roctriton): Scaled gemm tests need rework
// Test case: ScaleA with invalid rank (rank 1)
func.func @gemm_scaleA_wrong_rank(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                                  %scaleA: tensor<128xf8E8M0FNU>,
                                  %scaleB: tensor<128x32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA must be a rank 2 or rank 3 tensor representing [G,] M, K}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB features = mfma
    : tensor<64x128xf4E2M1FN> scaled by tensor<128xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<128x32xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// Test case: ScaleA with invalid rank (rank 4)
func.func @gemm_scaleA_rank4(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                             %scaleA: tensor<1x2x64x128xf8E8M0FNU>,
                             %scaleB: tensor<128x32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA must be a rank 2 or rank 3 tensor representing [G,] M, K}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB features = mfma
    : tensor<64x128xf4E2M1FN> scaled by tensor<1x2x64x128xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<128x32xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// Test case: ScaleB with invalid rank (rank 1)
func.func @gemm_scaleB_wrong_rank(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                                  %scaleA: tensor<64x128xf8E8M0FNU>,
                                  %scaleB: tensor<32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB must be a rank 2 or rank 3 tensor representing [G,] M, K}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB features = mfma
    : tensor<64x128xf4E2M1FN> scaled by tensor<64x128xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// Test case: ScaleB with invalid rank (rank 4)
func.func @gemm_scaleB_rank4(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                             %scaleA: tensor<64x128xf8E8M0FNU>,
                             %scaleB: tensor<1x2x128x32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB must be a rank 2 or rank 3 tensor representing [G,] M, K}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB features = mfma
    : tensor<64x128xf4E2M1FN> scaled by tensor<64x128xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<1x2x128x32xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

func.func @gemm_scale_presence_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x128xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  rock.gemm %a scaled by %scaleA * %b features = mfma
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x128xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_type_invalid(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA_bad: tensor<2x64x128xf8E4M3FN>, %scaleB: tensor<2x128x32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{'rock.gemm' op operand #2 must be Constraints the type to be either a Tensor or MemRef of certain types of elements., but got 'tensor<2x64x128xf8E4M3FN>'}}
  rock.gemm %a scaled by %scaleA_bad * %b scaled by %scaleB features = mfma
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x128xf8E4M3FN> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x128x32xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_k_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA_kbad: tensor<2x64x127xf8E8M0FNU>, %scaleB: tensor<2x128x32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA's K dimension must match matrix A's K dimension}}
  rock.gemm %a scaled by %scaleA_kbad * %b scaled by %scaleB features = mfma
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x127xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x128x32xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_m_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA_mbad: tensor<2x63x128xf8E8M0FNU>, %scaleB: tensor<2x128x32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA's M dimension must match matrix A's M dimension}}
  rock.gemm %a scaled by %scaleA_mbad * %b scaled by %scaleB features = mfma
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x63x128xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x128x32xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_g_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA_gbad: tensor<3x64x128xf8E8M0FNU>, %scaleB: tensor<2x128x32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA's G dimension must match matrix A's G dimension}}
  rock.gemm %a scaled by %scaleA_gbad * %b scaled by %scaleB features = mfma
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<3x64x128xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x128x32xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleB_k_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x128xf8E8M0FNU>, %scaleB_kbad: tensor<2x127x32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB's K dimension must match matrix B's K dimension}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB_kbad features = mfma
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x128xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x127x32xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleB_n_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x128xf8E8M0FNU>, %scaleB_nbad: tensor<2x128x31xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB's N dimension must match matrix B's N dimension}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB_nbad features = mfma
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x128xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x128x31xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleB_g_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x128xf8E8M0FNU>, %scaleB_gbad: tensor<3x128x32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB's G dimension must match matrix B's G dimension}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB_gbad features = mfma
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x128xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<3x128x32xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleB_type_invalid(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA : tensor<2x64x128xf8E8M0FNU>, %scaleB_bad : tensor<2x128x32xf8E4M3FN>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{'rock.gemm' op operand #3 must be Constraints the type to be either a Tensor or MemRef of certain types of elements., but got 'tensor<2x128x32xf8E4M3FN>'}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB_bad features = mfma
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x128xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x128x32xf8E4M3FN> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_transposed_k_mismatch(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>,
  %scaleA_tbad: tensor<127x64xf8E8M0FNU>, %scaleB: tensor<128x32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA's K dimension must match matrix A's K dimension}}
  rock.gemm %a scaled by tr %scaleA_tbad * %b scaled by %scaleB features = mfma
  : tensor<64x128xf4E2M1FN> scaled by tensor<127x64xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<128x32xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

func.func @gemm_scaleB_transposed_k_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x128xf8E8M0FNU>, %scaleB_kbad: tensor<2x32x127xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB's K dimension must match matrix B's K dimension}}
  rock.gemm %a scaled by %scaleA * %b scaled by tr %scaleB_kbad features = mfma
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x128xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x32x127xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @rock_scaled_gemm_invalid_arch(%a : tensor<32x64xf4E2M1FN>, %b : tensor<1x32x128xf4E2M1FN>, %scaleA : tensor<32x64xf8E8M0FNU>, %scaleB : tensor<1x32x128xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  // expected-error @+1 {{'rock.gemm' op Mfma does not support Float4E2M1FN data type}}
  rock.gemm tr %a scaled by tr %scaleA * %b scaled by %scaleB features = mfma
  : tensor<32x64xf4E2M1FN> scaled by tensor<32x64xf8E8M0FNU> * tensor<1x32x128xf4E2M1FN> scaled by tensor<1x32x128xf8E8M0FNU> -> tensor<64x128xf32>
  func.return
}

func.func @gemm_scaled_inputs_not_float4e2m1(%a: tensor<2x64x128xf16>,
                                            %b: tensor<2x128x32xf16>,
                                            %scaleA: tensor<2x64x128xf8E8M0FNU>,
                                            %scaleB: tensor<2x128x32xf8E8M0FNU>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{Scaled GEMMs are only supported for Float4E2M1FN input type}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB features = mfma
    : tensor<2x64x128xf16> scaled by tensor<2x64x128xf8E8M0FNU> *
      tensor<2x128x32xf16> scaled by tensor<2x128x32xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

// -----------------------------------------------------------------------------
// Gridwise gemm tests 
// -----------------------------------------------------------------------------

// TODO(roctriton): Scaled gemm tests need rework
#common_params = #rock.gemm_params<
  kPerBlock = 4,
  kpack = 4,
  mPerBlock = 64,
  nPerBlock = 64,
  numWaves = 4,
  matrixInstrNonkdim = 32,
  splitKFactor = 1,
  numStages = 2,
  wavesPerEU = 0, gridGroupSize = 0,
  numCTAs = 1>

func.func @gridwise_gemm_accel_scale_presence_a_only(%A: tensor<1x4x8xf4E2M1FN>, %B: tensor<1x4x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleA: tensor<1x4x8xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : tensor<1x4x8xf4E2M1FN>, tensor<1x4x16xf4E2M1FN>, tensor<1x4x8xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// Scale presence B only
func.func @gridwise_gemm_accel_scale_presence_b_only(%A: tensor<1x4x8xf4E2M1FN>, %B: tensor<1x4x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleB: tensor<1x4x16xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  %result = rock.gridwise_gemm(%A, %B, %scaleB) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : tensor<1x4x8xf4E2M1FN>, tensor<1x4x16xf4E2M1FN>, tensor<1x4x16xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleA type invalid
func.func @gridwise_gemm_accel_scaleA_type_invalid(%A: tensor<1x4x8xf4E2M1FN>, %B: tensor<1x4x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleA_bad: tensor<1x4x8xf8E4M3FN>, %scaleB: tensor<1x4x16xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{'rock.gridwise_gemm' op operand #2 must be 3D tensor of f8E8M0FNU type values, but got 'tensor<1x4x8xf8E4M3FN>'}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA_bad, %scaleB) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : tensor<1x4x8xf4E2M1FN>, tensor<1x4x16xf4E2M1FN>, tensor<1x4x8xf8E4M3FN>, tensor<1x4x16xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleA dims mismatch
func.func @gridwise_gemm_accel_scaleA_dims_mismatch(%A: tensor<1x4x8xf4E2M1FN>, %B: tensor<1x4x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleA_bad_dims: tensor<1x4x7xf8E8M0FNU>, %scaleB: tensor<1x4x16xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{ScaleA shape must match matrixA shape.}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA_bad_dims, %scaleB) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : tensor<1x4x8xf4E2M1FN>, tensor<1x4x16xf4E2M1FN>, tensor<1x4x7xf8E8M0FNU>, tensor<1x4x16xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleA input type invalid
func.func @gridwise_gemm_accel_scaleA_input_type_invalid(%A: tensor<1x4x8xf16>, %B: tensor<1x4x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleA: tensor<1x4x8xf8E8M0FNU>, %scaleB: tensor<1x4x16xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{For the scaled GEMMs, matrixA must be of type Float4E2M1FNType.}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : tensor<1x4x8xf16>, tensor<1x4x16xf4E2M1FN>, tensor<1x4x8xf8E8M0FNU>, tensor<1x4x16xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleB type invalid
func.func @gridwise_gemm_accel_scaleB_type_invalid(%A: tensor<1x4x8xf4E2M1FN>, %B: tensor<1x4x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleA: tensor<1x4x8xf8E8M0FNU>, %scaleB_bad: tensor<1x4x16xf8E4M3FN>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{'rock.gridwise_gemm' op operand #3 must be 3D tensor of f8E8M0FNU type values, but got 'tensor<1x4x16xf8E4M3FN>'}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB_bad) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : tensor<1x4x8xf4E2M1FN>, tensor<1x4x16xf4E2M1FN>, tensor<1x4x8xf8E8M0FNU>, tensor<1x4x16xf8E4M3FN> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleB dims mismatch
func.func @gridwise_gemm_accel_scaleB_dims_mismatch(%A: tensor<1x4x8xf4E2M1FN>, %B: tensor<1x4x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleA: tensor<1x4x8xf8E8M0FNU>, %scaleB_bad_dims: tensor<1x4x15xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{ScaleB shape must match matrixB shape.}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB_bad_dims) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : tensor<1x4x8xf4E2M1FN>, tensor<1x4x16xf4E2M1FN>, tensor<1x4x8xf8E8M0FNU>, tensor<1x4x15xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleB input type invalid
func.func @gridwise_gemm_accel_scaleB_input_type_invalid(%A: tensor<1x4x8xf4E2M1FN>, %B: tensor<1x4x16xf16>, %C: tensor<1x8x16xf32>, %scaleA: tensor<1x4x8xf8E8M0FNU>, %scaleB: tensor<1x4x16xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{For the scaled GEMMs, matrixB must be of type Float4E2M1FNType.}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : tensor<1x4x8xf4E2M1FN>, tensor<1x4x16xf16>, tensor<1x4x8xf8E8M0FNU>, tensor<1x4x16xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// Invalid arch 
func.func @rock_gridwise_gemm_accel_invalid_arch(%A: tensor<2x1024x1024xf4E2M1FN>, %B: tensor<2x1024x2048xf4E2M1FN>, %C: tensor<2x1024x2048xf32>, %scaleA : tensor<2x1024x1024xf8E8M0FNU>, %scaleB : tensor<2x1024x2048xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", numCU = 304 : i32} {
  // expected-error @+1 {{'rock.gridwise_gemm' op Mfma does not support Float4E2M1FN data type}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB) features = mfma {
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : tensor<2x1024x1024xf4E2M1FN>, tensor<2x1024x2048xf4E2M1FN>, tensor<2x1024x1024xf8E8M0FNU>, tensor<2x1024x2048xf8E8M0FNU> -> tensor<2x1024x2048xf32>
  %stored = rock.store %result to %C by set : tensor<2x1024x2048xf32> -> tensor<2x1024x2048xf32> to tensor<2x1024x2048xf32>
  return
}

// out data type invalid
func.func @rock_gridwise_gemm_accel_invalid_out_dtype(%A: tensor<2x1024x1024xf4E2M1FN>, %B: tensor<2x1024x2048xf4E2M1FN>, %C: tensor<2x1024x2048xf16>, %scaleA : tensor<2x1024x1024xf8E8M0FNU>, %scaleB : tensor<2x1024x2048xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", numCU = 256 : i32} {
  // expected-error @+1 {{'rock.gridwise_gemm' op 4-bit or 8-bit float input requires f32 output}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB) features = mfma {
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : tensor<2x1024x1024xf4E2M1FN>, tensor<2x1024x2048xf4E2M1FN>, tensor<2x1024x1024xf8E8M0FNU>, tensor<2x1024x2048xf8E8M0FNU> -> tensor<2x1024x2048xf16>
  %stored = rock.store %result to %C by set : tensor<2x1024x2048xf16> -> tensor<2x1024x2048xf16> to tensor<2x1024x2048xf16>
  return
}

// -----------------------------------------------------------------------------
// Blockwise gemm tests 
// -----------------------------------------------------------------------------
// TODO(roctriton): Scaled gemm tests need rework
// #blockwise_params = #rock.gemm_params<
//   kPerBlock = 2,
//   kpack = 2,
//   mPerBlock = 128,
//   nPerBlock = 128,
//   numWaves = 4,
//   matrixInstrNonkdim = 32,
//   splitKFactor = 1,
//   numStages = 2,
//   wavesPerEU = 0, gridGroupSize = 0,
//   numCTAs = 1>
// 
// func.func @blockwise_gemm_accel_scale_buffer_presence_a_only(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{scaleA and scaleB buffers must both be present or both be null.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA
//     scaled by %bufferScaleA
//     * %bufferB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       blockSize = 256 : i32,
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_loadA_scaleA_lds_only(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleB: memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{If scaleA is loaded from LDS, scaleA buffer must be non-null.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA
//     scaled by from %matrixScaleA
//     * %bufferB from %matrixB
//     scaled by %bufferScaleB from %matrixScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       blockSize = 256 : i32,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_scaleA_lds_shape_mismatch(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA_bad: memref<128xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA: memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>,
//   %bufferScaleB: memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{If scaleA is loaded from LDS, its shape must match matrixA's shape.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA
//     scaled by %bufferScaleA from %matrixScaleA_bad
//     * %bufferB from %matrixB
//     scaled by %bufferScaleB from %matrixScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       blockSize = 256 : i32,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>> from memref<128xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_scaleA_lds_type_bad(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA_bad: memref<256xvector<2xf8E4M3FN>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA: memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>,
//   %bufferScaleB: memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{ScaleA must be of type Float8E8M0FNU.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA
//     scaled by %bufferScaleA from %matrixScaleA_bad
//     * %bufferB from %matrixB
//     scaled by %bufferScaleB from %matrixScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       blockSize = 256 : i32,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>> from memref<256xvector<2xf8E4M3FN>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_matrixA_type_bad(
//   %matrixA_bad: memref<256xvector<2xf16>, #gpu.address_space<workgroup>>,
//   %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{For the scaled GEMMs, matrixA must be of type Float4E2M1FNType.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA_bad
//     scaled by %bufferScaleA from %matrixScaleA
//     * %bufferB from %matrixB
//     scaled by %bufferScaleB from %matrixScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f16, elementTypeLoad = f16, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       blockSize = 256 : i32,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf16>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_scaleA_buffer_shape_bad(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA_bad: memref<5xf8E8M0FNU, #gpu.address_space<private>>,
//   %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{If scaleA buffer is non-null, its shape must match bufferA's shape.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA
//     scaled by %bufferScaleA_bad from %matrixScaleA
//     * %bufferB from %matrixB
//     scaled by %bufferScaleB from %matrixScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       blockSize = 256 : i32,
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<5xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_scaleA_buffer_type_bad(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA_bad: memref<4xf8E4M3FN, #gpu.address_space<private>>,
//   %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{ScaleA buffer must be of type Float8E8M0FNU.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA
//     scaled by %bufferScaleA_bad from %matrixScaleA
//     * %bufferB from %matrixB
//     scaled by %bufferScaleB from %matrixScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       blockSize = 256 : i32,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E4M3FN, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_bufferA_type_bad(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA_bad: memref<4xf16, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{For the scaled GEMMs, bufferA must be of type Float4E2M1FNType.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA_bad from %matrixA
//     scaled by %bufferScaleA from %matrixScaleA
//     * %bufferB from %matrixB
//     scaled by %bufferScaleB from %matrixScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       blockSize = 256 : i32,
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf16, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_scale_buffer_presence_b_only(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{scaleA and scaleB buffers must both be present or both be null.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA
//     * %bufferB
//     scaled by %bufferScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       blockSize = 256 : i32,
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_scaleB_lds_shape_mismatch(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleB_bad: memref<255xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{If scaleB is loaded from LDS, its shape must match matrixB's shape.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA
//     scaled by %bufferScaleA from %matrixScaleA
//     * %bufferB from %matrixB
//     scaled by %bufferScaleB from %matrixScaleB_bad
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       blockSize = 256 : i32,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<255xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_scaleB_lds_type_bad(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleB_bad: memref<256xvector<2xf8E4M3FN>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{ScaleB must be of type Float8E8M0FNU.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA
//     scaled by %bufferScaleA from %matrixScaleA
//     * %bufferB from %matrixB
//     scaled by %bufferScaleB from %matrixScaleB_bad
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       blockSize = 256 : i32,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E4M3FN>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_matrixB_type_bad(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixB_bad: memref<256xvector<2xf16>, #gpu.address_space<workgroup>>,
//   %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{For the scaled GEMMs, matrixB must be of type Float4E2M1FNType.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA
//     scaled by %bufferScaleA from %matrixScaleA
//     * %bufferB from %matrixB_bad
//     scaled by %bufferScaleB from %matrixScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f16, elementTypeLoad = f16, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       blockSize = 256 : i32,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf16>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_scaleB_buffer_shape_bad(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %bufferScaleB_bad: memref<5xf8E8M0FNU, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{If scaleB buffer is non-null, its shape must match bufferB's shape.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA
//     scaled by %bufferScaleA from %matrixScaleA
//     * %bufferB from %matrixB
//     scaled by %bufferScaleB_bad from %matrixScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       blockSize = 256 : i32,
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<5xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_scaleB_buffer_type_bad(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %bufferScaleB_bad: memref<4xf8E4M3FN, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{ScaleB buffer must be of type Float8E8M0FNU.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA
//     scaled by %bufferScaleA from %matrixScaleA
//     * %bufferB from %matrixB
//     scaled by %bufferScaleB_bad from %matrixScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       blockSize = 256 : i32,
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E4M3FN, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_bufferB_type_bad(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB_bad: memref<4xf16, #gpu.address_space<private>>,
//   %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{For the scaled GEMMs, bufferB must be of type Float4E2M1FNType.}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA
//     scaled by %bufferScaleA from %matrixScaleA
//     * %bufferB_bad from %matrixB
//     scaled by %bufferScaleB from %matrixScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       blockSize = 256 : i32,
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       params = #blockwise_params
//     } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>>  from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf16, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }
// 
// func.func @blockwise_gemm_accel_invalid_arch(
//   %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
//   %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
//   %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
//   %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
//   %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
// ) {
//   // expected-disabled-error @+1 {{'rock.blockwise_gemm' op Mfma does not support Float4E2M1FN data type}}
//   rock.blockwise_gemm
//     %matrixC
//     += %bufferA from %matrixA
//     scaled by %bufferScaleA from %matrixScaleA
//     * %bufferB from %matrixB
//     scaled by %bufferScaleB from %matrixScaleB
//     features = mfma {
//       arch = "amdgcn-amd-amdhsa:gfx942",
//       loadAfromLDS,
//       loadBfromLDS,
//       blockSize = 256 : i32,
//       matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//       matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//       params = #blockwise_params    
//       } : memref<4xvector<16xf32>, #gpu.address_space<private>>
//         += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//         * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
//         scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
//   return
// }

//===----------------------------------------------------------------------===//
// Test cases for rock.threadwise_gemm
//===----------------------------------------------------------------------===//

// // TODO(roctriton): Scaled gemm tests need rework
// #params = #rock.gemm_params<
//   mPerBlock = 256,
//   nPerBlock = 256,
//   kPerBlock = 16,
//   numWaves = 8,
//   matrixInstrNonkdim = 32,
//   kpack = 1,
//   splitKFactor = 1, 
//   numStages = 2,
//   wavesPerEU = 0, gridGroupSize = 0,
//   numCTAs = 1>
// 
// // Error case: Only scaleA provided
// func.func @threadwise_gemm_accel_scale_mismatch1(
//   %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
//   %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
//   %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
//   %scaleA: memref<2x4xf8E8M0FNU, 5>      // matches matrixA
// ) {
//   %c0 = arith.constant 0 : index
//   // expected-disabled-error @+1 {{ScaleA and ScaleB must both be present or both be null.}}
//   rock.threadwise_gemm 
//     %matrixC += %matrixA scaled by %scaleA * %matrixB at [%c0, %c0, %c0] 
//     features = mfma{
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       params = #params
//     } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5>
//   return
// }
// 
// // Error case: Only scaleB provided
// func.func @threadwise_gemm_accel_scale_mismatch2(
//   %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
//   %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
//   %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
//   %scaleB: memref<3x4xf8E8M0FNU, 5>      // matches matrixB
// ) {
//   %c0 = arith.constant 0 : index
//   // expected-disabled-error @+1 {{ScaleA and ScaleB must both be present or both be null.}}
//   rock.threadwise_gemm 
//     %matrixC += %matrixA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] 
//     features = mfma{
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       params = #params
//     } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5>
//   return
// }
// 
// // Error case: Wrong scale type for scaleA
// func.func @threadwise_gemm_accel_wrong_scale_type_A(
//   %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
//   %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
//   %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
//   %scaleA_wrong: memref<2x4xf8E4M3FN, 5>,  // Wrong type
//   %scaleB: memref<3x4xf8E8M0FNU, 5>      // matches matrixB
// ) {
//   %c0 = arith.constant 0 : index
//   // expected-disabled-error @+1 {{ScaleA must be of type Float8E8M0FNU.}}
//   rock.threadwise_gemm 
//     %matrixC += %matrixA scaled by %scaleA_wrong * %matrixB scaled by %scaleB at [%c0, %c0, %c0] 
//     features = mfma{
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       params = #params
//     } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E4M3FN, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5>
//   return
// }
// 
// // Error case: Wrong scale type for scaleB
// func.func @threadwise_gemm_accel_wrong_scale_type_B(
//   %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
//   %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
//   %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
//   %scaleA: memref<2x4xf8E8M0FNU, 5>,     // matches matrixA
//   %scaleB_wrong: memref<3x4xf8E4M3FN, 5>  // Wrong type
// ) {
//   %c0 = arith.constant 0 : index
//   // expected-disabled-error @+1 {{ScaleB must be of type Float8E8M0FNU.}}
//   rock.threadwise_gemm 
//     %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB_wrong at [%c0, %c0, %c0] 
//     features = mfma{
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       params = #params
//     } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E4M3FN, 5>
//   return
// }
// 
// // Error case: Wrong input type for matrixA with scaling
// func.func @threadwise_gemm_accel_wrong_matrix_type_A(
//   %matrixA_wrong: memref<2x4xf16, 5>,    // Not f4E2M1FN
//   %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
//   %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
//   %scaleA: memref<2x4xf8E8M0FNU, 5>,     // matches matrixA dimensions
//   %scaleB: memref<3x4xf8E8M0FNU, 5>      // matches matrixB
// ) {
//   %c0 = arith.constant 0 : index
//   // expected-disabled-error @+1 {{For the scaled GEMMs, matrixA must be of type Float4E2M1FNType.}}
//   rock.threadwise_gemm 
//     %matrixC += %matrixA_wrong scaled by %scaleA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] 
//     features = mfma{
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       params = #params
//     } : memref<2x3xf32, 5> += memref<2x4xf16, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5>
//   return
// }
// 
// // Error case: Wrong input type for matrixB with scaling
// func.func @threadwise_gemm_accel_wrong_matrix_type_B(
//   %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
//   %matrixB_wrong: memref<3x4xf16, 5>,    // Not f4E2M1FN
//   %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
//   %scaleA: memref<2x4xf8E8M0FNU, 5>,     // matches matrixA
//   %scaleB: memref<3x4xf8E8M0FNU, 5>      // matches matrixB dimensions
// ) {
//   %c0 = arith.constant 0 : index
//   // expected-disabled-error @+1 {{For the scaled GEMMs, matrixB must be of type Float4E2M1FNType.}}
//   rock.threadwise_gemm 
//     %matrixC += %matrixA scaled by %scaleA * %matrixB_wrong scaled by %scaleB at [%c0, %c0, %c0] 
//     features = mfma{
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       params = #params
//     } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf16, 5> scaled by memref<3x4xf8E8M0FNU, 5>
//   return
// }
// 
// // Error case: Scale A shape doesn't match matrixA shape
// func.func @threadwise_gemm_accel_scale_shape_mismatch_A(
//   %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
//   %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
//   %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
//   %scaleA_wrong: memref<3x4xf8E8M0FNU, 5>,  // Wrong shape (m dimension mismatch)
//   %scaleB: memref<3x4xf8E8M0FNU, 5>      // matches matrixB
// ) {
//   %c0 = arith.constant 0 : index
//   // expected-disabled-error @+1 {{ScaleA shape must match matrixA shape.}}
//   rock.threadwise_gemm 
//     %matrixC += %matrixA scaled by %scaleA_wrong * %matrixB scaled by %scaleB at [%c0, %c0, %c0] 
//     features = mfma{
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       params = #params
//     } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5>
//   return
// }
// 
// // Error case: Scale B shape doesn't match matrixB shape
// func.func @threadwise_gemm_accel_scale_shape_mismatch_B(
//   %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
//   %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
//   %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
//   %scaleA: memref<2x4xf8E8M0FNU, 5>,     // matches matrixA
//   %scaleB_wrong: memref<4x4xf8E8M0FNU, 5>  // Wrong shape (n dimension mismatch)
// ) {
//   %c0 = arith.constant 0 : index
//   // expected-disabled-error @+1 {{ScaleB shape must match matrixB shape.}}
//   rock.threadwise_gemm 
//     %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB_wrong at [%c0, %c0, %c0] 
//     features = mfma{
//       arch = "amdgcn-amd-amdhsa:gfx950",
//       params = #params
//     } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<4x4xf8E8M0FNU, 5>
//   return
// }
// 
// // Error case: Architecture not supporting Float4E2M1FN
// func.func @threadwise_gemm_accel_unsupported_arch(
//   %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
//   %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
//   %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
//   %scaleA: memref<2x4xf8E8M0FNU, 5>,     // matches matrixA
//   %scaleB: memref<3x4xf8E8M0FNU, 5>      // matches matrixB
// ) {
//   %c0 = arith.constant 0 : index
//   // expected-disabled-error @+1 {{Mfma does not support Float4E2M1FN data type}}
//   rock.threadwise_gemm 
//     %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] 
//     features = mfma{
//       arch = "amdgcn-amd-amdhsa:gfx942", // Unsupported architecture for Float4E2M1FN
//       params = #params
//     } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5>
//   return
// }

