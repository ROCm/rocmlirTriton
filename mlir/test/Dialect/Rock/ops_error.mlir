// RUN: rocmlir-opt -verify-diagnostics %s

// // TODO(roctriton): We need to unbufferize attention
// func.func @gridwise_attn_atomic_add_fail(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x64x384xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x64xf32>) attributes {rock.block_size = 64 : i32, grid_size = 24 : i32, kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
//   %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : tensor<1x384x64xf32> to tensor<1x64x384xf32>
//   
//   // expected-disabled-error @below {{Only set store method is supported for attention.}}
//   rock.gridwise_attention(%0, %arg1, %arg2, %arg3) preSoftmaxOps = {} {
//     blockSize = 64 : i32,
//     gridSize = 24 : i32,
//     params0 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
//     params1 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
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
// func.func @gridwise_attn_prefix_offset_requires_causal(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x64x384xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x64xf32>, %arg4: tensor<1xi32>) attributes {rock.block_size = 64 : i32, grid_size = 24 : i32, kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
//   %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : tensor<1x384x64xf32> to tensor<1x64x384xf32>
//   
//   // expected-disabled-error @below {{prefixOffset requires causal to be enabled}}
//   rock.gridwise_attention(%0, %arg1, %arg2, %arg4, %arg3) preSoftmaxOps = {} {
//     blockSize = 64 : i32,
//     gridSize = 24 : i32,
//     params0 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
//     params1 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
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
// func.func @attention_nonset(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @below {{Only set store method is supported for attention.}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod atomic_add>}
//   return
// }
// 
// func.func @attention_numheadskv_negative(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @below {{numHeadsKV must be positive}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = -1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }
// 
// func.func @attention_numheadsq_negative(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @below {{numHeadsQ must be positive}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = -1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }
// 
// func.func @attention_numheadsq_not_divisible(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @below {{numHeadsQ is not divisible by numHeadsKV}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 3 : i32, numHeadsQ = 4 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }
// 
// func.func @attention_numheadsq_smaller_than_numheadskv(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @below {{numHeadsQ is not divisible by numHeadsKV}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 4 : i32, numHeadsQ = 2 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }
// 
// func.func @attention_prefix_offset_requires_causal(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>, %arg4: tensor<1xi32>) attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @below {{prefixOffset requires causal to be enabled}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    prefixOffset = (%arg4 : tensor<1xi32>)
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }

// -----------------------------------------------------------------------------
// gemm tests 
// -----------------------------------------------------------------------------

// Test case: Matrix A with invalid rank (rank 1)
func.func @gemm_matrixA_wrong_rank(%a: tensor<64xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{Matrix A must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a * %b
    : tensor<64xf32> * tensor<128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Matrix A with invalid rank (rank 4)
func.func @gemm_matrixA_rank4(%a: tensor<1x2x64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{Matrix A must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a * %b
    : tensor<1x2x64x128xf32> * tensor<128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Matrix B with invalid rank (rank 1)
func.func @gemm_matrixB_wrong_rank(%a: tensor<64x128xf32>, %b: tensor<32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{Matrix B must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a * %b
    : tensor<64x128xf32> * tensor<32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Matrix B with invalid rank (rank 4)
func.func @gemm_matrixB_rank4(%a: tensor<64x128xf32>, %b: tensor<1x2x128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{Matrix B must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a * %b
    : tensor<64x128xf32> * tensor<1x2x128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Matrix C with invalid rank (rank 1)
func.func @gemm_matrixC_wrong_rank(%a: tensor<64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{op Result must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a * %b
    : tensor<64x128xf32> * tensor<128x32xf32> -> tensor<64xf32>
  func.return
}

// Test case: Matrix C with invalid rank (rank 4)
func.func @gemm_matrixC_rank4(%a: tensor<64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{op Result must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a * %b
    : tensor<64x128xf32> * tensor<128x32xf32> -> tensor<1x2x64x32xf32>
  func.return
}

// Test case: Mixed ranks - A is rank 3, B and C are rank 2
func.func @gemm_mixed_ranks1(%a: tensor<2x64x128xf32>, %b: tensor<128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{group dimensions don't match g_a = 2 g_b = 1 g_result = 1}}
  rock.gemm %a * %b
    : tensor<2x64x128xf32> * tensor<128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: Mixed ranks - B is rank 3, A and C are rank 2
func.func @gemm_mixed_ranks2(%a: tensor<64x128xf32>, %b: tensor<2x128x32xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{group dimensions don't match g_a = 1 g_b = 2 g_result = 1}}
  rock.gemm %a * %b
    : tensor<64x128xf32> * tensor<2x128x32xf32> -> tensor<64x32xf32>
  func.return
}

// Test case: missing quantBlockSize
func.func @gemm_scaleA_wrong_rank(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                                  %scaleA: tensor<64x4xf8E8M0FNU>,
                                  %scaleB: tensor<32x4xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{quantBlockSize not defined}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB
    : tensor<64x128xf4E2M1FN> scaled by tensor<64x4xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// Test case: ScaleA with invalid rank (rank 1)
func.func @gemm_scaleA_wrong_rank(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                                  %scaleA: tensor<64xf8E8M0FNU>,
                                  %scaleB: tensor<32x4xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB {quantBlockSize = 32 : i64}
    : tensor<64x128xf4E2M1FN> scaled by tensor<64xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// Test case: ScaleA with invalid rank (rank 4)
func.func @gemm_scaleA_rank4(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                             %scaleA: tensor<1x2x64x128xf8E8M0FNU>,
                             %scaleB: tensor<32x4xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB {quantBlockSize = 32 : i64}
    : tensor<64x128xf4E2M1FN> scaled by tensor<1x2x64x128xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// Test case: ScaleB with invalid rank (rank 1)
func.func @gemm_scaleB_wrong_rank(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                                  %scaleA: tensor<64x4xf8E8M0FNU>,
                                  %scaleB: tensor<32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB {quantBlockSize = 32 : i64}
    : tensor<64x128xf4E2M1FN> scaled by tensor<64x4xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// Test case: ScaleB with invalid rank (rank 4)
func.func @gemm_scaleB_rank4(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>, 
                             %scaleA: tensor<64x4xf8E8M0FNU>,
                             %scaleB: tensor<1x2x128x32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB must be a rank 2 or rank 3 tensor representing [G,] D, K}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB {quantBlockSize = 32 : i64}
    : tensor<64x128xf4E2M1FN> scaled by tensor<64x4xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<1x2x128x32xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

func.func @gemm_scale_presence_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x4xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  rock.gemm %a scaled by %scaleA * %b {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_k_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA_kbad: tensor<2x64x3xf8E8M0FNU>, %scaleB: tensor<2x32x4xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA's K dimension must match matrix A's K dimension}}
  rock.gemm %a scaled by %scaleA_kbad * %b scaled by %scaleB {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x3xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x32x4xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_m_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA_mbad: tensor<2x63x4xf8E8M0FNU>, %scaleB: tensor<2x32x4xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA's M dimension must match matrix A's M dimension}}
  rock.gemm %a scaled by %scaleA_mbad * %b scaled by %scaleB {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x63x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x32x4xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_g_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA_gbad: tensor<3x64x4xf8E8M0FNU>, %scaleB: tensor<2x32x4xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA's G dimension must match matrix A's G dimension}}
  rock.gemm %a scaled by %scaleA_gbad * %b scaled by %scaleB {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<3x64x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x32x4xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleB_k_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x4xf8E8M0FNU>, %scaleB_kbad: tensor<2x32x3xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB's K dimension must match matrix B's K dimension}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB_kbad {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x32x3xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleB_n_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x4xf8E8M0FNU>, %scaleB_nbad: tensor<2x31x4xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB's N dimension must match matrix B's N dimension}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB_nbad {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x31x4xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleB_g_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x4xf8E8M0FNU>, %scaleB_gbad: tensor<3x32x4xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB's G dimension must match matrix B's G dimension}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB_gbad {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<3x32x4xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

func.func @gemm_scaleA_transposed_k_mismatch(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>,
  %scaleA_tbad: tensor<3x64xf8E8M0FNU>, %scaleB: tensor<32x4xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA's K dimension must match matrix A's K dimension}}
  rock.gemm %a scaled by tr %scaleA_tbad * %b scaled by %scaleB {quantBlockSize = 32 : i64}
  : tensor<64x128xf4E2M1FN> scaled by tensor<3x64xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

func.func @gemm_scaleB_transposed_k_mismatch(%a: tensor<2x64x128xf4E2M1FN>, %b: tensor<2x128x32xf4E2M1FN>,
  %scaleA: tensor<2x64x4xf8E8M0FNU>, %scaleB_kbad: tensor<2x3x32xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB's K dimension must match matrix B's K dimension}}
  rock.gemm %a scaled by %scaleA * %b scaled by tr %scaleB_kbad {quantBlockSize = 32 : i64}
  : tensor<2x64x128xf4E2M1FN> scaled by tensor<2x64x4xf8E8M0FNU> * tensor<2x128x32xf4E2M1FN> scaled by tensor<2x3x32xf8E8M0FNU> -> tensor<2x64x32xf32>
  func.return
}

// scaleA type must be f8E8M0FNU
func.func @gemm_scaleA_type_invalid(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>,
  %scaleA_bad: tensor<64x4xf8E4M3FN>, %scaleB: tensor<32x4xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{'rock.gemm' op operand #2 must be tensor of f8E8M0FNU type values, but got 'tensor<64x4xf8E4M3FN>'}}
  rock.gemm %a scaled by %scaleA_bad * %b scaled by %scaleB {quantBlockSize = 32 : i64}
  : tensor<64x128xf4E2M1FN> scaled by tensor<64x4xf8E4M3FN> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E8M0FNU> -> tensor<64x32xf32>
  func.return
}

// scaleB type must be f8E8M0FNU
func.func @gemm_scaleB_type_invalid(%a: tensor<64x128xf4E2M1FN>, %b: tensor<128x32xf4E2M1FN>,
  %scaleA: tensor<64x4xf8E8M0FNU>, %scaleB_bad: tensor<32x4xf8E4M3FN>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{'rock.gemm' op operand #3 must be tensor of f8E8M0FNU type values, but got 'tensor<32x4xf8E4M3FN>'}}
  rock.gemm %a scaled by %scaleA * %b scaled by %scaleB_bad {quantBlockSize = 32 : i64}
  : tensor<64x128xf4E2M1FN> scaled by tensor<64x4xf8E8M0FNU> * tensor<128x32xf4E2M1FN> scaled by tensor<32x4xf8E4M3FN> -> tensor<64x32xf32>
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

func.func @gridwise_gemm_accel_scale_presence_a_only(%A: tensor<1x4x8xf4E2M1FN>, %B: tensor<1x4x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleA: tensor<1x4x8xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA) {
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
  %result = rock.gridwise_gemm(%A, %B, %scaleB) {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : tensor<1x4x8xf4E2M1FN>, tensor<1x4x16xf4E2M1FN>, tensor<1x4x16xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleA dims mismatch
func.func @gridwise_gemm_accel_scaleA_dims_mismatch(%A: tensor<1x8x32xf4E2M1FN>, %B: tensor<1x32x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleA_bad_dims: tensor<1x8x7xf8E8M0FNU>, %scaleB: tensor<1x16x1xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{ScaleA shape must match matrixA shape.}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA_bad_dims, %scaleB) {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    quantBlockSize = 32 : i64,
    params = #common_params
  } : tensor<1x8x32xf4E2M1FN>, tensor<1x32x16xf4E2M1FN>, tensor<1x8x7xf8E8M0FNU>, tensor<1x16x1xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleB dims mismatch
func.func @gridwise_gemm_accel_scaleB_dims_mismatch(%A: tensor<1x8x32xf4E2M1FN>, %B: tensor<1x32x16xf4E2M1FN>, %C: tensor<1x8x16xf32>, %scaleA_bad_dims: tensor<1x8x1xf8E8M0FNU>, %scaleB: tensor<1x16x2xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{ScaleB shape must match matrixB shape.}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA_bad_dims, %scaleB) {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    quantBlockSize = 32 : i64,
    params = #common_params
  } : tensor<1x8x32xf4E2M1FN>, tensor<1x32x16xf4E2M1FN>, tensor<1x8x1xf8E8M0FNU>, tensor<1x16x2xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleA type must be f8E8M0FNU
func.func @gridwise_gemm_scaleA_type_invalid(%A: tensor<1x8x32xf4E2M1FN>, %B: tensor<1x32x16xf4E2M1FN>, %C: tensor<1x8x16xf32>,
    %scaleA_bad: tensor<1x8x1xf8E4M3FN>, %scaleB: tensor<1x16x1xf8E8M0FNU>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{'rock.gridwise_gemm' op operand #2 must be 3D tensor of f8E8M0FNU type values, but got 'tensor<1x8x1xf8E4M3FN>'}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA_bad, %scaleB) {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    quantBlockSize = 32 : i64,
    params = #common_params
  } : tensor<1x8x32xf4E2M1FN>, tensor<1x32x16xf4E2M1FN>, tensor<1x8x1xf8E4M3FN>, tensor<1x16x1xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// scaleB type must be f8E8M0FNU
func.func @gridwise_gemm_scaleB_type_invalid(%A: tensor<1x8x32xf4E2M1FN>, %B: tensor<1x32x16xf4E2M1FN>, %C: tensor<1x8x16xf32>,
    %scaleA: tensor<1x8x1xf8E8M0FNU>, %scaleB_bad: tensor<1x16x1xf8E4M3FN>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{'rock.gridwise_gemm' op operand #3 must be 3D tensor of f8E8M0FNU type values, but got 'tensor<1x16x1xf8E4M3FN>'}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB_bad) {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    quantBlockSize = 32 : i64,
    params = #common_params
  } : tensor<1x8x32xf4E2M1FN>, tensor<1x32x16xf4E2M1FN>, tensor<1x8x1xf8E8M0FNU>, tensor<1x16x1xf8E4M3FN> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}

// -----------------------------------------------------------------------------
// Blockwise gemm tests 
// -----------------------------------------------------------------------------
// TODO(roctriton): Scaled gemm tests need rework
// #blockwise_params = #rock.gemm_params<
//   kPerBlock = 2,
//   kpack = 1,
//   mPerBlock = 128,
//   nPerBlock = 128,
//   numWaves = 4,
//   matrixInstrNonkdim = 0,
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//     {
//       rock.arch = "amdgcn-amd-amdhsa:gfx942",
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
//   matrixInstrNonkdim = 0,
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
//    {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//    {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//    {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//    {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//    {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//    {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//    {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//    {
//       rock.arch = "amdgcn-amd-amdhsa:gfx950",
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
//    {
//       rock.arch = "amdgcn-amd-amdhsa:gfx942", // Unsupported architecture for Float4E2M1FN
//       params = #params
//     } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5>
//   return
// }

// =============================================================================
// rock.store tests
// =============================================================================

// Element type mismatch between source, dest, and result
func.func @store_elem_type_mismatch(
    %source: tensor<4x4xf32>, %dest: tensor<4x4xf32>) -> tensor<4x4xf16> {
  // expected-error @+1 {{failed to verify that all of {source, dest, result} have same element type}}
  %out = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf16> to tensor<4x4xf32>
  return %out : tensor<4x4xf16>
}

// Shape mismatch between source and dest
func.func @store_shape_mismatch(
    %source: tensor<4x4xf32>, %dest: tensor<8x8xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{source and dest shapes must match}}
  %out = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<8x8xf32>
  return %out : tensor<4x4xf32>
}

// Result used by a non-return op
func.func @store_result_not_returned(
    %source: tensor<4x4xf32>, %dest: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{result must be used directly by a func.return}}
  %out = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  %neg = arith.negf %out : tensor<4x4xf32>
  return %neg : tensor<4x4xf32>
}

// Result has multiple uses
func.func @store_result_multiple_uses(
    %source: tensor<4x4xf32>, %dest: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  // expected-error @+1 {{result must have at most one use (a func.return)}}
  %out = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  return %out, %out : tensor<4x4xf32>, tensor<4x4xf32>
}

// =============================================================================
// rock.cast_to_ptr tests
// =============================================================================

// Source is not i32
func.func @cast_to_ptr_src_not_i32(%src: tensor<64x64xf32>) -> tensor<64x64x!tt.ptr<f16>> {
  // expected-error @+1 {{operand #0 must be ranked tensor of 32-bit signless integer values}}
  %0 = rock.cast_to_ptr %src : tensor<64x64xf32> -> tensor<64x64x!tt.ptr<f16>>
  return %0 : tensor<64x64x!tt.ptr<f16>>
}

// Result is not a pointer tensor
func.func @cast_to_ptr_result_not_ptr(%src: tensor<64x64xi32>) -> tensor<64x64xf16> {
  // expected-error @+1 {{result must be a tensor of !tt.ptr}}
  %0 = rock.cast_to_ptr %src : tensor<64x64xi32> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// Shape mismatch between src and result
func.func @cast_to_ptr_shape_mismatch(%src: tensor<32x64xi32>) -> tensor<64x64x!tt.ptr<f16>> {
  // expected-error @+1 {{failed to verify that all of {src, result} have same shape}}
  %0 = rock.cast_to_ptr %src : tensor<32x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  return %0 : tensor<64x64x!tt.ptr<f16>>
}

// =============================================================================
// rock.extract_ptr tests
// =============================================================================

// Source is not a block argument
func.func @extract_ptr_not_block_arg(%src: tensor<64x64xf32>) -> i32 {
  %cst = arith.constant dense<0.0> : tensor<64x64xf32>
  // expected-error @+1 {{source must be a block argument}}
  %0 = rock.extract_ptr %cst : tensor<64x64xf32> -> i32
  return %0 : i32
}

// =============================================================================
// rock.blockwise_reduce tests
// =============================================================================

// Axis out of range
func.func @blockwise_reduce_axis_oob(%input: tensor<64x64xf32>) -> tensor<64xf32> {
  // expected-error @+1 {{axis is out of range}}
  %0 = rock.blockwise_reduce sum %input {axis = 3 : index} : tensor<64x64xf32> -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// Rank mismatch (output rank must be input rank - 1)
func.func @blockwise_reduce_rank_mismatch(%input: tensor<64x64xf32>) -> tensor<64x64xf32> {
  // expected-error @+1 {{output rank must be input rank - 1}}
  %0 = rock.blockwise_reduce sum %input {axis = 0 : index} : tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// Non-reduction dimension mismatch
func.func @blockwise_reduce_non_red_dim_mismatch(%input: tensor<64x64xf32>) -> tensor<32xf32> {
  // expected-error @+1 {{non-reduction dimension size mismatch at output dim 0}}
  %0 = rock.blockwise_reduce sum %input {axis = 1 : index} : tensor<64x64xf32> -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// Element type mismatch
func.func @blockwise_reduce_elem_type_mismatch(%input: tensor<64x64xf32>) -> tensor<64xf16> {
  // expected-error @+1 {{failed to verify that all of {input, result} have same element type}}
  %0 = rock.blockwise_reduce sum %input {axis = 1 : index} : tensor<64x64xf32> -> tensor<64xf16>
  return %0 : tensor<64xf16>
}

// =============================================================================
// rock.transforms_to_ptr tests
// =============================================================================

// Pointers element type not i32
func.func @transforms_to_ptr_ptr_not_i32(%src: tensor<64x64xf32>) -> (tensor<64x64xf16>, tensor<64x64xi1>) {
  // expected-error @+1 {{result #0 must be ranked tensor of 32-bit signless integer values}}
  %ptrs, %mask = rock.transforms_to_ptr %src : tensor<64x64xf32> -> tensor<64x64xf16>, tensor<64x64xi1>
  return %ptrs, %mask : tensor<64x64xf16>, tensor<64x64xi1>
}

// Mask element type not i1
func.func @transforms_to_ptr_mask_not_i1(%src: tensor<64x64xf32>) -> (tensor<64x64xi32>, tensor<64x64xi32>) {
  // expected-error @+1 {{result #1 must be ranked tensor of 1-bit signless integer values}}
  %ptrs, %mask = rock.transforms_to_ptr %src : tensor<64x64xf32> -> tensor<64x64xi32>, tensor<64x64xi32>
  return %ptrs, %mask : tensor<64x64xi32>, tensor<64x64xi32>
}

// Shape mismatch between pointers and mask
func.func @transforms_to_ptr_shape_mismatch(%src: tensor<64x64xf32>) -> (tensor<32x32xi32>, tensor<64x64xi1>) {
  // expected-error @+1 {{failed to verify that all of {pointers, mask} have same shape}}
  %ptrs, %mask = rock.transforms_to_ptr %src : tensor<64x64xf32> -> tensor<32x32xi32>, tensor<64x64xi1>
  return %ptrs, %mask : tensor<32x32xi32>, tensor<64x64xi1>
}

// Rank mismatch: extraIndices + pointers.rank != source.rank
func.func @transforms_to_ptr_rank_mismatch(
    %src: tensor<4x1x1x2x64x64xf16>, %i0: i32) -> (tensor<64x64xi32>, tensor<64x64xi1>) {
  // expected-error @+1 {{extraIndices.size() + pointers rank must equal source rank}}
  %ptrs, %mask = rock.transforms_to_ptr %src[%i0] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  return %ptrs, %mask : tensor<64x64xi32>, tensor<64x64xi1>
}

// Source last dimensions mismatch with pointers shape
func.func @transforms_to_ptr_last_dims_mismatch(
    %src: tensor<4x1x1x2x64x64xf16>, %i0: i32, %i1: i32, %i2: i32, %i3: i32) -> (tensor<32x64xi32>, tensor<32x64xi1>) {
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
    %i0: i32, %i1: i32, %i2: i32) -> tensor<32768xf16> {
  // expected-error @+1 {{failed to verify that all of {source, dest, result} have same element type}}
  %0 = rock.blockwise_store %src -> %dest[%i0, %i1, %i2] by set
    : tensor<16x16xf32> -> tensor<3x4x32x16x16xf32> -> tensor<32768xf16>
  return %0 : tensor<32768xf16>
}

// Rank mismatch: extraIndices + source.rank != dest.rank
func.func @blockwise_store_rank_mismatch(
    %src: tensor<16x16xf32>, %dest: tensor<3x4x32x16x16xf32>,
    %i0: i32) -> tensor<32768xf32> {
  // expected-error @+1 {{extraIndices.size() + source rank must equal dest rank}}
  %0 = rock.blockwise_store %src -> %dest[%i0] by set
    : tensor<16x16xf32> -> tensor<3x4x32x16x16xf32> -> tensor<32768xf32>
  return %0 : tensor<32768xf32>
}

// Result not used by return
func.func @blockwise_store_not_returned(
    %src: tensor<16x16xf32>, %dest: tensor<3x16x16xf32>,
    %i0: i32) -> tensor<768xf32> {
  // expected-error @+1 {{result must be used directly by a func.return}}
  %0 = rock.blockwise_store %src -> %dest[%i0] by set
    : tensor<16x16xf32> -> tensor<3x16x16xf32> -> tensor<768xf32>
  %neg = arith.negf %0 : tensor<768xf32>
  return %neg : tensor<768xf32>
}

// Result has multiple uses
func.func @blockwise_store_multiple_uses(
    %src: tensor<16x16xf32>, %dest: tensor<3x16x16xf32>,
    %i0: i32) -> (tensor<768xf32>, tensor<768xf32>) {
  // expected-error @+1 {{result must have at most one use (a func.return)}}
  %0 = rock.blockwise_store %src -> %dest[%i0] by set
    : tensor<16x16xf32> -> tensor<3x16x16xf32> -> tensor<768xf32>
  return %0, %0 : tensor<768xf32>, tensor<768xf32>
}

// Dest last dimensions shape mismatch
func.func @blockwise_store_shape_mismatch(
    %src: tensor<16x32xf32>, %dest: tensor<3x4x32x16x16xf32>,
    %i0: i32, %i1: i32, %i2: i32) -> tensor<32768xf32> {
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
    %src: tensor<4x1x1x2x64x64xf16>, %i0: i32, %i1: i32, %i2: i32, %i3: i32) -> tensor<64x64xf32> {
  // expected-error @+1 {{failed to verify that all of {source, result} have same element type}}
  %0 = rock.blockwise_load %src[%i0, %i1, %i2, %i3] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// Rank mismatch: sourceIndices.size() + result.rank != source.rank
func.func @blockwise_load_rank_mismatch(
    %src: tensor<4x1x1x2x64x64xf16>, %i0: i32) -> tensor<64x64xf16> {
  // expected-error @+1 {{sourceIndices.size() + result rank must equal source rank}}
  %0 = rock.blockwise_load %src[%i0] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// Input last dimensions shape mismatch
func.func @blockwise_load_shape_mismatch(
    %src: tensor<4x1x1x2x64x64xf16>, %i0: i32, %i1: i32, %i2: i32, %i3: i32) -> tensor<64x32xf16> {
  // expected-error @+1 {{Input last dimensions must match with result shape}}
  %0 = rock.blockwise_load %src[%i0, %i1, %i2, %i3] : tensor<4x1x1x2x64x64xf16> -> tensor<64x32xf16>
  return %0 : tensor<64x32xf16>
}

// =============================================================================
// rock.blockwise_load_ptr tests
// =============================================================================

// Pointer tensor element type not i32
func.func @blockwise_load_ptr_ptr_not_i32(
    %ptrs: tensor<64x64xf32>, %mask: tensor<64x64xi1>) -> tensor<64x64xf16> {
  // expected-error @+1 {{operand #0 must be ranked tensor of 32-bit signless integer values}}
  %0 = rock.blockwise_load_ptr %ptrs[%mask] : tensor<64x64xf32>, tensor<64x64xi1> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// Mask tensor element type not i1
func.func @blockwise_load_ptr_mask_not_i1(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi32>) -> tensor<64x64xf16> {
  // expected-error @+1 {{operand #1 must be ranked tensor of 1-bit signless integer values}}
  %0 = rock.blockwise_load_ptr %ptrs[%mask] : tensor<64x64xi32>, tensor<64x64xi32> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// Shape mismatch between pointers and result
func.func @blockwise_load_ptr_shape_mismatch(
    %ptrs: tensor<32x32xi32>, %mask: tensor<64x64xi1>) -> tensor<64x64xf16> {
  // expected-error @+1 {{failed to verify that all of {pointerTensor, maskTensor, result} have same shape}}
  %0 = rock.blockwise_load_ptr %ptrs[%mask] : tensor<32x32xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// =============================================================================
// rock.blockwise_store_ptr tests
// =============================================================================

// Pointer tensor element type not i32
func.func @blockwise_store_ptr_ptr_not_i32(
    %src: tensor<64x64xf32>, %ptrs: tensor<64x64xf16>, %mask: tensor<64x64xi1>) -> tensor<64x64xf32> {
  // expected-error @+1 {{operand #0 must be ranked tensor of 32-bit signless integer values}}
  %0 = rock.blockwise_store_ptr %src -> %ptrs(%mask) by set
    : tensor<64x64xf32> -> tensor<64x64xf16>(tensor<64x64xi1>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// Mask tensor element type not i1
func.func @blockwise_store_ptr_mask_not_i1(
    %src: tensor<64x64xf32>, %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi32>) -> tensor<64x64xf32> {
  // expected-error @+1 {{operand #1 must be ranked tensor of 1-bit signless integer values}}
  %0 = rock.blockwise_store_ptr %src -> %ptrs(%mask) by set
    : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// Shape mismatch between pointers, mask, and source
func.func @blockwise_store_ptr_shape_mismatch(
    %src: tensor<64x64xf32>, %ptrs: tensor<32x32xi32>, %mask: tensor<64x64xi1>) -> tensor<64x64xf32> {
  // expected-error @+1 {{failed to verify that all of {pointerTensor, maskTensor, source} have same shape}}
  %0 = rock.blockwise_store_ptr %src -> %ptrs(%mask) by set
    : tensor<64x64xf32> -> tensor<32x32xi32>(tensor<64x64xi1>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// Element type mismatch between source and result
func.func @blockwise_store_ptr_elem_mismatch(
    %src: tensor<64x64xf32>, %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>) -> tensor<64x64xf16> {
  // expected-error @+1 {{failed to verify that all of {source, result} have same element type}}
  %0 = rock.blockwise_store_ptr %src -> %ptrs(%mask) by set
    : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// Result not used by return
func.func @blockwise_store_ptr_not_returned(
    %src: tensor<64x64xf32>, %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>) -> tensor<64x64xf32> {
  // expected-error @+1 {{result must be used directly by a func.return}}
  %0 = rock.blockwise_store_ptr %src -> %ptrs(%mask) by set
    : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<64x64xf32>
  %neg = arith.negf %0 : tensor<64x64xf32>
  return %neg : tensor<64x64xf32>
}

// =============================================================================
// rock.load_marker tests
// =============================================================================

#load_marker_tmap = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 64 + d1, d2)> by [<Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 1] -> ["k"] at [0]>, <PassThrough ["n"] at [2] -> ["n"] at [1]>] bounds = [4, 64, 128] -> [256, 128]>

// Element type mismatch between source and result
func.func @load_marker_elem_type_mismatch(%src: tensor<256x128xf16>, %i0: i32) -> tensor<64x128xf32> {
  // expected-error @+1 {{failed to verify that all of {source, result} have same element type}}
  %0 = rock.load_marker %src views [#load_marker_tmap] [%i0] : tensor<256x128xf16> -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// Upper dims != result rank + extraIndices count
func.func @load_marker_rank_mismatch(%src: tensor<256x128xf16>, %i0: i32, %i1: i32) -> tensor<64x128xf16> {
  // expected-error @+1 {{upper bounds must equal tensor rank + extraIndices count}}
  %0 = rock.load_marker %src views [#load_marker_tmap] [%i0, %i1] : tensor<256x128xf16> -> tensor<64x128xf16>
  return %0 : tensor<64x128xf16>
}

// Upper bounds last dimensions mismatch with result shape
// #load_marker_tmap upper bounds = [4, 64, 128], result rank 2 → take_back(2) = [64, 128]
func.func @load_marker_upper_shape_mismatch(%src: tensor<256x128xf16>, %i0: i32) -> tensor<32x128xf16> {
  // expected-error @+1 {{Upper bounds last dimensions must match with result shape}}
  %0 = rock.load_marker %src views [#load_marker_tmap] [%i0] : tensor<256x128xf16> -> tensor<32x128xf16>
  return %0 : tensor<32x128xf16>
}

// Lower bounds mismatch with source shape
// #load_marker_tmap lower bounds = [256, 128], source is [128, 128]
func.func @load_marker_lower_shape_mismatch(%src: tensor<128x128xf16>, %i0: i32) -> tensor<64x128xf16> {
  // expected-error @+1 {{Lower bounds must match with input shape}}
  %0 = rock.load_marker %src views [#load_marker_tmap] [%i0] : tensor<128x128xf16> -> tensor<64x128xf16>
  return %0 : tensor<64x128xf16>
}

// =============================================================================
// rock.store_marker tests
// =============================================================================

#store_marker_tmap = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 2, 64, 64] -> [1, 64, 128]>

// Element type mismatch between source and result
func.func @store_marker_elem_type_mismatch(%src: tensor<64x64xf32>, %i0: i32, %i1: i32, %i2: i32) -> tensor<1x64x128xf16> {
  // expected-error @+1 {{failed to verify that all of {source, result} have same element type}}
  %0 = rock.store_marker %src views [#store_marker_tmap] [%i0, %i1, %i2] : tensor<64x64xf32> -> tensor<1x64x128xf16>
  return %0 : tensor<1x64x128xf16>
}

// Upper dims != source rank + extraIndices count
func.func @store_marker_rank_mismatch(%src: tensor<64x64xf32>, %i0: i32) -> tensor<1x64x128xf32> {
  // expected-error @+1 {{upper bounds must equal tensor rank + extraIndices count}}
  %0 = rock.store_marker %src views [#store_marker_tmap] [%i0] : tensor<64x64xf32> -> tensor<1x64x128xf32>
  return %0 : tensor<1x64x128xf32>
}

// Upper bounds last dimensions mismatch with source shape
// #store_marker_tmap upper bounds = [1, 1, 2, 64, 64], source rank 2 → take_back(2) = [64, 64]
func.func @store_marker_upper_shape_mismatch(%src: tensor<32x32xf32>, %i0: i32, %i1: i32, %i2: i32) -> tensor<1x64x128xf32> {
  // expected-error @+1 {{Upper bounds last dimensions must match with result shape}}
  %0 = rock.store_marker %src views [#store_marker_tmap] [%i0, %i1, %i2] : tensor<32x32xf32> -> tensor<1x64x128xf32>
  return %0 : tensor<1x64x128xf32>
}

// Lower bounds mismatch with result shape
// #store_marker_tmap lower bounds = [1, 64, 128], result is [2, 64, 128]
func.func @store_marker_lower_shape_mismatch(%src: tensor<64x64xf32>, %i0: i32, %i1: i32, %i2: i32) -> tensor<2x64x128xf32> {
  // expected-error @+1 {{Lower bounds must match with input shape}}
  %0 = rock.store_marker %src views [#store_marker_tmap] [%i0, %i1, %i2] : tensor<64x64xf32> -> tensor<2x64x128xf32>
  return %0 : tensor<2x64x128xf32>
}

// =============================================================================
// rock.untile tests
// =============================================================================

// Source rank greater than result rank
func.func @untile_source_rank_greater(%src: tensor<4x64x128xf16>) -> tensor<64x128xf16> {
  // expected-error @+1 {{source rank is greater than result rank}}
  %0 = rock.untile %src : tensor<4x64x128xf16> -> tensor<64x128xf16>
  return %0 : tensor<64x128xf16>
}

// =============================================================================
// rock.transform tests
// =============================================================================

#xform = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 64 + d1, d2)> by [<Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 1] -> ["k"] at [0]>, <PassThrough ["n"] at [2] -> ["n"] at [1]>] bounds = [4, 64, 128] -> [256, 128]>

// Element type mismatch
func.func @transform_elem_type_mismatch(%arg0: tensor<256x128xf16>) -> tensor<4x64x128xf32> {
  // expected-error @+1 {{failed to verify that all of {input, output} have same element type}}
  %0 = rock.transform %arg0 by #xform : tensor<256x128xf16> to tensor<4x64x128xf32>
  return %0 : tensor<4x64x128xf32>
}

// Input shape doesn't match lower bounds
func.func @transform_input_shape_mismatch(%arg0: tensor<128x128xf16>) -> tensor<4x64x128xf16> {
  // expected-error @+1 {{input shape must match transform lower bounds}}
  %0 = rock.transform %arg0 by #xform : tensor<128x128xf16> to tensor<4x64x128xf16>
  return %0 : tensor<4x64x128xf16>
}

// Output shape doesn't match upper bounds
func.func @transform_output_shape_mismatch(%arg0: tensor<256x128xf16>) -> tensor<8x32x128xf16> {
  // expected-error @+1 {{output shape must match transform upper bounds}}
  %0 = rock.transform %arg0 by #xform : tensor<256x128xf16> to tensor<8x32x128xf16>
  return %0 : tensor<8x32x128xf16>
}

