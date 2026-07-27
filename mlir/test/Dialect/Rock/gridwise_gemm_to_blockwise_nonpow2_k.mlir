// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

// CHECK-NOT: rock.gridwise_gemm
// CHECK-LABEL: @gemm_nonpow2_kperblock_2_segments
// CHECK-SAME: -> tensor<1x64x64xf32>
func.func @gemm_nonpow2_kperblock_2_segments(%arg0: tensor<1x64x96xf16>, %arg1: tensor<1x96x64xf16>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32, rock.kernel} {
  // CHECK-DAG: %[[ACC0:.+]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  // CHECK:     %[[OUT:.+]] = scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[ACC0]]) -> (tensor<64x64xf32>)

  // CHECK: %[[ACC1:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC]]) : tensor<64x32xf16>, tensor<32x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK: %[[ACC2:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC1]]) : tensor<64x16xf16>, tensor<16x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK: scf.yield %[[ACC2]] : tensor<64x64xf32>

  // CHECK: rock.store_marker %[[OUT]] {{.*}} : tensor<64x64xf32> -> tensor<1x64x64xf32>
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 48, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x96xf16>, tensor<1x96x64xf16> -> tensor<1x64x64xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}

// -----

// CHECK-NOT: rock.gridwise_gemm
// CHECK-LABEL: @gemm_nonpow2_kperblock_3_segments
// CHECK-SAME: -> tensor<1x64x64xf32>
func.func @gemm_nonpow2_kperblock_3_segments(%arg0: tensor<1x64x224xf16>, %arg1: tensor<1x224x64xf16>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32, rock.kernel} {
  // CHECK-DAG: %[[ACC0:.+]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  // CHECK:     %[[OUT:.+]] = scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[ACC0]]) -> (tensor<64x64xf32>)

  // CHECK: %[[ACC1:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC]]) : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK: %[[ACC2:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC1]]) : tensor<64x32xf16>, tensor<32x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK: %[[ACC3:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC2]]) : tensor<64x16xf16>, tensor<16x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK: scf.yield %[[ACC3]] : tensor<64x64xf32>

  // CHECK: rock.store_marker %[[OUT]] {{.*}} : tensor<64x64xf32> -> tensor<1x64x64xf32>
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 112, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x224xf16>, tensor<1x224x64xf16> -> tensor<1x64x64xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}

// -----

// CHECK-LABEL: @gemm_pow2_kperblock_no_peel
// CHECK: %[[ACC0:.+]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NOT: rock.transform
// CHECK: %[[OUT:.+]] = scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[ACC0]]) -> (tensor<64x64xf32>)
// CHECK: %[[ACC1:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC]]) : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: scf.yield %[[ACC1]] : tensor<64x64xf32>
// CHECK-NOT: rock.blockwise_gemm
func.func @gemm_pow2_kperblock_no_peel(%arg0: tensor<1x64x128xf16>, %arg1: tensor<1x128x64xf16>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32, rock.kernel} {
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 64, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x128xf16>, tensor<1x128x64xf16> -> tensor<1x64x64xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}

// -----

// kPerBlock = 20 -> {16, 4}: accepted.

// CHECK-LABEL: @gemm_nonpow2_kperblock_i8
func.func @gemm_nonpow2_kperblock_i8(%arg0: tensor<1x64x80xi8>, %arg1: tensor<1x80x64xi8>, %arg2: tensor<1x64x64xi32>) -> tensor<1x64x64xi32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32, rock.kernel} {
  // CHECK: %[[ACC1:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %{{.*}}) : tensor<64x16xi8>, tensor<16x64xi8>, tensor<64x64xi32> -> tensor<64x64xi32>
  // CHECK: rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC1]]) : tensor<64x4xi8>, tensor<4x64xi8>, tensor<64x64xi32> -> tensor<64x64xi32>
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 20, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x80xi8>, tensor<1x80x64xi8> -> tensor<1x64x64xi32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xi32> -> tensor<1x64x64xi32> to tensor<1x64x64xi32>
  return %out : tensor<1x64x64xi32>
}

// -----

// 18 -> {16, 2}: the 2-wide segment has no integer dot instruction, so Triton
// would legalize it as f32 and round-trip the i32 accumulator through
// sitofp/fptosi, losing bits past f32's exact-integer range: rejected.

func.func @gemm_nonpow2_kperblock_i8_narrow_segment_unsupported(%arg0: tensor<1x64x72xi8>, %arg1: tensor<1x72x64xi8>, %arg2: tensor<1x64x64xi32>) -> tensor<1x64x64xi32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32, rock.kernel} {
  // expected-error @+2 {{non-power-of-two kPerBlock is not supported for integer operands when it decomposes into K segments narrower than 4}}
  // expected-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 18, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x72xi8>, tensor<1x72x64xi8> -> tensor<1x64x64xi32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xi32> -> tensor<1x64x64xi32> to tensor<1x64x64xi32>
  return %out : tensor<1x64x64xi32>
}

// -----

func.func @gemm_nonpow2_kperblock_scaled_unsupported(%arg0: tensor<1x64x96xf4E2M1FN>, %arg1: tensor<1x96x64xf4E2M1FN>, %scaleA: tensor<1x64x6xf8E8M0FNU>, %scaleB: tensor<1x64x6xf8E8M0FNU>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i32, rock.kernel} {
  // expected-error @+2 {{non-power-of-two kPerBlock is not supported for scaled gemm}}
  // expected-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
  %result = rock.gridwise_gemm(%arg0, %arg1, %scaleA, %scaleB) {quantBlockSize = 16 : i64, params = #rock.gemm_params<kPerBlock = 48, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x96xf4E2M1FN>, tensor<1x96x64xf4E2M1FN>, tensor<1x64x6xf8E8M0FNU>, tensor<1x64x6xf8E8M0FNU> -> tensor<1x64x64xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}

// -----

func.func @gemm_nonpow2_kperblock_f4_unsupported(%arg0: tensor<1x64x96xf4E2M1FN>, %arg1: tensor<1x96x64xf4E2M1FN>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i32, rock.kernel} {
  // expected-error @+2 {{non-power-of-two kPerBlock is not supported for 4-bit operands}}
  // expected-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 48, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x96xf4E2M1FN>, tensor<1x96x64xf4E2M1FN> -> tensor<1x64x64xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}

// -----

func.func @gemm_nonpow2_kperblock_i4_unsupported(%arg0: tensor<1x64x96xi4>, %arg1: tensor<1x96x64xi4>, %arg2: tensor<1x64x64xi32>) -> tensor<1x64x64xi32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i32, rock.kernel} {
  // expected-error @+2 {{non-power-of-two kPerBlock is not supported for 4-bit operands}}
  // expected-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 48, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x96xi4>, tensor<1x96x64xi4> -> tensor<1x64x64xi32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xi32> -> tensor<1x64x64xi32> to tensor<1x64x64xi32>
  return %out : tensor<1x64x64xi32>
}

// -----

// The GEMM operand is f16 here, but it is dequantized from an i4 kernel
// argument, so the same packing runs and the same restriction applies.

func.func @gemm_nonpow2_kperblock_dequant_i4_unsupported(%arg0: tensor<6144xi4>, %arg1: tensor<1x96x64xf16>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i32, rock.kernel} {
  %scale = arith.constant dense<2.500000e-01> : tensor<1x64x96xf16>
  %a = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 96 + d2)>
    by [<Unmerge{64, 96} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 96] -> [6144]>
    : tensor<6144xi4> to tensor<1x64x96xi4>
  %a_f16 = arith.sitofp %a : tensor<1x64x96xi4> to tensor<1x64x96xf16>
  %a_dequant = arith.mulf %a_f16, %scale : tensor<1x64x96xf16>
  // expected-error @+2 {{non-power-of-two kPerBlock is not supported for 4-bit operands}}
  // expected-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
  %result = rock.gridwise_gemm(%a_dequant, %arg1) {params = #rock.gemm_params<kPerBlock = 48, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x96xf16>, tensor<1x96x64xf16> -> tensor<1x64x64xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}
