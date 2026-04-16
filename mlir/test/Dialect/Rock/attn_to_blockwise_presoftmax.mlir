// RUN: rocmlir-opt -rock-gridwise-attn-to-blockwise -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @attn_mask_reshape_nonzero_qk_idx
// The load_marker for the mask should bridge from tile coords to the
// flat tensor<4096xi8> (Unmerge transform is on the external input,
// captured by untransform into the LoadMarkerOp views).
// CHECK: rock.load_marker {{.*}}tensor<4096xi8> -> tensor<32x32xi8>
// The arith.trunci from the body should be cloned with tile-sized types.
// CHECK: arith.trunci {{.*}} tensor<32x32xi8> to tensor<32x32xi1>
func.func @attn_mask_reshape_nonzero_qk_idx(
    %q: tensor<1x64x32xf32>,
    %k: tensor<1x32x64xf32>,
    %v: tensor<1x64x32xf32>,
    %mask: tensor<4096xi8>) -> tensor<1x64x32xf32>
    attributes {
      rock.block_size = 256 : i32,
      rock.grid_size = 2 : i32,
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"
    } {
  %mask_reshaped = rock.transform %mask by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 64 + d2)> by [<Unmerge{1, 64, 64} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 64, 64] -> [4096]> : tensor<4096xi8> to tensor<1x64x64xi8>
  %result = rock.gridwise_attention(%q, %k, %v, %mask_reshaped) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x64x64xf32>, %arg_mask: tensor<1x64x64xi8>):
    %0 = arith.trunci %arg_mask : tensor<1x64x64xi8> to tensor<1x64x64xi1>
    rock.yield %arg_qk : tensor<1x64x64xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x64x32xf32>, tensor<1x32x64xf32>, tensor<1x64x32xf32>, tensor<1x64x64xi8> -> tensor<1x64x32xf32>
  return %result : tensor<1x64x32xf32>
}

// -----

// CHECK-LABEL: func @attn_scalar_broadcast
// The load_marker for the scale should bridge from tile coords to the
// scalar tensor<1xf32> (broadcast + reshape transforms on external input,
// captured by untransform into the LoadMarkerOp views).
// CHECK: rock.load_marker {{.*}}tensor<1xf32> -> tensor<32x32xf32>
// The mulf should use tile-sized types.
// CHECK: arith.mulf {{.*}} tensor<32x32xf32>
func.func @attn_scalar_broadcast(
    %q: tensor<1x64x32xf32>,
    %k: tensor<1x32x64xf32>,
    %v: tensor<1x64x32xf32>,
    %scale: tensor<1xf32>) -> tensor<1x64x32xf32>
    attributes {
      rock.block_size = 256 : i32,
      rock.grid_size = 2 : i32,
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"
    } {
  %s0 = rock.transform %scale by <affine_map<(d0, d1, d2) -> (d0 + d1 + d2)> by [<Unmerge{1, 1, 1} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 1, 1] -> [1]> : tensor<1xf32> to tensor<1x1x1xf32>
  %s1 = rock.transform %s0 by <affine_map<(d0, d1, d2) -> (d0, 0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [1, 64, 64] -> [1, 1, 1]> : tensor<1x1x1xf32> to tensor<1x64x64xf32>
  %result = rock.gridwise_attention(%q, %k, %v, %s1) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x64x64xf32>, %arg_scale: tensor<1x64x64xf32>):
    %0 = arith.mulf %arg_qk, %arg_scale : tensor<1x64x64xf32>
    rock.yield %0 : tensor<1x64x64xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x64x32xf32>, tensor<1x32x64xf32>, tensor<1x64x32xf32>, tensor<1x64x64xf32> -> tensor<1x64x32xf32>
  return %result : tensor<1x64x32xf32>
}

// -----

// CHECK-LABEL: func @attn_external_splat_constants
// Splat constants defined outside the preSoftmaxBody (captured by closure)
// must be replaced with tile-shaped versions when cloning region ops.
// CHECK: arith.constant dense<1.250000e-01> : tensor<32x32xf32>
// CHECK: arith.constant dense<1.000000e+01> : tensor<32x32xf32>
// CHECK: arith.mulf {{.*}} tensor<32x32xf32>
// CHECK: arith.select {{.*}} tensor<32x32xf32>
func.func @attn_external_splat_constants(
    %q: tensor<1x64x32xf32>,
    %k: tensor<1x32x64xf32>,
    %v: tensor<1x64x32xf32>,
    %mask: tensor<4096xi8>) -> tensor<1x64x32xf32>
    attributes {
      rock.block_size = 256 : i32,
      rock.grid_size = 2 : i32,
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"
    } {
  %cst_scale = arith.constant dense<1.250000e-01> : tensor<1x64x64xf32>
  %cst_fill = arith.constant dense<1.000000e+01> : tensor<1x64x64xf32>
  %mask_reshaped = rock.transform %mask by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 64 + d2)> by [<Unmerge{1, 64, 64} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 64, 64] -> [4096]> : tensor<4096xi8> to tensor<1x64x64xi8>
  %result = rock.gridwise_attention(%q, %k, %v, %mask_reshaped) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x64x64xf32>, %arg_mask: tensor<1x64x64xi8>):
    %0 = arith.trunci %arg_mask : tensor<1x64x64xi8> to tensor<1x64x64xi1>
    %1 = arith.mulf %arg_qk, %cst_scale : tensor<1x64x64xf32>
    %2 = arith.select %0, %1, %cst_fill : tensor<1x64x64xi1>, tensor<1x64x64xf32>
    rock.yield %2 : tensor<1x64x64xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x64x32xf32>, tensor<1x32x64xf32>, tensor<1x64x32xf32>, tensor<1x64x64xi8> -> tensor<1x64x32xf32>
  return %result : tensor<1x64x32xf32>
}

// -----

// CHECK-LABEL: func @attn_splat_constants_in_body
// Splat constants defined inside the preSoftmaxBody must also be resized.
// CHECK: arith.constant dense<1.250000e-01> : tensor<32x32xf32>
// CHECK: arith.constant dense<1.000000e+01> : tensor<32x32xf32>
// CHECK: arith.mulf {{.*}} tensor<32x32xf32>
// CHECK: arith.select {{.*}} tensor<32x32xf32>
func.func @attn_splat_constants_in_body(
    %q: tensor<1x64x32xf32>,
    %k: tensor<1x32x64xf32>,
    %v: tensor<1x64x32xf32>,
    %mask: tensor<4096xi8>) -> tensor<1x64x32xf32>
    attributes {
      rock.block_size = 256 : i32,
      rock.grid_size = 2 : i32,
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"
    } {
  %mask_reshaped = rock.transform %mask by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 64 + d2)> by [<Unmerge{1, 64, 64} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 64, 64] -> [4096]> : tensor<4096xi8> to tensor<1x64x64xi8>
  %result = rock.gridwise_attention(%q, %k, %v, %mask_reshaped) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x64x64xf32>, %arg_mask: tensor<1x64x64xi8>):
    %cst_scale = arith.constant dense<1.250000e-01> : tensor<1x64x64xf32>
    %cst_fill = arith.constant dense<1.000000e+01> : tensor<1x64x64xf32>
    %0 = arith.trunci %arg_mask : tensor<1x64x64xi8> to tensor<1x64x64xi1>
    %1 = arith.mulf %arg_qk, %cst_scale : tensor<1x64x64xf32>
    %2 = arith.select %0, %1, %cst_fill : tensor<1x64x64xi1>, tensor<1x64x64xf32>
    rock.yield %2 : tensor<1x64x64xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x64x32xf32>, tensor<1x32x64xf32>, tensor<1x64x32xf32>, tensor<1x64x64xi8> -> tensor<1x64x32xf32>
  return %result : tensor<1x64x32xf32>
}

// -----

// CHECK-LABEL: func @attn_no_presoftmax_ops
// CHECK-NOT: rock.gridwise_attention
// CHECK: rock.store_marker
func.func @attn_no_presoftmax_ops(
    %q: tensor<1x64x32xf32>,
    %k: tensor<1x32x64xf32>,
    %v: tensor<1x64x32xf32>) -> tensor<1x64x32xf32>
    attributes {
      rock.block_size = 256 : i32,
      rock.grid_size = 2 : i32,
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x64x64xf32>):
    rock.yield %arg_qk : tensor<1x64x64xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x64x32xf32>, tensor<1x32x64xf32>, tensor<1x64x32xf32> -> tensor<1x64x32xf32>
  return %result : tensor<1x64x32xf32>
}

// -----

// CHECK-LABEL: func @attn_external_transform_chain
// Transforms and elementwise add happen outside the body. untransform()
// peels the broadcast off the extra input, finding the addf result
// (tensor<1x1x1xf32>) as the root for the LoadMarkerOp.
// CHECK: arith.addf {{.*}} tensor<1x1x1xf32>
// CHECK: rock.load_marker {{.*}}tensor<1x1x1xf32> -> tensor<32x32xf32>
// The body mulf should be cloned with tile-sized types.
// CHECK: arith.mulf {{.*}} tensor<32x32xf32>
func.func @attn_external_transform_chain(
    %q: tensor<1x64x32xf32>,
    %k: tensor<1x32x64xf32>,
    %v: tensor<1x64x32xf32>,
    %s0: tensor<1xf32>,
    %s1: tensor<1xf32>) -> tensor<1x64x32xf32>
    attributes {
      rock.block_size = 256 : i32,
      rock.grid_size = 2 : i32,
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"
    } {
  %t0 = rock.transform %s0 by <affine_map<(d0, d1, d2) -> (d0 + d1 + d2)> by [<Unmerge{1, 1, 1} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 1, 1] -> [1]> : tensor<1xf32> to tensor<1x1x1xf32>
  %t1 = rock.transform %s1 by <affine_map<(d0, d1, d2) -> (d0 + d1 + d2)> by [<Unmerge{1, 1, 1} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 1, 1] -> [1]> : tensor<1xf32> to tensor<1x1x1xf32>
  %sum = arith.addf %t0, %t1 : tensor<1x1x1xf32>
  %bcast = rock.transform %sum by <affine_map<(d0, d1, d2) -> (d0, 0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [1, 64, 64] -> [1, 1, 1]> : tensor<1x1x1xf32> to tensor<1x64x64xf32>
  %result = rock.gridwise_attention(%q, %k, %v, %bcast) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x64x64xf32>, %arg_extra: tensor<1x64x64xf32>):
    %r = arith.mulf %arg_qk, %arg_extra : tensor<1x64x64xf32>
    rock.yield %r : tensor<1x64x64xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x64x32xf32>, tensor<1x32x64xf32>, tensor<1x64x32xf32>, tensor<1x64x64xf32> -> tensor<1x64x32xf32>
  return %result : tensor<1x64x32xf32>
}

// -----

// Error: an external non-splat constant captured by the preSoftmaxBody
// cannot be retiled to tile-level shapes.
func.func @error_external_non_splat_constant(
    %q: tensor<1x4x4xf32>,
    %k: tensor<1x4x4xf32>,
    %v: tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    attributes {
      rock.block_size = 256 : i32,
      rock.grid_size = 1 : i32,
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"
    } {
  %cst = arith.constant dense<[[[1.0, 2.0, 3.0, 4.0],
                                 [5.0, 6.0, 7.0, 8.0],
                                 [9.0, 10.0, 11.0, 12.0],
                                 [13.0, 14.0, 15.0, 16.0]]]> : tensor<1x4x4xf32>
  // expected-error @below {{'rock.gridwise_attention' op non-splat constant in preSoftmaxBody cannot be tiled}}
  // expected-error @below {{Failed to post process first GEMM output}}
  // expected-error @below {{failed to legalize operation 'rock.gridwise_attention' that was explicitly marked illegal}}
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x4x4xf32>):
    %0 = arith.mulf %arg_qk, %cst : tensor<1x4x4xf32>
    rock.yield %0 : tensor<1x4x4xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 4, nPerBlock = 4, kPerBlock = 4, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 4, nPerBlock = 4, kPerBlock = 4, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x4x4xf32>, tensor<1x4x4xf32>, tensor<1x4x4xf32> -> tensor<1x4x4xf32>
  return %result : tensor<1x4x4xf32>
}

// -----

// Error: a non-splat constant defined inside the preSoftmaxBody
// cannot be retiled to tile-level shapes.
func.func @error_body_non_splat_constant(
    %q: tensor<1x4x4xf32>,
    %k: tensor<1x4x4xf32>,
    %v: tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    attributes {
      rock.block_size = 256 : i32,
      rock.grid_size = 1 : i32,
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"
    } {
  // expected-error @below {{'rock.gridwise_attention' op non-splat constant in preSoftmaxBody cannot be tiled}}
  // expected-error @below {{Failed to post process first GEMM output}}
  // expected-error @below {{failed to legalize operation 'rock.gridwise_attention' that was explicitly marked illegal}}
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x4x4xf32>):
    %cst = arith.constant dense<[[[1.0, 2.0, 3.0, 4.0],
                                   [5.0, 6.0, 7.0, 8.0],
                                   [9.0, 10.0, 11.0, 12.0],
                                   [13.0, 14.0, 15.0, 16.0]]]> : tensor<1x4x4xf32>
    %0 = arith.mulf %arg_qk, %cst : tensor<1x4x4xf32>
    rock.yield %0 : tensor<1x4x4xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 4, nPerBlock = 4, kPerBlock = 4, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 4, nPerBlock = 4, kPerBlock = 4, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x4x4xf32>, tensor<1x4x4xf32>, tensor<1x4x4xf32> -> tensor<1x4x4xf32>
  return %result : tensor<1x4x4xf32>
}

// -----

// CHECK-LABEL: func @attn_single_chiplet_odd_cu
// CHECK-NOT: rock.gridwise_attention
// CHECK: rock.store_marker
func.func @attn_single_chiplet_odd_cu(
    %q: tensor<1x64x32xf32>,
    %k: tensor<1x32x64xf32>,
    %v: tensor<1x64x32xf32>) -> tensor<1x64x32xf32>
    attributes {
      rock.block_size = 256 : i32,
      rock.grid_size = 2 : i32,
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-",
      rock.num_cu = 35 : i64,
      rock.num_chiplets = 1 : i64
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x64x64xf32>):
    rock.yield %arg_qk : tensor<1x64x64xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x64x32xf32>, tensor<1x32x64xf32>, tensor<1x64x32xf32> -> tensor<1x64x32xf32>
  return %result : tensor<1x64x32xf32>
}

// -----

// CHECK-LABEL: func @attn_multi_chiplet_even
// CHECK-NOT: rock.gridwise_attention
// CHECK: rock.store_marker
func.func @attn_multi_chiplet_even(
    %q: tensor<1x64x32xf32>,
    %k: tensor<1x32x64xf32>,
    %v: tensor<1x64x32xf32>) -> tensor<1x64x32xf32>
    attributes {
      rock.block_size = 256 : i32,
      rock.grid_size = 2 : i32,
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-",
      rock.num_cu = 304 : i64,
      rock.num_chiplets = 8 : i64
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x64x64xf32>):
    rock.yield %arg_qk : tensor<1x64x64xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x64x32xf32>, tensor<1x32x64xf32>, tensor<1x64x32xf32> -> tensor<1x64x32xf32>
  return %result : tensor<1x64x32xf32>
}
