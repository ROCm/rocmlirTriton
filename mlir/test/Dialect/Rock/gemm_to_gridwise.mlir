// Ensures that the padding application, group application, etc. in gemm-to-gridwise
// function as expected.

// RUN: rocmlir-opt -rock-lower-reduce -rock-fusion-splitk-regularization -rock-gemm-to-gridwise -mlir-print-local-scope %s | FileCheck %s

#general_gemm_params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 8, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#general_gemm_params_splitk = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 8, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 2, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#general_gemm_params1 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xdlops_gemm_params0 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xdlops_gemm_params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 4, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xdlops_gemm_params3 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 3, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// CHECK-LABEL: func.func @gemm_easy_case_from_conv
// CHECK-SAME: (%[[a:.*]]: tensor<1x72x128xf32>, %[[b:.*]]: tensor<1x72x512xf32>, %[[c:.*]]: tensor<1x128x512xf32>)
// CHECK-SAME: rock.grid_size = 4
func.func @gemm_easy_case_from_conv(%a: tensor<1x72x128xf32>, %b: tensor<1x72x512xf32>, %c: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK: %[[transA:.*]] = rock.transform %[[a]] by {{.*}} : tensor<1x72x128xf32> to tensor<1x128x72xf32>
  // CHECK: rock.gridwise_gemm(%[[transA]], %[[b]]) 
  %result = rock.gemm tr %a * %b {
    params = #general_gemm_params0
  } : tensor<1x72x128xf32> * tensor<1x72x512xf32> -> tensor<1x128x512xf32>
  %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
  func.return %out : tensor<1x128x512xf32>
}

// CHECK-LABEL: func.func @gemm_splitk
// CHECK-SAME: (%[[a:.*]]: tensor<1x72x128xf32>, %[[b:.*]]: tensor<1x72x512xf32>, %[[c:.*]]: tensor<1x128x512xf32> {rock.prefill = {{.*}} : f32})
// CHECK-SAME: rock.grid_size = 8 : i32
func.func @gemm_splitk(%a: tensor<1x72x128xf32>, %b: tensor<1x72x512xf32>, %c: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.gridwise_gemm({{.*}}, {{.*}})
  // CHECK: rock.store {{.*}} by atomic_add
  %result = rock.gemm tr %a * %b {
    params = #general_gemm_params_splitk
  } : tensor<1x72x128xf32> * tensor<1x72x512xf32> -> tensor<1x128x512xf32>
  %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
  func.return %out : tensor<1x128x512xf32>
}

// Stream-K: streamKMultiple >= 1 makes gemm-to-gridwise choose a partition dim
// and pad it up to whole tiles so RockStreamKDecompose can always decompose.
// Here G=1, mBlocks=4096/32=128, nBlocks=4096/128=32, num_cu=70, targetP=140.
// The M-partition (span=floor(140/32)=4) is cheapest; mBlocks 128 is padded to
// 129 (rb=1) so M -> 4128 and grid_size = 129*32 = 4128. The chosen dim (M)
// is recorded in rock.streamk_part_dim for the decompose pass.
#streamk_params = #rock.gemm_params<mPerBlock = 32, nPerBlock = 128, kPerBlock = 16, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 3, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, streamKMultiple = 2>

// CHECK-LABEL: func.func @gemm_streamk_pads_partition_dim
// CHECK-SAME: rock.grid_size = 4128 : i32
func.func @gemm_streamk_pads_partition_dim(%a: tensor<1x4096x14336xf16>, %b: tensor<1x14336x4096xf16>, %c: tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.num_cu = 70 : i32} {
  // CHECK: rock.gridwise_gemm({{.*}}) {{.*}}rock.streamk_part_dim = #rock<rock.streamk_part_dim m>{{.*}} -> tensor<1x4128x4096xf32>
  %result = rock.gemm %a * %b {
    params = #streamk_params
  } : tensor<1x4096x14336xf16> * tensor<1x14336x4096xf16> -> tensor<1x4096x4096xf32>
  %out = rock.store %result to %c by set : tensor<1x4096x4096xf32> -> tensor<1x4096x4096xf32> to tensor<1x4096x4096xf32>
  func.return %out : tensor<1x4096x4096xf32>
}

// Stream-K K padding: G=1, mBlocks=nBlocks=10 (gridFull=100), num_cu=80,
// streamKMultiple=1 -> N-partition with splitK=4. K = 192 is a multiple of
// kPerBlock=64 but not of splitK*kPerBlock=256, so gemm-to-gridwise zero-pads K
// up to 256 (Pad{0, 64}) on A and B so the decompose pass's remainder folds
// evenly. The partition dim (N) needs no tile padding here.
#streamk_kpad_params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, streamKMultiple = 1>

// CHECK-LABEL: func.func @gemm_streamk_pads_k
// CHECK-SAME: rock.grid_size = 100 : i32
func.func @gemm_streamk_pads_k(%a: tensor<1x1280x192xf16>, %b: tensor<1x192x1280xf16>, %c: tensor<1x1280x1280xf32>) -> tensor<1x1280x1280xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 80 : i32} {
  // CHECK-DAG: Pad{0, 64} ["gemmKPad"]
  // CHECK: rock.gridwise_gemm({{.*}}) {{.*}}rock.streamk_part_dim = #rock<rock.streamk_part_dim n>{{.*}} : tensor<1x1280x256xf16>, tensor<1x256x1280xf16> -> tensor<1x1280x1280xf32>
  %result = rock.gemm %a * %b {
    params = #streamk_kpad_params
  } : tensor<1x1280x192xf16> * tensor<1x192x1280xf16> -> tensor<1x1280x1280xf32>
  %out = rock.store %result to %c by set : tensor<1x1280x1280xf32> -> tensor<1x1280x1280xf32> to tensor<1x1280x1280xf32>
  func.return %out : tensor<1x1280x1280xf32>
}

// Stream-K M + K padding: G=1, mBlocks=4096/32=128, nBlocks=4096/128=32,
// num_cu=70, streamKMultiple=2 (targetP=140). N-partition span=floor(140/128)=1
// is too small, so M is chosen (span=4, splitK=4); mBlocks 128 -> 129 pads M to
// 4128. K=14352 is a multiple of kPerBlock=16 but not of splitK*kPerBlock=64, so
// K is also zero-padded up to 14400 (Pad{0, 48}). So both M and K are padded.
#streamk_mkpad_params = #rock.gemm_params<mPerBlock = 32, nPerBlock = 128, kPerBlock = 16, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 3, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, streamKMultiple = 2>

// CHECK-LABEL: func.func @gemm_streamk_pads_m_and_k
// CHECK-SAME: rock.grid_size = 4128 : i32
func.func @gemm_streamk_pads_m_and_k(%a: tensor<1x4096x14352xf16>, %b: tensor<1x14352x4096xf16>, %c: tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.num_cu = 70 : i32} {
  // CHECK-DAG: Pad{0, 48} ["gemmKPad"]
  // CHECK: rock.gridwise_gemm({{.*}}) {{.*}}rock.streamk_part_dim = #rock<rock.streamk_part_dim m>{{.*}} : tensor<1x4128x14400xf16>, tensor<1x14400x4096xf16> -> tensor<1x4128x4096xf32>
  %result = rock.gemm %a * %b {
    params = #streamk_mkpad_params
  } : tensor<1x4096x14352xf16> * tensor<1x14352x4096xf16> -> tensor<1x4096x4096xf32>
  %out = rock.store %result to %c by set : tensor<1x4096x4096xf32> -> tensor<1x4096x4096xf32> to tensor<1x4096x4096xf32>
  func.return %out : tensor<1x4096x4096xf32>
}

// Stream-K N + K padding: G=1, mBlocks=1280/128=10, nBlocks=1408/128=11,
// num_cu=80, streamKMultiple=1 (targetP=80). N-partition (span=8) is chosen with
// remBlocks=4/splitK=2; nBlocks 11 -> 12 pads N to 1536. K=192 is a multiple of
// kPerBlock=64 but not of splitK*kPerBlock=128, so K is also zero-padded up to
// 256 (Pad{0, 64}). So both N and K are padded.
#streamk_nkpad_params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, streamKMultiple = 1>

// CHECK-LABEL: func.func @gemm_streamk_pads_n_and_k
// CHECK-SAME: rock.grid_size = 120 : i32
func.func @gemm_streamk_pads_n_and_k(%a: tensor<1x1280x192xf16>, %b: tensor<1x192x1408xf16>, %c: tensor<1x1280x1408xf32>) -> tensor<1x1280x1408xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 80 : i32} {
  // CHECK-DAG: Pad{0, 64} ["gemmKPad"]
  // CHECK: rock.gridwise_gemm({{.*}}) {{.*}}rock.streamk_part_dim = #rock<rock.streamk_part_dim n>{{.*}} : tensor<1x1280x256xf16>, tensor<1x256x1536xf16> -> tensor<1x1280x1536xf32>
  %result = rock.gemm %a * %b {
    params = #streamk_nkpad_params
  } : tensor<1x1280x192xf16> * tensor<1x192x1408xf16> -> tensor<1x1280x1408xf32>
  %out = rock.store %result to %c by set : tensor<1x1280x1408xf32> -> tensor<1x1280x1408xf32> to tensor<1x1280x1408xf32>
  func.return %out : tensor<1x1280x1408xf32>
}

// CHECK-LABEL: func.func @gemm_easy_case_from_conv_xdlops
// CHECK-SAME: (%[[a:.*]]: tensor<1x72x128xf32>, %[[b:.*]]: tensor<1x72x512xf32>, %[[c:.*]]: tensor<1x128x512xf32>)
// CHECK-SAME: rock.grid_size = 16 : i32
func.func @gemm_easy_case_from_conv_xdlops(%a: tensor<1x72x128xf32>, %b: tensor<1x72x512xf32>, %c: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: %[[transA:.*]] = rock.transform %[[a]] by {{.*}} : tensor<1x72x128xf32> to tensor<1x128x72xf32>
  // CHECK: rock.gridwise_gemm(%[[transA]], %[[b]])
  %result = rock.gemm tr %a * %b {
    params = #xdlops_gemm_params0
  } : tensor<1x72x128xf32> * tensor<1x72x512xf32> -> tensor<1x128x512xf32>
  %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
  func.return %out : tensor<1x128x512xf32>
}

// CHECK-LABEL: func.func @gemm_most_general_padding_case
// CHECK-SAME: (%[[a:.*]]: tensor<1x1x1xf32>, %[[b:.*]]: tensor<1x1x1xf32>, %[[c:.*]]: tensor<1x1x1xf32>)
// CHECK-SAME: rock.grid_size = 1
func.func @gemm_most_general_padding_case(%a: tensor<1x1x1xf32>, %b: tensor<1x1x1xf32>, %c: tensor<1x1x1xf32>) -> tensor<1x1x1xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK: rock.transform %[[a]] by {{.*}} : tensor<1x1x1xf32> to tensor<1x1x1xf32>
  // CHECK: rock.transform {{.*}} by {{.*}} to tensor<1x64x16xf32>
  // CHECK: rock.transform %[[b]] by {{.*}} : tensor<1x1x1xf32> to tensor<1x16x64xf32>
  // CHECK: rock.transform %[[c]] by {{.*}} : tensor<1x1x1xf32> to tensor<1x64x64xf32>
  // CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) 
  %result = rock.gemm tr %a * %b {
    params = #general_gemm_params1
  } : tensor<1x1x1xf32> * tensor<1x1x1xf32> -> tensor<1x1x1xf32>
  %out = rock.store %result to %c by set : tensor<1x1x1xf32> -> tensor<1x1x1xf32> to tensor<1x1x1xf32>
  func.return %out : tensor<1x1x1xf32>
}

// CHECK-LABEL: func.func @gemm_in_standard_form
// CHECK-SAME: (%[[a:.*]]: tensor<128x72xf32>, %[[b:.*]]: tensor<72x512xf32>, %[[c:.*]]: tensor<128x512xf32>)
// CHECK-SAME: rock.grid_size = 4
func.func @gemm_in_standard_form(%a: tensor<128x72xf32>, %b: tensor<72x512xf32>, %c: tensor<128x512xf32>) -> tensor<128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK: %[[normalizeA:.*]] = rock.transform %[[a]] by {{.*}} : tensor<128x72xf32> to tensor<1x128x72xf32>
  // CHECK: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : tensor<72x512xf32> to tensor<1x72x512xf32>
  // CHECK: %[[normalizeC:.*]] = rock.transform %[[c]] by {{.*}} : tensor<128x512xf32> to tensor<1x128x512xf32>
  // CHECK: rock.gridwise_gemm(%[[normalizeA]], %[[normalizeB]]) 
  %result = rock.gemm %a * %b {
    params = #general_gemm_params0
  } : tensor<128x72xf32> * tensor<72x512xf32> -> tensor<128x512xf32>
  %out = rock.store %result to %c by set : tensor<128x512xf32> -> tensor<128x512xf32> to tensor<128x512xf32>
  func.return %out : tensor<128x512xf32>
}

// CHECK-LABEL: func.func @gemm_transposed_from_gridwise
// CHECK-SAME: (%[[a:.*]]: tensor<1x128x72xf32>, %[[b:.*]]: tensor<1x512x72xf32>, %[[c:.*]]: tensor<1x512x128xf32>)
// CHECK-SAME: grid_size = 4
func.func @gemm_transposed_from_gridwise(%a: tensor<1x128x72xf32>, %b: tensor<1x512x72xf32>, %c: tensor<1x512x128xf32>) -> tensor<1x512x128xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] {{.*}} : tensor<1x512x72xf32> to tensor<1x72x512xf32{{.*}}>
  // CHECK-DAG: %[[normalizeC:.*]] = rock.transform %[[c]] {{.*}} : tensor<1x512x128xf32> to tensor<1x128x512xf32{{.*}}>
  // CHECK: rock.gridwise_gemm(%[[a]], %[[normalizeB]])
  %result = rock.gemm %a * tr %b {
    params = #general_gemm_params0,
    oTransposed
  } : tensor<1x128x72xf32> * tensor<1x512x72xf32> -> tensor<1x512x128xf32>
  %out = rock.store %result to %c by set : tensor<1x512x128xf32> -> tensor<1x512x128xf32> to tensor<1x512x128xf32>
  func.return %out : tensor<1x512x128xf32>
}

// CHECK-LABEL: func.func @gemm_pad_for_split_k
// CHECK-SAME: (%[[a:.*]]: tensor<1x128x238xf32>, %[[b:.*]]: tensor<1x238x512xf32>, %[[c:.*]]: tensor<1x128x512xf32> {rock.prefill = {{.*}} : f32})
// CHECK-SAME: rock.grid_size = 48
func.func @gemm_pad_for_split_k(%a: tensor<1x128x238xf32>, %b: tensor<1x238x512xf32>, %c: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.transform %[[a]] by {{.*}} : tensor<1x128x238xf32> to tensor<1x128x240xf32>
  // CHECK: rock.transform %[[b]] by {{.*}} : tensor<1x238x512xf32> to tensor<1x240x512xf32>
  %alloc = tensor.empty() : tensor<1x128x512xf32>
  // CHECK: rock.gridwise_gemm({{.*}}, {{.*}})
  // CHECK: rock.store {{.*}} by atomic_add
  %result = rock.gemm %a * %b {
    params = #xdlops_gemm_params3
  } : tensor<1x128x238xf32> * tensor<1x238x512xf32> -> tensor<1x128x512xf32>
  %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
  func.return %out : tensor<1x128x512xf32>
}

// CHECK-LABEL: func.func @gemm_reduce_and_split_k
// CHECK-SAME: (%[[a:.*]]: tensor<1x128x238xf32>, %[[b:.*]]: tensor<1x238x512xf32>, %[[c:.*]]: tensor<1x128x1xf32> {rock.prefill = {{.*}} : f32}, %[[d:.*]]: tensor<1x128x512xf32> {rock.prefill = {{.*}} : f32})
// CHECK-SAME: grid_size = 48
func.func @gemm_reduce_and_split_k(%a: tensor<1x128x238xf32>, %b: tensor<1x238x512xf32>, %c: tensor<1x128x1xf32>, %d: tensor<1x128x512xf32>) -> (tensor<1x128x1xf32>, tensor<1x128x512xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: tensor<3x128x80xf32>, tensor<3x80x512xf32> -> tensor<3x128x512xf32>
  // CHECK: rock.store {{.*}} by atomic_add
  // CHECK: rock.store {{.*}} by atomic_add
  %result = rock.gemm %a * %b {
    params = #xdlops_gemm_params3
  } : tensor<1x128x238xf32> * tensor<1x238x512xf32> -> tensor<1x128x512xf32>
  %reduced = rock.reduce  sum %result {axis = 2 : index} : tensor<1x128x512xf32> -> tensor<1x128x1xf32>
  %out_d = rock.store %result to %d by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
  %out_c = rock.store %reduced to %c by set : tensor<1x128x1xf32> -> tensor<1x128x1xf32> to tensor<1x128x1xf32>
  func.return %out_c, %out_d : tensor<1x128x1xf32>, tensor<1x128x512xf32>
}

// CHECK-LABEL: func.func @gemm_reduce_and_split_k_return_reduce_directly
// CHECK-SAME: (%[[a:.*]]: tensor<1x128x238xf32>, %[[b:.*]]: tensor<1x238x512xf32>, %[[c:.*]]: tensor<1x128x1xf32> {rock.prefill = {{.*}} : f32}, %[[d:.*]]: tensor<1x128x512xf32> {rock.prefill = {{.*}} : f32})
// CHECK-SAME: grid_size = 48
func.func @gemm_reduce_and_split_k_return_reduce_directly(%a: tensor<1x128x238xf32>, %b: tensor<1x238x512xf32>, %c: tensor<1x128x1xf32>, %d: tensor<1x128x512xf32>) -> (tensor<1x128x1xf32>, tensor<1x128x512xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: tensor<3x128x80xf32>, tensor<3x80x512xf32> -> tensor<3x128x512xf32>
  // CHECK: rock.store {{.*}} by atomic_add
  // CHECK: rock.store {{.*}} by atomic_add
  %result = rock.gemm %a * %b {
    params = #xdlops_gemm_params3
  } : tensor<1x128x238xf32> * tensor<1x238x512xf32> -> tensor<1x128x512xf32>
  %reduced = rock.reduce  sum %result {axis = 2 : index} : tensor<1x128x512xf32> -> tensor<1x128x1xf32>
  %out_d = rock.store %result to %d by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
  %out_c = rock.store %reduced to %c by set : tensor<1x128x1xf32> -> tensor<1x128x1xf32> to tensor<1x128x1xf32>
  func.return %out_c, %out_d : tensor<1x128x1xf32>, tensor<1x128x512xf32>
}

// CHECK-LABEL: func.func @gemm_fusion_to_f32_split_k
// CHECK-SAME: (%[[a:.*]]: tensor<1x5x4xf16>, %[[b:.*]]: tensor<1x4x3xf16>, %[[c:.*]]: tensor<1x5x3xf16>, %[[d:.*]]: tensor<1x5x3xf32> {rock.prefill = 0.000000e+00 : f32})
// CHECK-SAME: rock.grid_size = 3
func.func @gemm_fusion_to_f32_split_k(%arg0: tensor<1x5x4xf16>, %arg1: tensor<1x4x3xf16>, %arg2: tensor<1x5x3xf16>, %arg3: tensor<1x5x3xf32>) -> tensor<1x5x3xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.gridwise_gemm
  // CHECK: arith.addf
  // CHECK: arith.extf {{.*}} to tensor<3x64x64xf32>
  // CHECK: rock.store {{.*}} by atomic_add
  %result = rock.gemm %arg0 * %arg1 {
    params = #xdlops_gemm_params3
  } : tensor<1x5x4xf16> * tensor<1x4x3xf16> -> tensor<1x5x3xf16>
  %added = arith.addf %result, %arg2 : tensor<1x5x3xf16>
  %extended = arith.extf %added : tensor<1x5x3xf16> to tensor<1x5x3xf32>
  %out = rock.store %extended to %arg3 by set : tensor<1x5x3xf32> -> tensor<1x5x3xf32> to tensor<1x5x3xf32>
  return %out : tensor<1x5x3xf32>
}

// CHECK-LABEL: func.func @gemm_fusion_to_f16_split_k
// CHECK-SAME: (%[[a:.*]]: tensor<1x5x4xf32>, %[[b:.*]]: tensor<1x4x3xf32>, %[[c:.*]]: tensor<1x5x3xf32>, %[[d:.*]]: tensor<1x5x3xf16> {rock.prefill = 0.000000e+00 : f16})
// CHECK-SAME: rock.grid_size = 3 : i32
func.func @gemm_fusion_to_f16_split_k(%arg0: tensor<1x5x4xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x5x3xf32>, %arg3: tensor<1x5x3xf16>) -> tensor<1x5x3xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.gridwise_gemm
  // CHECK: arith.addf
  // CHECK: arith.truncf {{.*}} to tensor<3x64x64xf16>
  // CHECK: rock.store {{.*}} by atomic_add
  %result = rock.gemm %arg0 * %arg1 {
    params = #xdlops_gemm_params3
  } : tensor<1x5x4xf32> * tensor<1x4x3xf32> -> tensor<1x5x3xf32>
  %added = arith.addf %result, %arg2 : tensor<1x5x3xf32>
  %trunc = arith.truncf %added : tensor<1x5x3xf32> to tensor<1x5x3xf16>
  %out = rock.store %trunc to %arg3 by set : tensor<1x5x3xf16> -> tensor<1x5x3xf16> to tensor<1x5x3xf16>
  return %out : tensor<1x5x3xf16>
}

// -----

// Tests for scaled GEMM 

// CHECK-LABEL: func.func @gemm_scaled_fp4_already_f8e8m0
// CHECK-SAME: (%[[a:.*]]: tensor<1x72x128xf4E2M1FN>, %[[b:.*]]: tensor<1x72x512xf4E2M1FN>, %[[c:.*]]: tensor<1x128x512xf32>, %[[scaleA:.*]]: tensor<1x128x72xf8E8M0FNU>, %[[scaleB:.*]]: tensor<1x72x512xf8E8M0FNU>)
// CHECK-SAME: grid_size = 16 : i32
func.func @gemm_scaled_fp4_already_f8e8m0(%a: tensor<1x72x128xf4E2M1FN>, %b: tensor<1x72x512xf4E2M1FN>, %c: tensor<1x128x512xf32>, %scaleA: tensor<1x128x72xf8E8M0FNU>, %scaleB: tensor<1x72x512xf8E8M0FNU>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[a]] {{.*}} : tensor<1x72x128xf4E2M1FN> to tensor<1x128x72xf4E2M1FN{{.*}}>
  // CHECK-DAG: %[[normalizeScaleB:.*]] = rock.transform %[[scaleB]] {{.*}} : tensor<1x72x512xf8E8M0FNU> to tensor<1x512x72xf8E8M0FNU{{.*}}>
  // CHECK: rock.gridwise_gemm(%[[normalizeA]], %[[b]], %[[scaleA]], %[[normalizeScaleB]])
  %result = rock.gemm tr %a scaled by %scaleA * %b scaled by tr %scaleB {
    params = #xdlops_gemm_params0,
    quantBlockSize = 1 : i64
  } : tensor<1x72x128xf4E2M1FN> scaled by tensor<1x128x72xf8E8M0FNU> * tensor<1x72x512xf4E2M1FN> scaled by tensor<1x72x512xf8E8M0FNU> -> tensor<1x128x512xf32>
  %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
  func.return %out : tensor<1x128x512xf32>
}

// CHECK-LABEL: func.func @gemm_scaled_fp4_with_padding
// CHECK-SAME: (%[[a:.*]]: tensor<1x1x1xf4E2M1FN>, %[[b:.*]]: tensor<1x1x1xf4E2M1FN>, %[[c:.*]]: tensor<1x1x1xf32>, %[[scaleA:.*]]: tensor<1x1x1xf8E8M0FNU>, %[[scaleB:.*]]: tensor<1x1x1xf8E8M0FNU>)
// CHECK-SAME: grid_size = 1
func.func @gemm_scaled_fp4_with_padding(%a: tensor<1x1x1xf4E2M1FN>, %b: tensor<1x1x1xf4E2M1FN>, %c: tensor<1x1x1xf32>, %scaleA: tensor<1x1x1xf8E8M0FNU>, %scaleB: tensor<1x1x1xf8E8M0FNU>) -> tensor<1x1x1xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK-DAG: rock.transform {{.*}} : tensor<1x1x1xf4E2M1FN> to tensor<1x64x8xf4E2M1FN{{.*}}>
  // CHECK-DAG: rock.transform %[[b]] by {{.*}} : tensor<1x1x1xf4E2M1FN> to tensor<1x8x64xf4E2M1FN{{.*}}>
  // CHECK-DAG: rock.transform %[[scaleA]] by {{.*}} : tensor<1x1x1xf8E8M0FNU> to tensor<1x64x8xf8E8M0FNU{{.*}}>
  // CHECK-DAG: rock.transform {{.*}} : tensor<1x1x1xf8E8M0FNU> to tensor<1x64x8xf8E8M0FNU{{.*}}>
  // CHECK: rock.gridwise_gemm({{.*}}, {{.*}}, {{.*}}, {{.*}}) {{.*}} : tensor<1x64x8xf4E2M1FN>, tensor<1x8x64xf4E2M1FN>, tensor<1x64x8xf8E8M0FNU>, tensor<1x64x8xf8E8M0FNU> -> tensor<1x64x64xf32>
  %result = rock.gemm tr %a scaled by %scaleA * %b scaled by tr %scaleB {
    params = #xdlops_gemm_params0,
    quantBlockSize = 1 : i64
  } : tensor<1x1x1xf4E2M1FN> scaled by tensor<1x1x1xf8E8M0FNU> * tensor<1x1x1xf4E2M1FN> scaled by tensor<1x1x1xf8E8M0FNU> -> tensor<1x1x1xf32>
  %out = rock.store %result to %c by set : tensor<1x1x1xf32> -> tensor<1x1x1xf32> to tensor<1x1x1xf32>
  func.return %out : tensor<1x1x1xf32>
}

// CHECK-LABEL: func.func @gemm_scaled_fp4_transposed
// CHECK-SAME: (%[[a:.*]]: tensor<1x128x72xf4E2M1FN>, %[[b:.*]]: tensor<1x512x72xf4E2M1FN>, %[[c:.*]]: tensor<1x512x128xf32>, %[[scaleA:.*]]: tensor<1x72x128xf8E8M0FNU>, %[[scaleB:.*]]: tensor<1x72x512xf8E8M0FNU>)
// CHECK-SAME: grid_size = 16 : i32
func.func @gemm_scaled_fp4_transposed(%a: tensor<1x128x72xf4E2M1FN>, %b: tensor<1x512x72xf4E2M1FN>, %c: tensor<1x512x128xf32>, %scaleA: tensor<1x72x128xf8E8M0FNU>, %scaleB: tensor<1x72x512xf8E8M0FNU>) -> tensor<1x512x128xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] {{.*}} : tensor<1x512x72xf4E2M1FN> to tensor<1x72x512xf4E2M1FN{{.*}}>
  // CHECK-DAG: %[[normalizeScaleA:.*]] = rock.transform %[[scaleA]] {{.*}} : tensor<1x72x128xf8E8M0FNU> to tensor<1x128x72xf8E8M0FNU{{.*}}>
  // CHECK-DAG: %[[normalizeScaleB:.*]] = rock.transform %[[scaleB]] {{.*}} : tensor<1x72x512xf8E8M0FNU> to tensor<1x512x72xf8E8M0FNU{{.*}}>
  // CHECK: rock.gridwise_gemm(%[[a]], %[[normalizeB]], %[[normalizeScaleA]], %[[normalizeScaleB]])
  %result = rock.gemm %a scaled by tr %scaleA * tr %b scaled by tr %scaleB {
    params = #xdlops_gemm_params0,
    oTransposed,
    quantBlockSize = 1 : i64
  } : tensor<1x128x72xf4E2M1FN> scaled by tensor<1x72x128xf8E8M0FNU> * tensor<1x512x72xf4E2M1FN> scaled by tensor<1x72x512xf8E8M0FNU> -> tensor<1x512x128xf32>
  %out = rock.store %result to %c by set : tensor<1x512x128xf32> -> tensor<1x512x128xf32> to tensor<1x512x128xf32>
  func.return %out : tensor<1x512x128xf32>
}

// -----

// CHECK-LABEL: func.func @gemm_scaled_fp4_splitk
// CHECK-SAME: (%[[a:.*]]: tensor<1x72x128xf4E2M1FN>, %[[b:.*]]: tensor<1x72x512xf4E2M1FN>, %[[c:.*]]: tensor<1x128x512xf32> {rock.prefill = 0.000000e+00 : f32}, %[[scaleA:.*]]: tensor<1x128x72xf8E8M0FNU>, %[[scaleB:.*]]: tensor<1x72x512xf8E8M0FNU>)
// CHECK-SAME: grid_size = 32 : i32
func.func @gemm_scaled_fp4_splitk(%a: tensor<1x72x128xf4E2M1FN>, %b: tensor<1x72x512xf4E2M1FN>, %c: tensor<1x128x512xf32>, %scaleA: tensor<1x128x72xf8E8M0FNU>, %scaleB: tensor<1x72x512xf8E8M0FNU>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // Transpose A: 1x72x128 -> 1x128x72 (M-K layout)
  // CHECK-DAG: rock.transform %[[a]] by {{.*}} : tensor<1x72x128xf4E2M1FN> to tensor<1x128x72xf4E2M1FN>
  // Transpose scaleB: 1x72x512 -> 1x512x72 (N-K layout)
  // CHECK-DAG: rock.transform %[[scaleB]] by {{.*}} : tensor<1x72x512xf8E8M0FNU> to tensor<1x512x72xf8E8M0FNU>

  // Split K=72 into 2 parts of 36 + pad to 40 per split
  // CHECK-DAG: rock.transform {{.*}} : tensor<2x128x36xf4E2M1FN> to tensor<2x128x40xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : tensor<2x36x512xf4E2M1FN> to tensor<2x40x512xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : tensor<2x128x36xf8E8M0FNU> to tensor<2x128x40xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : tensor<2x512x36xf8E8M0FNU> to tensor<2x512x40xf8E8M0FNU>

  // Split and merge C
  // CHECK-DAG: rock.transform %[[c]] by {{.*}} : tensor<1x128x512xf32> to tensor<1x2x128x512xf32>
  // CHECK-DAG: rock.transform {{.*}} : tensor<1x2x128x512xf32> to tensor<2x128x512xf32>

  // CHECK: rock.gridwise_gemm({{.*}}, {{.*}}, {{.*}}, {{.*}}) {{.*}} : tensor<2x128x40xf4E2M1FN>, tensor<2x40x512xf4E2M1FN>, tensor<2x128x40xf8E8M0FNU>, tensor<2x512x40xf8E8M0FNU>
  // CHECK: rock.store {{.*}} by atomic_add
  %result = rock.gemm tr %a scaled by %scaleA * %b scaled by tr %scaleB {
    params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 2, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    quantBlockSize = 1 : i64
  } : tensor<1x72x128xf4E2M1FN> scaled by tensor<1x128x72xf8E8M0FNU> * tensor<1x72x512xf4E2M1FN> scaled by tensor<1x72x512xf8E8M0FNU> -> tensor<1x128x512xf32>
  %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
  func.return %out : tensor<1x128x512xf32>
}

// CHECK-LABEL: func.func @gemm_scaled_fp4_splitk_odd
// CHECK-SAME: (%[[aRaw:.*]]: tensor<589824xf4E2M1FN>, %[[bRaw:.*]]: tensor<589824xf4E2M1FN>, %[[cRaw:.*]]: tensor<196608xf32> {rock.prefill = 0.000000e+00 : f32}, %[[scaleARaw:.*]]: tensor<18432xf8E8M0FNU>, %[[scaleBRaw:.*]]: tensor<18432xf8E8M0FNU>)
// CHECK-SAME: grid_size = 240 : i32
// Note: the store dest is the rank-3 view %2 (not the rank-1 %arg2), because
// rock-gemm-to-gridwise's normalizeMatrix only supports rank-2 or rank-3 outputs.
func.func @gemm_scaled_fp4_splitk_odd(%arg0: tensor<589824xf4E2M1FN>, %arg1: tensor<589824xf4E2M1FN>, %arg2: tensor<196608xf32>, %arg3: tensor<18432xf8E8M0FNU>, %arg4: tensor<18432xf8E8M0FNU>) -> tensor<196608xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK-DAG: rock.transform %[[aRaw]] by {{.*}} : tensor<589824xf4E2M1FN> to tensor<3x256x768xf4E2M1FN>
  // CHECK-DAG: rock.transform %[[bRaw]] by {{.*}} : tensor<589824xf4E2M1FN> to tensor<3x768x256xf4E2M1FN>
  // CHECK-DAG: rock.transform %[[cRaw]] by {{.*}} : tensor<196608xf32> to tensor<3x256x256xf32>
  // CHECK-DAG: rock.transform %[[scaleARaw]] by {{.*}} : tensor<18432xf8E8M0FNU> to tensor<3x256x24xf8E8M0FNU>
  // CHECK-DAG: rock.transform %[[scaleBRaw]] by {{.*}} : tensor<18432xf8E8M0FNU> to tensor<3x24x256xf8E8M0FNU>

  // Scale broadcasting through AddDim, Broadcast, and Merge transformations
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x24xf8E8M0FNU> to tensor<3x256x24x1xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x24x1xf8E8M0FNU> to tensor<3x256x24x32xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x24x32xf8E8M0FNU> to tensor<3x256x768xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x24x256xf8E8M0FNU> to tensor<3x24x1x256xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x24x1x256xf8E8M0FNU> to tensor<3x24x32x256xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x24x32x256xf8E8M0FNU> to tensor<3x768x256xf8E8M0FNU>

  // Transpose scaleB from KxN to NxK
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x768x256xf8E8M0FNU> to tensor<3x256x768xf8E8M0FNU>

  // Padding K from 768 to 770 on both inputs and both scales
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x768xf4E2M1FN> to tensor<3x256x770xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x768x256xf4E2M1FN> to tensor<3x770x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x768xf8E8M0FNU> to tensor<3x256x770xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x768xf8E8M0FNU> to tensor<3x256x770xf8E8M0FNU>

  // Split K=770 into 5 parts of 154 (and merge G with KSplit -> 15)
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x770xf4E2M1FN> to tensor<3x256x5x154xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x5x154xf4E2M1FN> to tensor<15x256x154xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x770x256xf4E2M1FN> to tensor<3x5x154x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x5x154x256xf4E2M1FN> to tensor<15x154x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x770xf8E8M0FNU> to tensor<3x256x5x154xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x5x154xf8E8M0FNU> to tensor<15x256x154xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x770xf8E8M0FNU> to tensor<3x256x5x154xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x5x154xf8E8M0FNU> to tensor<15x256x154xf8E8M0FNU>

  // Split and merge C
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x256x256xf32> to tensor<3x5x256x256xf32>
  // CHECK-DAG: rock.transform {{.*}} : tensor<3x5x256x256xf32> to tensor<15x256x256xf32>

  // Final padding K from 154 to 512 (kPerBlock)
  // CHECK-DAG: rock.transform {{.*}} : tensor<15x256x154xf4E2M1FN> to tensor<15x256x512xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : tensor<15x154x256xf4E2M1FN> to tensor<15x512x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : tensor<15x256x154xf8E8M0FNU> to tensor<15x256x512xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : tensor<15x256x154xf8E8M0FNU> to tensor<15x256x512xf8E8M0FNU>

  // CHECK: rock.gridwise_gemm({{.*}}, {{.*}}, {{.*}}, {{.*}}) {{.*}} : tensor<15x256x512xf4E2M1FN>, tensor<15x512x256xf4E2M1FN>, tensor<15x256x512xf8E8M0FNU>, tensor<15x256x512xf8E8M0FNU>
  // CHECK: rock.store {{.*}} by atomic_add
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 768 + d2)> by [<Unmerge{3, 256, 768} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 768] -> [589824]> : tensor<589824xf4E2M1FN> to tensor<3x256x768xf4E2M1FN>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 768 + d1) * 256 + d2)> by [<Unmerge{3, 768, 256} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 768, 256] -> [589824]> : tensor<589824xf4E2M1FN> to tensor<3x768x256xf4E2M1FN>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 256 + d2)> by [<Unmerge{3, 256, 256} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 256] -> [196608]> : tensor<196608xf32> to tensor<3x256x256xf32>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 24 + d2)> by [<Unmerge{3, 256, 24} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 24] -> [18432]> : tensor<18432xf8E8M0FNU> to tensor<3x256x24xf8E8M0FNU>
  %4 = rock.transform %arg4 by <affine_map<(d0, d1, d2) -> ((d0 * 24 + d1) * 256 + d2)> by [<Unmerge{3, 24, 256} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 24, 256] -> [18432]> : tensor<18432xf8E8M0FNU> to tensor<3x24x256xf8E8M0FNU>
  %5 = rock.transform %3 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)> by [<AddDim{1} ["block"] at [3] -> [] at []>, <PassThrough ["g", "m", "kScale"] at [0, 1, 2] -> ["g", "m", "kScale"] at [0, 1, 2]>] bounds = [3, 256, 24, 1] -> [3, 256, 24]> : tensor<3x256x24xf8E8M0FNU> to tensor<3x256x24x1xf8E8M0FNU>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)> by [<Broadcast{1} ["block"] at [3] -> ["block"] at [3]>, <PassThrough ["g", "m", "kScale"] at [0, 1, 2] -> ["g", "m", "kScale"] at [0, 1, 2]>] bounds = [3, 256, 24, 32] -> [3, 256, 24, 1]> : tensor<3x256x24x1xf8E8M0FNU> to tensor<3x256x24x32xf8E8M0FNU>
  %7 = rock.transform %6 by <affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 32, d2 mod 32)> by [<Merge{24, 32} ["k"] at [2] -> ["kScale", "block"] at [2, 3]>, <PassThrough ["g", "m"] at [0, 1] -> ["g", "m"] at [0, 1]>] bounds = [3, 256, 768] -> [3, 256, 24, 32]> : tensor<3x256x24x32xf8E8M0FNU> to tensor<3x256x768xf8E8M0FNU>
  %8 = rock.transform %4 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)> by [<AddDim{1} ["block"] at [2] -> [] at []>, <PassThrough ["g", "kScale", "n"] at [0, 1, 3] -> ["g", "kScale", "n"] at [0, 1, 2]>] bounds = [3, 24, 1, 256] -> [3, 24, 256]> : tensor<3x24x256xf8E8M0FNU> to tensor<3x24x1x256xf8E8M0FNU>
  %9 = rock.transform %8 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)> by [<Broadcast{1} ["block"] at [2] -> ["block"] at [2]>, <PassThrough ["g", "kScale", "n"] at [0, 1, 3] -> ["g", "kScale", "n"] at [0, 1, 3]>] bounds = [3, 24, 32, 256] -> [3, 24, 1, 256]> : tensor<3x24x1x256xf8E8M0FNU> to tensor<3x24x32x256xf8E8M0FNU>
  %10 = rock.transform %9 by <affine_map<(d0, d1, d2) -> (d0, d1 floordiv 32, d1 mod 32, d2)> by [<PassThrough ["g", "n"] at [0, 2] -> ["g", "n"] at [0, 3]>, <Merge{24, 32} ["k"] at [1] -> ["kScale", "block"] at [1, 2]>] bounds = [3, 768, 256] -> [3, 24, 32, 256]> : tensor<3x24x32x256xf8E8M0FNU> to tensor<3x768x256xf8E8M0FNU>
  %result = rock.gemm %0 scaled by %7 * %1 scaled by tr %10 {
    params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 512, kpack = 32, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 5, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    quantBlockSize = 1 : i64
  } : tensor<3x256x768xf4E2M1FN> scaled by tensor<3x256x768xf8E8M0FNU> * tensor<3x768x256xf4E2M1FN> scaled by tensor<3x768x256xf8E8M0FNU> -> tensor<3x256x256xf32>
  %out = rock.store %result to %2 by set : tensor<3x256x256xf32> -> tensor<196608xf32> to tensor<3x256x256xf32>
  func.return %out : tensor<196608xf32>
}
