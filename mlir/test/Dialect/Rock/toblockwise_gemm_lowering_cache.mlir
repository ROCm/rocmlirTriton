// Cache-modifier heuristics for rock-gridwise-gemm-to-blockwise.
//
// A is reused across nBlocks workgroups, B across mBlocks. When one GEMM
// dimension is skinny (a single block) the operand along the *other*
// dimension is read only once, so it is streamed (CacheModifier cs) to avoid
// evicting the reused operand -- but only under cache pressure (A + B do not
// fit in the last-level cache). An operand whose load reloads data (a
// non-injective view such as a broadcast) relies on caching and is never
// streamed. All tests use gfx90a (8 MiB LLC) so cache pressure is reachable
// with modest tensors.

// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

#gemm_params = #rock.gemm_params<kPerBlock = 64, mPerBlock = 128, nPerBlock = 128, kpack = 8, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// M is skinny (mBlocks = 128/128 = 1) and N is not (nBlocks = 256/128 = 2).
// A (4 MiB) + B (8 MiB) = 12 MiB > 8 MiB LLC -> cache pressure. B has no reuse
// (read once), so B is streamed; A keeps its reuse across N and stays cached.
// The tiling orders each tile with gemmK first for B, a KxN matrix, and second
// for A, so this case also pins the tile shape each marker produces. The
// remaining cases only check the cache modifier.
// CHECK-LABEL: @m_skinny_pressure_streams_b
// CHECK-DAG: rock.load_marker %arg1 {{.*}}{cacheModifier = #rock<CacheModifier cs>} : tensor<1x8192x256xf32> -> tensor<64x128xf32>
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>} : tensor<1x128x8192xf32> -> tensor<128x64xf32>
func.func @m_skinny_pressure_streams_b(%arg0: tensor<1x128x8192xf32>, %arg1: tensor<1x8192x256xf32>, %arg2: tensor<1x128x256xf32>) -> tensor<1x128x256xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 2 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.num_cu = 104 : i32} {
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #gemm_params} : tensor<1x128x8192xf32>, tensor<1x8192x256xf32> -> tensor<1x128x256xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x128x256xf32> -> tensor<1x128x256xf32> to tensor<1x128x256xf32>
  return %out : tensor<1x128x256xf32>
}

// -----

#gemm_params = #rock.gemm_params<kPerBlock = 64, mPerBlock = 128, nPerBlock = 128, kpack = 8, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// N is skinny (nBlocks = 128/128 = 1) and M is not (mBlocks = 256/128 = 2).
// Under cache pressure A has no reuse (read once) and is streamed; B stays
// cached for its reuse across M.
// CHECK-LABEL: @n_skinny_pressure_streams_a
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier cs>{{.*}}} : tensor<1x256x8192xf32> -> tensor<128x64xf32>
// CHECK-DAG: rock.load_marker %arg1 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x8192x128xf32> -> tensor<64x128xf32>
func.func @n_skinny_pressure_streams_a(%arg0: tensor<1x256x8192xf32>, %arg1: tensor<1x8192x128xf32>, %arg2: tensor<1x256x128xf32>) -> tensor<1x256x128xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 2 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.num_cu = 104 : i32} {
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #gemm_params} : tensor<1x256x8192xf32>, tensor<1x8192x128xf32> -> tensor<1x256x128xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x256x128xf32> -> tensor<1x256x128xf32> to tensor<1x256x128xf32>
  return %out : tensor<1x256x128xf32>
}

// -----

#gemm_params = #rock.gemm_params<kPerBlock = 64, mPerBlock = 128, nPerBlock = 128, kpack = 8, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// M is skinny but A (64 KiB) + B (128 KiB) fits easily in the 8 MiB LLC, so
// there is no cache pressure and nothing is streamed: both stay cached.
// CHECK-LABEL: @m_skinny_no_pressure
// CHECK-DAG: rock.load_marker %arg1 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x128x256xf32> -> tensor<64x128xf32>
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x128x128xf32> -> tensor<128x64xf32>
func.func @m_skinny_no_pressure(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x256xf32>, %arg2: tensor<1x128x256xf32>) -> tensor<1x128x256xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 2 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.num_cu = 104 : i32} {
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #gemm_params} : tensor<1x128x128xf32>, tensor<1x128x256xf32> -> tensor<1x128x256xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x128x256xf32> -> tensor<1x128x256xf32> to tensor<1x128x256xf32>
  return %out : tensor<1x128x256xf32>
}

// -----

#gemm_params = #rock.gemm_params<kPerBlock = 64, mPerBlock = 128, nPerBlock = 128, kpack = 8, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// B is fed through a Broadcast view (its single K row is replicated to K=8192):
// iterating K re-reads the same element, a non-injective ("reload") view. Even
// though M is skinny and under cache pressure, B relies on caching for its
// repeated reads and is NOT streamed. A (injective, no reuse along N's single
// extra block) stays cached too.
// CHECK-LABEL: @m_skinny_pressure_noninjective_b
// CHECK-DAG: rock.load_marker %{{.*}} {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x8192x256xf32> -> tensor<64x128xf32>
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x128x8192xf32> -> tensor<128x64xf32>
#bcastK = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, 0, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Broadcast{1} ["k"] at [1] -> ["k"] at [1]>, <PassThrough ["n"] at [2] -> ["n"] at [2]>] bounds = [1, 8192, 256] -> [1, 1, 256]>
func.func @m_skinny_pressure_noninjective_b(%arg0: tensor<1x128x8192xf32>, %arg1: tensor<1x1x256xf32>, %arg2: tensor<1x128x256xf32>) -> tensor<1x128x256xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 2 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.num_cu = 104 : i32} {
  %b = rock.transform %arg1 by #bcastK : tensor<1x1x256xf32> to tensor<1x8192x256xf32>
  %result = rock.gridwise_gemm(%arg0, %b) {params = #gemm_params} : tensor<1x128x8192xf32>, tensor<1x8192x256xf32> -> tensor<1x128x256xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x128x256xf32> -> tensor<1x128x256xf32> to tensor<1x128x256xf32>
  return %out : tensor<1x128x256xf32>
}

// -----

#gemm_params = #rock.gemm_params<kPerBlock = 64, mPerBlock = 128, nPerBlock = 128, kpack = 8, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// The skinny test is on the *block* count, not the raw size: M = N = 256 are
// small but each spans 2 blocks (mBlocks = nBlocks = 256/128 = 2), so neither
// operand is read once. Even under cache pressure nothing is streamed.
// CHECK-LABEL: @small_but_multiblock_no_stream
// CHECK-DAG: rock.load_marker %arg1 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x8192x256xf32> -> tensor<64x128xf32>
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x256x8192xf32> -> tensor<128x64xf32>
func.func @small_but_multiblock_no_stream(%arg0: tensor<1x256x8192xf32>, %arg1: tensor<1x8192x256xf32>, %arg2: tensor<1x256x256xf32>) -> tensor<1x256x256xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 4 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.num_cu = 104 : i32} {
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #gemm_params} : tensor<1x256x8192xf32>, tensor<1x8192x256xf32> -> tensor<1x256x256xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x256x256xf32> -> tensor<1x256x256xf32> to tensor<1x256x256xf32>
  return %out : tensor<1x256x256xf32>
}

// -----

#gemm_params = #rock.gemm_params<kPerBlock = 64, mPerBlock = 128, nPerBlock = 128, kpack = 8, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// B is built by an AddDim of size 8192 along K: the added iteration dim has no
// lower correspondent, so every K step re-reads the same element (a reload).
// Even with M skinny and under pressure, B is NOT streamed.
// CHECK-LABEL: @m_skinny_pressure_adddim_b
// CHECK-DAG: rock.load_marker %{{.*}} {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x8192x256xf32> -> tensor<64x128xf32>
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x128x8192xf32> -> tensor<128x64xf32>
#adddimK = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <AddDim{8192} ["k"] at [1] -> [] at []>, <PassThrough ["n"] at [2] -> ["n"] at [1]>] bounds = [1, 8192, 256] -> [1, 256]>
func.func @m_skinny_pressure_adddim_b(%arg0: tensor<1x128x8192xf32>, %arg1: tensor<1x256xf32>, %arg2: tensor<1x128x256xf32>) -> tensor<1x128x256xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 2 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.num_cu = 104 : i32} {
  %b = rock.transform %arg1 by #adddimK : tensor<1x256xf32> to tensor<1x8192x256xf32>
  %result = rock.gridwise_gemm(%arg0, %b) {params = #gemm_params} : tensor<1x128x8192xf32>, tensor<1x8192x256xf32> -> tensor<1x128x256xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x128x256xf32> -> tensor<1x128x256xf32> to tensor<1x128x256xf32>
  return %out : tensor<1x128x256xf32>
}

// -----

#gemm_params = #rock.gemm_params<kPerBlock = 64, mPerBlock = 128, nPerBlock = 128, kpack = 8, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// Input-fusion chain rock.transform -> fusion -> rock.transform -> fusion ->
// rock.transform on B, all views injective. The walk traverses the fusion ops
// and finds no reload, so with M skinny and under pressure B is still streamed.
// CHECK-LABEL: @m_skinny_pressure_fusion_chain_injective
// CHECK-DAG: rock.load_marker %{{.*}} {{.*}}{cacheModifier = #rock<CacheModifier cs>{{.*}}} : tensor<1x8192x256xf32> -> tensor<64x128xf32>
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x128x8192xf32> -> tensor<128x64xf32>
#id = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["g", "k", "n"] at [0, 1, 2] -> ["g", "k", "n"] at [0, 1, 2]>] bounds = [1, 8192, 256] -> [1, 8192, 256]>
func.func @m_skinny_pressure_fusion_chain_injective(%arg0: tensor<1x128x8192xf32>, %arg1: tensor<1x8192x256xf32>, %arg2: tensor<1x128x256xf32>) -> tensor<1x128x256xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 2 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.num_cu = 104 : i32} {
  %t1 = rock.transform %arg1 by #id : tensor<1x8192x256xf32> to tensor<1x8192x256xf32>
  %f1 = math.absf %t1 : tensor<1x8192x256xf32>
  %t2 = rock.transform %f1 by #id : tensor<1x8192x256xf32> to tensor<1x8192x256xf32>
  %f2 = arith.addf %t2, %t2 : tensor<1x8192x256xf32>
  %t3 = rock.transform %f2 by #id : tensor<1x8192x256xf32> to tensor<1x8192x256xf32>
  %result = rock.gridwise_gemm(%arg0, %t3) {params = #gemm_params} : tensor<1x128x8192xf32>, tensor<1x8192x256xf32> -> tensor<1x128x256xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x128x256xf32> -> tensor<1x128x256xf32> to tensor<1x128x256xf32>
  return %out : tensor<1x128x256xf32>
}

// -----

#gemm_params = #rock.gemm_params<kPerBlock = 64, mPerBlock = 128, nPerBlock = 128, kpack = 8, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// Same shape of chain, but a Broadcast (reload) sits behind the fusion ops. The
// walk must see through the fusions to the broadcast: B is NOT streamed.
// CHECK-LABEL: @m_skinny_pressure_fusion_chain_reload
// CHECK-DAG: rock.load_marker %{{.*}} {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x8192x256xf32> -> tensor<64x128xf32>
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x128x8192xf32> -> tensor<128x64xf32>
#id2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["g", "k", "n"] at [0, 1, 2] -> ["g", "k", "n"] at [0, 1, 2]>] bounds = [1, 8192, 256] -> [1, 8192, 256]>
#bcastK2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, 0, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Broadcast{1} ["k"] at [1] -> ["k"] at [1]>, <PassThrough ["n"] at [2] -> ["n"] at [2]>] bounds = [1, 8192, 256] -> [1, 1, 256]>
func.func @m_skinny_pressure_fusion_chain_reload(%arg0: tensor<1x128x8192xf32>, %arg1: tensor<1x1x256xf32>, %arg2: tensor<1x128x256xf32>) -> tensor<1x128x256xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 2 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.num_cu = 104 : i32} {
  %b0 = rock.transform %arg1 by #bcastK2 : tensor<1x1x256xf32> to tensor<1x8192x256xf32>
  %f1 = math.absf %b0 : tensor<1x8192x256xf32>
  %t2 = rock.transform %f1 by #id2 : tensor<1x8192x256xf32> to tensor<1x8192x256xf32>
  %f2 = arith.addf %t2, %t2 : tensor<1x8192x256xf32>
  %t3 = rock.transform %f2 by #id2 : tensor<1x8192x256xf32> to tensor<1x8192x256xf32>
  %result = rock.gridwise_gemm(%arg0, %t3) {params = #gemm_params} : tensor<1x128x8192xf32>, tensor<1x8192x256xf32> -> tensor<1x128x256xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x128x256xf32> -> tensor<1x128x256xf32> to tensor<1x128x256xf32>
  return %out : tensor<1x128x256xf32>
}

// -----

// Realistic im2col convolution lowered to a gridwise_gemm (output of
// rocmlir-gen -operation conv ... | rocmlir-driver -rock-conv-to-gemm
// -rock-gemm-to-gridwise). The input operand (gemmK x gemmN, tensor
// <1x1152x6272xf32>) reaches the kernel arg through Embed views (the conv
// im2col), i.e. a reload. gemmM = 64 = mPerBlock -> M skinny, and the input
// footprint (~29 MiB) exceeds the 8 MiB LLC, so absent the reload the input
// would be streamed; because it reloads, it stays cached. The filter is also
// cached (nBlocks is large).
// CHECK-LABEL: @rock_conv_gkc01_ngc01_ngk01
// CHECK-DAG: rock.load_marker %{{.*}} {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x1152x6272xf32> -> tensor<32x32xf32>
// CHECK-DAG: rock.load_marker %{{.*}} {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x64x1152xf32> -> tensor<64x32xf32>
func.func @rock_conv_gkc01_ngc01_ngk01(%arg0: tensor<73728xf32>, %arg1: tensor<1048576xf32>, %arg2: tensor<401408xf32>) -> tensor<401408xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.block_size = 256 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 196 : i32, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 104 : i32} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 128 + d2) * 3 + d3) * 3 + d4)> by [<Unmerge{64, 128, 3, 3} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 128, 3, 3] -> [73728]> : tensor<73728xf32> to tensor<1x64x128x3x3xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 128 + d2) * 16 + d3) * 16 + d4)> by [<Unmerge{32, 128, 16, 16} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [32, 1, 128, 16, 16] -> [1048576]> : tensor<1048576xf32> to tensor<32x1x128x16x16xf32>
  %2 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 9, (d1 mod 9) floordiv 3, d1 mod 3)> by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <Merge{128, 3, 3} ["gemmK"] at [1] -> ["c", "0", "1"] at [2, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["k"] at [1]>] bounds = [1, 1152, 64] -> [1, 64, 128, 3, 3]> : tensor<1x64x128x3x3xf32> to tensor<1x1152x64xf32>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM", "gemmK"] at [1, 2] -> ["gemmM", "gemmK"] at [2, 1]>] bounds = [1, 64, 1152] -> [1, 1152, 64]> : tensor<1x1152x64xf32> to tensor<1x64x1152xf32>
  %4 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)> by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, <PassThrough ["gi"] at [1] -> ["gi"] at [1]>, <PassThrough ["ci"] at [2] -> ["ci"] at [2]>, <Pad{0, 0, 0, 0} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>] bounds = [32, 1, 128, 16, 16] -> [32, 1, 128, 16, 16]> : tensor<32x1x128x16x16xf32> to tensor<32x1x128x16x16xf32>
  %5 = rock.transform %4 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 1} ["0", "0o"] at [3, 4] -> ["0ipad"] at [3]>, <Embed{1, 1} ["1", "1o"] at [5, 6] -> ["1ipad"] at [4]>] bounds = [32, 1, 128, 3, 14, 3, 14] -> [32, 1, 128, 16, 16]> : tensor<32x1x128x16x16xf32> to tensor<32x1x128x3x14x3x14xf32>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2) -> (d2 floordiv 196, d0, d1 floordiv 9, (d1 mod 9) floordiv 3, (d2 mod 196) floordiv 14, d1 mod 3, d2 mod 14)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{128, 3, 3} ["gemmK"] at [1] -> ["ci", "0", "1"] at [2, 3, 5]>, <Merge{32, 14, 14} ["gemmN"] at [2] -> ["ni", "0o", "1o"] at [0, 4, 6]>] bounds = [1, 1152, 6272] -> [32, 1, 128, 3, 14, 3, 14]> : tensor<32x1x128x3x14x3x14xf32> to tensor<1x1152x6272xf32>
  %7 = rock.gridwise_gemm(%3, %6) {params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x64x1152xf32>, tensor<1x1152x6272xf32> -> tensor<1x64x6272xf32>
  %8 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 64 + d2) * 14 + d3) * 14 + d4)> by [<Unmerge{32, 64, 14, 14} ["no", "ko", "0o", "1o"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [32, 1, 64, 14, 14] -> [401408]> : tensor<401408xf32> to tensor<32x1x64x14x14xf32>
  %9 = rock.transform %8 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)> by [<PassThrough ["no", "go", "ko", "0o", "1o"] at [0, 1, 2, 3, 4] -> ["no", "go", "ko", "0o", "1o"] at [0, 1, 2, 3, 4]>] bounds = [32, 1, 64, 14, 14] -> [32, 1, 64, 14, 14]> : tensor<32x1x64x14x14xf32> to tensor<32x1x64x14x14xf32>
  %10 = rock.transform %9 by <affine_map<(d0, d1, d2) -> (d2 floordiv 196, d0, d1, (d2 mod 196) floordiv 14, d2 mod 14)> by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <PassThrough ["gemmM"] at [1] -> ["ko"] at [2]>, <Merge{32, 14, 14} ["gemmN"] at [2] -> ["no", "0o", "1o"] at [0, 3, 4]>] bounds = [1, 64, 6272] -> [32, 1, 64, 14, 14]> : tensor<32x1x64x14x14xf32> to tensor<1x64x6272xf32>
  %12 = rock.store %7 to %10 alias %arg2 by set : tensor<1x64x6272xf32> -> tensor<401408xf32> to tensor<1x64x6272xf32> alias tensor<401408xf32>
  return %12 : tensor<401408xf32>
}
