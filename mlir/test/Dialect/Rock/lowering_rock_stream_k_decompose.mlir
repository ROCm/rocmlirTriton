// Tests for the rock-stream-k-decompose pass.
//
// The pass runs at the gridwise layer (after gemm/attn-to-gridwise, before
// gridwise-to-blockwise). It splits one rock.gridwise_gemm into data-parallel
// wave sub-gemms (stored `set`) plus a split-K remainder sub-gemm (K folded
// into G, accumulated `atomic_add`), all sharing a single persistent grid of
// P = streamKMultiple * num_cu workgroups. A wave spans the other two dims
// fully and `span` blocks along the partition dim. Output fusion is replicated
// per cell.
//
// The pass is gated by the `streamKMultiple` tuning parameter (0 = disabled,
// >= 1 = that multiple of num_cu) and consumes the `rock.streamk_part_dim`
// attribute GemmToGridwise records on the gridwise_gemm (which also pads the
// grid so the decomposition is exact); these hand-written inputs set it
// directly. -canonicalize drops the now-dead original gridwise_gemm / fusion /
// store (the real pipeline DCEs them via addWithDCE).

// RUN: rocmlir-opt -rock-stream-k-decompose -canonicalize -split-input-file -mlir-print-local-scope %s | FileCheck %s

#params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, streamKMultiple = 1>

// ============================================================
// num_cu = 80, streamKMultiple = 1 -> P = 80. G=1, mBlocks=nBlocks=10
// (gridFull=100). N is tried first:
//   span = P/(G*mBlocks) = 80/10 = 8 n-blocks/wave -> 1 data-parallel wave
//     covering N cols [0,1024) -> tensor<1x1280x1024xf32> (A shared, B sliced).
//   remBlocks = 10 - 8 = 2 -> remTiles = 20, splitK = 80/20 = 4
//     -> G folds to 4, K folds to 64 -> tensor<4x1280x256xf32>, atomic_add.
//   func grid_size becomes P = 80.
// ============================================================

// CHECK-LABEL: func.func @stream_k_hybrid
// CHECK-SAME: rock.grid_size = 80 : i32
// CHECK-DAG: rock.gridwise_gemm{{.*}} -> tensor<1x1280x1024xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}} -> tensor<4x1280x256xf32>
// CHECK-DAG: rock.store{{.*}}by set
// CHECK-DAG: rock.store{{.*}}by atomic_add
func.func @stream_k_hybrid(%a: tensor<1x1280x256xf16>, %b: tensor<1x256x1280xf16>, %c: tensor<1x1280x1280xf32>) -> tensor<1x1280x1280xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 80 : i32, rock.grid_size = 100 : i32} {
  %r = rock.gridwise_gemm(%a, %b) {params = #params, rock.streamk_part_dim = #rock<rock.streamk_part_dim n>} : tensor<1x1280x256xf16>, tensor<1x256x1280xf16> -> tensor<1x1280x1280xf32>
  %out = rock.store %r to %c by set : tensor<1x1280x1280xf32> -> tensor<1x1280x1280xf32> to tensor<1x1280x1280xf32>
  return %out : tensor<1x1280x1280xf32>
}

// -----

#params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, streamKMultiple = 1>

// ============================================================
// Skinny-M: G=1, mBlocks=1, nBlocks=90 (gridFull=90), num_cu=80. Partition N:
//   span = P/(G*mBlocks) = 80 n-blocks/wave -> 1 wave -> tensor<1x128x10240xf32>.
//   remBlocks = 90 - 80 = 10 -> remTiles = 10, splitK = 8 -> tensor<8x128x1280xf32>.
// This case previously bailed under M-only slicing.
// ============================================================

// CHECK-LABEL: func.func @stream_k_skinny_m
// CHECK-SAME: rock.grid_size = 80 : i32
// CHECK-DAG: rock.gridwise_gemm{{.*}} -> tensor<1x128x10240xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}} -> tensor<8x128x1280xf32>
// CHECK-DAG: rock.store{{.*}}by set
// CHECK-DAG: rock.store{{.*}}by atomic_add
func.func @stream_k_skinny_m(%a: tensor<1x128x512xf16>, %b: tensor<1x512x11520xf16>, %c: tensor<1x128x11520xf32>) -> tensor<1x128x11520xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 80 : i32, rock.grid_size = 90 : i32} {
  %r = rock.gridwise_gemm(%a, %b) {params = #params, rock.streamk_part_dim = #rock<rock.streamk_part_dim n>} : tensor<1x128x512xf16>, tensor<1x512x11520xf16> -> tensor<1x128x11520xf32>
  %out = rock.store %r to %c by set : tensor<1x128x11520xf32> -> tensor<1x128x11520xf32> to tensor<1x128x11520xf32>
  return %out : tensor<1x128x11520xf32>
}

// -----

#params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, streamKMultiple = 1>

// ============================================================
// Output fusion: the gemm result is combined with two extra inputs (add then
// mul) before the store. The fusion chain is replicated per cell (one
// data-parallel wave + one split-K remainder), with each extra input sliced to
// match. The data-parallel cell stores `set`; the remainder cell accumulates
// `atomic_add`.
// ============================================================

// CHECK-LABEL: func.func @stream_k_output_fusion
// CHECK-SAME: rock.grid_size = 80 : i32
// CHECK-DAG: rock.gridwise_gemm{{.*}} -> tensor<1x1280x1024xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}} -> tensor<4x1280x256xf32>
// CHECK-COUNT-2: arith.addf
// CHECK-COUNT-2: arith.mulf
// CHECK-DAG: rock.store{{.*}}by set
// CHECK-DAG: rock.store{{.*}}by atomic_add
func.func @stream_k_output_fusion(%a: tensor<1x1280x256xf16>, %b: tensor<1x256x1280xf16>, %c: tensor<1x1280x1280xf32>, %bias0: tensor<1x1280x1280xf32>, %bias1: tensor<1x1280x1280xf32>) -> tensor<1x1280x1280xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 80 : i32, rock.grid_size = 100 : i32} {
  %r = rock.gridwise_gemm(%a, %b) {params = #params, rock.streamk_part_dim = #rock<rock.streamk_part_dim n>} : tensor<1x1280x256xf16>, tensor<1x256x1280xf16> -> tensor<1x1280x1280xf32>
  %add = arith.addf %r, %bias0 : tensor<1x1280x1280xf32>
  %mul = arith.mulf %add, %bias1 : tensor<1x1280x1280xf32>
  %out = rock.store %mul to %c by set : tensor<1x1280x1280xf32> -> tensor<1x1280x1280xf32> to tensor<1x1280x1280xf32>
  return %out : tensor<1x1280x1280xf32>
}

// -----

#params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, streamKMultiple = 1>

// No rock.streamk_part_dim attribute (the grid fits within one persistent wave,
// gridFull = 4 <= P = 80, so GemmToGridwise records no partition dim), so the
// pass leaves the gemm unchanged.

// CHECK-LABEL: func.func @fits_one_wave
// CHECK: rock.gridwise_gemm{{.*}} -> tensor<1x256x256xf32>
// CHECK-NOT: by atomic_add
func.func @fits_one_wave(%a: tensor<1x256x256xf16>, %b: tensor<1x256x256xf16>, %c: tensor<1x256x256xf32>) -> tensor<1x256x256xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 80 : i32, rock.grid_size = 4 : i32} {
  %r = rock.gridwise_gemm(%a, %b) {params = #params} : tensor<1x256x256xf16>, tensor<1x256x256xf16> -> tensor<1x256x256xf32>
  %out = rock.store %r to %c by set : tensor<1x256x256xf32> -> tensor<1x256x256xf32> to tensor<1x256x256xf32>
  return %out : tensor<1x256x256xf32>
}

// -----

#params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, streamKMultiple = 1>

// ============================================================
// Output-fusion K-reduction regularization: a fused `add gemmOut, bias` is
// applied once per K-fold, but the split-K remainder atomic_adds `splitK`
// partial products into the same tile, so `bias` would be added `splitK` times.
// The decompose pass divides `bias` by splitK (here 4) *only* in the remainder
// cell; the data-parallel wave (stored `set`) adds `bias` unscaled.
//
//   remainder: add(gemmOut_rem, bias_rem / 4.0) -> atomic_add
//   wave:      add(gemmOut_wave, bias_wave)      -> set
// ============================================================

// CHECK-LABEL: func.func @stream_k_fusion_regularization
// CHECK-SAME: rock.grid_size = 80 : i32
// The bias addend is divided by splitK = 4 exactly once (the remainder cell),
// then added; the data-parallel wave adds the bias unscaled.
// CHECK-DAG: %[[SK:.*]] = arith.constant dense<4.000000e+00>
// CHECK-DAG: arith.addf %{{.*}}, %{{.*}} : tensor<1x1280x1024xf32>
// CHECK-DAG: %[[DIV:.*]] = arith.divf %{{.*}}, %[[SK]] : tensor<4x1280x256xf32>
// CHECK-DAG: arith.addf %{{.*}}, %[[DIV]] : tensor<4x1280x256xf32>
// CHECK-DAG: rock.store{{.*}}by set
// CHECK-DAG: rock.store{{.*}}by atomic_add
func.func @stream_k_fusion_regularization(%a: tensor<1x1280x256xf16>, %b: tensor<1x256x1280xf16>, %c: tensor<1x1280x1280xf32>, %bias: tensor<1x1280x1280xf32>) -> tensor<1x1280x1280xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 80 : i32, rock.grid_size = 100 : i32} {
  %r = rock.gridwise_gemm(%a, %b) {params = #params, rock.streamk_part_dim = #rock<rock.streamk_part_dim n>} : tensor<1x1280x256xf16>, tensor<1x256x1280xf16> -> tensor<1x1280x1280xf32>
  %add = arith.addf %r, %bias : tensor<1x1280x1280xf32>
  %out = rock.store %add to %c by set : tensor<1x1280x1280xf32> -> tensor<1x1280x1280xf32> to tensor<1x1280x1280xf32>
  return %out : tensor<1x1280x1280xf32>
}

// -----

#params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, streamKMultiple = 2>

// ============================================================
// Smaller-P search: num_cu = 80, streamKMultiple = 2 -> target P = 160.
// G=1, mBlocks=6, nBlocks=28 (gridFull=168). No partition refills P=160
// exactly (160 % (G*mBlocks)=160%6 != 0, 160 % (G*nBlocks)=160%28 != 0), so the
// pass picks the largest valid P = 156 (N-partition: span=26 -> 3328 cols,
// remBlocks=2, splitK=13). The func grid_size becomes the reduced P = 156.
// K = 832 = splitK(13) * kPerBlock(64), the alignment GemmToGridwise guarantees.
// ============================================================

// CHECK-LABEL: func.func @stream_k_smaller_p
// CHECK-SAME: rock.grid_size = 156 : i32
// CHECK-DAG: rock.gridwise_gemm{{.*}} -> tensor<1x768x3328xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}} -> tensor<13x768x256xf32>
// CHECK-DAG: rock.store{{.*}}by set
// CHECK-DAG: rock.store{{.*}}by atomic_add
func.func @stream_k_smaller_p(%a: tensor<1x768x832xf16>, %b: tensor<1x832x3584xf16>, %c: tensor<1x768x3584xf32>) -> tensor<1x768x3584xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 80 : i32, rock.grid_size = 168 : i32} {
  %r = rock.gridwise_gemm(%a, %b) {params = #params, rock.streamk_part_dim = #rock<rock.streamk_part_dim n>} : tensor<1x768x832xf16>, tensor<1x832x3584xf16> -> tensor<1x768x3584xf32>
  %out = rock.store %r to %c by set : tensor<1x768x3584xf32> -> tensor<1x768x3584xf32> to tensor<1x768x3584xf32>
  return %out : tensor<1x768x3584xf32>
}

// -----

#params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// Disabled: streamKMultiple defaults to 0, so the pass is a no-op even though
// the grid (gridFull = 100) overflows num_cu = 80.

// CHECK-LABEL: func.func @stream_k_disabled
// CHECK-SAME: rock.grid_size = 100 : i32
// CHECK: rock.gridwise_gemm{{.*}} -> tensor<1x1280x1280xf32>
// CHECK-NOT: by atomic_add
func.func @stream_k_disabled(%a: tensor<1x1280x256xf16>, %b: tensor<1x256x1280xf16>, %c: tensor<1x1280x1280xf32>) -> tensor<1x1280x1280xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 80 : i32, rock.grid_size = 100 : i32} {
  %r = rock.gridwise_gemm(%a, %b) {params = #params} : tensor<1x1280x256xf16>, tensor<1x256x1280xf16> -> tensor<1x1280x1280xf32>
  %out = rock.store %r to %c by set : tensor<1x1280x1280xf32> -> tensor<1x1280x1280xf32> to tensor<1x1280x1280xf32>
  return %out : tensor<1x1280x1280xf32>
}
