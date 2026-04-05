// RUN: rocmlir-opt -rock-gridwise-elementwise-to-blockwise -mlir-print-local-scope %s | FileCheck %s

// Non-elementwise kernel should be skipped.
// CHECK-LABEL: func.func @noop_gemm
// CHECK-NOT: tt.get_program_id
// CHECK-NOT: rock.load_marker
// CHECK: rock.gemm
func.func @noop_gemm(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = rock.store %0 to %arg2 by set : tensor<8x32xf32> -> tensor<8x32xf32> to tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}

// Simple 1D, single tile: load_markers on inputs, store_marker on output.
// CHECK-LABEL: func.func @simple_1d
// CHECK: %[[BID:.*]] = tt.get_program_id x
// CHECK: %[[LM1:.*]] = rock.load_marker %arg1 views [{{.*}}Unmerge{1, 256}{{.*}}][%[[BID]]]
// CHECK: %[[LM0:.*]] = rock.load_marker %arg0 views [{{.*}}Unmerge{1, 256}{{.*}}][%[[BID]]]
// CHECK: %[[ADD:.*]] = arith.addf %[[LM0]], %[[LM1]] : tensor<256xf32>
// CHECK: %[[SM:.*]] = rock.store_marker %[[ADD]] views [{{.*}}Unmerge{1, 256}{{.*}}][%[[BID]]]
// CHECK: rock.store %[[SM]] to %arg2
func.func @simple_1d(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32, rock.grid_size = 1 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
  %1 = rock.store %0 to %arg2 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
  return %1 : tensor<256xf32>
}

// Multi-tile: gridSize = 3, tileSize = 256, Unmerge{3, 256}.
// CHECK-LABEL: func.func @multi_tile
// CHECK: %[[BID:.*]] = tt.get_program_id x
// CHECK: rock.load_marker %arg1 views [{{.*}}Unmerge{3, 256}{{.*}}][%[[BID]]]
// CHECK: rock.load_marker %arg0 views [{{.*}}Unmerge{3, 256}{{.*}}][%[[BID]]]
// CHECK: arith.addf {{.*}} : tensor<256xf32>
// CHECK: rock.store_marker {{.*}} views [{{.*}}Unmerge{3, 256}{{.*}}][%[[BID]]]
func.func @multi_tile(%arg0: tensor<768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>) -> tensor<768xf32> attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32, rock.grid_size = 3 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<768xf32>
  %1 = rock.store %0 to %arg2 by set : tensor<768xf32> -> tensor<768xf32> to tensor<768xf32>
  return %1 : tensor<768xf32>
}

// Chained fusions: load_markers on all leaf inputs, store_marker on final.
// Intermediate results (addf output) do NOT get markers.
// CHECK-LABEL: func.func @chained
// CHECK: %[[BID:.*]] = tt.get_program_id x
// CHECK: %[[LM2:.*]] = rock.load_marker %arg2 views
// CHECK: %[[LM1:.*]] = rock.load_marker %arg1 views
// CHECK: %[[LM0:.*]] = rock.load_marker %arg0 views
// CHECK: %[[ADD:.*]] = arith.addf %[[LM0]], %[[LM1]]
// CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %[[LM2]]
// CHECK: %[[SM:.*]] = rock.store_marker %[[MUL]] views
// CHECK: rock.store %[[SM]] to
func.func @chained(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32, rock.grid_size = 1 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
  %1 = arith.mulf %0, %arg2 : tensor<256xf32>
  %2 = rock.store %1 to %arg3 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
  return %2 : tensor<256xf32>
}

// Constant inputs get load_markers too.
// CHECK-LABEL: func.func @constant_input
// CHECK: %[[BID:.*]] = tt.get_program_id x
// CHECK: rock.load_marker %arg0 views
// CHECK: arith.constant
// CHECK: rock.load_marker %cst views
// CHECK: arith.addf
// CHECK: rock.store_marker
func.func @constant_input(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32, rock.grid_size = 1 : i32} {
  %cst = arith.constant dense<1.0> : tensor<256xf32>
  %0 = arith.addf %arg0, %cst : tensor<256xf32>
  %1 = rock.store %0 to %arg1 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
  return %1 : tensor<256xf32>
}

// propagateOutputType through a long chain: add -> mul -> sub.
// load_markers on all four leaf inputs, tile type propagated through
// all three fusions, store_marker only on the final result.
// CHECK-LABEL: func.func @propagate_long_chain
// CHECK: %[[BID:.*]] = tt.get_program_id x
// CHECK: %[[LM3:.*]] = rock.load_marker %arg3 views [{{.*}}Unmerge{1, 256}{{.*}}][%[[BID]]] : tensor<256xf32> -> tensor<256xf32>
// CHECK: %[[LM2:.*]] = rock.load_marker %arg2 views [{{.*}}Unmerge{1, 256}{{.*}}][%[[BID]]] : tensor<256xf32> -> tensor<256xf32>
// CHECK: %[[LM1:.*]] = rock.load_marker %arg1 views [{{.*}}Unmerge{1, 256}{{.*}}][%[[BID]]] : tensor<256xf32> -> tensor<256xf32>
// CHECK: %[[LM0:.*]] = rock.load_marker %arg0 views [{{.*}}Unmerge{1, 256}{{.*}}][%[[BID]]] : tensor<256xf32> -> tensor<256xf32>
// CHECK: %[[ADD:.*]] = arith.addf %[[LM0]], %[[LM1]] : tensor<256xf32>
// CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %[[LM2]] : tensor<256xf32>
// CHECK: %[[SUB:.*]] = arith.subf %[[MUL]], %[[LM3]] : tensor<256xf32>
// CHECK: %[[SM:.*]] = rock.store_marker %[[SUB]] views [{{.*}}Unmerge{1, 256}{{.*}}][%[[BID]]] : tensor<256xf32> -> tensor<256xf32>
// CHECK: rock.store %[[SM]] to %arg4
func.func @propagate_long_chain(
    %arg0: tensor<256xf32>, %arg1: tensor<256xf32>,
    %arg2: tensor<256xf32>, %arg3: tensor<256xf32>,
    %arg4: tensor<256xf32>) -> tensor<256xf32>
    attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32, rock.grid_size = 1 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
  %1 = arith.mulf %0, %arg2 : tensor<256xf32>
  %2 = arith.subf %1, %arg3 : tensor<256xf32>
  %3 = rock.store %2 to %arg4 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
  return %3 : tensor<256xf32>
}

// Type-changing fusion: f32 inputs -> addf -> truncf -> f16 output.
// propagateOutputType preserves element types: addf stays f32, truncf
// produces f16 tile, store_marker carries f16.
// CHECK-LABEL: func.func @type_change_f32_to_f16
// CHECK: %[[BID:.*]] = tt.get_program_id x
// CHECK: %[[LM1:.*]] = rock.load_marker %arg1 views {{.*}} : tensor<256xf32> -> tensor<256xf32>
// CHECK: %[[LM0:.*]] = rock.load_marker %arg0 views {{.*}} : tensor<256xf32> -> tensor<256xf32>
// CHECK: %[[ADD:.*]] = arith.addf %[[LM0]], %[[LM1]] : tensor<256xf32>
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[ADD]] : tensor<256xf32> to tensor<256xf16>
// CHECK: %[[SM:.*]] = rock.store_marker %[[TRUNC]] views {{.*}} : tensor<256xf16> -> tensor<256xf16>
// CHECK: rock.store %[[SM]] to %arg2 {{.*}} : tensor<256xf16> -> tensor<256xf16> to tensor<256xf16>
func.func @type_change_f32_to_f16(
    %arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf16>)
    -> tensor<256xf16>
    attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32, rock.grid_size = 1 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
  %1 = arith.truncf %0 : tensor<256xf32> to tensor<256xf16>
  %2 = rock.store %1 to %arg2 by set : tensor<256xf16> -> tensor<256xf16> to tensor<256xf16>
  return %2 : tensor<256xf16>
}

// Two outputs: each store gets its own store_marker, both sharing the same bid.
// CHECK-LABEL: func.func @two_stores
// CHECK: %[[BID:.*]] = tt.get_program_id x
// CHECK: %[[LM1:.*]] = rock.load_marker %arg1 views
// CHECK: %[[LM0:.*]] = rock.load_marker %arg0 views
// CHECK: %[[ADD:.*]] = arith.addf %[[LM0]], %[[LM1]] : tensor<256xf32>
// CHECK: %[[MUL:.*]] = arith.mulf %[[LM0]], %[[LM1]] : tensor<256xf32>
// CHECK: %[[SM0:.*]] = rock.store_marker %[[ADD]] views {{.*}}[%[[BID]]]
// CHECK: rock.store %[[SM0]] to %arg2
// CHECK: %[[SM1:.*]] = rock.store_marker %[[MUL]] views {{.*}}[%[[BID]]]
// CHECK: rock.store %[[SM1]] to %arg3
func.func @two_stores(
    %arg0: tensor<256xf32>, %arg1: tensor<256xf32>,
    %arg2: tensor<256xf32>, %arg3: tensor<256xf32>)
    -> (tensor<256xf32>, tensor<256xf32>)
    attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32, rock.grid_size = 1 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
  %1 = arith.mulf %arg0, %arg1 : tensor<256xf32>
  %2 = rock.store %0 to %arg2 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
  %3 = rock.store %1 to %arg3 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
  return %2, %3 : tensor<256xf32>, tensor<256xf32>
}
