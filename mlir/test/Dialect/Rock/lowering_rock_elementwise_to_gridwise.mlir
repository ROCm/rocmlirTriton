// RUN: rocmlir-opt -rock-elementwise-to-gridwise -mlir-print-local-scope %s | FileCheck %s

// Non-elementwise kernel should be skipped.
// CHECK-LABEL: func.func @noop_gemm
// CHECK-NOT: rock.grid_size
// CHECK: rock.gemm
func.func @noop_gemm(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = rock.store %0 to %arg2 by set : tensor<8x32xf32> -> tensor<8x32xf32> to tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}

// 1D input that exactly fills one tile: no padding, gridSize = 1.
// CHECK-LABEL: func.func @exact_tile
// CHECK-SAME: rock.grid_size = 1 : i32
// CHECK-NOT: Pad
// CHECK: arith.addf %arg0, %arg1 : tensor<256xf32>
func.func @exact_tile(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
  %1 = rock.store %0 to %arg2 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
  return %1 : tensor<256xf32>
}

// 1D input smaller than tile: padded from 12 to 256, gridSize = 1.
// Store dest (%arg2) and both inputs are padded.
// CHECK-LABEL: func.func @pad_1d
// CHECK-SAME: rock.grid_size = 1 : i32
// CHECK: rock.transform %arg2 by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf32> to tensor<256xf32>
// CHECK: rock.transform %arg1 by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf32> to tensor<256xf32>
// CHECK: rock.transform %arg0 by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf32> to tensor<256xf32>
// CHECK: arith.addf {{.*}} : tensor<256xf32>
// CHECK: rock.store {{.*}} by set : tensor<256xf32> -> tensor<12xf32> to tensor<256xf32>
func.func @pad_1d(%arg0: tensor<12xf32>, %arg1: tensor<12xf32>, %arg2: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<12xf32>
  %1 = rock.store %0 to %arg2 by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
  return %1 : tensor<12xf32>
}

// 2D input: merged to 1D then padded. 3x4=12 -> pad to 256, gridSize = 1.
// Both inputs and the store dest get Merge then Pad.
// CHECK-LABEL: func.func @merge_and_pad_2d
// CHECK-SAME: rock.grid_size = 1 : i32
// Store dest is merged + padded.
// CHECK: rock.transform %arg2 by {{.*}}Merge{3, 4}{{.*}} : tensor<3x4xf32> to tensor<12xf32>
// CHECK: rock.transform %{{.*}} by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf32> to tensor<256xf32>
// Inputs are merged + padded.
// CHECK: rock.transform %arg0 by {{.*}}Merge{3, 4}{{.*}} : tensor<3x4xf32> to tensor<12xf32>
// CHECK: arith.addf {{.*}} : tensor<256xf32>
// Store: padded source, original buffer type preserved.
// CHECK: rock.store {{.*}} by set : tensor<256xf32> -> tensor<3x4xf32> to tensor<256xf32>
func.func @merge_and_pad_2d(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<3x4xf32>
  %1 = rock.store %0 to %arg2 by set : tensor<3x4xf32> -> tensor<3x4xf32> to tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// Multiple tiles: 600 elements / 256 tile = ceil -> gridSize = 3, padded to 768.
// Both inputs and the store dest are padded.
// CHECK-LABEL: func.func @multi_tile
// CHECK-SAME: rock.grid_size = 3 : i32
// Store dest padded.
// CHECK: rock.transform %arg2 by {{.*}}Pad{0, 168}{{.*}} : tensor<600xf32> to tensor<768xf32>
// Inputs padded.
// CHECK: rock.transform %arg1 by {{.*}}Pad{0, 168}{{.*}} : tensor<600xf32> to tensor<768xf32>
// CHECK: rock.transform %arg0 by {{.*}}Pad{0, 168}{{.*}} : tensor<600xf32> to tensor<768xf32>
// CHECK: arith.addf {{.*}} : tensor<768xf32>
// Store: padded source, original buffer type preserved.
// CHECK: rock.store {{.*}} by set : tensor<768xf32> -> tensor<600xf32> to tensor<768xf32>
func.func @multi_tile(%arg0: tensor<600xf32>, %arg1: tensor<600xf32>, %arg2: tensor<600xf32>) -> tensor<600xf32> attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<600xf32>
  %1 = rock.store %0 to %arg2 by set : tensor<600xf32> -> tensor<600xf32> to tensor<600xf32>
  return %1 : tensor<600xf32>
}

// Chained fusions: add then mul. Only leaf inputs are padded; intermediate
// results are propagated. Store destination is also padded.
// CHECK-LABEL: func.func @chained_fusions
// CHECK-SAME: rock.grid_size = 1 : i32
// Store dest merged + padded.
// CHECK: rock.transform %arg3 by {{.*}}Merge{3, 4}{{.*}} : tensor<3x4xf32> to tensor<12xf32>
// CHECK: rock.transform %{{.*}} by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf32> to tensor<256xf32>
// Fusion ops on padded tensors.
// CHECK: arith.addf {{.*}} : tensor<256xf32>
// CHECK: arith.mulf {{.*}} : tensor<256xf32>
// Store: padded source, original buffer type preserved.
// CHECK: rock.store {{.*}} by set : tensor<256xf32> -> tensor<3x4xf32> to tensor<256xf32>
func.func @chained_fusions(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<3x4xf32>, %arg3: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<3x4xf32>
  %1 = arith.mulf %0, %arg2 : tensor<3x4xf32>
  %2 = rock.store %1 to %arg3 by set : tensor<3x4xf32> -> tensor<3x4xf32> to tensor<3x4xf32>
  return %2 : tensor<3x4xf32>
}

// Constants as inputs: arith.constant operands are also flattened and padded.
// CHECK-LABEL: func.func @constant_input
// CHECK-SAME: rock.grid_size = 1 : i32
// Store dest padded.
// CHECK: rock.transform %arg1 by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf32> to tensor<256xf32>
// Input arg padded.
// CHECK: rock.transform %arg0 by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf32> to tensor<256xf32>
// Constant created then padded.
// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<12xf32>
// CHECK: rock.transform %[[CST]] by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf32> to tensor<256xf32>
// CHECK: arith.addf {{.*}} : tensor<256xf32>
// Store: padded source, original buffer type preserved.
// CHECK: rock.store {{.*}} by set : tensor<256xf32> -> tensor<12xf32> to tensor<256xf32>
func.func @constant_input(%arg0: tensor<12xf32>, %arg1: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32} {
  %cst = arith.constant dense<1.0> : tensor<12xf32>
  %0 = arith.addf %arg0, %cst : tensor<12xf32>
  %1 = rock.store %0 to %arg1 by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
  return %1 : tensor<12xf32>
}

// Two independent subgraphs with different sizes. maxElements comes from the
// larger subgraph (600), so gridSize = ceil(600/256) = 3, paddedSize = 768.
// The smaller subgraph (100 elements) is padded to 768 as well.
// CHECK-LABEL: func.func @two_subgraphs_max_elements
// CHECK-SAME: rock.grid_size = 3 : i32
// Outputs: larger gets Pad{0,168}, smaller gets Pad{0,668}.
// CHECK: rock.transform %arg5 by {{.*}}Pad{0, 168}{{.*}} : tensor<600xf32> to tensor<768xf32>
// CHECK: rock.transform %arg4 by {{.*}}Pad{0, 668}{{.*}} : tensor<100xf32> to tensor<768xf32>
// Inputs: the 600-element args get Pad{0,168}, the 100-element args get Pad{0,668}.
// CHECK: rock.transform %arg3 by {{.*}}Pad{0, 168}{{.*}} : tensor<600xf32> to tensor<768xf32>
// CHECK: rock.transform %arg2 by {{.*}}Pad{0, 168}{{.*}} : tensor<600xf32> to tensor<768xf32>
// CHECK: rock.transform %arg1 by {{.*}}Pad{0, 668}{{.*}} : tensor<100xf32> to tensor<768xf32>
// CHECK: rock.transform %arg0 by {{.*}}Pad{0, 668}{{.*}} : tensor<100xf32> to tensor<768xf32>
// Both fusions operate on padded 768-element tensors.
// CHECK: arith.addf {{.*}} : tensor<768xf32>
// CHECK: arith.addf {{.*}} : tensor<768xf32>
// Stores: source is 768, dest types are preserved (original buffer shapes).
// CHECK: rock.store {{.*}} to {{.*}} by set : tensor<768xf32> -> tensor<100xf32> to tensor<768xf32>
// CHECK: rock.store {{.*}} to {{.*}} by set : tensor<768xf32> -> tensor<600xf32> to tensor<768xf32>
func.func @two_subgraphs_max_elements(
    %arg0: tensor<100xf32>, %arg1: tensor<100xf32>,
    %arg2: tensor<600xf32>, %arg3: tensor<600xf32>,
    %arg4: tensor<100xf32>, %arg5: tensor<600xf32>)
    -> (tensor<100xf32>, tensor<600xf32>)
    attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<100xf32>
  %1 = arith.addf %arg2, %arg3 : tensor<600xf32>
  %2 = rock.store %0 to %arg4 by set : tensor<100xf32> -> tensor<100xf32> to tensor<100xf32>
  %3 = rock.store %1 to %arg5 by set : tensor<600xf32> -> tensor<600xf32> to tensor<600xf32>
  return %2, %3 : tensor<100xf32>, tensor<600xf32>
}

// Type-changing fusion: f32 inputs -> arith.truncf -> f16 output.
// propagateOutputType preserves element types while changing shapes.
// CHECK-LABEL: func.func @type_change_f32_to_f16
// CHECK-SAME: rock.grid_size = 1 : i32
// f16 output destination is padded but keeps f16 element type.
// CHECK: rock.transform %arg2 by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf16> to tensor<256xf16>
// f32 inputs are padded and keep f32 element type.
// CHECK: rock.transform %arg1 by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf32> to tensor<256xf32>
// CHECK: rock.transform %arg0 by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf32> to tensor<256xf32>
// addf operates on padded f32 tensors.
// CHECK: %[[ADD:.*]] = arith.addf {{.*}} : tensor<256xf32>
// truncf converts padded f32 to padded f16.
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[ADD]] : tensor<256xf32> to tensor<256xf16>
// Store source is f16, dest buffer type preserved.
// CHECK: rock.store %[[TRUNC]] to {{.*}} by set : tensor<256xf16> -> tensor<12xf16> to tensor<256xf16>
func.func @type_change_f32_to_f16(
    %arg0: tensor<12xf32>, %arg1: tensor<12xf32>, %arg2: tensor<12xf16>)
    -> tensor<12xf16>
    attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<12xf32>
  %1 = arith.truncf %0 : tensor<12xf32> to tensor<12xf16>
  %2 = rock.store %1 to %arg2 by set : tensor<12xf16> -> tensor<12xf16> to tensor<12xf16>
  return %2 : tensor<12xf16>
}

// Long chain: propagateOutputType propagates through add -> mul -> sub,
// all fusion ops get the padded shape while preserving element type.
// CHECK-LABEL: func.func @propagate_long_chain
// CHECK-SAME: rock.grid_size = 1 : i32
// Store dest merged + padded.
// CHECK: rock.transform %arg4 by {{.*}}Merge{3, 4}{{.*}} : tensor<3x4xf32> to tensor<12xf32>
// CHECK: rock.transform %{{.*}} by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf32> to tensor<256xf32>
// Leaf inputs are merged + padded.
// CHECK: rock.transform %arg0 by {{.*}}Merge{3, 4}{{.*}} : tensor<3x4xf32> to tensor<12xf32>
// CHECK: rock.transform %{{.*}} by {{.*}}Pad{0, 244}{{.*}} : tensor<12xf32> to tensor<256xf32>
// All three fusions operate on padded 256-element tensors.
// CHECK: %[[ADD:.*]] = arith.addf {{.*}} : tensor<256xf32>
// CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], {{.*}} : tensor<256xf32>
// CHECK: %[[SUB:.*]] = arith.subf %[[MUL]], {{.*}} : tensor<256xf32>
// Store: padded source, original buffer type preserved.
// CHECK: rock.store %[[SUB]] to {{.*}} by set : tensor<256xf32> -> tensor<3x4xf32> to tensor<256xf32>
func.func @propagate_long_chain(
    %arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>,
    %arg2: tensor<3x4xf32>, %arg3: tensor<3x4xf32>,
    %arg4: tensor<3x4xf32>) -> tensor<3x4xf32>
    attributes {rock.kernel, perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>, rock.block_size = 256 : i32} {
  %0 = arith.addf %arg0, %arg1 : tensor<3x4xf32>
  %1 = arith.mulf %0, %arg2 : tensor<3x4xf32>
  %2 = arith.subf %1, %arg3 : tensor<3x4xf32>
  %3 = rock.store %2 to %arg4 by set : tensor<3x4xf32> -> tensor<3x4xf32> to tensor<3x4xf32>
  return %3 : tensor<3x4xf32>
}
