// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --rock-conv-to-gemm --mlir-print-local-scope --split-input-file | FileCheck %s

// Backward convolution coverage for the rock-conv-to-gemm pass. Each function
// below is the IR that `rocmlir-gen` followed by the earlier pipeline steps
// (`--rock-affix-params --rock-lower-reduce --rock-regularize-output
// --rock-regularize-inter-gemm-fusion`) hands to `rock-conv-to-gemm`. Keeping
// the pre-cooked form here means the RUN line only exercises conv-to-gemm,
// matching the convention used by `conv_to_gemm.mlir`.

// ============================================================================
// @bwd_data_basic
// Happy-path single-GEMM lowering for rock.conv_bwd_data.
// 2x1x4x8x8 input grad, 1x4x4x3x3 filter, 2x1x4x6x6 output grad, stride 1.
// Pins:
//   - "tilda" / "dot" decomposition on the filter (Embed{1,1} + Slice)
//   - Embed{-1, 1} on the output grad (negation matches stride-1 phase)
//   - Pad{0,0,0,0} on the input grad (no spatial padding)
//   - gemmK = filter K * dotslice = 4 * 3 * 3 = 36
//   - gemmM = filter C * tildaslice = 4 * 1 * 1 = 4
//   - gemmN = batch * islice = 2 * 8 * 8 = 128
//   - Single rock.gemm followed by a single rock.store ... by set
// ============================================================================
// CHECK-LABEL: @bwd_data_basic
// CHECK-NOT: rock.conv_bwd_data
// CHECK: <Embed{1, 1} ["0dot", "0tilda"] at [3, 4] -> ["0"] at [3]>, <Embed{1, 1} ["1dot", "1tilda"] at [5, 6] -> ["1"] at [4]>
// CHECK: <Slice{0, 3, 0, 3} ["0dotslice", "1dotslice"] at [3, 5] -> ["0dot", "1dot"] at [3, 5]>, <Slice{0, 1, 0, 1} ["0tildaslice", "1tildaslice"] at [4, 6] -> ["0tilda", "1tilda"] at [4, 6]>
// CHECK: <Merge{4, 3, 3} ["gemmK"]
// CHECK-SAME: <Merge{4, 1, 1} ["gemmM"]
// CHECK: <Pad{0, 0, 0, 0} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>
// CHECK: <Embed{-1, 1} ["0dot", "0tilda"] at [3, 4] -> ["0o"] at [3]>, <Embed{-1, 1} ["1dot", "1tilda"] at [5, 6] -> ["1o"] at [4]>
// CHECK: rock.gemm tr {{.*}} : tensor<1x36x4xf32> * tensor<1x36x128xf32> -> tensor<1x4x128xf32>
// CHECK: rock.store {{.*}} by set
// CHECK-NOT: rock.gemm
// CHECK-NOT: rock.store
func.func @bwd_data_basic(%arg0: tensor<144xf32>, %arg1: tensor<288xf32>, %arg2: tensor<512xf32>) -> tensor<512xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.block_size = 128 : i32, rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 4 + d2) * 3 + d3) * 3 + d4)> by [<Unmerge{4, 4, 3, 3} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4, 4, 3, 3] -> [144]> : tensor<144xf32> to tensor<1x4x4x3x3xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 6 + d3) * 6 + d4)> by [<Unmerge{2, 4, 6, 6} ["no", "ko", "0o", "1o"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [2, 1, 4, 6, 6] -> [288]> : tensor<288xf32> to tensor<2x1x4x6x6xf32>
  %2 = rock.conv_bwd_data(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>, strides = [1 : index, 1 : index]} : tensor<1x4x4x3x3xf32>, tensor<2x1x4x6x6xf32> -> tensor<2x1x4x8x8xf32>
  %3 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 8 + d3) * 8 + d4)> by [<Unmerge{2, 4, 8, 8} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [2, 1, 4, 8, 8] -> [512]> : tensor<512xf32> to tensor<2x1x4x8x8xf32>
  %4 = rock.store %2 to %3 by set : tensor<2x1x4x8x8xf32> -> tensor<512xf32> to tensor<2x1x4x8x8xf32>
  return %4 : tensor<512xf32>
}

// -----

// ============================================================================
// @bwd_data_padded
// Same shapes as @bwd_data_basic but with padding_h = padding_w = 1.
// Input grad pads from 8x8 to 10x10 before the implicit-GEMM transforms, and
// the output grad keeps its 8x8 spatial size. The conv-to-gemm pass must:
//   - emit a Pad{1, 1, 1, 1} on the input-grad layout
//   - still produce a single rock.gemm + rock.store ... by set
// ============================================================================
// CHECK-LABEL: @bwd_data_padded
// CHECK-NOT: rock.conv_bwd_data
// CHECK: <Pad{1, 1, 1, 1} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>
// CHECK: <Embed{1, 1} ["0ftilda", "0itilda"] at [3, 4] -> ["0ipad"] at [3]>, <Embed{1, 1} ["1ftilda", "1itilda"] at [5, 6] -> ["1ipad"] at [4]>
// CHECK: rock.gemm tr
// CHECK: rock.store {{.*}} by set
// CHECK-NOT: rock.gemm
// CHECK-NOT: rock.store
func.func @bwd_data_padded(%arg0: tensor<144xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> tensor<512xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.block_size = 128 : i32, rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 4 + d2) * 3 + d3) * 3 + d4)> by [<Unmerge{4, 4, 3, 3} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4, 4, 3, 3] -> [144]> : tensor<144xf32> to tensor<1x4x4x3x3xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 8 + d3) * 8 + d4)> by [<Unmerge{2, 4, 8, 8} ["no", "ko", "0o", "1o"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [2, 1, 4, 8, 8] -> [512]> : tensor<512xf32> to tensor<2x1x4x8x8xf32>
  %2 = rock.conv_bwd_data(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>, strides = [1 : index, 1 : index]} : tensor<1x4x4x3x3xf32>, tensor<2x1x4x8x8xf32> -> tensor<2x1x4x8x8xf32>
  %3 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 8 + d3) * 8 + d4)> by [<Unmerge{2, 4, 8, 8} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [2, 1, 4, 8, 8] -> [512]> : tensor<512xf32> to tensor<2x1x4x8x8xf32>
  %4 = rock.store %2 to %3 by set : tensor<2x1x4x8x8xf32> -> tensor<512xf32> to tensor<2x1x4x8x8xf32>
  return %4 : tensor<512xf32>
}

// -----

// ============================================================================
// @bwd_data_dilated
// dilation_h = dilation_w = 2. Stride is still 1, so filTilda = {1, 1} and
// only kernel ID 0 is emitted (single GEMM). The dilation surfaces as the
// first coefficient of the spatial Embed maps:
//   - filter side and input-grad side use Embed{2, 1}  (positive dilation)
//   - output-grad side uses Embed{-2, 1}               (negated stride/dot)
// ============================================================================
// CHECK-LABEL: @bwd_data_dilated
// CHECK-NOT: rock.conv_bwd_data
// CHECK: <Embed{2, 1} ["0ftilda", "0itilda"] at [3, 4] -> ["0ipad"] at [3]>, <Embed{2, 1} ["1ftilda", "1itilda"] at [5, 6] -> ["1ipad"] at [4]>
// CHECK: <Embed{-2, 1} ["0dot", "0tilda"] at [3, 4] -> ["0o"] at [3]>, <Embed{-2, 1} ["1dot", "1tilda"] at [5, 6] -> ["1o"] at [4]>
// CHECK: rock.gemm tr
// CHECK: rock.store {{.*}} by set
// CHECK-NOT: rock.gemm
// CHECK-NOT: rock.store
func.func @bwd_data_dilated(%arg0: tensor<144xf32>, %arg1: tensor<128xf32>, %arg2: tensor<512xf32> {rock.prefill = 0.000000e+00 : f32}) -> tensor<512xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.block_size = 128 : i32, rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 4 + d2) * 3 + d3) * 3 + d4)> by [<Unmerge{4, 4, 3, 3} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4, 4, 3, 3] -> [144]> : tensor<144xf32> to tensor<1x4x4x3x3xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 4 + d3) * 4 + d4)> by [<Unmerge{2, 4, 4, 4} ["no", "ko", "0o", "1o"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [2, 1, 4, 4, 4] -> [128]> : tensor<128xf32> to tensor<2x1x4x4x4xf32>
  %2 = rock.conv_bwd_data(%0, %1) {dilations = [2 : index, 2 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>, strides = [1 : index, 1 : index]} : tensor<1x4x4x3x3xf32>, tensor<2x1x4x4x4xf32> -> tensor<2x1x4x8x8xf32>
  %3 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 8 + d3) * 8 + d4)> by [<Unmerge{2, 4, 8, 8} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [2, 1, 4, 8, 8] -> [512]> : tensor<512xf32> to tensor<2x1x4x8x8xf32>
  %4 = rock.store %2 to %3 by set : tensor<2x1x4x8x8xf32> -> tensor<512xf32> to tensor<2x1x4x8x8xf32>
  return %4 : tensor<512xf32>
}

// -----

// ============================================================================
// @bwd_data_strided_multi_kernel
// stride_h = stride_w = 2 with a 3x3 filter gives filTilda = {2, 2}, so the
// pass emits one rock.gemm per (iTilda_h, iTilda_w) phase = 4 GEMMs. Each
// phase has its own filter/output slices and a distinct gemmK product:
//   - phase (0,0): dotslice 2x2 -> gemmK = 4 * 2 * 2 = 16
//   - phase (0,1)/(1,0): dotslice 2x1 / 1x2 -> gemmK = 4 * 2 = 8
//   - phase (1,1): dotslice 1x1 -> gemmK = 4
// All four phases write the same destination via `rock.store ... by set`
// (the result tensors are non-overlapping slices, no atomic add needed).
// Sibling of conv_to_gemm_bwd_data_empty_filter_slice.mlir, which exercises
// the divide-ceil bug fix for phases whose dotslice is *empty*; here we lock
// in the count for a "fully populated" stride>1 case.
//
// `rock.store` is `Pure`, so each per-phase store result must be kept alive
// in the SSA chain. The BwdData branch in `RockConvToGemmPass` routes the
// first store through the original `func.return` operand and appends the
// remaining three as additional return operands, expanding the kernel's
// result list from 1 to 4 (all aliasing the same `%arg2` output buffer).
// ============================================================================
// CHECK-LABEL: @bwd_data_strided_multi_kernel
// CHECK-SAME: -> (tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>)
// CHECK-NOT: rock.conv_bwd_data
// CHECK: rock.gemm tr
// CHECK: %[[STORE0:.*]] = rock.store {{.*}} by set
// CHECK: rock.gemm tr
// CHECK: %[[STORE1:.*]] = rock.store {{.*}} by set
// CHECK: rock.gemm tr
// CHECK: %[[STORE2:.*]] = rock.store {{.*}} by set
// CHECK: rock.gemm tr
// CHECK: %[[STORE3:.*]] = rock.store {{.*}} by set
// CHECK-NOT: rock.gemm
// CHECK-NOT: rock.store
// CHECK: return %[[STORE0]], %[[STORE1]], %[[STORE2]], %[[STORE3]]
func.func @bwd_data_strided_multi_kernel(%arg0: tensor<144xf32>, %arg1: tensor<72xf32>, %arg2: tensor<512xf32>) -> tensor<512xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.block_size = 128 : i32, rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 4 + d2) * 3 + d3) * 3 + d4)> by [<Unmerge{4, 4, 3, 3} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4, 4, 3, 3] -> [144]> : tensor<144xf32> to tensor<1x4x4x3x3xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 3 + d3) * 3 + d4)> by [<Unmerge{2, 4, 3, 3} ["no", "ko", "0o", "1o"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [2, 1, 4, 3, 3] -> [72]> : tensor<72xf32> to tensor<2x1x4x3x3xf32>
  %2 = rock.conv_bwd_data(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>, strides = [2 : index, 2 : index]} : tensor<1x4x4x3x3xf32>, tensor<2x1x4x3x3xf32> -> tensor<2x1x4x8x8xf32>
  %3 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 8 + d3) * 8 + d4)> by [<Unmerge{2, 4, 8, 8} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [2, 1, 4, 8, 8] -> [512]> : tensor<512xf32> to tensor<2x1x4x8x8xf32>
  %4 = rock.store %2 to %3 by set : tensor<2x1x4x8x8xf32> -> tensor<512xf32> to tensor<2x1x4x8x8xf32>
  return %4 : tensor<512xf32>
}

// -----

// ============================================================================
// @bwd_weight_non_atomic
// Backward weight on gfx1100 with small shapes -> kBlocks = 1, so
// isWrWAtomicKernel returns false and the pass takes the non-atomic path:
// a single rock.gemm followed by a `by set` store. Pins the conv->gemm
// transform shape (filter as gemmM/gemmN, input as gemmK/gemmN, output as
// gemmK/gemmM) the BwdWeight lowering builds.
// ============================================================================
// CHECK-LABEL: @bwd_weight_non_atomic
// CHECK-NOT: rock.conv_bwd_weight
// CHECK: <PassThrough ["gemmM"] at [1] -> ["k"] at [1]>, <Merge{4, 3, 3} ["gemmN"] at [2] -> ["c", "0", "1"] at [2, 3, 4]>
// CHECK: <Pad{0, 0, 0, 0} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>
// CHECK: <Merge{2, 6, 6} ["gemmK"]
// CHECK-SAME: <Merge{4, 3, 3} ["gemmN"]
// CHECK: <Merge{2, 6, 6} ["gemmK"]
// CHECK-SAME: <PassThrough ["gemmM"] at [2] -> ["ko"] at [2]>
// CHECK: rock.gemm tr {{.*}} : tensor<1x72x4xf32> * tensor<1x72x36xf32> -> tensor<1x4x36xf32>
// CHECK: rock.store {{.*}} by set
// CHECK-NOT: by atomic_add
func.func @bwd_weight_non_atomic(%arg0: tensor<512xf32>, %arg1: tensor<288xf32>, %arg2: tensor<144xf32>) -> tensor<144xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.block_size = 128 : i32, rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 8 + d3) * 8 + d4)> by [<Unmerge{2, 4, 8, 8} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [2, 1, 4, 8, 8] -> [512]> : tensor<512xf32> to tensor<2x1x4x8x8xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 6 + d3) * 6 + d4)> by [<Unmerge{2, 4, 6, 6} ["no", "ko", "0o", "1o"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [2, 1, 4, 6, 6] -> [288]> : tensor<288xf32> to tensor<2x1x4x6x6xf32>
  %2 = rock.conv_bwd_weight(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kBlocks = 1 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>, strides = [1 : index, 1 : index]} : tensor<2x1x4x8x8xf32>, tensor<2x1x4x6x6xf32> -> tensor<1x4x4x3x3xf32>
  %3 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 4 + d2) * 3 + d3) * 3 + d4)> by [<Unmerge{4, 4, 3, 3} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4, 4, 3, 3] -> [144]> : tensor<144xf32> to tensor<1x4x4x3x3xf32>
  %4 = rock.store %2 to %3 by set : tensor<1x4x4x3x3xf32> -> tensor<144xf32> to tensor<1x4x4x3x3xf32>
  return %4 : tensor<144xf32>
}

// -----

// ============================================================================
// @bwd_weight_atomic
// Backward weight on gfx908 with larger shapes (N=128, K=C=64, 14x14 ->
// 12x12 with 3x3 filter). isFastAtomicAddSupported(gfx908, f32) is true and
// affix-params picked kBlocks = 32, so the pass takes the
// `backwardWeightAtomicAdd` path:
//   - filter destination carries `rock.prefill = 0.000000e+00`
//   - filter gets an extra AddDim{32} ["kBlock"] dimension that becomes part
//     of gemmG (1 * 32 = 32)
//   - input/output operands split N into n0=32 / n1=4 via Unmerge{32, 4}
//   - emitted gemm has gemmG=32 and the store uses `by atomic_add`
// This is the only test that pins the kBlock split + atomic_add interaction.
// ============================================================================
// CHECK-LABEL: @bwd_weight_atomic
// CHECK-SAME: rock.prefill = 0.000000e+00 : f32
// CHECK-NOT: rock.conv_bwd_weight
// CHECK: <AddDim{32} ["kBlock"] at [1] -> [] at []>
// CHECK: <Merge{1, 32} ["gemmG"] at [0] -> ["g", "kBlock"] at [0, 1]>
// CHECK: <Unmerge{32, 4} ["n0", "n1"] at [0, 1] -> ["ni"] at [0]>
// CHECK: <Unmerge{32, 4} ["n0", "n1"] at [0, 1] -> ["no"] at [0]>
// CHECK: rock.gemm tr {{.*}} : tensor<32x576x64xf32> * tensor<32x576x576xf32> -> tensor<32x64x576xf32>
// CHECK: rock.store {{.*}} by atomic_add
// CHECK-NOT: by set
func.func @bwd_weight_atomic(%arg0: tensor<1605632xf32>, %arg1: tensor<1179648xf32>, %arg2: tensor<36864xf32> {rock.prefill = 0.000000e+00 : f32}) -> tensor<36864xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 256 : i32, rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 64 + d2) * 14 + d3) * 14 + d4)> by [<Unmerge{128, 64, 14, 14} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [128, 1, 64, 14, 14] -> [1605632]> : tensor<1605632xf32> to tensor<128x1x64x14x14xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 64 + d2) * 12 + d3) * 12 + d4)> by [<Unmerge{128, 64, 12, 12} ["no", "ko", "0o", "1o"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [128, 1, 64, 12, 12] -> [1179648]> : tensor<1179648xf32> to tensor<128x1x64x12x12xf32>
  %2 = rock.conv_bwd_weight(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kBlocks = 32 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>, strides = [1 : index, 1 : index]} : tensor<128x1x64x14x14xf32>, tensor<128x1x64x12x12xf32> -> tensor<1x64x64x3x3xf32>
  %3 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 64 + d2) * 3 + d3) * 3 + d4)> by [<Unmerge{64, 64, 3, 3} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64, 3, 3] -> [36864]> : tensor<36864xf32> to tensor<1x64x64x3x3xf32>
  %4 = rock.store %2 to %3 by set : tensor<1x64x64x3x3xf32> -> tensor<36864xf32> to tensor<1x64x64x3x3xf32>
  return %4 : tensor<36864xf32>
}

// -----

// ============================================================================
// @bwd_weight_atomic_fp16
// Backward weight on gfx942 with fp16 element type. Mirrors
// @bwd_weight_atomic but with f16 instead of f32. After removing the
// historic fp32 workspace + cast-kernel pair, fp16 BwdWeight uses a single
// kernel that atomic-adds partial sums directly into the f16 filter buffer.
// `isFastAtomicAddSupported(gfx942, f16)` is true, so the atomic path runs;
// the lowering must NOT materialize any f32 workspace tensor.
// ============================================================================
// CHECK-LABEL: @bwd_weight_atomic_fp16
// CHECK-SAME: tensor<{{[0-9]+}}xf16> {rock.prefill = 0.000000e+00 : f16}
// CHECK-NOT: tensor<{{[0-9]+}}xf32>
// CHECK-NOT: rock.conv_bwd_weight
// CHECK: <AddDim{32} ["kBlock"] at [1] -> [] at []>
// CHECK: <Merge{1, 32} ["gemmG"] at [0] -> ["g", "kBlock"] at [0, 1]>
// CHECK: rock.gemm tr {{.*}} : tensor<32x576x64xf16> * tensor<32x576x576xf16> -> tensor<32x64x576xf16>
// CHECK: rock.store {{.*}} by atomic_add
// CHECK-NOT: by set
func.func @bwd_weight_atomic_fp16(%arg0: tensor<1605632xf16>, %arg1: tensor<1179648xf16>, %arg2: tensor<36864xf16> {rock.prefill = 0.000000e+00 : f16}) -> tensor<36864xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.block_size = 256 : i32, rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 64 + d2) * 14 + d3) * 14 + d4)> by [<Unmerge{128, 64, 14, 14} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [128, 1, 64, 14, 14] -> [1605632]> : tensor<1605632xf16> to tensor<128x1x64x14x14xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 64 + d2) * 12 + d3) * 12 + d4)> by [<Unmerge{128, 64, 12, 12} ["no", "ko", "0o", "1o"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [128, 1, 64, 12, 12] -> [1179648]> : tensor<1179648xf16> to tensor<128x1x64x12x12xf16>
  %2 = rock.conv_bwd_weight(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kBlocks = 32 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>, strides = [1 : index, 1 : index]} : tensor<128x1x64x14x14xf16>, tensor<128x1x64x12x12xf16> -> tensor<1x64x64x3x3xf16>
  %3 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 64 + d2) * 3 + d3) * 3 + d4)> by [<Unmerge{64, 64, 3, 3} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64, 3, 3] -> [36864]> : tensor<36864xf16> to tensor<1x64x64x3x3xf16>
  %4 = rock.store %2 to %3 by set : tensor<1x64x64x3x3xf16> -> tensor<36864xf16> to tensor<1x64x64x3x3xf16>
  return %4 : tensor<36864xf16>
}
