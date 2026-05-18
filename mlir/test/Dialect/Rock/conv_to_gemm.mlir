// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --rock-conv-to-gemm --mlir-print-local-scope --split-input-file | FileCheck %s

// CHECK-LABEL: @nhwc_1x1
// CHECK: <AddDim{1} ["0"] at [1] -> [] at []>, <PassThrough ["0o"] at [2] -> ["0ipad"] at [1]>, <AddDim{1} ["1"] at [3] -> [] at []>, <PassThrough ["1o"] at [4] -> ["1ipad"] at [2]>
// CHECK-NOT: Embed
// CHECK: rock.gemm
func.func @nhwc_1x1(%arg0: tensor<16384xf16>, %arg1: tensor<802816xf16>, %arg2: tensor<3211264xf16>) -> tensor<64x14x14x1x256xf16> attributes {rock.block_size = 128 : i32, rock.kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : tensor<16384xf16> to tensor<1x256x1x1x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf16> to tensor<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : tensor<3211264xf16> to tensor<64x14x14x1x256xf16>
  %result = rock.conv(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<kPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, strides = [1 : index, 1 : index]} : tensor<1x256x1x1x64xf16>, tensor<64x14x14x1x64xf16> -> tensor<64x14x14x1x256xf16>
  %out = rock.store %result to %2 by set : tensor<64x14x14x1x256xf16> -> tensor<64x14x14x1x256xf16> to tensor<64x14x14x1x256xf16>
  return %out : tensor<64x14x14x1x256xf16>
}

// CHECK-LABEL: @nhwc_1x1_stride_2
// CHECK: <AddDim{1} ["0"] at [1] -> [] at []>, <Embed{2} ["0o"] at [2] -> ["0ipad"] at [1]>, <AddDim{1} ["1"] at [3] -> [] at []>, <Embed{2} ["1o"] at [4] -> ["1ipad"] at [2]>
// CHECK: rock.gemm
func.func @nhwc_1x1_stride_2(%arg0: tensor<16384xf16>, %arg1: tensor<802816xf16>, %arg2: tensor<802816xf16>) -> tensor<64x7x7x1x256xf16> attributes {rock.block_size = 128 : i32, rock.kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : tensor<16384xf16> to tensor<1x256x1x1x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf16> to tensor<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 7 + d1) * 7 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 7, 7, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 7, 7, 1, 256] -> [802816]> : tensor<802816xf16> to tensor<64x7x7x1x256xf16>
  %result = rock.conv(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<kPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, strides = [2 : index, 2 : index]} : tensor<1x256x1x1x64xf16>, tensor<64x14x14x1x64xf16> -> tensor<64x7x7x1x256xf16>
  %out = rock.store %result to %2 by set : tensor<64x7x7x1x256xf16> -> tensor<64x7x7x1x256xf16> to tensor<64x7x7x1x256xf16>
  return %out : tensor<64x7x7x1x256xf16>
}

// CHECK-LABEL: @nhwc_3x3
// CHECK: <Embed{1, 1} ["0", "0o"] at [1, 2] -> ["0ipad"] at [1]>, <Embed{1, 1} ["1", "1o"] at [3, 4] -> ["1ipad"] at [2]>
// CHECK: rock.gemm
func.func @nhwc_3x3(%arg0: tensor<147456xf16>, %arg1: tensor<802816xf16>, %arg2: tensor<2359296xf16>) -> tensor<64x12x12x1x256xf16> attributes {rock.block_size = 128 : i32, rock.kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((((d0 * 256 + d1) * 3 + d2) * 3 + d3) * 64 + d4)> by [<Unmerge{1, 256, 3, 3, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 3, 3, 64] -> [147456]> : tensor<147456xf16> to tensor<1x256x3x3x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf16> to tensor<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 12 + d1) * 12 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 12, 12, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 12, 12, 1, 256] -> [2359296]> : tensor<2359296xf16> to tensor<64x12x12x1x256xf16>
  %result = rock.conv(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<kPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, strides = [1 : index, 1 : index]} : tensor<1x256x3x3x64xf16>, tensor<64x14x14x1x64xf16> -> tensor<64x12x12x1x256xf16>
  %out = rock.store %result to %2 by set : tensor<64x12x12x1x256xf16> -> tensor<64x12x12x1x256xf16> to tensor<64x12x12x1x256xf16>
  return %out : tensor<64x12x12x1x256xf16>
}

// Lock in the `Pad{1, 1, 1, 1}` transform produced when the conv has non-zero
// spatial padding. Output spatial = 14 + 2*1 - 3 + 1 = 14, so the output buffer
// keeps shape 64x14x14x1x256 (same as @nhwc_3x3 but with padding).
// CHECK-LABEL: @nhwc_3x3_padded
// CHECK: <Pad{1, 1, 1, 1} ["0ipad", "1ipad"] at [1, 2] -> ["0i", "1i"] at [1, 2]>
// CHECK: <Embed{1, 1} ["0", "0o"] at [1, 2] -> ["0ipad"] at [1]>, <Embed{1, 1} ["1", "1o"] at [3, 4] -> ["1ipad"] at [2]>
// CHECK: rock.gemm
func.func @nhwc_3x3_padded(%arg0: tensor<147456xf16>, %arg1: tensor<802816xf16>, %arg2: tensor<3211264xf16>) -> tensor<64x14x14x1x256xf16> attributes {rock.block_size = 128 : i32, rock.kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((((d0 * 256 + d1) * 3 + d2) * 3 + d3) * 64 + d4)> by [<Unmerge{1, 256, 3, 3, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 3, 3, 64] -> [147456]> : tensor<147456xf16> to tensor<1x256x3x3x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf16> to tensor<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : tensor<3211264xf16> to tensor<64x14x14x1x256xf16>
  %result = rock.conv(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], output_layout = ["no", "0o", "1o", "go", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], params = #rock.gemm_params<kPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, strides = [1 : index, 1 : index]} : tensor<1x256x3x3x64xf16>, tensor<64x14x14x1x64xf16> -> tensor<64x14x14x1x256xf16>
  %out = rock.store %result to %2 by set : tensor<64x14x14x1x256xf16> -> tensor<64x14x14x1x256xf16> to tensor<64x14x14x1x256xf16>
  return %out : tensor<64x14x14x1x256xf16>
}

// Lock in the dilated `Embed{2, 2}` transform produced when dilation > 1.
// Effective receptive field is (3-1)*2 + 1 = 5, so output spatial = 14 - 5 + 1
// = 10 and the output buffer is 64x10x10x1x256 (1638400 f16 elements).
// CHECK-LABEL: @nhwc_3x3_dilated
// CHECK: <Embed{2, 1} ["0", "0o"] at [1, 2] -> ["0ipad"] at [1]>, <Embed{2, 1} ["1", "1o"] at [3, 4] -> ["1ipad"] at [2]>
// CHECK: rock.gemm
func.func @nhwc_3x3_dilated(%arg0: tensor<147456xf16>, %arg1: tensor<802816xf16>, %arg2: tensor<1638400xf16>) -> tensor<64x10x10x1x256xf16> attributes {rock.block_size = 128 : i32, rock.kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((((d0 * 256 + d1) * 3 + d2) * 3 + d3) * 64 + d4)> by [<Unmerge{1, 256, 3, 3, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 3, 3, 64] -> [147456]> : tensor<147456xf16> to tensor<1x256x3x3x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf16> to tensor<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 10 + d1) * 10 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 10, 10, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 10, 10, 1, 256] -> [1638400]> : tensor<1638400xf16> to tensor<64x10x10x1x256xf16>
  %result = rock.conv(%0, %1) {dilations = [2 : index, 2 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<kPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, strides = [1 : index, 1 : index]} : tensor<1x256x3x3x64xf16>, tensor<64x14x14x1x64xf16> -> tensor<64x10x10x1x256xf16>
  %out = rock.store %result to %2 by set : tensor<64x10x10x1x256xf16> -> tensor<64x10x10x1x256xf16> to tensor<64x10x10x1x256xf16>
  return %out : tensor<64x10x10x1x256xf16>
}

// Lock in multi-group convolution (G=2). Filter is [2, 128, 1, 1, 32]
// (per-group K=128, C=32). Total elements: 2*128*1*1*32=8192 filter,
// 64*14*14*2*32=802816 input, 64*14*14*2*128=3211264 output.
// CHECK-LABEL: @grouped_nhwc_1x1
// CHECK: rock.gemm
// CHECK-SAME: tensor<2x{{[0-9]+}}x{{[0-9]+}}xf16> * tensor<2x{{[0-9]+}}x{{[0-9]+}}xf16>
func.func @grouped_nhwc_1x1(%arg0: tensor<8192xf16>, %arg1: tensor<802816xf16>, %arg2: tensor<3211264xf16>) -> tensor<64x14x14x2x128xf16> attributes {rock.block_size = 128 : i32, rock.kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 128 + d1 + d2 + d3) * 32 + d4)> by [<Unmerge{2, 128, 1, 1, 32} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [2, 128, 1, 1, 32] -> [8192]> : tensor<8192xf16> to tensor<2x128x1x1x32xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> ((((d0 * 14 + d1) * 14 + d2) * 2 + d3) * 32 + d4)> by [<Unmerge{64, 14, 14, 2, 32} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 2, 32] -> [802816]> : tensor<802816xf16> to tensor<64x14x14x2x32xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> ((((d0 * 14 + d1) * 14 + d2) * 2 + d3) * 128 + d4)> by [<Unmerge{64, 14, 14, 2, 128} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 2, 128] -> [3211264]> : tensor<3211264xf16> to tensor<64x14x14x2x128xf16>
  %result = rock.conv(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<kPerBlock = 4, mPerBlock = 128, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, strides = [1 : index, 1 : index]} : tensor<2x128x1x1x32xf16>, tensor<64x14x14x2x32xf16> -> tensor<64x14x14x2x128xf16>
  %out = rock.store %result to %2 by set : tensor<64x14x14x2x128xf16> -> tensor<64x14x14x2x128xf16> to tensor<64x14x14x2x128xf16>
  return %out : tensor<64x14x14x2x128xf16>
}

// Verify that conv-to-gemm handles type-changing output fusions.
// The conv produces f32 but the store destination is i32 (through arith.fptoui).
// The gemm must produce f32 (matching conv inputs), not i32.
// CHECK-LABEL: @nhwc_1x1_fptoui_fusion
// CHECK: rock.gemm
// CHECK-SAME: -> tensor<{{[0-9x]+}}xf32>
// CHECK: arith.fptoui {{.*}} : tensor<{{[0-9x]+}}xf32> to tensor<{{[0-9x]+}}xi32>
// CHECK: rock.store
func.func @nhwc_1x1_fptoui_fusion(%arg0: tensor<16384xf32>, %arg1: tensor<802816xf32>, %arg2: tensor<3211264xi32>) -> tensor<64x14x14x1x256xi32> attributes {rock.block_size = 128 : i32, rock.kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : tensor<16384xf32> to tensor<1x256x1x1x64xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf32> to tensor<64x14x14x1x64xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : tensor<3211264xi32> to tensor<64x14x14x1x256xi32>
  %result = rock.conv(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<kPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, strides = [1 : index, 1 : index]} : tensor<1x256x1x1x64xf32>, tensor<64x14x14x1x64xf32> -> tensor<64x14x14x1x256xf32>
  %cast = arith.fptoui %result : tensor<64x14x14x1x256xf32> to tensor<64x14x14x1x256xi32>
  %out = rock.store %cast to %2 by set : tensor<64x14x14x1x256xi32> -> tensor<64x14x14x1x256xi32> to tensor<64x14x14x1x256xi32>
  return %out : tensor<64x14x14x1x256xi32>
}

// Sibling of @nhwc_1x1_fptoui_fusion: same shape-from-dest / eltype-from-conv
// behavior in getResultType + updateStoreOpForGemm, exercised through a
// different output cast (signed). The conv produces f32 but the destination is
// i32 via arith.fptosi; the gemm must still produce f32.
// CHECK-LABEL: @nhwc_1x1_fptosi_fusion
// CHECK: rock.gemm
// CHECK-SAME: -> tensor<{{[0-9x]+}}xf32>
// CHECK: arith.fptosi {{.*}} : tensor<{{[0-9x]+}}xf32> to tensor<{{[0-9x]+}}xi32>
// CHECK: rock.store
func.func @nhwc_1x1_fptosi_fusion(%arg0: tensor<16384xf32>, %arg1: tensor<802816xf32>, %arg2: tensor<3211264xi32>) -> tensor<64x14x14x1x256xi32> attributes {rock.block_size = 128 : i32, rock.kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : tensor<16384xf32> to tensor<1x256x1x1x64xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf32> to tensor<64x14x14x1x64xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : tensor<3211264xi32> to tensor<64x14x14x1x256xi32>
  %result = rock.conv(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<kPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, strides = [1 : index, 1 : index]} : tensor<1x256x1x1x64xf32>, tensor<64x14x14x1x64xf32> -> tensor<64x14x14x1x256xf32>
  %cast = arith.fptosi %result : tensor<64x14x14x1x256xf32> to tensor<64x14x14x1x256xi32>
  %out = rock.store %cast to %2 by set : tensor<64x14x14x1x256xi32> -> tensor<64x14x14x1x256xi32> to tensor<64x14x14x1x256xi32>
  return %out : tensor<64x14x14x1x256xi32>
}

// Output fusion with an *extra* fusion input: the canonical bias-add pattern
// `arith.addf %conv, %bias`. Exercises both helpers at the end of
// ConvRewritePattern in ConvToGemm.cpp:
//   * propagateOutputType: replaces arith.addf's conv-result operand with the
//     new gemm result and updates the addf result type from the conv's
//     5-D NHWC shape to the gemm's [G, M, N] = [1, 256, 12544].
//   * replaceFusionExtraInputs: rewires arith.addf's *bias* operand from the
//     original 5-D %bias to the layout-merged 3-D bias view that
//     commonConvRewrite produced for it (so both addf operands match the
//     gemm shape).
// CHECK-LABEL: @nhwc_1x1_bias_add
// CHECK-NOT: rock.conv
// CHECK: %[[BIAS_VIEW:.*]] = rock.transform %{{.*}} by {{.*}}Merge{64, 14, 14} ["gemmN"]{{.*}}: tensor<64x14x14x1x256xf16> to tensor<1x256x12544xf16>
// CHECK: %[[GEMM:.*]] = rock.gemm tr
// CHECK-SAME: -> tensor<1x256x12544xf16>
// CHECK: %[[ADD:.*]] = arith.addf %[[GEMM]], %[[BIAS_VIEW]] : tensor<1x256x12544xf16>
// CHECK: rock.store %[[ADD]] to %{{.*}} by set : tensor<1x256x12544xf16>
func.func @nhwc_1x1_bias_add(%arg0: tensor<16384xf16>, %arg1: tensor<802816xf16>, %arg2: tensor<3211264xf16>, %arg3: tensor<3211264xf16>) -> tensor<64x14x14x1x256xf16> attributes {rock.block_size = 128 : i32, rock.kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : tensor<16384xf16> to tensor<1x256x1x1x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf16> to tensor<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : tensor<3211264xf16> to tensor<64x14x14x1x256xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : tensor<3211264xf16> to tensor<64x14x14x1x256xf16>
  %result = rock.conv(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<kPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, strides = [1 : index, 1 : index]} : tensor<1x256x1x1x64xf16>, tensor<64x14x14x1x64xf16> -> tensor<64x14x14x1x256xf16>
  %added = arith.addf %result, %2 : tensor<64x14x14x1x256xf16>
  %out = rock.store %added to %3 by set : tensor<64x14x14x1x256xf16> -> tensor<64x14x14x1x256xf16> to tensor<64x14x14x1x256xf16>
  return %out : tensor<64x14x14x1x256xf16>
}

// Multi-op output fusion chain: `arith.addf` (binary, has an extra input)
// followed by `math.absf` (unary). Pins the *recursive* walk in
// propagateOutputType: it must update the result type of BOTH the addf and
// the downstream absf to the gemm shape, even though only the addf has an
// entry in fusionInputMap (absf has no extra inputs).
// CHECK-LABEL: @nhwc_1x1_bias_add_then_absf
// CHECK-NOT: rock.conv
// CHECK: %[[BIAS_VIEW:.*]] = rock.transform %{{.*}} by {{.*}}Merge{64, 14, 14} ["gemmN"]{{.*}}: tensor<64x14x14x1x256xf16> to tensor<1x256x12544xf16>
// CHECK: %[[GEMM:.*]] = rock.gemm tr
// CHECK-SAME: -> tensor<1x256x12544xf16>
// CHECK: %[[ADD:.*]] = arith.addf %[[GEMM]], %[[BIAS_VIEW]] : tensor<1x256x12544xf16>
// CHECK: %[[ABS:.*]] = math.absf %[[ADD]] : tensor<1x256x12544xf16>
// CHECK: rock.store %[[ABS]] to %{{.*}} by set : tensor<1x256x12544xf16>
func.func @nhwc_1x1_bias_add_then_absf(%arg0: tensor<16384xf16>, %arg1: tensor<802816xf16>, %arg2: tensor<3211264xf16>, %arg3: tensor<3211264xf16>) -> tensor<64x14x14x1x256xf16> attributes {rock.block_size = 128 : i32, rock.kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : tensor<16384xf16> to tensor<1x256x1x1x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf16> to tensor<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : tensor<3211264xf16> to tensor<64x14x14x1x256xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : tensor<3211264xf16> to tensor<64x14x14x1x256xf16>
  %result = rock.conv(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<kPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, strides = [1 : index, 1 : index]} : tensor<1x256x1x1x64xf16>, tensor<64x14x14x1x64xf16> -> tensor<64x14x14x1x256xf16>
  %added = arith.addf %result, %2 : tensor<64x14x14x1x256xf16>
  %absed = math.absf %added : tensor<64x14x14x1x256xf16>
  %out = rock.store %absed to %3 by set : tensor<64x14x14x1x256xf16> -> tensor<64x14x14x1x256xf16> to tensor<64x14x14x1x256xf16>
  return %out : tensor<64x14x14x1x256xf16>
}

// Element-type-changing fan-out fusion with TWO output tensors:
//   conv (f16) -> addf %conv, %bias (f16) -> { store_f16, fptoui -> store_i8 }
// Stresses several interacting paths in ConvRewritePattern at once:
//   * traceOutputsAndFusionInputs collects BOTH rock.store ops as `stores`
//     (the fusion chain forks at the addf result), so the store-rewrite loop
//     must run twice and rewire each store to the right post-propagation
//     source (addf -> f16 store, fptoui -> i8 store).
//   * getResultType picks the conv's element type (f16) for the gemm even
//     though one of the two store destinations is i8.
//   * propagateOutputType walks the fork: it updates BOTH the addf result
//     type AND the (downstream) fptoui result type to the gemm shape
//     [1, 256, 12544], preserving each fusion op's *own* element type
//     (addf stays f16, fptoui stays i8).
//   * replaceFusionExtraInputs rewires the bias operand of addf even though
//     the chain fans out below it.
// CHECK-LABEL: @nhwc_1x1_bias_add_two_outputs
// CHECK-NOT: rock.conv
// CHECK: %[[BIAS_VIEW:.*]] = rock.transform %{{.*}} by {{.*}}Merge{64, 14, 14} ["gemmN"]{{.*}}: tensor<64x14x14x1x256xf16> to tensor<1x256x12544xf16>
// CHECK: %[[GEMM:.*]] = rock.gemm tr
// CHECK-SAME: -> tensor<1x256x12544xf16>
// CHECK: %[[ADD:.*]] = arith.addf %[[GEMM]], %[[BIAS_VIEW]] : tensor<1x256x12544xf16>
// CHECK: %[[CAST:.*]] = arith.fptoui %[[ADD]] : tensor<1x256x12544xf16> to tensor<1x256x12544xi8>
// CHECK: rock.store %[[ADD]] to %{{.*}} by set : tensor<1x256x12544xf16>
// CHECK: rock.store %[[CAST]] to %{{.*}} by set : tensor<1x256x12544xi8>
func.func @nhwc_1x1_bias_add_two_outputs(%arg0: tensor<16384xf16>, %arg1: tensor<802816xf16>, %arg2: tensor<3211264xf16>, %arg3: tensor<3211264xf16>, %arg4: tensor<3211264xi8>) -> (tensor<64x14x14x1x256xf16>, tensor<64x14x14x1x256xi8>) attributes {rock.block_size = 128 : i32, rock.kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : tensor<16384xf16> to tensor<1x256x1x1x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf16> to tensor<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : tensor<3211264xf16> to tensor<64x14x14x1x256xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : tensor<3211264xf16> to tensor<64x14x14x1x256xf16>
  %4 = rock.transform %arg4 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : tensor<3211264xi8> to tensor<64x14x14x1x256xi8>
  %result = rock.conv(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.gemm_params<kPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, strides = [1 : index, 1 : index]} : tensor<1x256x1x1x64xf16>, tensor<64x14x14x1x64xf16> -> tensor<64x14x14x1x256xf16>
  %added = arith.addf %result, %2 : tensor<64x14x14x1x256xf16>
  %cast = arith.fptoui %added : tensor<64x14x14x1x256xf16> to tensor<64x14x14x1x256xi8>
  %s_f16 = rock.store %added to %3 by set : tensor<64x14x14x1x256xf16> -> tensor<64x14x14x1x256xf16> to tensor<64x14x14x1x256xf16>
  %s_i8 = rock.store %cast to %4 by set : tensor<64x14x14x1x256xi8> -> tensor<64x14x14x1x256xi8> to tensor<64x14x14x1x256xi8>
  return %s_f16, %s_i8 : tensor<64x14x14x1x256xf16>, tensor<64x14x14x1x256xi8>
}

// CHECK-LABEL: @conv_gemm_nhwc_1x1
// CHECK: <AddDim{1} ["0"] at [1] -> [] at []>, <PassThrough ["0o"] at [2] -> ["0ipad"] at [1]>, <AddDim{1} ["1"] at [3] -> [] at []>, <PassThrough ["1o"] at [4] -> ["1ipad"] at [2]>
// CHECK-NOT: Embed
// CHECK: rock.gemm_elementwise_gemm
func.func @conv_gemm_nhwc_1x1(%arg0: tensor<16384xf32>, %arg1: tensor<802816xf32>, %arg2: tensor<65536xf32>, %arg3: tensor<3211264xf32>) -> tensor<1x12544x256xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : tensor<16384xf32> to tensor<1x256x1x1x64xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf32> to tensor<64x14x14x1x64xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{256, 256} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 256] -> [65536]> : tensor<65536xf32> to tensor<1x256x256xf32>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{12544, 256} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 12544, 256] -> [3211264]> : tensor<3211264xf32> to tensor<1x12544x256xf32>
  %result = rock.conv_elementwise_gemm{
   ab = conv(%0, %1) : tensor<1x256x1x1x64xf32>, tensor<64x14x14x1x64xf32>
   out = ab * %2 : tensor<1x256x256xf32>
  } {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} -> tensor<1x12544x256xf32>
  %out = rock.store %result to %3 by set : tensor<1x12544x256xf32> -> tensor<1x12544x256xf32> to tensor<1x12544x256xf32>
  return %out : tensor<1x12544x256xf32>
}

// CHECK-LABEL: @conv_gemm_nhwc_1x1_stride_2
// CHECK: <AddDim{1} ["0"] at [1] -> [] at []>, <Embed{2} ["0o"] at [2] -> ["0ipad"] at [1]>, <AddDim{1} ["1"] at [3] -> [] at []>, <Embed{2} ["1o"] at [4] -> ["1ipad"] at [2]>
// CHECK: rock.gemm_elementwise_gemm
func.func @conv_gemm_nhwc_1x1_stride_2(%arg0: tensor<16384xf32>, %arg1: tensor<802816xf32>, %arg2: tensor<65536xf32>, %arg3: tensor<802816xf32>) -> tensor<1x3136x256xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : tensor<16384xf32> to tensor<1x256x1x1x64xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf32> to tensor<64x14x14x1x64xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{256, 256} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 256] -> [65536]> : tensor<65536xf32> to tensor<1x256x256xf32>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{3136, 256} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 3136, 256] -> [802816]> : tensor<802816xf32> to tensor<1x3136x256xf32>
  %result = rock.conv_elementwise_gemm{
   ab = conv(%0, %1) : tensor<1x256x1x1x64xf32>, tensor<64x14x14x1x64xf32>
   out = ab * %2 : tensor<1x256x256xf32>
  } {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [2 : index, 2 : index]} -> tensor<1x3136x256xf32>
  %out = rock.store %result to %3 by set : tensor<1x3136x256xf32> -> tensor<1x3136x256xf32> to tensor<1x3136x256xf32>
  return %out : tensor<1x3136x256xf32>
}

// CHECK-LABEL: @conv_gemm_nhwc_3x3
// CHECK: <Embed{1, 1} ["0", "0o"] at [1, 2] -> ["0ipad"] at [1]>, <Embed{1, 1} ["1", "1o"] at [3, 4] -> ["1ipad"] at [2]>
// CHECK: rock.gemm_elementwise_gemm
func.func @conv_gemm_nhwc_3x3(%arg0: tensor<147456xf32>, %arg1: tensor<802816xf32>, %arg2: tensor<65536xf32>, %arg3: tensor<2359296xf32>) -> tensor<1x9216x256xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((((d0 * 256 + d1) * 3 + d2) * 3 + d3) * 64 + d4)> by [<Unmerge{1, 256, 3, 3, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 3, 3, 64] -> [147456]> : tensor<147456xf32> to tensor<1x256x3x3x64xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf32> to tensor<64x14x14x1x64xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{256, 256} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 256] -> [65536]> : tensor<65536xf32> to tensor<1x256x256xf32>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{9216, 256} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 9216, 256] -> [2359296]> : tensor<2359296xf32> to tensor<1x9216x256xf32>
  %result = rock.conv_elementwise_gemm{
   ab = conv(%0, %1) : tensor<1x256x3x3x64xf32>, tensor<64x14x14x1x64xf32>
   out = ab * %2 : tensor<1x256x256xf32>
  } {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} -> tensor<1x9216x256xf32>
  %out = rock.store %result to %3 by set : tensor<1x9216x256xf32> -> tensor<1x9216x256xf32> to tensor<1x9216x256xf32>
  return %out : tensor<1x9216x256xf32>
}

// CHECK-LABEL: @conv_gemm_nhwc_3x3_atomicadd
// CHECK: <Embed{1, 1} ["0", "0o"] at [1, 2] -> ["0ipad"] at [1]>, <Embed{1, 1} ["1", "1o"] at [3, 4] -> ["1ipad"] at [2]>
// CHECK: rock.gemm_elementwise_gemm
// CHECK: rock.store {{.*}} by atomic_add
func.func @conv_gemm_nhwc_3x3_atomicadd(%arg0: tensor<147456xf32>, %arg1: tensor<802816xf32>, %arg2: tensor<65536xf32>, %arg3: tensor<2359296xf32>) -> tensor<1x9216x256xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((((d0 * 256 + d1) * 3 + d2) * 3 + d3) * 64 + d4)> by [<Unmerge{1, 256, 3, 3, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 3, 3, 64] -> [147456]> : tensor<147456xf32> to tensor<1x256x3x3x64xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf32> to tensor<64x14x14x1x64xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{256, 256} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 256] -> [65536]> : tensor<65536xf32> to tensor<1x256x256xf32>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{9216, 256} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 9216, 256] -> [2359296]> : tensor<2359296xf32> to tensor<1x9216x256xf32>
  %result = rock.conv_elementwise_gemm{
   ab = conv(%0, %1) : tensor<1x256x3x3x64xf32>, tensor<64x14x14x1x64xf32>
   out = ab * %2 : tensor<1x256x256xf32>
  } {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} -> tensor<1x9216x256xf32>
  %out = rock.store %result to %3 by atomic_add : tensor<1x9216x256xf32> -> tensor<1x9216x256xf32> to tensor<1x9216x256xf32>
  return %out : tensor<1x9216x256xf32>
}

// Exercise the optional pre-second-GEMM region of `rock.conv_elementwise_gemm`.
// The conv-to-gemm pass must inline the region into the resulting
// `rock.gemm_elementwise_gemm` (it uses inlineRegionBefore).
// CHECK-LABEL: @conv_gemm_nhwc_1x1_elemwise
// CHECK: rock.gemm_elementwise_gemm
// CHECK: ab = elementwise
// CHECK: arith.negf
// CHECK: rock.yield
func.func @conv_gemm_nhwc_1x1_elemwise(%arg0: tensor<16384xf32>, %arg1: tensor<802816xf32>, %arg2: tensor<65536xf32>, %arg3: tensor<3211264xf32>) -> tensor<1x12544x256xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : tensor<16384xf32> to tensor<1x256x1x1x64xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf32> to tensor<64x14x14x1x64xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{256, 256} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 256] -> [65536]> : tensor<65536xf32> to tensor<1x256x256xf32>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{12544, 256} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 12544, 256] -> [3211264]> : tensor<3211264xf32> to tensor<1x12544x256xf32>
  %result = rock.conv_elementwise_gemm{
   ab = conv(%0, %1) : tensor<1x256x1x1x64xf32>, tensor<64x14x14x1x64xf32>
   ab = elementwise {
   ^bb0(%arg4: tensor<1x256x12544xf32>):
     %neg = arith.negf %arg4 : tensor<1x256x12544xf32>
     rock.yield %neg : tensor<1x256x12544xf32>
   }
   out = ab * %2 : tensor<1x256x256xf32>
  } {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} -> tensor<1x12544x256xf32>
  %out = rock.store %result to %3 by set : tensor<1x12544x256xf32> -> tensor<1x12544x256xf32> to tensor<1x12544x256xf32>
  return %out : tensor<1x12544x256xf32>
}

// Exercise the `cTransposed` unit attribute on `rock.conv_elementwise_gemm`,
// i.e. `c` is `[G, O, N]` instead of `[G, N, O]`. The resulting
// `rock.gemm_elementwise_gemm` must propagate `cTransposed` (printed as
// `* tr %c`). Here O=128, N=256, so c has shape [1, 128, 256] and the result
// has shape [1, 12544, 128] (1605632 f32 elements).
// CHECK-LABEL: @conv_gemm_nhwc_1x1_cTransposed
// CHECK: rock.gemm_elementwise_gemm
// CHECK: out = ab * tr
func.func @conv_gemm_nhwc_1x1_cTransposed(%arg0: tensor<16384xf32>, %arg1: tensor<802816xf32>, %arg2: tensor<32768xf32>, %arg3: tensor<1605632xf32>) -> tensor<1x12544x128xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : tensor<16384xf32> to tensor<1x256x1x1x64xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf32> to tensor<64x14x14x1x64xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{128, 256} ["o", "gemmN"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 256] -> [32768]> : tensor<32768xf32> to tensor<1x128x256xf32>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{12544, 128} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 12544, 128] -> [1605632]> : tensor<1605632xf32> to tensor<1x12544x128xf32>
  %result = rock.conv_elementwise_gemm{
   ab = conv(%0, %1) : tensor<1x256x1x1x64xf32>, tensor<64x14x14x1x64xf32>
   out = ab * tr %2 : tensor<1x128x256xf32>
  } {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} -> tensor<1x12544x128xf32>
  %out = rock.store %result to %3 by set : tensor<1x12544x128xf32> -> tensor<1x12544x128xf32> to tensor<1x12544x128xf32>
  return %out : tensor<1x12544x128xf32>
}
