// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-transforms-to-pointer-arith --split-input-file --verify-diagnostics | FileCheck %s

// Verifies transforms_to_ptr is lowered to pointer arithmetic for loads
// CHECK-LABEL: @test_transforms_to_ptr_load
// CHECK-SAME: (%[[ARG0:.*]]: tensor<32768xf16>)
//      CHECK:   %[[BASE_PTR:.*]] = rock.extract_ptr %[[ARG0]] : tensor<32768xf16> -> i32
//      CHECK:   tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
//      CHECK:   tt.expand_dims
//      CHECK:   tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
//      CHECK:   tt.expand_dims
//      CHECK:   arith.muli
//      CHECK:   arith.addi
//      CHECK:   tt.splat %[[BASE_PTR]] : i32 -> tensor<64x64xi32>
//      CHECK:   arith.addi {{.*}} : tensor<64x64xi32>
//      CHECK:   tt.splat {{.*}} : i1 -> tensor<64x64xi1>
//      CHECK:   rock.blockwise_load_ptr {{.*}} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_transforms_to_ptr_load(%arg0: tensor<32768xf16>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  // Transforms similar to GEMM lowering
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]> : tensor<32768xf16> to tensor<1x256x128xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]> : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>

  // transforms_to_ptr should be expanded to pointer arithmetic
  %pointers, %mask = rock.transforms_to_ptr %1[%c0_i32, %c0_i32, %c0_i32, %c1_i32] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  %2 = rock.blockwise_load_ptr %pointers[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

  return %2 : tensor<64x64xf16>
}

// -----

// Verifies transforms_to_ptr is lowered to pointer arithmetic for stores
// CHECK-LABEL: @test_transforms_to_ptr_store
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x64xf32>, %[[ARG1:.*]]: tensor<8192xf32>)
//      CHECK:   %[[BASE_PTR:.*]] = rock.extract_ptr %[[ARG1]] : tensor<8192xf32> -> i32
//      CHECK:   tt.make_range
//      CHECK:   tt.expand_dims
//      CHECK:   tt.make_range
//      CHECK:   tt.expand_dims
//      CHECK:   arith.muli
//      CHECK:   arith.addi
//      CHECK:   tt.splat %[[BASE_PTR]] : i32 -> tensor<64x64xi32>
//      CHECK:   arith.addi {{.*}} : tensor<64x64xi32>
//      CHECK:   tt.splat {{.*}} : i1 -> tensor<64x64xi1>
//      CHECK:   rock.blockwise_store_ptr {{.*}} by  set
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_transforms_to_ptr_store(%arg0: tensor<64x64xf32>, %arg1: tensor<8192xf32>) -> tensor<8192xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  // Output transforms similar to GEMM lowering
  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{64, 128} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 128] -> [8192]> : tensor<8192xf32> to tensor<1x64x128xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 2, 64, 64] -> [1, 64, 128]> : tensor<1x64x128xf32> to tensor<1x1x2x64x64xf32>

  // transforms_to_ptr should be expanded to pointer arithmetic
  %pointers, %mask = rock.transforms_to_ptr %1[%c0_i32, %c0_i32, %c1_i32] : tensor<1x1x2x64x64xf32> -> tensor<64x64xi32>, tensor<64x64xi1>
  %2 = rock.blockwise_store_ptr %arg0 -> %pointers(%mask) by set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<8192xf32>

  return %2 : tensor<8192xf32>
}


// -----

// Verifies extract_ptr is hoisted to function entry for function arguments
// CHECK-LABEL: @test_extract_ptr_hoisting
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8192xf16>, %[[ARG1:.*]]: tensor<8192xf16>)
//      CHECK:   %[[PTR1:.*]] = rock.extract_ptr %[[ARG1]] : tensor<8192xf16> -> i32
//      CHECK:   %[[PTR0:.*]] = rock.extract_ptr %[[ARG0]] : tensor<8192xf16> -> i32
//      CHECK:   tt.make_range
//      CHECK:   tt.splat %[[PTR0]]
//      CHECK:   rock.blockwise_load_ptr
//      CHECK:   tt.splat %[[PTR1]]
//      CHECK:   rock.blockwise_load_ptr
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_extract_ptr_hoisting(%arg0: tensor<8192xf16>, %arg1: tensor<8192xf16>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %c0_i32 = arith.constant 0 : i32

  // First buffer transforms
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 128 + d1)> by [<Unmerge{64, 128} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 128] -> [8192]> : tensor<8192xf16> to tensor<64x128xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d2, d1 * 64 + d3)> by [<Unmerge{1, 64} ["m_block", "m_iter"] at [0, 2] -> ["m"] at [0]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [1, 3] -> ["n"] at [1]>] bounds = [1, 2, 64, 64] -> [64, 128]> : tensor<64x128xf16> to tensor<1x2x64x64xf16>
  %pointers0, %mask0 = rock.transforms_to_ptr %1[%c0_i32, %c0_i32] : tensor<1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  %2 = rock.blockwise_load_ptr %pointers0[%mask0] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

  // Second buffer transforms
  %3 = rock.transform %arg1 by <affine_map<(d0, d1) -> (d0 * 128 + d1)> by [<Unmerge{64, 128} ["k", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 128] -> [8192]> : tensor<8192xf16> to tensor<64x128xf16>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d2, d1 * 64 + d3)> by [<Unmerge{1, 64} ["k_block", "k_iter"] at [0, 2] -> ["k"] at [0]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [1, 3] -> ["n"] at [1]>] bounds = [1, 2, 64, 64] -> [64, 128]> : tensor<64x128xf16> to tensor<1x2x64x64xf16>
  %pointers1, %mask1 = rock.transforms_to_ptr %4[%c0_i32, %c0_i32] : tensor<1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  %5 = rock.blockwise_load_ptr %pointers1[%mask1] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

  // Return first load result
  return %2 : tensor<64x64xf16>
}

// -----

// Verifies Pad transforms produce non-trivial validity masks with bounds checks
// CHECK-LABEL: @test_pad_mask
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4032xf16>)
//      CHECK:   %[[BASE_PTR:.*]] = rock.extract_ptr %[[ARG0]] : tensor<4032xf16> -> i32
//      CHECK:   tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
//      CHECK:   tt.expand_dims {{.*}} : tensor<64xi32> -> tensor<64x1xi32>
//      CHECK:   tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
//      CHECK:   tt.expand_dims {{.*}} : tensor<64xi32> -> tensor<1x64xi32>
//      CHECK:   %[[BOUND:.*]] = arith.constant 63 : i32
//      CHECK:   arith.cmpi ult, {{.*}}, {{.*}} : tensor<64x1xi32>
//      CHECK:   arith.andi {{.*}} : tensor<64x1xi1>
//      CHECK:   tt.splat %[[BASE_PTR]] : i32 -> tensor<64x64xi32>
//      CHECK:   arith.addi {{.*}} : tensor<64x64xi32>
//      CHECK:   tt.broadcast {{.*}} : tensor<64x1xi1> -> tensor<64x64xi1>
//      CHECK:   rock.blockwise_load_ptr {{.*}} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_pad_mask(%arg0: tensor<4032xf16>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // Unmerge into 63x64, then Pad the first dim by 1 on the right.
  // The view is 64x64 but the last row is out-of-bounds.
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{63, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [63, 64] -> [4032]> : tensor<4032xf16> to tensor<63x64xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1) -> (d0, d1)> by [<Pad{0, 1} ["m_padded"] at [0] -> ["m"] at [0]>, <PassThrough ["n"] at [1] -> ["n"] at [1]>] bounds = [64, 64] -> [63, 64]> : tensor<63x64xf16> to tensor<64x64xf16>

  %pointers, %mask = rock.transforms_to_ptr %1 : tensor<64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  %2 = rock.blockwise_load_ptr %pointers[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

  return %2 : tensor<64x64xf16>
}

// -----

// Verifies constant buffer root uses base pointer of 0 (no extract_ptr).
// This mirrors attention where a fakeTensor is used to compute a mask
// for out-of-bounds padding, not to actually load memory.
// CHECK-LABEL: @test_constant_buffer
//  CHECK-NOT:   rock.extract_ptr
//      CHECK:   tt.make_range
//      CHECK:   tt.splat %{{.*}} : i32 -> tensor<64x64xi32>
//      CHECK:   arith.addi {{.*}} : tensor<64x64xi32>
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_constant_buffer(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %cst = arith.constant dense<0.0> : tensor<1xf16>

  %0 = rock.transform %cst by <affine_map<(d0, d1) -> (0)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [1]> : tensor<1xf16> to tensor<64x64xf16>

  %pointers, %mask = rock.transforms_to_ptr %0 : tensor<64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  %1 = arith.select %mask, %arg0, %arg1 : tensor<64x64xi1>, tensor<64x64xf16>

  return %1 : tensor<64x64xf16>
}

// -----

// Verifies no extra indices case (source rank = output rank)
// CHECK-LABEL: @test_no_extra_indices
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4096xf16>)
//      CHECK:   %[[BASE_PTR:.*]] = rock.extract_ptr %[[ARG0]] : tensor<4096xf16> -> i32
//      CHECK:   tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
//      CHECK:   tt.expand_dims
//      CHECK:   tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
//      CHECK:   tt.expand_dims
//      CHECK:   arith.muli
//      CHECK:   arith.addi
//      CHECK:   tt.splat %[[BASE_PTR]] : i32 -> tensor<64x64xi32>
//      CHECK:   arith.addi {{.*}} : tensor<64x64xi32>
//      CHECK:   tt.splat {{.*}} : i1 -> tensor<64x64xi1>
//      CHECK:   rock.blockwise_load_ptr {{.*}} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_no_extra_indices(%arg0: tensor<4096xf16>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf16> to tensor<64x64xf16>

  %pointers, %mask = rock.transforms_to_ptr %0 : tensor<64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  %1 = rock.blockwise_load_ptr %pointers[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

  return %1 : tensor<64x64xf16>
}

// -----

// Verifies non-square tile shapes produce correct make_range bounds
// CHECK-LABEL: @test_nonsquare_tile
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4096xf16>)
//      CHECK:   tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
//      CHECK:   tt.expand_dims {{.*}} {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
//      CHECK:   tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
//      CHECK:   tt.expand_dims {{.*}} {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
//      CHECK:   tt.broadcast {{.*}} : tensor<32x1xi32> -> tensor<32x128xi32>
//      CHECK:   tt.broadcast {{.*}} : tensor<1x128xi32> -> tensor<32x128xi32>
//      CHECK:   rock.blockwise_load_ptr {{.*}} : tensor<32x128xi32>, tensor<32x128xi1> -> tensor<32x128xf16>
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_nonsquare_tile(%arg0: tensor<4096xf16>) -> tensor<32x128xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %c0_i32 = arith.constant 0 : i32

  // Simple transform with AddDim to create extra dimensions for indices
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (d2 * 128 + d3)> by [<Unmerge{32, 128} ["m_iter", "n_iter"] at [2, 3] -> ["raw"] at [0]>, <AddDim{1} ["m_block"] at [0] -> [] at []>, <AddDim{1} ["n_block"] at [1] -> [] at []>] bounds = [1, 1, 32, 128] -> [4096]> : tensor<4096xf16> to tensor<1x1x32x128xf16>

  %pointers, %mask = rock.transforms_to_ptr %0[%c0_i32, %c0_i32] : tensor<1x1x32x128xf16> -> tensor<32x128xi32>, tensor<32x128xi1>
  %1 = rock.blockwise_load_ptr %pointers[%mask] : tensor<32x128xi32>, tensor<32x128xi1> -> tensor<32x128xf16>

  return %1 : tensor<32x128xf16>
}

// -----

// Verifies Pad + Embed (convolution-style sliding window) produces
// bounds checks from Embed and floordiv/mod from Merge
// CHECK-LABEL: @test_embed_conv_style
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2048xf16>)
//      CHECK:   rock.extract_ptr %[[ARG0]]
//      CHECK:   tt.make_range {end = 4 : i32, start = 0 : i32}
//      CHECK:   tt.expand_dims
//      CHECK:   tt.make_range {end = 4 : i32, start = 0 : i32}
//      CHECK:   tt.expand_dims
//      CHECK:   arith.divsi
//      CHECK:   arith.remsi
//      CHECK:   arith.cmpi ult, {{.*}} : tensor<4x4xi32>
//      CHECK:   arith.andi {{.*}} : tensor<4x4xi1>
//      CHECK:   rock.blockwise_load_ptr
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_embed_conv_style(%arg0: tensor<2048xf16>) -> tensor<4x4xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %c0_i32 = arith.constant 0 : i32

  // 1. Unmerge raw into c, h, w
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0 * 256 + d1 * 8 + d2)> by [<Unmerge{8, 32, 8} ["c", "h", "w"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [8, 32, 8] -> [2048]> : tensor<2048xf16> to tensor<8x32x8xf16>

  // 2. Pad h by 1 on each side
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["c"] at [0] -> ["c"] at [0]>, <Pad{1, 1} ["h_pad"] at [1] -> ["h"] at [1]>, <PassThrough ["w"] at [2] -> ["w"] at [2]>] bounds = [8, 34, 8] -> [8, 32, 8]> : tensor<8x32x8xf16> to tensor<8x34x8xf16>

  // 3. Embed for 3-tap sliding window (stride 1)
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 + d2, d3)> by [<PassThrough ["c"] at [0] -> ["c"] at [0]>, <Embed{1, 1} ["fh", "oh"] at [1, 2] -> ["h_pad"] at [1]>, <PassThrough ["w"] at [3] -> ["w"] at [3]>] bounds = [8, 3, 32, 8] -> [8, 34, 8]> : tensor<8x34x8xf16> to tensor<8x3x32x8xf16>

  // 4. Merge c and fh into k
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0 floordiv 3, d0 mod 3, d1, d2)> by [<Merge{8, 3} ["k"] at [0] -> ["c", "fh"] at [0, 1]>, <PassThrough ["oh"] at [1] -> ["oh"] at [2]>, <PassThrough ["w"] at [2] -> ["w"] at [3]>] bounds = [24, 32, 8] -> [8, 3, 32, 8]> : tensor<8x3x32x8xf16> to tensor<24x32x8xf16>

  // 5. Tile with extra indices
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 4 + d3, d1 * 4 + d4, d2)> by [<Unmerge{6, 4} ["k_loop", "k_iter"] at [0, 3] -> ["k"] at [0]>, <Unmerge{8, 4} ["oh_block", "oh_iter"] at [1, 4] -> ["oh"] at [1]>, <AddDim{1} ["w_block"] at [2] -> [] at []>] bounds = [6, 8, 1, 4, 4] -> [24, 32, 8]> : tensor<24x32x8xf16> to tensor<6x8x1x4x4xf16>

  %pointers, %mask = rock.transforms_to_ptr %4[%c0_i32, %c0_i32, %c0_i32] : tensor<6x8x1x4x4xf16> -> tensor<4x4xi32>, tensor<4x4xi1>
  %5 = rock.blockwise_load_ptr %pointers[%mask] : tensor<4x4xi32>, tensor<4x4xi1> -> tensor<4x4xf16>

  return %5 : tensor<4x4xf16>
}
