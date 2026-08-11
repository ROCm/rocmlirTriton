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
//      CHECK:   tt.splat {{.*}} : i1 -> tensor<64x64xi1>
//      CHECK:   tt.splat %[[BASE_PTR]] : i32 -> tensor<64x64xi32>
//      CHECK:   arith.addi {{.*}} : tensor<64x64xi32>
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
  %2 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

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
//      CHECK:   tt.splat {{.*}} : i1 -> tensor<64x64xi1>
//      CHECK:   tt.splat %[[BASE_PTR]] : i32 -> tensor<64x64xi32>
//      CHECK:   arith.addi {{.*}} : tensor<64x64xi32>
//      CHECK:   rock.blockwise_store_ptr {{.*}} by  set
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_transforms_to_ptr_store(%arg0: tensor<64x64xf32>, %arg1: tensor<8192xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  // Output transforms similar to GEMM lowering
  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{64, 128} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 128] -> [8192]> : tensor<8192xf32> to tensor<1x64x128xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 2, 64, 64] -> [1, 64, 128]> : tensor<1x64x128xf32> to tensor<1x1x2x64x64xf32>

  // transforms_to_ptr should be expanded to pointer arithmetic
  %pointers, %mask = rock.transforms_to_ptr %1[%c0_i32, %c0_i32, %c1_i32] : tensor<1x1x2x64x64xf32> -> tensor<64x64xi32>, tensor<64x64xi1>
  rock.blockwise_store_ptr %arg0 -> %pointers(%mask) by set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>)

  return
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
  %2 = rock.blockwise_load_ptr %pointers0[%mask0] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

  // Second buffer transforms
  %3 = rock.transform %arg1 by <affine_map<(d0, d1) -> (d0 * 128 + d1)> by [<Unmerge{64, 128} ["k", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 128] -> [8192]> : tensor<8192xf16> to tensor<64x128xf16>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d2, d1 * 64 + d3)> by [<Unmerge{1, 64} ["k_block", "k_iter"] at [0, 2] -> ["k"] at [0]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [1, 3] -> ["n"] at [1]>] bounds = [1, 2, 64, 64] -> [64, 128]> : tensor<64x128xf16> to tensor<1x2x64x64xf16>
  %pointers1, %mask1 = rock.transforms_to_ptr %4[%c0_i32, %c0_i32] : tensor<1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  %5 = rock.blockwise_load_ptr %pointers1[%mask1] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

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
//      CHECK:   tt.broadcast {{.*}} : tensor<64x1xi1> -> tensor<64x64xi1>
//      CHECK:   tt.splat %[[BASE_PTR]] : i32 -> tensor<64x64xi32>
//      CHECK:   arith.addi {{.*}} : tensor<64x64xi32>
//      CHECK:   rock.blockwise_load_ptr {{.*}} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_pad_mask(%arg0: tensor<4032xf16>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  // Unmerge into 63x64, then Pad the first dim by 1 on the right.
  // The view is 64x64 but the last row is out-of-bounds.
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{63, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [63, 64] -> [4032]> : tensor<4032xf16> to tensor<63x64xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1) -> (d0, d1)> by [<Pad{0, 1} ["m_padded"] at [0] -> ["m"] at [0]>, <PassThrough ["n"] at [1] -> ["n"] at [1]>] bounds = [64, 64] -> [63, 64]> : tensor<63x64xf16> to tensor<64x64xf16>

  %pointers, %mask = rock.transforms_to_ptr %1 : tensor<64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  %2 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

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
  %cst = arith.constant dense<0.0> : tensor<4096xf16>

  %0 = rock.transform %cst by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf16> to tensor<64x64xf16>

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
//      CHECK:   tt.splat {{.*}} : i1 -> tensor<64x64xi1>
//      CHECK:   tt.splat %[[BASE_PTR]] : i32 -> tensor<64x64xi32>
//      CHECK:   arith.addi {{.*}} : tensor<64x64xi32>
//      CHECK:   rock.blockwise_load_ptr {{.*}} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_no_extra_indices(%arg0: tensor<4096xf16>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf16> to tensor<64x64xf16>

  %pointers, %mask = rock.transforms_to_ptr %0 : tensor<64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  %1 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

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
  %1 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x128xi32>, tensor<32x128xi1> -> tensor<32x128xf16>

  return %1 : tensor<32x128xf16>
}

// -----

// Verifies conv-style input transforms with padding=1 (same-size output).
// Pad{1,1} produces bounds checks; Embed + Merge produce divui/remui (the
// affine mod/floordiv lowering is unsigned, see visitModExpr).
// CHECK-LABEL: @test_embed_conv_style
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1048576xf32>)
//      CHECK:   rock.extract_ptr %[[ARG0]]
//      CHECK:   tt.make_range
//      CHECK:   tt.expand_dims
//      CHECK:   tt.make_range
//      CHECK:   tt.expand_dims
//      CHECK:   arith.divui
//      CHECK:   arith.remui
//      CHECK:   arith.cmpi
//      CHECK:   arith.andi
//      CHECK:   rock.blockwise_load_ptr
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_embed_conv_style(%arg0: tensor<1048576xf32>) -> tensor<8x64xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %c0_i32 = arith.constant 0 : i32

  // 1. Unmerge raw into ni, ci, 0i, 1i with AddDim for gi
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 8 + d2) * 32 + d3) * 32 + d4)> by [<Unmerge{128, 8, 32, 32} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [128, 1, 8, 32, 32] -> [1048576]> : tensor<1048576xf32> to tensor<128x1x8x32x32xf32>

  // 2. Pad spatial dims by 1 on each side (same-size conv)
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 1, d4 - 1)> by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, <PassThrough ["gi"] at [1] -> ["gi"] at [1]>, <PassThrough ["ci"] at [2] -> ["ci"] at [2]>, <Pad{1, 1, 1, 1} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>] bounds = [128, 1, 8, 34, 34] -> [128, 1, 8, 32, 32]> : tensor<128x1x8x32x32xf32> to tensor<128x1x8x34x34xf32>

  // 3. Embed for 3x3 sliding window (stride 1), output spatial = 32x32
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 1} ["0", "0o"] at [3, 4] -> ["0ipad"] at [3]>, <Embed{1, 1} ["1", "1o"] at [5, 6] -> ["1ipad"] at [4]>] bounds = [128, 1, 8, 3, 32, 3, 32] -> [128, 1, 8, 34, 34]> : tensor<128x1x8x34x34xf32> to tensor<128x1x8x3x32x3x32xf32>

  // 4. Merge into gemmG, gemmK, gemmN (gemmN = 128*32*32 = 131072)
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d2 floordiv 1024, d0, d1 floordiv 9, (d1 mod 9) floordiv 3, (d2 mod 1024) floordiv 32, d1 mod 3, d2 mod 32)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{8, 3, 3} ["gemmK"] at [1] -> ["ci", "0", "1"] at [2, 3, 5]>, <Merge{128, 32, 32} ["gemmN"] at [2] -> ["ni", "0o", "1o"] at [0, 4, 6]>] bounds = [1, 72, 131072] -> [128, 1, 8, 3, 32, 3, 32]> : tensor<128x1x8x3x32x3x32xf32> to tensor<1x72x131072xf32>

  // 5. Tile into blocks
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 8 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{9, 8} ["k_block", "k_iter"] at [1, 3] -> ["gemmK"] at [1]>, <Unmerge{2048, 64} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 9, 2048, 8, 64] -> [1, 72, 131072]> : tensor<1x72x131072xf32> to tensor<1x9x2048x8x64xf32>

  %pointers, %mask = rock.transforms_to_ptr %4[%c0_i32, %c0_i32, %c0_i32] : tensor<1x9x2048x8x64xf32> -> tensor<8x64xi32>, tensor<8x64xi1>
  %5 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<8x64xi32>, tensor<8x64xi1> -> tensor<8x64xf32>

  return %5 : tensor<8x64xf32>
}

// -----

// Verifies 64-bit index arithmetic is emitted when the transform chain
// requires it (here the underlying buffer exceeds the 32-bit byte range, and
// the linearized index domain exceeds INT32_MAX). The base placeholder is
// extracted directly as i64 (rock.extract_ptr -> i64) and the make_range
// coordinates (necessarily i32) are sign-extended to i64, so all offset
// arithmetic happens in 64 bits and cannot overflow before tt.addptr. No
// base-widening extension is needed since the base already matches the offset
// width.
// CHECK-LABEL: @test_i64_large_buffer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8589934592xf16>)
//      CHECK:   %[[BASE_PTR:.*]] = rock.extract_ptr %[[ARG0]] : tensor<8589934592xf16> -> i64
//      CHECK:   %[[RANGE:.*]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
//      CHECK:   arith.extsi %[[RANGE]] : tensor<64xi32> to tensor<64xi64>
//      CHECK:   arith.muli {{.*}} : i64
//      CHECK:   arith.addi {{.*}} : tensor<64xi64>
//      CHECK:   %[[BASE_SPLAT:.*]] = tt.splat %[[BASE_PTR]] : i64 -> tensor<64xi64>
//  CHECK-NOT:   arith.extsi %[[BASE_SPLAT]]
//      CHECK:   arith.addi {{.*}} : tensor<64xi64>
//      CHECK:   rock.blockwise_load_ptr {{.*}} : tensor<64xi64>, tensor<64xi1> -> tensor<64xf16>
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_i64_large_buffer(%arg0: tensor<8589934592xf16>) -> tensor<64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %c0_i32 = arith.constant 0 : i32

  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{134217728, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [134217728, 64] -> [8589934592]> : tensor<8589934592xf16> to tensor<134217728x64xf16>

  %pointers, %mask = rock.transforms_to_ptr %0[%c0_i32] : tensor<134217728x64xf16> -> tensor<64xi64>, tensor<64xi1>
  %1 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64xi64>, tensor<64xi1> -> tensor<64xf16>

  return %1 : tensor<64xf16>
}

// -----

// Verifies i64 is chosen from the *index domain*, not the buffer size. Here the
// buffer is tiny (131072 elements, 256 KB) but a Pad inflates the m dimension to
// 2147483648 (> INT32_MAX). Even though every reachable offset is small, the
// transform-chain bound overflows 32 bits, so TransformsToPtrOp must emit i64
// offset arithmetic (base extracted as i64, coordinates/validity in i64).
// CHECK-LABEL: @test_i64_overpad_small_buffer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<131072xf16>)
//      CHECK:   %[[BASE_PTR:.*]] = rock.extract_ptr %[[ARG0]] : tensor<131072xf16> -> i64
//      CHECK:   %[[RANGE:.*]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
//      CHECK:   arith.extsi %[[RANGE]] : tensor<64xi32> to tensor<64xi64>
//      CHECK:   arith.cmpi ult, {{.*}} : i64
//      CHECK:   %[[BASE_SPLAT:.*]] = tt.splat %[[BASE_PTR]] : i64 -> tensor<64xi64>
//  CHECK-NOT:   arith.extsi %[[BASE_SPLAT]]
//      CHECK:   arith.addi {{.*}} : tensor<64xi64>
//      CHECK:   rock.blockwise_load_ptr {{.*}} : tensor<64xi64>, tensor<64xi1> -> tensor<64xf16>
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @test_i64_overpad_small_buffer(%arg0: tensor<131072xf16>) -> tensor<64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %c0_i32 = arith.constant 0 : i32

  // Unmerge the small buffer into 2048x64, then Pad m up to 2147483648 (> INT32_MAX).
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{2048, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [2048, 64] -> [131072]> : tensor<131072xf16> to tensor<2048x64xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1) -> (d0, d1)> by [<Pad{0, 2147481600} ["m_padded"] at [0] -> ["m"] at [0]>, <PassThrough ["n"] at [1] -> ["n"] at [1]>] bounds = [2147483648, 64] -> [2048, 64]> : tensor<2048x64xf16> to tensor<2147483648x64xf16>

  %pointers, %mask = rock.transforms_to_ptr %1[%c0_i32] : tensor<2147483648x64xf16> -> tensor<64xi64>, tensor<64xi1>
  %2 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64xi64>, tensor<64xi1> -> tensor<64xf16>

  return %2 : tensor<64xf16>
}

// -----

// Dense non-splat constants use a real pointer base and a row-major flattening
// map. Unit dimensions must not be duplicated in that map.
// CHECK-LABEL: @test_dense_constant_flattening
// CHECK: %[[VALUES:.*]] = arith.constant dense<{{.*}}> : tensor<1x1x2x2xf32>
// CHECK: %[[BASE:.*]] = rock.extract_ptr %[[VALUES]] : tensor<1x1x2x2xf32> -> i32
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: tt.splat %[[BASE]] : i32 -> tensor<1x1x2x2xi32>
// CHECK-NOT: rock.transforms_to_ptr
func.func @test_dense_constant_flattening() attributes {rock.arch = "##TOKEN_ARCH##"} {
  %values = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>
  %pointers, %mask = rock.transforms_to_ptr %values : tensor<1x1x2x2xf32> -> tensor<1x1x2x2xi32>, tensor<1x1x2x2xi1>
  return
}

// -----

// Rank-N splat constants use the same implicit row-major flattening map, but
// retain their synthetic zero base because they do not require memory storage.
// CHECK-LABEL: @test_splat_constant_flattening
// CHECK-NOT: rock.extract_ptr
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: tt.splat %{{.*}} : i32 -> tensor<2x2xi32>
// CHECK-NOT: rock.transforms_to_ptr
func.func @test_splat_constant_flattening() attributes {rock.arch = "##TOKEN_ARCH##"} {
  %values = arith.constant dense<0.0> : tensor<2x2xf32>
  %pointers, %mask = rock.transforms_to_ptr %values : tensor<2x2xf32> -> tensor<2x2xi32>, tensor<2x2xi1>
  return
}
