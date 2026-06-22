// Unit tests for the sub-byte i4 handling in rock-legalize-float-types.
// These test the case where the GEMM operand is wider (e.g. f16) but root
// block args are i4, requiring broadcast transforms and sub-byte extraction.

// RUN: rocmlir-opt -rock-legalize-float-types -mlir-print-local-scope --split-input-file %s | FileCheck %s

// Test 1: Sub-byte uint4 behind a dequant fusion chain.
// The GEMM operand is f16, but the data block arg is i4 (packed uint4).
// strideFactor == 1 so the FAST path fires: the existing transform chain is
// halved in-place (no Broadcast/Merge inserted) and the load tile shape is
// halved.  After the load, andi (low nibble) + shrui (high nibble) + tt.join
// + tt.reshape rebuild the original logical tile shape.

// CHECK-LABEL: func.func @test_i4_sub_byte_extui
// Block arg halved to i8
// CHECK-SAME: (%[[DATA:.*]]: tensor<8xi8>,
// Existing transforms (Unmerge / AddDim) survive, just with halved sizes.
// CHECK: rock.transform %[[DATA]]{{.*}}AddDim
// Load tile is the HALVED i8 shape (last dim 4 -> 2 along packing axis).
// CHECK: %[[LOADED:.*]] = rock.blockwise_load %{{.*}} : tensor<1x1x1x1x4x2xi8> -> tensor<4x2xi8>
// Sub-byte extraction: andi (low nibble) before shrui (high nibble).
// CHECK: arith.andi %[[LOADED]], %{{.*}} : tensor<4x2xi8>
// CHECK: arith.shrui %[[LOADED]], %{{.*}} : tensor<4x2xi8>
// Re-interleave low/high nibbles back to the original logical tile.
// CHECK: tt.join
// CHECK: tt.reshape
// CHECK-NOT: arith.extui
func.func @test_i4_sub_byte_extui(
    %data: tensor<16xi4>,
    %scale: tensor<16xf16>,
    %out: tensor<16xf32>) -> tensor<16xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32

  // Data i4 chain: 1D(16) -> 3D(1x4x4) -> 6D(1x1x1x1x4x4) -> load -> extui
  %data_3d = rock.transform %data by <affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
    by [<Unmerge{4, 4} ["k", "n"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 4, 4] -> [16]>
    : tensor<16xi4> to tensor<1x4x4xi4>
  %data_6d = rock.transform %data_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 4 + d4, d3 * 4 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 4} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 4, 4] -> [1, 4, 4]>
    : tensor<1x4x4xi4> to tensor<1x1x1x1x4x4xi4>
  %data_tile = rock.blockwise_load %data_6d[%c0, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
    : tensor<1x1x1x1x4x4xi4> -> tensor<4x4xi4>
  %ext = arith.extui %data_tile : tensor<4x4xi4> to tensor<4x4xi8>
  %ext_f16 = arith.uitofp %ext : tensor<4x4xi8> to tensor<4x4xf16>

  // Scale chain (f16 block arg, no i4 involved)
  %scale_3d = rock.transform %scale by <affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
    by [<Unmerge{4, 4} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 4, 4] -> [16]>
    : tensor<16xf16> to tensor<1x4x4xf16>
  %scale_6d = rock.transform %scale_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 4 + d4, d3 * 4 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 4} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 4, 4] -> [1, 4, 4]>
    : tensor<1x4x4xf16> to tensor<1x1x1x1x4x4xf16>
  %scale_tile = rock.blockwise_load %scale_6d[%c0, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
    : tensor<1x1x1x1x4x4xf16> -> tensor<4x4xf16>

  // Dequant fusion: scale * data
  %dequant = arith.mulf %ext_f16, %scale_tile : tensor<4x4xf16>

  // matrixA (non-i4, plain f16)
  %a_3d = rock.transform %scale by <affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
    by [<Unmerge{4, 4} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 4, 4] -> [16]>
    : tensor<16xf16> to tensor<1x4x4xf16>
  %a_6d = rock.transform %a_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3 * 4 + d4, d0 * 4 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 4} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 4} ["m_block", "m_iter"] at [3, 4] -> ["m"] at [1]>,
        <AddDim{1} ["n_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 4, 4] -> [1, 4, 4]>
    : tensor<1x4x4xf16> to tensor<1x1x1x1x4x4xf16>
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
    : tensor<1x1x1x1x4x4xf16> -> tensor<4x4xf16>

  %cst = arith.constant dense<0.0> : tensor<4x4xf32>
  %result = rock.blockwise_gemm(%a_tile, %dequant, %cst)
    : tensor<4x4xf16>, tensor<4x4xf16>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %out : tensor<16xf32>
}

// -----

// Test 2: Sub-byte int4 (signed) behind a dequant fusion chain.
// Same structure as uint4 test above, but uses arith.extsi instead of extui.
// Fast path again (strideFactor == 1).  After the unpack (andi + shrui +
// tt.join + tt.reshape), the extsi user is replaced by an arith.shli+shrsi
// pair on the unpacked tile to sign-extend each 4-bit nibble.

// CHECK-LABEL: func.func @test_i4_sub_byte_extsi
// CHECK-SAME: (%[[DATA:.*]]: tensor<8xi8>,
// CHECK: rock.transform %[[DATA]]{{.*}}AddDim
// CHECK: %[[LOADED:.*]] = rock.blockwise_load %{{.*}} : tensor<1x1x1x1x4x2xi8> -> tensor<4x2xi8>
// Fast-path nibble unpack: andi (low) + shrui (high) on the halved tile.
// CHECK: arith.andi %[[LOADED]], %{{.*}} : tensor<4x2xi8>
// CHECK: arith.shrui %[[LOADED]], %{{.*}} : tensor<4x2xi8>
// CHECK: tt.join
// CHECK: tt.reshape
// Sign-extend each nibble of the unpacked tile via shli<<4 then shrsi>>4.
// CHECK: arith.shli %{{.*}}, %{{.*}} : tensor<4x4xi8>
// CHECK: arith.shrsi %{{.*}}, %{{.*}} : tensor<4x4xi8>
// CHECK-NOT: arith.extsi
func.func @test_i4_sub_byte_extsi(
    %data: tensor<16xi4>,
    %scale: tensor<16xf16>,
    %out: tensor<16xf32>) -> tensor<16xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32

  %data_3d = rock.transform %data by <affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
    by [<Unmerge{4, 4} ["k", "n"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 4, 4] -> [16]>
    : tensor<16xi4> to tensor<1x4x4xi4>
  %data_6d = rock.transform %data_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 4 + d4, d3 * 4 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 4} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 4, 4] -> [1, 4, 4]>
    : tensor<1x4x4xi4> to tensor<1x1x1x1x4x4xi4>
  %data_tile = rock.blockwise_load %data_6d[%c0, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
    : tensor<1x1x1x1x4x4xi4> -> tensor<4x4xi4>
  %ext = arith.extsi %data_tile : tensor<4x4xi4> to tensor<4x4xi8>
  %ext_f16 = arith.sitofp %ext : tensor<4x4xi8> to tensor<4x4xf16>

  %scale_3d = rock.transform %scale by <affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
    by [<Unmerge{4, 4} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 4, 4] -> [16]>
    : tensor<16xf16> to tensor<1x4x4xf16>
  %scale_6d = rock.transform %scale_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 4 + d4, d3 * 4 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 4} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 4, 4] -> [1, 4, 4]>
    : tensor<1x4x4xf16> to tensor<1x1x1x1x4x4xf16>
  %scale_tile = rock.blockwise_load %scale_6d[%c0, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
    : tensor<1x1x1x1x4x4xf16> -> tensor<4x4xf16>

  %dequant = arith.mulf %ext_f16, %scale_tile : tensor<4x4xf16>

  %a_3d = rock.transform %scale by <affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
    by [<Unmerge{4, 4} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 4, 4] -> [16]>
    : tensor<16xf16> to tensor<1x4x4xf16>
  %a_6d = rock.transform %a_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3 * 4 + d4, d0 * 4 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 4} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 4} ["m_block", "m_iter"] at [3, 4] -> ["m"] at [1]>,
        <AddDim{1} ["n_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 4, 4] -> [1, 4, 4]>
    : tensor<1x4x4xf16> to tensor<1x1x1x1x4x4xf16>
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
    : tensor<1x1x1x1x4x4xf16> -> tensor<4x4xf16>

  %cst = arith.constant dense<0.0> : tensor<4x4xf32>
  %result = rock.blockwise_gemm(%a_tile, %dequant, %cst)
    : tensor<4x4xf16>, tensor<4x4xf16>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %out : tensor<16xf32>
}

// -----

// Test 3: Sub-byte i4 with arith.constant in the fusion chain.
// When the dequant chain includes an inlined constant (e.g. scale literal),
// collectOperandInputs should skip it without error.  Same fast path as
// tests 1/2 (strideFactor == 1).

// CHECK-LABEL: func.func @test_i4_constant_in_fusion_chain
// CHECK-SAME: (%[[DATA:.*]]: tensor<8xi8>,
// CHECK: rock.transform %[[DATA]]{{.*}}AddDim
// Fast-path nibble unpack: andi (low) before shrui (high) on the halved tile.
// CHECK: arith.andi %{{.*}}, %{{.*}} : tensor<4x2xi8>
// CHECK: arith.shrui %{{.*}}, %{{.*}} : tensor<4x2xi8>
// CHECK: tt.join
// CHECK: tt.reshape
// CHECK-NOT: arith.extui
func.func @test_i4_constant_in_fusion_chain(
    %data: tensor<16xi4>,
    %out: tensor<16xf32>) -> tensor<16xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32

  %data_3d = rock.transform %data by <affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
    by [<Unmerge{4, 4} ["k", "n"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 4, 4] -> [16]>
    : tensor<16xi4> to tensor<1x4x4xi4>
  %data_6d = rock.transform %data_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 4 + d4, d3 * 4 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 4} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 4, 4] -> [1, 4, 4]>
    : tensor<1x4x4xi4> to tensor<1x1x1x1x4x4xi4>
  %data_tile = rock.blockwise_load %data_6d[%c0, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
    : tensor<1x1x1x1x4x4xi4> -> tensor<4x4xi4>
  %ext = arith.extui %data_tile : tensor<4x4xi4> to tensor<4x4xi8>
  %ext_f16 = arith.uitofp %ext : tensor<4x4xi8> to tensor<4x4xf16>

  // Inlined constant scale (no block arg) in the fusion chain
  %scale_cst = arith.constant dense<2.0> : tensor<4x4xf16>
  %dequant = arith.mulf %ext_f16, %scale_cst : tensor<4x4xf16>

  // matrixA: constant f16
  %a_cst = arith.constant dense<1.0> : tensor<4x4xf16>

  %cst = arith.constant dense<0.0> : tensor<4x4xf32>
  %result = rock.blockwise_gemm(%a_cst, %dequant, %cst)
    : tensor<4x4xf16>, tensor<4x4xf16>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %out : tensor<16xf32>
}

// -----

// Test 4: Sub-byte uint4 with a broadcast period that exceeds the K-tile.
//
// This mimics the AWQ zero-points pattern from the Llama-2-7b int4 reproducer:
// the i4 block arg is widened by an AddDim+Broadcast and then a Merge folds
// the broadcast factor into the K axis, so the in-tile nibble-select pattern
// is degenerate (every element of a single tile shares the same shift).  In
// that regime the high/low nibble alternation is driven solely by the outer
// K-loop induction variable, so the pass must emit a loop-variant `tt.splat`
// of an arith-computed scalar shift inside the scf.for body (otherwise LICM
// would hoist a constant-zero shift out of the loop, silently picking the
// low nibble for every iteration).
//
// Chain (block arg -> load source):
//   <16xi4>   --Unmerge{4,4}-->        <4x4xi4>
//             --AddDim{1}     -->      <4x4x1xi4>
//             --Broadcast{1}  -->      <4x4x4xi4>     (broadcast factor 4)
//             --Merge{4,4}    -->      <4x16xi4>      (strideFactor *= 4)
//             --AddDim/permute-->      <1x16x4xi4>
//             --Unmerge{4,4}+Unmerge{1,4}+AddDim --> <4x1x1x1x4x4xi4>
// Tile axis K = 4, strideFactor = 4 -> 4 >= 4 so the dynamic path fires.

// CHECK-LABEL: func.func @test_i4_sub_byte_dynamic_shift
// Block arg halved to i8.
// CHECK-SAME: (%[[DATA:.*]]: tensor<8xi8>,
// Broadcast transform chain prepended ahead of the original chain.
// CHECK: rock.transform %[[DATA]]{{.*}}AddDim
// CHECK: rock.transform{{.*}}Broadcast
// CHECK: rock.transform{{.*}}Merge
// The shift sequence lives *inside* the scf.for and depends on the iv.
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args({{.*}}) -> (tensor<4x4xf32>) : i32 {
// CHECK: arith.muli %[[IV]], %{{.*}} : i32
// CHECK: arith.divui %{{.*}}, %{{.*}} : i32
// CHECK: arith.remui %{{.*}}, %{{.*}} : i32
// CHECK: arith.muli %{{.*}}, %{{.*}} : i32
// CHECK: arith.trunci %{{.*}} : i32 to i8
// CHECK: tt.splat %{{.*}} : i8 -> tensor<4x4xi8>
// CHECK: arith.shrui %{{.*}}, %{{.*}} : tensor<4x4xi8>
// CHECK: arith.andi %{{.*}}, %{{.*}} : tensor<4x4xi8>
// The static in-tile shift form (tt.make_range + ...) must NOT be emitted,
// the alternation depends on the K-loop iv, so it has to live inside scf.for.
// CHECK-NOT: tt.make_range
// CHECK-NOT: arith.extui %{{.*}} : tensor<4x4xi4>
func.func @test_i4_sub_byte_dynamic_shift(
    %data: tensor<16xi4>,
    %scale: tensor<16xf16>,
    %out: tensor<16xf32>) -> tensor<16xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32

  // Bottom: 1D(16) -> 2D(4, 4) with the stride-1 axis as the LAST upper dim.
  %data_2d = rock.transform %data by <affine_map<(d0, d1) -> (d0 * 4 + d1)>
    by [<Unmerge{4, 4} ["outer", "packed"] at [0, 1] -> ["raw"] at [0]>]
    bounds = [4, 4] -> [16]>
    : tensor<16xi4> to tensor<4x4xi4>
  // Add a unit "broad" dim ahead of the broadcast.
  %data_2d_addbroad = rock.transform %data_2d by <affine_map<(d0, d1, d2) -> (d0, d1)>
    by [<PassThrough ["outer"] at [0] -> ["outer"] at [0]>,
        <PassThrough ["packed"] at [1] -> ["packed"] at [1]>,
        <AddDim{1} ["broad"] at [2] -> [] at []>]
    bounds = [4, 4, 1] -> [4, 4]>
    : tensor<4x4xi4> to tensor<4x4x1xi4>
  // Broadcast "broad" from 1 to 4: every byte is reused 4x along K.
  %data_bcast = rock.transform %data_2d_addbroad by <affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    by [<PassThrough ["outer"] at [0] -> ["outer"] at [0]>,
        <PassThrough ["packed"] at [1] -> ["packed"] at [1]>,
        <Broadcast{1} ["broad"] at [2] -> ["broad"] at [2]>]
    bounds = [4, 4, 4] -> [4, 4, 1]>
    : tensor<4x4x1xi4> to tensor<4x4x4xi4>
  // Fold {packed, broad} into a single K-axis: this is the Merge whose tail
  // size (params[1] = 4) bumps the sub-byte stride factor up to the tile
  // axis length, forcing the dynamic path.
  %data_merged = rock.transform %data_bcast by <affine_map<(d0, d1) -> (d0, d1 floordiv 4, d1 mod 4)>
    by [<PassThrough ["outer"] at [0] -> ["outer"] at [0]>,
        <Merge{4, 4} ["k_full"] at [1] -> ["packed", "broad"] at [1, 2]>]
    bounds = [4, 16] -> [4, 4, 4]>
    : tensor<4x4x4xi4> to tensor<4x16xi4>
  // 2D(4, 16) -> 3D(g=1, k=16, n=4) by adding a g dim and swapping outer -> n.
  %data_3d = rock.transform %data_merged by <affine_map<(d0, d1, d2) -> (d2, d1)>
    by [<AddDim{1} ["g"] at [0] -> [] at []>,
        <PassThrough ["k"] at [1] -> ["k_full"] at [1]>,
        <PassThrough ["n"] at [2] -> ["outer"] at [0]>]
    bounds = [1, 16, 4] -> [4, 16]>
    : tensor<4x16xi4> to tensor<1x16x4xi4>
  // 3D(1, 16, 4) -> 6D(k_loop=4, g_block=1, m_block=1, n_block=1, k_iter=4, n_iter=4).
  %data_6d = rock.transform %data_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 4 + d4, d3 * 4 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{4, 4} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [4, 1, 1, 1, 4, 4] -> [1, 16, 4]>
    : tensor<1x16x4xi4> to tensor<4x1x1x1x4x4xi4>

  // Scale chain (f16 block arg, no i4 involved).
  %scale_3d = rock.transform %scale by <affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
    by [<Unmerge{4, 4} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 4, 4] -> [16]>
    : tensor<16xf16> to tensor<1x4x4xf16>
  %scale_6d = rock.transform %scale_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 4 + d4, d3 * 4 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 4} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 4, 4] -> [1, 4, 4]>
    : tensor<1x4x4xf16> to tensor<1x1x1x1x4x4xf16>

  // matrixA chain (plain f16, no sub-byte).
  %a_3d = rock.transform %scale by <affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
    by [<Unmerge{4, 4} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 4, 4] -> [16]>
    : tensor<16xf16> to tensor<1x4x4xf16>
  %a_6d = rock.transform %a_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3 * 4 + d4, d0 * 4 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 4} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 4} ["m_block", "m_iter"] at [3, 4] -> ["m"] at [1]>,
        <AddDim{1} ["n_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 4, 4] -> [1, 4, 4]>
    : tensor<1x4x4xf16> to tensor<1x1x1x1x4x4xf16>

  // Wrap the sub-byte load and gemm in an scf.for with an i32 iv: this is
  // mandatory for the dynamic path -- buildSubByteShiftDynamic looks up the
  // enclosing scf.for and uses its induction variable as the K-loop counter.
  %cst = arith.constant dense<0.0> : tensor<4x4xf32>
  %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %cst) -> (tensor<4x4xf32>)  : i32 {
    %data_tile = rock.blockwise_load %data_6d[%iv, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
      : tensor<4x1x1x1x4x4xi4> -> tensor<4x4xi4>
    %ext = arith.extui %data_tile : tensor<4x4xi4> to tensor<4x4xi8>
    %ext_f16 = arith.uitofp %ext : tensor<4x4xi8> to tensor<4x4xf16>
    %scale_tile = rock.blockwise_load %scale_6d[%c0, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
      : tensor<1x1x1x1x4x4xf16> -> tensor<4x4xf16>
    %dequant = arith.mulf %ext_f16, %scale_tile : tensor<4x4xf16>
    %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
      : tensor<1x1x1x1x4x4xf16> -> tensor<4x4xf16>
    %g = rock.blockwise_gemm(%a_tile, %dequant, %acc)
      : tensor<4x4xf16>, tensor<4x4xf16>, tensor<4x4xf32> -> tensor<4x4xf32>
    scf.yield %g : tensor<4x4xf32>
  }
  return %out : tensor<16xf32>
}

// -----

// Test 5: Sub-byte uint4 with a broadcast period that is > 1 but still less
// than the K-tile axis length.  This is the "static in-tile" fallback case:
// the nibble alternation can be expressed entirely within a single tile, so
// the pass emits a loop-invariant make_range-based shift tensor (no scf.for
// dependency).  Previously this was a marker-tagged `arith.constant` lowered
// later in RockToTTIR; the IR is now emitted directly here.
//
// Chain (block arg -> load source):
//   <16xi4>   --Unmerge{4,4}-->        <4x4xi4>
//             --AddDim{1}     -->      <4x4x1xi4>
//             --Broadcast{1}  -->      <4x4x2xi4>     (broadcast factor 2)
//             --Merge{4,2}    -->      <4x8xi4>       (strideFactor *= 2)
//             --AddDim/permute-->      <1x8x4xi4>
//             --Unmerge{1,8}+Unmerge{1,4}+AddDim --> <1x1x1x1x8x4xi4>
// Tile axis K = 8, strideFactor = 2 -> 2 < 8 so the static path fires.

// CHECK-LABEL: func.func @test_i4_sub_byte_static_shift
// Block arg halved to i8.
// CHECK-SAME: (%[[DATA:.*]]: tensor<8xi8>,
// CHECK: rock.transform %[[DATA]]{{.*}}AddDim
// CHECK: rock.transform{{.*}}Broadcast
// CHECK: rock.transform{{.*}}Merge
// The shift sequence is loop-invariant: tt.make_range + arith ops, no scf.for
// induction variable in the shift computation.
// CHECK: tt.make_range {end = 8 : i32, start = 0 : i32}
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK: arith.muli
// CHECK: arith.trunci
// CHECK: tt.expand_dims
// CHECK: tt.broadcast
// Sub-byte extraction over the loaded i8 tile.
// CHECK: arith.shrui %{{.*}}, %{{.*}} : tensor<8x4xi8>
// CHECK: arith.andi %{{.*}}, %{{.*}} : tensor<8x4xi8>
// No tt.splat-based shift (that would be the dynamic path).
// CHECK-NOT: tt.splat %{{.*}} : i8 -> tensor<8x4xi8>
// CHECK-NOT: arith.extui %{{.*}} : tensor<8x4xi4>
func.func @test_i4_sub_byte_static_shift(
    %data: tensor<16xi4>,
    %scale: tensor<32xf16>,
    %out: tensor<16xf32>) -> tensor<16xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32

  // Bottom: 1D(16) -> 2D(4, 4).
  %data_2d = rock.transform %data by <affine_map<(d0, d1) -> (d0 * 4 + d1)>
    by [<Unmerge{4, 4} ["outer", "packed"] at [0, 1] -> ["raw"] at [0]>]
    bounds = [4, 4] -> [16]>
    : tensor<16xi4> to tensor<4x4xi4>
  // Add a unit "broad" dim ahead of the broadcast.
  %data_2d_addbroad = rock.transform %data_2d by <affine_map<(d0, d1, d2) -> (d0, d1)>
    by [<PassThrough ["outer"] at [0] -> ["outer"] at [0]>,
        <PassThrough ["packed"] at [1] -> ["packed"] at [1]>,
        <AddDim{1} ["broad"] at [2] -> [] at []>]
    bounds = [4, 4, 1] -> [4, 4]>
    : tensor<4x4xi4> to tensor<4x4x1xi4>
  // Broadcast "broad" from 1 to 2: every byte is reused 2x along K.
  %data_bcast = rock.transform %data_2d_addbroad by <affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    by [<PassThrough ["outer"] at [0] -> ["outer"] at [0]>,
        <PassThrough ["packed"] at [1] -> ["packed"] at [1]>,
        <Broadcast{1} ["broad"] at [2] -> ["broad"] at [2]>]
    bounds = [4, 4, 2] -> [4, 4, 1]>
    : tensor<4x4x1xi4> to tensor<4x4x2xi4>
  // Fold {packed, broad} into K with tail size 2 -> strideFactor = 2.
  %data_merged = rock.transform %data_bcast by <affine_map<(d0, d1) -> (d0, d1 floordiv 2, d1 mod 2)>
    by [<PassThrough ["outer"] at [0] -> ["outer"] at [0]>,
        <Merge{4, 2} ["k_full"] at [1] -> ["packed", "broad"] at [1, 2]>]
    bounds = [4, 8] -> [4, 4, 2]>
    : tensor<4x4x2xi4> to tensor<4x8xi4>
  // 2D(4, 8) -> 3D(g=1, k=8, n=4) by adding a g dim and swapping outer -> n.
  %data_3d = rock.transform %data_merged by <affine_map<(d0, d1, d2) -> (d2, d1)>
    by [<AddDim{1} ["g"] at [0] -> [] at []>,
        <PassThrough ["k"] at [1] -> ["k_full"] at [1]>,
        <PassThrough ["n"] at [2] -> ["outer"] at [0]>]
    bounds = [1, 8, 4] -> [4, 8]>
    : tensor<4x8xi4> to tensor<1x8x4xi4>
  // 3D(1, 8, 4) -> 6D(k_loop=1, g_block=1, m_block=1, n_block=1, k_iter=8, n_iter=4).
  %data_6d = rock.transform %data_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 8 + d4, d3 * 4 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 8} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 8, 4] -> [1, 8, 4]>
    : tensor<1x8x4xi4> to tensor<1x1x1x1x8x4xi4>
  %data_tile = rock.blockwise_load %data_6d[%c0, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
    : tensor<1x1x1x1x8x4xi4> -> tensor<8x4xi4>
  %ext = arith.extui %data_tile : tensor<8x4xi4> to tensor<8x4xi8>
  %ext_f16 = arith.uitofp %ext : tensor<8x4xi8> to tensor<8x4xf16>

  // Scale chain (f16, no sub-byte).  Lower bounds are [1, 8, 4] so that the
  // 6D unmerges line up with the data chain (k=8 at lower dim 1, n=4 at 2).
  %scale_3d = rock.transform %scale by <affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
    by [<Unmerge{8, 4} ["k", "n"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 8, 4] -> [32]>
    : tensor<32xf16> to tensor<1x8x4xf16>
  %scale_6d = rock.transform %scale_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 8 + d4, d3 * 4 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 8} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 8, 4] -> [1, 8, 4]>
    : tensor<1x8x4xf16> to tensor<1x1x1x1x8x4xf16>
  %scale_tile = rock.blockwise_load %scale_6d[%c0, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
    : tensor<1x1x1x1x8x4xf16> -> tensor<8x4xf16>

  %dequant = arith.mulf %ext_f16, %scale_tile : tensor<8x4xf16>

  // matrixA chain (f16, no sub-byte).  Tile is [M=4, K=8].
  %a_3d = rock.transform %scale by <affine_map<(d0, d1, d2) -> (d1 * 8 + d2)>
    by [<Unmerge{4, 8} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 4, 8] -> [32]>
    : tensor<32xf16> to tensor<1x4x8xf16>
  %a_6d = rock.transform %a_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3 * 4 + d4, d0 * 8 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 8} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 4} ["m_block", "m_iter"] at [3, 4] -> ["m"] at [1]>,
        <AddDim{1} ["n_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 4, 8] -> [1, 4, 8]>
    : tensor<1x4x8xf16> to tensor<1x1x1x1x4x8xf16>
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0] {cacheModifier = #rock<CacheModifier none>}
    : tensor<1x1x1x1x4x8xf16> -> tensor<4x8xf16>

  // GEMM: A=[M,K]=[4,8] x B=[K,N]=[8,4] -> C=[M,N]=[4,4].
  %cst = arith.constant dense<0.0> : tensor<4x4xf32>
  %result = rock.blockwise_gemm(%a_tile, %dequant, %cst)
    : tensor<4x8xf16>, tensor<8x4xf16>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %out : tensor<16xf32>
}
