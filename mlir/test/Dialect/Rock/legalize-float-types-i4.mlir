// Unit tests for the sub-byte i4 handling in rock-legalize-float-types.
// These test the case where the GEMM operand is wider (e.g. f16) but root
// block args are i4, requiring broadcast transforms and sub-byte extraction.

// RUN: rocmlir-opt -rock-legalize-float-types -mlir-print-local-scope --split-input-file %s | FileCheck %s

// Test 1: Sub-byte uint4 behind a dequant fusion chain.
// The GEMM operand is f16, but the data block arg is i4 (packed uint4).
// The pass should:
//   1. Halve the i4 block arg: tensor<16 x i4> -> tensor<8 x i8>
//   2. Insert broadcast transforms: [8] -> [8,1] -> [8,2] -> [16]
//   3. Replace arith.extui with shift-and-mask sub-byte extraction

// CHECK-LABEL: func.func @test_i4_sub_byte_extui
// Block arg halved to i8
// CHECK-SAME: (%[[DATA:.*]]: tensor<8xi8>,
// Broadcast transform chain inserted between halved arg and bottom transform
// CHECK: rock.transform %[[DATA]]{{.*}}AddDim
// CHECK: rock.transform{{.*}}Broadcast
// CHECK: rock.transform{{.*}}Merge
// Load result type is i8
// CHECK: %[[LOADED:.*]] = rock.blockwise_load %{{.*}} : tensor<1x1x1x1x4x4xi8> -> tensor<4x4xi8>
// Sub-byte extraction: shrui + andi (no arith.extui)
// CHECK: arith.shrui %[[LOADED]], %{{.*}} : tensor<4x4xi8>
// CHECK: arith.andi %{{.*}}, %{{.*}} : tensor<4x4xi8>
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
  %data_tile = rock.blockwise_load %data_6d[%c0, %c0, %c0, %c0]
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
  %scale_tile = rock.blockwise_load %scale_6d[%c0, %c0, %c0, %c0]
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
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x4x4xf16> -> tensor<4x4xf16>

  %cst = arith.constant dense<0.0> : tensor<4x4xf32>
  %result = rock.blockwise_gemm(%a_tile, %dequant, %cst)
    : tensor<4x4xf16>, tensor<4x4xf16>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %out : tensor<16xf32>
}

// -----

// Test 2: Sub-byte int4 (signed) behind a dequant fusion chain.
// Same structure as uint4 test above, but uses arith.extsi instead of extui.
// The pass should replace arith.extsi with the sign-extending variant:
//   (loaded >> shifts) & 0xF  then  (<< 4) >>s 4

// CHECK-LABEL: func.func @test_i4_sub_byte_extsi
// CHECK-SAME: (%[[DATA:.*]]: tensor<8xi8>,
// CHECK: rock.transform %[[DATA]]{{.*}}AddDim
// CHECK: rock.transform{{.*}}Broadcast
// CHECK: rock.transform{{.*}}Merge
// CHECK: %[[LOADED:.*]] = rock.blockwise_load %{{.*}} : tensor<1x1x1x1x4x4xi8> -> tensor<4x4xi8>
// Sub-byte extraction + sign extension: shrui, andi, shli, shrsi
// CHECK: arith.shrui %[[LOADED]], %{{.*}} : tensor<4x4xi8>
// CHECK: arith.andi %{{.*}}, %{{.*}} : tensor<4x4xi8>
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
  %data_tile = rock.blockwise_load %data_6d[%c0, %c0, %c0, %c0]
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
  %scale_tile = rock.blockwise_load %scale_6d[%c0, %c0, %c0, %c0]
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
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x4x4xf16> -> tensor<4x4xf16>

  %cst = arith.constant dense<0.0> : tensor<4x4xf32>
  %result = rock.blockwise_gemm(%a_tile, %dequant, %cst)
    : tensor<4x4xf16>, tensor<4x4xf16>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %out : tensor<16xf32>
}

// -----

// Test 3: Sub-byte i4 with arith.constant in the fusion chain.
// When the dequant chain includes an inlined constant (e.g. scale literal),
// collectOperandInputs should skip it without error.

// CHECK-LABEL: func.func @test_i4_constant_in_fusion_chain
// CHECK-SAME: (%[[DATA:.*]]: tensor<8xi8>,
// CHECK: rock.transform %[[DATA]]{{.*}}AddDim
// CHECK: arith.shrui
// CHECK: arith.andi
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
  %data_tile = rock.blockwise_load %data_6d[%c0, %c0, %c0, %c0]
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
