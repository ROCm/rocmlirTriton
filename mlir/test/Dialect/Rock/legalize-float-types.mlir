// Unit tests for rock-legalize-float-types pass.
// Verifies that non-TT_Float types used in scaled GEMM kernels and wrappers
// are legalized to integer types.

// RUN: rocmlir-opt -rock-legalize-float-types -mlir-print-local-scope --split-input-file %s | FileCheck %s

// Test 1: f8E8M0FNU scales -> i8 (data types are TT_Float, no packing needed)

// CHECK-LABEL: func.func @test_f8_scale_to_i8
// CHECK-SAME: (%{{.*}}: tensor<64x64xf8E4M3FN>, %{{.*}}: tensor<64x64xf8E4M3FN>,
// CHECK-SAME:  %[[SA:.*]]: tensor<64x2xi8>, %[[SB:.*]]: tensor<64x2xi8>,
// CHECK-SAME:  %{{.*}}: tensor<64x64xf32>)
//      CHECK: rock.blockwise_gemm
// CHECK-SAME: scaled by %[[SA]]
// CHECK-SAME: scaled by %[[SB]]
// CHECK-SAME: tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>
// CHECK-SAME: tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>
// matrixA/B types (f8E4M3FN) are TT_Float, so no origElemType attr is set.
// CHECK-NOT: matrixAOrigElemType
// CHECK-NOT: matrixBOrigElemType
func.func @test_f8_scale_to_i8(
    %a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>,
    %scaleA: tensor<64x2xf8E8M0FNU>, %scaleB: tensor<64x2xf8E8M0FNU>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %result = rock.blockwise_gemm(%a scaled by %scaleA, %b scaled by %scaleB, %c)
    {quantBlockSize = 32 : i64}
    : tensor<64x64xf8E4M3FN> scaled by tensor<64x2xf8E8M0FNU>,
      tensor<64x64xf8E4M3FN> scaled by tensor<64x2xf8E8M0FNU>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test 2: Non-kernel functions (no rock.kernel) are unchanged

// CHECK-LABEL: func.func @test_non_kernel_unchanged
// CHECK-SAME: tensor<64x2xf8E8M0FNU>
func.func @test_non_kernel_unchanged(
    %a: tensor<64x64xf8E4M3FN>,
    %scaleA: tensor<64x2xf8E8M0FNU>) -> tensor<64x2xf8E8M0FNU> {
  return %scaleA : tensor<64x2xf8E8M0FNU>
}

// -----

// Test 3: Wrapper converts f8E8M0FNU memref via unrealized_conversion_cast

func.func @kernel_f8(%arg0: tensor<256xf8E8M0FNU>) -> tensor<256xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %cst = arith.constant dense<0.0> : tensor<256xf32>
  return %cst : tensor<256xf32>
}

// CHECK-LABEL: func.func @wrapper_f8_scale
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : memref<256xf8E8M0FNU> to memref<256xi8>
// CHECK: bufferization.to_tensor %[[CAST]]
func.func @wrapper_f8_scale(%mem: memref<256xf8E8M0FNU>) {
  %t = bufferization.to_tensor %mem : memref<256xf8E8M0FNU> to tensor<256xf8E8M0FNU>
  %r = func.call @kernel_f8(%t) : (tensor<256xf8E8M0FNU>) -> tensor<256xf32>
  return
}

// -----

// Test 4: f4E2M1FN packing with full transform chain.
// matrixA (MxK): K is stride-1 in raw -> K gets halved (matrixAKPack=true)
//   raw block arg: tensor<16384xf4E2M1FN> -> tensor<8192xi8>
//   3D: tensor<1x64x256> -> tensor<1x64x128> (K halved)
//   6D: k_iter halved 64 -> 32
//   tile: tensor<64x64> -> tensor<64x32>
// matrixB (KxN): N is stride-1 in raw -> N gets halved (matrixBKPack=false)
//   raw block arg: tensor<32768xf4E2M1FN> -> tensor<16384xi8>
//   3D: tensor<1x256x128> -> tensor<1x256x64> (N halved)
//   6D: n_iter halved 64 -> 32
//   tile: tensor<64x64> -> tensor<64x32>

// CHECK-LABEL: func.func @test_f4_packing
// CHECK-SAME: (%{{.*}}: tensor<8192xi8>, %{{.*}}: tensor<16384xi8>,
// CHECK-SAME:  %{{.*}}: tensor<512xi8>, %{{.*}}: tensor<1024xi8>,
// CHECK-SAME:  %{{.*}}: tensor<8192xf32>)
// B comes first in the output; n_iter halved from 64 to 32
// CHECK: rock.transform %{{.*}} by <{{.*}}Unmerge{2, 32}{{.*}}["n_block", "n_iter"]
// CHECK: rock.blockwise_load {{.*}} : tensor<4x1x1x2x64x32xi8> -> tensor<64x32xi8>
// A: k_iter halved from 64 to 32
// CHECK: rock.transform %{{.*}} by <{{.*}}Unmerge{4, 32}{{.*}}["k_loop", "k_iter"]
// CHECK: rock.blockwise_load {{.*}} : tensor<4x1x1x2x64x32xi8> -> tensor<64x32xi8>
//      CHECK: rock.blockwise_gemm
// CHECK-SAME: matrixAKPack = true
// CHECK-SAME: matrixAOrigElemType = f4E2M1FN
// CHECK-SAME: matrixBKPack = false
// CHECK-SAME: matrixBOrigElemType = f4E2M1FN
// CHECK-SAME: tensor<64x32xi8> scaled by tensor<64x2xi8>
// CHECK-SAME: tensor<64x32xi8> scaled by tensor<64x2xi8>
func.func @test_f4_packing(
    %arg0: tensor<16384xf4E2M1FN>,
    %arg1: tensor<32768xf4E2M1FN>,
    %arg2: tensor<512xf8E8M0FNU>,
    %arg3: tensor<1024xf8E8M0FNU>,
    %arg4: tensor<8192xf32>) -> tensor<8192xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // Scale transforms (f8E8M0FNU -> i8 by simple type swap)
  %sa_3d = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 8 + d2)>
    by [<Unmerge{64, 8} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 8] -> [512]>
    : tensor<512xf8E8M0FNU> to tensor<1x64x8xf8E8M0FNU>
  %sb_3d = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 8 + d2)>
    by [<Unmerge{128, 8} ["n", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 128, 8] -> [1024]>
    : tensor<1024xf8E8M0FNU> to tensor<1x128x8xf8E8M0FNU>
  %c0 = arith.constant 0 : i32

  // matrixB chain: 1D -> 3D -> 6D -> blockwise_load
  // raw = k * 128 + n, so N is stride-1 (fastest changing)
  %b_3d = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
    by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 256, 128] -> [32768]>
    : tensor<32768xf4E2M1FN> to tensor<1x256x128xf4E2M1FN>
  %b_6d = rock.transform %b_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]>
    : tensor<1x256x128xf4E2M1FN> to tensor<4x1x1x2x64x64xf4E2M1FN>
  %b_tile = rock.blockwise_load %b_6d[%c0, %c0, %c0, %c0]
    : tensor<4x1x1x2x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  // matrixA chain: 1D -> 3D -> 6D -> blockwise_load
  // raw = m * 256 + k, so K is stride-1 (fastest changing)
  %a_3d = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)>
    by [<Unmerge{64, 256} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 256] -> [16384]>
    : tensor<16384xf4E2M1FN> to tensor<1x64x256xf4E2M1FN>
  %a_6d = rock.transform %a_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>,
        <AddDim{2} ["n_block"] at [3] -> [] at []>]
    bounds = [4, 1, 1, 2, 64, 64] -> [1, 64, 256]>
    : tensor<1x64x256xf4E2M1FN> to tensor<4x1x1x2x64x64xf4E2M1FN>
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0]
    : tensor<4x1x1x2x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  // scaleB chain: 3D -> passthrough -> 6D -> load
  %sb_pt = rock.transform %sb_3d by <affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    by [<PassThrough ["g", "n", "kScale"] at [0, 1, 2] -> ["g", "n", "kScale"] at [0, 1, 2]>]
    bounds = [1, 128, 8] -> [1, 128, 8]>
    : tensor<1x128x8xf8E8M0FNU> to tensor<1x128x8xf8E8M0FNU>
  %sb_6d = rock.transform %sb_pt by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3 * 64 + d4, d0 * 2 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{4, 2} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 4] -> ["n"] at [1]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [4, 1, 1, 2, 64, 2] -> [1, 128, 8]>
    : tensor<1x128x8xf8E8M0FNU> to tensor<4x1x1x2x64x2xf8E8M0FNU>
  %sb_tile = rock.blockwise_load %sb_6d[%c0, %c0, %c0, %c0]
    : tensor<4x1x1x2x64x2xf8E8M0FNU> -> tensor<64x2xf8E8M0FNU>

  // scaleA chain: 3D -> passthrough -> 6D -> load
  %sa_pt = rock.transform %sa_3d by <affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    by [<PassThrough ["g", "m", "kScale"] at [0, 1, 2] -> ["g", "m", "kScale"] at [0, 1, 2]>]
    bounds = [1, 64, 8] -> [1, 64, 8]>
    : tensor<1x64x8xf8E8M0FNU> to tensor<1x64x8xf8E8M0FNU>
  %sa_6d = rock.transform %sa_pt by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 2 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{4, 2} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>,
        <AddDim{2} ["n_block"] at [3] -> [] at []>]
    bounds = [4, 1, 1, 2, 64, 2] -> [1, 64, 8]>
    : tensor<1x64x8xf8E8M0FNU> to tensor<4x1x1x2x64x2xf8E8M0FNU>
  %sa_tile = rock.blockwise_load %sa_6d[%c0, %c0, %c0, %c0]
    : tensor<4x1x1x2x64x2xf8E8M0FNU> -> tensor<64x2xf8E8M0FNU>

  %cst = arith.constant dense<0.0> : tensor<64x64xf32>
  %result = rock.blockwise_gemm(%a_tile scaled by %sa_tile, %b_tile scaled by %sb_tile, %cst)
    {quantBlockSize = 32 : i64}
    : tensor<64x64xf4E2M1FN> scaled by tensor<64x2xf8E8M0FNU>,
      tensor<64x64xf4E2M1FN> scaled by tensor<64x2xf8E8M0FNU>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %arg4 : tensor<8192xf32>
}

// -----

// Test 5: Pad transform on K dimension of matrixA.
// A raw: G=1, M=64, K=32 (padded to 64). K is stride-1.
// After halving K: Pad{0,32} -> Pad{0,16}, lower K 32->16, arg 2048->1024.

// CHECK-LABEL: func.func @test_f4_pad
// CHECK-SAME: (%[[A:.*]]: tensor<1024xi8>,
// A chain: check Pad params halved from {0,32} to {0,16}
// CHECK: rock.transform %[[A]]
// CHECK: Unmerge{64, 16}
// CHECK: rock.transform
// CHECK: Pad{0, 16}
// CHECK: rock.transform
// CHECK: Unmerge{1, 32}{{.*}}["k_loop", "k_iter"]
// CHECK: rock.blockwise_load {{.*}} -> tensor<64x32xi8>
// CHECK: matrixAKPack = true
func.func @test_f4_pad(
    %arg0: tensor<2048xf4E2M1FN>,
    %arg1: tensor<4096xf4E2M1FN>,
    %arg2: tensor<64x1xf8E8M0FNU>,
    %arg3: tensor<64x1xf8E8M0FNU>,
    %arg4: tensor<4096xf32>) -> tensor<4096xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32

  // A: 1D -> 3D (M=64, K=32, K is stride-1)
  %a_3d = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 32 + d2)>
    by [<Unmerge{64, 32} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 32] -> [2048]>
    : tensor<2048xf4E2M1FN> to tensor<1x64x32xf4E2M1FN>
  // A: Pad K from 32 to 64
  %a_pad = rock.transform %a_3d by <affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    by [<PassThrough ["g"] at [0] -> ["g"] at [0]>,
        <PassThrough ["m"] at [1] -> ["m"] at [1]>,
        <Pad{0, 32} ["kPad"] at [2] -> ["k"] at [2]>]
    bounds = [1, 64, 64] -> [1, 64, 32]>
    : tensor<1x64x32xf4E2M1FN> to tensor<1x64x64xf4E2M1FN>
  // A: 3D -> 6D
  %a_6d = rock.transform %a_pad by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>,
        <AddDim{1} ["n_block"] at [3] -> [] at []>]
    bounds = [1, 1, 1, 1, 64, 64] -> [1, 64, 64]>
    : tensor<1x64x64xf4E2M1FN> to tensor<1x1x1x1x64x64xf4E2M1FN>
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  // B: simple chain (K=64, N=64, N is stride-1)
  %b_3d = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
    by [<Unmerge{64, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 64] -> [4096]>
    : tensor<4096xf4E2M1FN> to tensor<1x64x64xf4E2M1FN>
  %b_6d = rock.transform %b_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 64, 64] -> [1, 64, 64]>
    : tensor<1x64x64xf4E2M1FN> to tensor<1x1x1x1x64x64xf4E2M1FN>
  %b_tile = rock.blockwise_load %b_6d[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  %sa_tile = rock.blockwise_load %arg2[]
    : tensor<64x1xf8E8M0FNU> -> tensor<64x1xf8E8M0FNU>
  %sb_tile = rock.blockwise_load %arg3[]
    : tensor<64x1xf8E8M0FNU> -> tensor<64x1xf8E8M0FNU>

  %cst = arith.constant dense<0.0> : tensor<64x64xf32>
  %result = rock.blockwise_gemm(%a_tile scaled by %sa_tile, %b_tile scaled by %sb_tile, %cst)
    {quantBlockSize = 64 : i64}
    : tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %arg4 : tensor<4096xf32>
}

// -----

// Test 6: Slice transform on K dimension of matrixA.
// A raw: G=1, M=64, K=128. Slice K [0, 64) -> K_sliced=64.
// After halving K: Slice{0,64} -> Slice{0,32}, lower K 128->64, arg 8192->4096.

// CHECK-LABEL: func.func @test_f4_slice
// CHECK-SAME: (%[[A:.*]]: tensor<4096xi8>,
// A chain: check Slice end halved from 64 to 32
// CHECK: rock.transform %[[A]]
// CHECK: Unmerge{64, 64}
// CHECK: rock.transform
// CHECK: Slice{0, 32}
// CHECK: rock.transform
// CHECK: Unmerge{1, 32}{{.*}}["k_loop", "k_iter"]
// CHECK: rock.blockwise_load {{.*}} -> tensor<64x32xi8>
// CHECK: matrixAKPack = true
func.func @test_f4_slice(
    %arg0: tensor<8192xf4E2M1FN>,
    %arg1: tensor<4096xf4E2M1FN>,
    %arg2: tensor<64x1xf8E8M0FNU>,
    %arg3: tensor<64x1xf8E8M0FNU>,
    %arg4: tensor<4096xf32>) -> tensor<4096xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32

  // A: 1D -> 3D (M=64, K=128, K is stride-1)
  %a_3d = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
    by [<Unmerge{64, 128} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 128] -> [8192]>
    : tensor<8192xf4E2M1FN> to tensor<1x64x128xf4E2M1FN>
  // A: Slice K from 128 to [0, 64)
  %a_slice = rock.transform %a_3d by <affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    by [<PassThrough ["g"] at [0] -> ["g"] at [0]>,
        <PassThrough ["m"] at [1] -> ["m"] at [1]>,
        <Slice{0, 64} ["kSlice"] at [2] -> ["k"] at [2]>]
    bounds = [1, 64, 64] -> [1, 64, 128]>
    : tensor<1x64x128xf4E2M1FN> to tensor<1x64x64xf4E2M1FN>
  // A: 3D -> 6D
  %a_6d = rock.transform %a_slice by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>,
        <AddDim{1} ["n_block"] at [3] -> [] at []>]
    bounds = [1, 1, 1, 1, 64, 64] -> [1, 64, 64]>
    : tensor<1x64x64xf4E2M1FN> to tensor<1x1x1x1x64x64xf4E2M1FN>
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  // B: simple chain (K=64, N=64, N is stride-1)
  %b_3d = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
    by [<Unmerge{64, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 64] -> [4096]>
    : tensor<4096xf4E2M1FN> to tensor<1x64x64xf4E2M1FN>
  %b_6d = rock.transform %b_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 64, 64] -> [1, 64, 64]>
    : tensor<1x64x64xf4E2M1FN> to tensor<1x1x1x1x64x64xf4E2M1FN>
  %b_tile = rock.blockwise_load %b_6d[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  %sa_tile = rock.blockwise_load %arg2[]
    : tensor<64x1xf8E8M0FNU> -> tensor<64x1xf8E8M0FNU>
  %sb_tile = rock.blockwise_load %arg3[]
    : tensor<64x1xf8E8M0FNU> -> tensor<64x1xf8E8M0FNU>

  %cst = arith.constant dense<0.0> : tensor<64x64xf32>
  %result = rock.blockwise_gemm(%a_tile scaled by %sa_tile, %b_tile scaled by %sb_tile, %cst)
    {quantBlockSize = 64 : i64}
    : tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %arg4 : tensor<4096xf32>
}

// -----

// Test 7: Merge transform on K dimension of matrixA.
// A raw: G=1, M=64, k0=4, k1=64 -> Merge{4,64} -> K=256.
// After halving K: Merge last param halved {4,64}->{4,32}, arg 16384->8192.

// CHECK-LABEL: func.func @test_f4_merge
// CHECK-SAME: (%[[A:.*]]: tensor<8192xi8>,
// A chain: check Merge params changed from {4,64} to {4,32}
// CHECK: rock.transform %[[A]]
// CHECK: Unmerge{64, 4, 32}
// CHECK: rock.transform
// CHECK: Merge{4, 32}
// CHECK: rock.transform
// CHECK: Unmerge{4, 32}{{.*}}["k_loop", "k_iter"]
// CHECK: rock.blockwise_load {{.*}} -> tensor<64x32xi8>
// CHECK: matrixAKPack = true
func.func @test_f4_merge(
    %arg0: tensor<16384xf4E2M1FN>,
    %arg1: tensor<4096xf4E2M1FN>,
    %arg2: tensor<64x1xf8E8M0FNU>,
    %arg3: tensor<64x1xf8E8M0FNU>,
    %arg4: tensor<4096xf32>) -> tensor<4096xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32

  // A: 1D -> 4D (M=64, k0=4, k1=64)
  %a_4d = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> ((d1 * 4 + d2) * 64 + d3)>
    by [<Unmerge{64, 4, 64} ["m", "k0", "k1"] at [1, 2, 3] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 4, 64] -> [16384]>
    : tensor<16384xf4E2M1FN> to tensor<1x64x4x64xf4E2M1FN>
  // A: Merge k0,k1 -> K=256
  %a_3d = rock.transform %a_4d by <affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 64, d2 mod 64)>
    by [<PassThrough ["g"] at [0] -> ["g"] at [0]>,
        <PassThrough ["m"] at [1] -> ["m"] at [1]>,
        <Merge{4, 64} ["k"] at [2] -> ["k0", "k1"] at [2, 3]>]
    bounds = [1, 64, 256] -> [1, 64, 4, 64]>
    : tensor<1x64x4x64xf4E2M1FN> to tensor<1x64x256xf4E2M1FN>
  // A: 3D -> 6D
  %a_6d = rock.transform %a_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>,
        <AddDim{1} ["n_block"] at [3] -> [] at []>]
    bounds = [4, 1, 1, 1, 64, 64] -> [1, 64, 256]>
    : tensor<1x64x256xf4E2M1FN> to tensor<4x1x1x1x64x64xf4E2M1FN>
  %a_tile_m = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0]
    : tensor<4x1x1x1x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  // B: simple chain (K=64, N=64, N is stride-1)
  %b_3d_m = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
    by [<Unmerge{64, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 64] -> [4096]>
    : tensor<4096xf4E2M1FN> to tensor<1x64x64xf4E2M1FN>
  %b_6d_m = rock.transform %b_3d_m by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 64, 64] -> [1, 64, 64]>
    : tensor<1x64x64xf4E2M1FN> to tensor<1x1x1x1x64x64xf4E2M1FN>
  %b_tile_m = rock.blockwise_load %b_6d_m[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  %sa_tile_m = rock.blockwise_load %arg2[]
    : tensor<64x1xf8E8M0FNU> -> tensor<64x1xf8E8M0FNU>
  %sb_tile_m = rock.blockwise_load %arg3[]
    : tensor<64x1xf8E8M0FNU> -> tensor<64x1xf8E8M0FNU>

  %cst_m = arith.constant dense<0.0> : tensor<64x64xf32>
  %result_m = rock.blockwise_gemm(%a_tile_m scaled by %sa_tile_m, %b_tile_m scaled by %sb_tile_m, %cst_m)
    {quantBlockSize = 64 : i64}
    : tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %arg4 : tensor<4096xf32>
}

// -----

// Test 8: PassThrough on the K dimension being halved.
// A raw: G=1, M=64, K=64. K is stride-1. A PassThrough sits between
// the initial Unmerge and the final 6D Unmerge, directly on K.
// After halving: PassThrough propagates halving from K=32 down to K=32.

// CHECK-LABEL: func.func @test_f4_passthrough_on_k
// CHECK-SAME: tensor<2048xi8>
// First Unmerge halves K from 64 to 32
// CHECK: rock.transform %{{.*}} by <{{.*}}Unmerge{64, 32}
// PassThrough propagates the halved K dimension
// CHECK: rock.transform %{{.*}} by <{{.*}}PassThrough ["g", "m", "k"]{{.*}}bounds = [1, 64, 32] -> [1, 64, 32]
// 6D Unmerge has k_iter halved to 32
// CHECK: rock.transform %{{.*}} by <{{.*}}Unmerge{1, 32}{{.*}}["k_loop", "k_iter"]
// CHECK: rock.blockwise_load {{.*}} -> tensor<64x32xi8>
// CHECK: matrixAKPack = true
func.func @test_f4_passthrough_on_k(
    %arg0: tensor<4096xf4E2M1FN>,
    %arg1: tensor<4096xf4E2M1FN>,
    %arg2: tensor<64x1xf8E8M0FNU>,
    %arg3: tensor<64x1xf8E8M0FNU>,
    %arg4: tensor<4096xf32>) -> tensor<4096xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32

  // A: 1D -> 3D (M=64, K=64, K is stride-1)
  %a_3d_pt = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
    by [<Unmerge{64, 64} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 64] -> [4096]>
    : tensor<4096xf4E2M1FN> to tensor<1x64x64xf4E2M1FN>
  // A: PassThrough on all dims including K
  %a_pt = rock.transform %a_3d_pt by <affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    by [<PassThrough ["g", "m", "k"] at [0, 1, 2] -> ["g", "m", "k"] at [0, 1, 2]>]
    bounds = [1, 64, 64] -> [1, 64, 64]>
    : tensor<1x64x64xf4E2M1FN> to tensor<1x64x64xf4E2M1FN>
  // A: 3D -> 6D
  %a_6d_pt = rock.transform %a_pt by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>,
        <AddDim{1} ["n_block"] at [3] -> [] at []>]
    bounds = [1, 1, 1, 1, 64, 64] -> [1, 64, 64]>
    : tensor<1x64x64xf4E2M1FN> to tensor<1x1x1x1x64x64xf4E2M1FN>
  %a_tile_pt = rock.blockwise_load %a_6d_pt[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  // B: simple chain (K=64, N=64, N is stride-1)
  %b_3d_pt = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
    by [<Unmerge{64, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 64] -> [4096]>
    : tensor<4096xf4E2M1FN> to tensor<1x64x64xf4E2M1FN>
  %b_6d_pt = rock.transform %b_3d_pt by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 64, 64] -> [1, 64, 64]>
    : tensor<1x64x64xf4E2M1FN> to tensor<1x1x1x1x64x64xf4E2M1FN>
  %b_tile_pt = rock.blockwise_load %b_6d_pt[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  %sa_tile_pt = rock.blockwise_load %arg2[]
    : tensor<64x1xf8E8M0FNU> -> tensor<64x1xf8E8M0FNU>
  %sb_tile_pt = rock.blockwise_load %arg3[]
    : tensor<64x1xf8E8M0FNU> -> tensor<64x1xf8E8M0FNU>

  %cst_pt = arith.constant dense<0.0> : tensor<64x64xf32>
  %result_pt = rock.blockwise_gemm(%a_tile_pt scaled by %sa_tile_pt, %b_tile_pt scaled by %sb_tile_pt, %cst_pt)
    {quantBlockSize = 64 : i64}
    : tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %arg4 : tensor<4096xf32>
}

// -----

// Test 9: Merge + reordering PassThrough (traceHalvingPath validation).
// The PassThrough swaps k0 and k1 positions. The tracer must follow the
// dim remapping to find that k1 (Merge lower idx 1, at reordered pos 2)
// maps through the PassThrough to bottom Unmerge dim 3 (stride-1).
//
// Bottom Unmerge: ["m","k0","k1"] at [1,2,3] -> k1 is stride-1 (last).
// PassThrough swaps: upper [g,m,k1,k0] at [0,1,2,3] -> lower [0,1,2,3]
//   so upper dim 2 (k1) -> lower dim 3, upper dim 3 (k0) -> lower dim 2.
// Merge{4,64} ["k"] -> ["k0","k1"] at [3,2] (k0 at reordered pos 3,
//   k1 at reordered pos 2). K = k0*64 + k1, inner = k1 = stride-1.

// CHECK-LABEL: func.func @test_f4_merge_passthrough_chain
// CHECK-SAME: (%[[A:.*]]: tensor<8192xi8>,
// Bottom Unmerge: k1 halved from 64 to 32
// CHECK: rock.transform %[[A]]
// CHECK: Unmerge{64, 4, 32}
// PassThrough: k1 (upper dim 2) halved, bounds reflect reorder
// CHECK: rock.transform
// CHECK: PassThrough ["k1"] at [2] -> ["k1"] at [3]
// CHECK-SAME: bounds = [1, 64, 32, 4]
// Merge halves k1: {4,64} -> {4,32}
// CHECK: rock.transform
// CHECK: Merge{4, 32}
// 6D Unmerge halves k_iter: {4,64} -> {4,32}
// CHECK: rock.transform
// CHECK: Unmerge{4, 32}{{.*}}["k_loop", "k_iter"]
// CHECK: rock.blockwise_load {{.*}} -> tensor<64x32xi8>
// CHECK: matrixAKPack = true
func.func @test_f4_merge_passthrough_chain(
    %arg0: tensor<16384xf4E2M1FN>,
    %arg1: tensor<4096xf4E2M1FN>,
    %arg2: tensor<64x1xf8E8M0FNU>,
    %arg3: tensor<64x1xf8E8M0FNU>,
    %arg4: tensor<4096xf32>) -> tensor<4096xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32

  // A: 1D -> 4D (M=64, k0=4, k1=64, k1 is stride-1)
  // raw = m * 256 + k0 * 64 + k1
  %a_4d = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> ((d1 * 4 + d2) * 64 + d3)>
    by [<Unmerge{64, 4, 64} ["m", "k0", "k1"] at [1, 2, 3] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 4, 64] -> [16384]>
    : tensor<16384xf4E2M1FN> to tensor<1x64x4x64xf4E2M1FN>
  // A: PassThrough that SWAPS k0 <-> k1 positions.
  // Upper: [g, m, k1, k0] with sizes [1, 64, 64, 4]
  // Lower: [g, m, k0, k1] with sizes [1, 64, 4, 64]
  %a_pt = rock.transform %a_4d by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
    by [<PassThrough ["g"] at [0] -> ["g"] at [0]>,
        <PassThrough ["m"] at [1] -> ["m"] at [1]>,
        <PassThrough ["k1"] at [2] -> ["k1"] at [3]>,
        <PassThrough ["k0"] at [3] -> ["k0"] at [2]>]
    bounds = [1, 64, 64, 4] -> [1, 64, 4, 64]>
    : tensor<1x64x4x64xf4E2M1FN> to tensor<1x64x64x4xf4E2M1FN>
  // A: Merge k0,k1 -> K=256. k0 at reordered pos 3, k1 at pos 2.
  %a_3d = rock.transform %a_pt by <affine_map<(d0, d1, d2) -> (d0, d1, d2 mod 64, d2 floordiv 64)>
    by [<PassThrough ["g"] at [0] -> ["g"] at [0]>,
        <PassThrough ["m"] at [1] -> ["m"] at [1]>,
        <Merge{4, 64} ["k"] at [2] -> ["k0", "k1"] at [3, 2]>]
    bounds = [1, 64, 256] -> [1, 64, 64, 4]>
    : tensor<1x64x64x4xf4E2M1FN> to tensor<1x64x256xf4E2M1FN>
  // A: 3D -> 6D
  %a_6d = rock.transform %a_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>,
        <AddDim{1} ["n_block"] at [3] -> [] at []>]
    bounds = [4, 1, 1, 1, 64, 64] -> [1, 64, 256]>
    : tensor<1x64x256xf4E2M1FN> to tensor<4x1x1x1x64x64xf4E2M1FN>
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0]
    : tensor<4x1x1x1x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  // B: simple chain (K=64, N=64, N is stride-1)
  %b_3d = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
    by [<Unmerge{64, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 64] -> [4096]>
    : tensor<4096xf4E2M1FN> to tensor<1x64x64xf4E2M1FN>
  %b_6d = rock.transform %b_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 64, 64] -> [1, 64, 64]>
    : tensor<1x64x64xf4E2M1FN> to tensor<1x1x1x1x64x64xf4E2M1FN>
  %b_tile = rock.blockwise_load %b_6d[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  %sa_tile = rock.blockwise_load %arg2[]
    : tensor<64x1xf8E8M0FNU> -> tensor<64x1xf8E8M0FNU>
  %sb_tile = rock.blockwise_load %arg3[]
    : tensor<64x1xf8E8M0FNU> -> tensor<64x1xf8E8M0FNU>

  %cst = arith.constant dense<0.0> : tensor<64x64xf32>
  %result = rock.blockwise_gemm(%a_tile scaled by %sa_tile, %b_tile scaled by %sb_tile, %cst)
    {quantBlockSize = 64 : i64}
    : tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %arg4 : tensor<4096xf32>
}
