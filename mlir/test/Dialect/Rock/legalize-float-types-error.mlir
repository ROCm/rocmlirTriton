// Negative tests for rock-legalize-float-types pass.

// RUN: rocmlir-opt -rock-legalize-float-types --split-input-file -verify-diagnostics %s

// Test: Broadcast on the halving dimension should fail.
// A has 64 real elements (M=64, K_raw=1); K is broadcast from 1 to 64.
// K-packing fails (broadcast dim has maxVec=1, no contiguous data).
// D-packing also fails: M=64 is the outer sub-dim in Unmerge{64,1}, so
// the halving path cannot reach the innermost (stride-1) sub-dim (k_raw).

func.func @test_f4_broadcast_on_k_fails(
    %arg0: tensor<64xf4E2M1FN>,
    %arg1: tensor<4096xf4E2M1FN>,
    %arg2: tensor<64x1xf8E8M0FNU>,
    %arg3: tensor<64x1xf8E8M0FNU>,
    %arg4: tensor<4096xf32>) -> tensor<4096xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32

  // A: 1D -> 3D  (G=1, M=64, K_raw=1)
  %a_3d = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 1 + d2)>
    by [<Unmerge{64, 1} ["m", "k_raw"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 1] -> [64]>
    : tensor<64xf4E2M1FN> to tensor<1x64x1xf4E2M1FN>
  // A: Broadcast K_raw from 1 to 64
  %a_3d_b = rock.transform %a_3d by <affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    by [<PassThrough ["g"] at [0] -> ["g"] at [0]>,
        <PassThrough ["m"] at [1] -> ["m"] at [1]>,
        <Broadcast{1} ["k"] at [2] -> ["k_raw"] at [2]>]
    bounds = [1, 64, 64] -> [1, 64, 1]>
    : tensor<1x64x1xf4E2M1FN> to tensor<1x64x64xf4E2M1FN>
  // A: 3D -> 6D
  %a_6d = rock.transform %a_3d_b by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>,
        <AddDim{1} ["n_block"] at [3] -> [] at []>]
    bounds = [1, 1, 1, 1, 64, 64] -> [1, 64, 64]>
    : tensor<1x64x64xf4E2M1FN> to tensor<1x1x1x1x64x64xf4E2M1FN>
  // expected-error @+1 {{could not trace halving path to stride-1 dimension}}
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

// Test: AddDim on the K dimension, with M=1 so D-packing also fails.
// K is created via AddDim (no backing data). M=1 can't be halved (odd).
// Both K and D fail -> error.

func.func @test_f4_adddim_on_k_fails(
    %arg0: tensor<1xf4E2M1FN>,
    %arg1: tensor<4096xf4E2M1FN>,
    %arg2: tensor<1x1xf8E8M0FNU>,
    %arg3: tensor<64x1xf8E8M0FNU>,
    %arg4: tensor<4096xf32>) -> tensor<4096xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32

  // A: 1D -> 2D (just M=1)
  %a_2d = rock.transform %arg0 by <affine_map<(d0, d1) -> (d1)>
    by [<Unmerge{1} ["m"] at [1] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 1] -> [1]>
    : tensor<1xf4E2M1FN> to tensor<1x1xf4E2M1FN>
  // A: Add K=64 via AddDim
  %a_3d = rock.transform %a_2d by <affine_map<(d0, d1, d2) -> (d0, d1)>
    by [<PassThrough ["g"] at [0] -> ["g"] at [0]>,
        <PassThrough ["m"] at [1] -> ["m"] at [1]>,
        <AddDim{64} ["k"] at [2] -> [] at []>]
    bounds = [1, 1, 64] -> [1, 1]>
    : tensor<1x1xf4E2M1FN> to tensor<1x1x64xf4E2M1FN>
  // A: 3D -> 6D
  %a_6d = rock.transform %a_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 1 + d4, d0 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 1} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>,
        <AddDim{1} ["n_block"] at [3] -> [] at []>]
    bounds = [1, 1, 1, 1, 1, 64] -> [1, 1, 64]>
    : tensor<1x1x64xf4E2M1FN> to tensor<1x1x1x1x1x64xf4E2M1FN>
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x1x64xf4E2M1FN> -> tensor<1x64xf4E2M1FN>

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
    : tensor<1x1xf8E8M0FNU> -> tensor<1x1xf8E8M0FNU>
  %sb_tile = rock.blockwise_load %arg3[]
    : tensor<64x1xf8E8M0FNU> -> tensor<64x1xf8E8M0FNU>

  %cst = arith.constant dense<0.0> : tensor<1x64xf32>
  // expected-error @+1 {{max vectorization of both D and K is 1}}
  %result = rock.blockwise_gemm(%a_tile scaled by %sa_tile, %b_tile scaled by %sb_tile, %cst)
    {quantBlockSize = 64 : i64}
    : tensor<1x64xf4E2M1FN> scaled by tensor<1x1xf8E8M0FNU>,
      tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<1x64xf32> -> tensor<1x64xf32>
  return %arg4 : tensor<4096xf32>
}

// Test: G (batch) is the fastest-changing dimension in the raw buffer.
// Layout: raw = m * K*G + k * G + g (G is stride-1 instead of K or N).
// Both K and D (M for A) have non-unit stride -> max vectorization = 1 for both.
// The pass cannot find a dimension to halve for f4 packing.
func.func @test_f4_batch_fastest(
    %a_raw: tensor<12288xf4E2M1FN>,
    %b_raw: tensor<12288xf4E2M1FN>,
    %sa: tensor<64x2xf8E8M0FNU>,
    %sb: tensor<64x2xf8E8M0FNU>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // matrixA: 1D -> 3D with G innermost (stride-1)
  // raw = (m * 64 + k) * 3 + g -> K has stride 3, M has stride 192
  %a_3d = rock.transform %a_raw by <affine_map<(d0, d1, d2) -> ((d1 * 64 + d2) * 3 + d0)>
    by [<Unmerge{64, 64, 3} ["m", "k", "g"] at [1, 2, 0] -> ["raw"] at [0]>]
    bounds = [3, 64, 64] -> [12288]>
    : tensor<12288xf4E2M1FN> to tensor<3x64x64xf4E2M1FN>
  %a_6d = rock.transform %a_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>,
        <AddDim{2} ["n_block"] at [3] -> [] at []>]
    bounds = [1, 3, 1, 2, 64, 64] -> [3, 64, 64]>
    : tensor<3x64x64xf4E2M1FN> to tensor<1x3x1x2x64x64xf4E2M1FN>
  %c0 = arith.constant 0 : i32
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0]
    : tensor<1x3x1x2x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  // matrixB: same G-innermost layout
  %b_3d = rock.transform %b_raw by <affine_map<(d0, d1, d2) -> ((d1 * 64 + d2) * 3 + d0)>
    by [<Unmerge{64, 64, 3} ["k", "n", "g"] at [1, 2, 0] -> ["raw"] at [0]>]
    bounds = [3, 64, 64] -> [12288]>
    : tensor<12288xf4E2M1FN> to tensor<3x64x64xf4E2M1FN>
  %b_6d = rock.transform %b_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 3, 1, 1, 64, 64] -> [3, 64, 64]>
    : tensor<3x64x64xf4E2M1FN> to tensor<1x3x1x1x64x64xf4E2M1FN>
  %b_tile = rock.blockwise_load %b_6d[%c0, %c0, %c0, %c0]
    : tensor<1x3x1x1x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  %cst = arith.constant dense<0.0> : tensor<64x64xf32>
  // expected-error @+1 {{max vectorization of both D and K is 1}}
  %result = rock.blockwise_gemm(%a_tile scaled by %sa, %b_tile scaled by %sb, %cst)
    {quantBlockSize = 32 : i64}
    : tensor<64x64xf4E2M1FN> scaled by tensor<64x2xf8E8M0FNU>,
      tensor<64x64xf4E2M1FN> scaled by tensor<64x2xf8E8M0FNU>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: Odd K dimension (K=33) that is not divisible by 2.
// Even though K is stride-1 in the raw buffer, K=33 is odd.
// For i4 types (4-bit), gcd(33, 128/4) = gcd(33, 32) = 1, so
// max vectorization = 1. M (stride K=33) is also not vectorizable.
// The pass cannot halve any dimension for f4 packing.
func.func @test_f4_odd_k(
    %a_raw: tensor<2112xf4E2M1FN>,
    %b_raw: tensor<2112xf4E2M1FN>,
    %sa: tensor<64x1xf8E8M0FNU>,
    %sb: tensor<64x1xf8E8M0FNU>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // matrixA: K=33 is stride-1 in raw
  %a_3d = rock.transform %a_raw by <affine_map<(d0, d1, d2) -> (d1 * 33 + d2)>
    by [<Unmerge{64, 33} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 33] -> [2112]>
    : tensor<2112xf4E2M1FN> to tensor<1x64x33xf4E2M1FN>
  %a_6d = rock.transform %a_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 33 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 33} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>,
        <AddDim{2} ["n_block"] at [3] -> [] at []>]
    bounds = [1, 1, 1, 2, 64, 33] -> [1, 64, 33]>
    : tensor<1x64x33xf4E2M1FN> to tensor<1x1x1x2x64x33xf4E2M1FN>
  %c0 = arith.constant 0 : i32
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x2x64x33xf4E2M1FN> -> tensor<64x33xf4E2M1FN>

  // matrixB: K=33, N=64
  %b_3d = rock.transform %b_raw by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
    by [<Unmerge{33, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 33, 64] -> [2112]>
    : tensor<2112xf4E2M1FN> to tensor<1x33x64xf4E2M1FN>
  %b_6d = rock.transform %b_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 33 + d4, d3 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 33} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>,
        <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>,
        <AddDim{1} ["m_block"] at [2] -> [] at []>]
    bounds = [1, 1, 1, 1, 33, 64] -> [1, 33, 64]>
    : tensor<1x33x64xf4E2M1FN> to tensor<1x1x1x1x33x64xf4E2M1FN>
  %b_tile = rock.blockwise_load %b_6d[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x33x64xf4E2M1FN> -> tensor<33x64xf4E2M1FN>

  %cst = arith.constant dense<0.0> : tensor<64x64xf32>
  // expected-error @+1 {{max vectorization of both D and K is 1}}
  %result = rock.blockwise_gemm(%a_tile scaled by %sa, %b_tile scaled by %sb, %cst)
    {quantBlockSize = 33 : i64}
    : tensor<64x33xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<33x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: Fusion op (arith.addf) on a 4-bit tensor between blockwise_load and
// blockwise_gemm. f4E2M1FN is an OCP MX storage-only format with no hardware
// arithmetic; fusions on packed 4-bit data are not supported.

func.func @test_f4_fusion_rejected(
    %arg0: tensor<4096xf4E2M1FN>,
    %arg1: tensor<4096xf4E2M1FN>,
    %arg2: tensor<64x1xf8E8M0FNU>,
    %arg3: tensor<64x1xf8E8M0FNU>,
    %arg4: tensor<4096xf32>) -> tensor<4096xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c0 = arith.constant 0 : i32

  %a_3d = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
    by [<Unmerge{64, 64} ["m", "k"] at [1, 2] -> ["raw"] at [0]>,
        <AddDim{1} ["g"] at [0] -> [] at []>]
    bounds = [1, 64, 64] -> [4096]>
    : tensor<4096xf4E2M1FN> to tensor<1x64x64xf4E2M1FN>
  %a_6d = rock.transform %a_3d by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 64 + d5)>
    by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
        <Unmerge{1, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>,
        <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>,
        <AddDim{1} ["n_block"] at [3] -> [] at []>]
    bounds = [1, 1, 1, 1, 64, 64] -> [1, 64, 64]>
    : tensor<1x64x64xf4E2M1FN> to tensor<1x1x1x1x64x64xf4E2M1FN>
  %a_tile = rock.blockwise_load %a_6d[%c0, %c0, %c0, %c0]
    : tensor<1x1x1x1x64x64xf4E2M1FN> -> tensor<64x64xf4E2M1FN>

  // expected-error @+1 {{fusion ops on 4-bit types are not supported}}
  %fused = arith.addf %a_tile, %a_tile : tensor<64x64xf4E2M1FN>

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
  %result = rock.blockwise_gemm(%fused scaled by %sa_tile, %b_tile scaled by %sb_tile, %cst)
    {quantBlockSize = 64 : i64}
    : tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf4E2M1FN> scaled by tensor<64x1xf8E8M0FNU>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %arg4 : tensor<4096xf32>
}
