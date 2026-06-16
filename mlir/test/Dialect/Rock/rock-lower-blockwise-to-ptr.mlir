// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-lower-blockwise-to-ptr | FileCheck %s

// CHECK-LABEL: @test_blockwise_load
// CHECK-SAME: (%[[ARG0:.*]]: tensor<32768xf16>, %[[DST:.*]]: tensor<4096xf16>)
//      CHECK:   %[[TRANS0:.*]] = rock.transform %[[ARG0]] by
//      CHECK:   %[[TRANS1:.*]] = rock.transform %[[TRANS0]] by
//      CHECK:   %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %[[TRANS1]][%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
// The cache modifier on the blockwise_load is propagated to blockwise_load_ptr.
//      CHECK:   %[[RESULT:.*]] = rock.blockwise_load_ptr %[[PTRS]][%[[MASK]]] {cacheModifier = #rock<CacheModifier cs>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
//      CHECK:   rock.blockwise_store_ptr %[[RESULT]] -> %{{.*}}({{.*}}) by  set
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_load
//  CHECK-NOT:   rock.blockwise_store
func.func @test_blockwise_load(%arg0: tensor<32768xf16>, %dst: tensor<4096xf16>) -> tensor<4096xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  // Transforms based similar to a GEMM lowering (for rock.blockwise_load)
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]> : tensor<32768xf16> to tensor<1x256x128xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]> : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>

  %2 = rock.blockwise_load %1[%c0_i32, %c0_i32, %c0_i32, %c1_i32] {cacheModifier = #rock<CacheModifier cs>} : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xf16>

  // Consume the loaded tile so the function is naturally void after lowering.
  %dstView = rock.transform %dst by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf16> to tensor<64x64xf16>
  %3 = rock.blockwise_store %2 -> %dstView by set : tensor<64x64xf16> -> tensor<64x64xf16> -> tensor<4096xf16>

  return %3 : tensor<4096xf16>
}

// -----

// CHECK-LABEL: @test_blockwise_store
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x64xf32>, %[[ARG1:.*]]: tensor<8192xf32>)
//      CHECK:   %[[TRANS0:.*]] = rock.transform %[[ARG1]] by
//      CHECK:   %[[TRANS1:.*]] = rock.transform %[[TRANS0]] by
//      CHECK:   %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %[[TRANS1]][%{{.*}}, %{{.*}}, %{{.*}}] : tensor<1x1x2x64x64xf32> -> tensor<64x64xi32>, tensor<64x64xi1>
//      CHECK:   rock.blockwise_store_ptr %[[ARG0]] -> %[[PTRS]](%[[MASK]]) by  set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>)
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_load
//  CHECK-NOT:   rock.blockwise_store
func.func @test_blockwise_store(%arg0: tensor<64x64xf32>, %arg1: tensor<8192xf32>) -> tensor<8192xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  // Transforms based similar to a GEMM lowering (for rock.blockwise_store)
  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{64, 128} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 128] -> [8192]> : tensor<8192xf32> to tensor<1x64x128xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 2, 64, 64] -> [1, 64, 128]> : tensor<1x64x128xf32> to tensor<1x1x2x64x64xf32>

  %2 = rock.blockwise_store %arg0 -> %1[%c0_i32, %c0_i32, %c1_i32] by set : tensor<64x64xf32> -> tensor<1x1x2x64x64xf32> -> tensor<8192xf32>

  return %2 : tensor<8192xf32>
}

// -----

// CHECK-LABEL: @test_multiple_ops
//       CHECK:   %{{.*}}, %{{.*}} = rock.transforms_to_ptr %{{.*}} : tensor<1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
//       CHECK:   rock.blockwise_load_ptr
//       CHECK:   %{{.*}}, %{{.*}} = rock.transforms_to_ptr %{{.*}} : tensor<1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
//       CHECK:   rock.blockwise_load_ptr
//       CHECK:   %{{.*}}, %{{.*}} = rock.transforms_to_ptr %{{.*}} : tensor<64x64xf32> -> tensor<64x64xi32>, tensor<64x64xi1>
//       CHECK:   rock.blockwise_store_ptr {{.*}} by  set
//   CHECK-NOT:   rock.blockwise_load
//   CHECK-NOT:   rock.blockwise_store
func.func @test_multiple_ops(%arg0: tensor<8192xf16>, %arg1: tensor<8192xf16>, %arg2: tensor<4096xf32>) -> tensor<4096xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %c0_i32 = arith.constant 0 : i32

  // First load (matrix A)
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 128 + d1)> by [<Unmerge{64, 128} ["m", "k"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 128] -> [8192]> : tensor<8192xf16> to tensor<64x128xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d2, d1 * 64 + d3)> by [<Unmerge{1, 64} ["m_block", "m_iter"] at [0, 2] -> ["m"] at [0]>, <Unmerge{2, 64} ["k_block", "k_iter"] at [1, 3] -> ["k"] at [1]>] bounds = [1, 2, 64, 64] -> [64, 128]> : tensor<64x128xf16> to tensor<1x2x64x64xf16>
  %2 = rock.blockwise_load %1[%c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x2x64x64xf16> -> tensor<64x64xf16>

  // Second load (matrix B)
  %3 = rock.transform %arg1 by <affine_map<(d0, d1) -> (d0 * 128 + d1)> by [<Unmerge{64, 128} ["k", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 128] -> [8192]> : tensor<8192xf16> to tensor<64x128xf16>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d2, d1 * 64 + d3)> by [<Unmerge{1, 64} ["k_block", "k_iter"] at [0, 2] -> ["k"] at [0]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [1, 3] -> ["n"] at [1]>] bounds = [1, 2, 64, 64] -> [64, 128]> : tensor<64x128xf16> to tensor<1x2x64x64xf16>
  %5 = rock.blockwise_load %4[%c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x2x64x64xf16> -> tensor<64x64xf16>

  // Computation
  %6 = arith.extf %2 : tensor<64x64xf16> to tensor<64x64xf32>

  // Store result (matrix C)
  %7 = rock.transform %arg2 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf32> to tensor<64x64xf32>
  %8 = rock.blockwise_store %6 -> %7 by set : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>

  return %8 : tensor<4096xf32>
}

// -----

// Store with atomic_add method (verifies storeMethod is preserved)
// CHECK-LABEL: @test_store_atomic_add
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x64xf32>, %[[ARG1:.*]]: tensor<4096xf32>)
//      CHECK:   %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}} : tensor<64x64xf32> -> tensor<64x64xi32>, tensor<64x64xi1>
//      CHECK:   rock.blockwise_store_ptr %[[ARG0]] -> %[[PTRS]](%[[MASK]]) by  atomic_add
//  CHECK-NOT:   rock.blockwise_store
func.func @test_store_atomic_add(%arg0: tensor<64x64xf32>, %arg1: tensor<4096xf32>) -> tensor<4096xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %c0_i32 = arith.constant 0 : i32

  %0 = rock.transform %arg1 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf32> to tensor<64x64xf32>
  %1 = rock.blockwise_store %arg0 -> %0 by atomic_add : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>

  return %1 : tensor<4096xf32>
}

// -----

// Store with atomic_max method (verifies storeMethod is preserved)
// CHECK-LABEL: @test_store_atomic_max
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x64xf32>, %[[ARG1:.*]]: tensor<4096xf32>)
//      CHECK:   rock.blockwise_store_ptr {{.*}} by  atomic_max
//  CHECK-NOT:   rock.blockwise_store
func.func @test_store_atomic_max(%arg0: tensor<64x64xf32>, %arg1: tensor<4096xf32>) -> tensor<4096xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %c0_i32 = arith.constant 0 : i32

  %0 = rock.transform %arg1 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf32> to tensor<64x64xf32>
  %1 = rock.blockwise_store %arg0 -> %0 by atomic_max : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>

  return %1 : tensor<4096xf32>
}

// -----

// Load with i8 element type (verifies element type handling)
// CHECK-LABEL: @test_load_i8
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4096xi8>, %[[DST:.*]]: tensor<4096xi8>)
//      CHECK:   %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}} : tensor<1x64x64xi8> -> tensor<64x64xi32>, tensor<64x64xi1>
//      CHECK:   %[[RESULT:.*]] = rock.blockwise_load_ptr %[[PTRS]][%[[MASK]]] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xi8>
//      CHECK:   rock.blockwise_store_ptr %[[RESULT]] -> %{{.*}}({{.*}}) by  set
//  CHECK-NOT:   rock.blockwise_load
//  CHECK-NOT:   rock.blockwise_store
func.func @test_load_i8(%arg0: tensor<4096xi8>, %dst: tensor<4096xi8>) -> tensor<4096xi8> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %c0_i32 = arith.constant 0 : i32

  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["block"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : tensor<4096xi8> to tensor<1x64x64xi8>
  %1 = rock.blockwise_load %0[%c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x64x64xi8> -> tensor<64x64xi8>

  %dstView = rock.transform %dst by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xi8> to tensor<64x64xi8>
  %2 = rock.blockwise_store %1 -> %dstView by set : tensor<64x64xi8> -> tensor<64x64xi8> -> tensor<4096xi8>

  return %2 : tensor<4096xi8>
}

// -----

// Non-square tile shape (verifies shape preservation)
// CHECK-LABEL: @test_nonsquare_tile
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4096xf16>, %[[DST:.*]]: tensor<4096xf16>)
//      CHECK:   %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}} : tensor<1x32x128xf16> -> tensor<32x128xi32>, tensor<32x128xi1>
//      CHECK:   %[[RESULT:.*]] = rock.blockwise_load_ptr %[[PTRS]][%[[MASK]]] {cacheModifier = #rock<CacheModifier none>} : tensor<32x128xi32>, tensor<32x128xi1> -> tensor<32x128xf16>
//      CHECK:   rock.blockwise_store_ptr %[[RESULT]] -> %{{.*}}({{.*}}) by  set
//  CHECK-NOT:   rock.blockwise_load
//  CHECK-NOT:   rock.blockwise_store
func.func @test_nonsquare_tile(%arg0: tensor<4096xf16>, %dst: tensor<4096xf16>) -> tensor<4096xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %c0_i32 = arith.constant 0 : i32

  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{32, 128} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["block"] at [0] -> [] at []>] bounds = [1, 32, 128] -> [4096]> : tensor<4096xf16> to tensor<1x32x128xf16>
  %1 = rock.blockwise_load %0[%c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x32x128xf16> -> tensor<32x128xf16>

  %dstView = rock.transform %dst by <affine_map<(d0, d1) -> (d0 * 128 + d1)> by [<Unmerge{32, 128} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [32, 128] -> [4096]> : tensor<4096xf16> to tensor<32x128xf16>
  %2 = rock.blockwise_store %1 -> %dstView by set : tensor<32x128xf16> -> tensor<32x128xf16> -> tensor<4096xf16>

  return %2 : tensor<4096xf16>
}

// -----

// Operations inside scf.for (verifies transformation in control flow)
// CHECK-LABEL: @test_inside_scf_for
//      CHECK:   scf.for
//      CHECK:     %{{.*}}, %{{.*}} = rock.transforms_to_ptr
//      CHECK:     rock.blockwise_load_ptr
//      CHECK:   rock.blockwise_store_ptr
//  CHECK-NOT:   rock.blockwise_load
//  CHECK-NOT:   rock.blockwise_store
func.func @test_inside_scf_for(%arg0: tensor<8192xf16>, %dst: tensor<4096xf16>) -> tensor<4096xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst = arith.constant dense<0.0> : tensor<64x64xf16>

  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 64 + d2)> by [<Unmerge{2, 64, 64} ["k_loop", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 64] -> [8192]> : tensor<8192xf16> to tensor<2x64x64xf16>

  %result = scf.for %i = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc = %cst) -> tensor<64x64xf16> : i32 {
    %1 = rock.blockwise_load %0[%i] {cacheModifier = #rock<CacheModifier none>} : tensor<2x64x64xf16> -> tensor<64x64xf16>
    %2 = arith.addf %acc, %1 : tensor<64x64xf16>
    scf.yield %2 : tensor<64x64xf16>
  }

  %dstView = rock.transform %dst by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf16> to tensor<64x64xf16>
  %3 = rock.blockwise_store %result -> %dstView by set : tensor<64x64xf16> -> tensor<64x64xf16> -> tensor<4096xf16>

  return %3 : tensor<4096xf16>
}

// -----

// blockwise_load without source indices
// CHECK-LABEL: @test_no_indices
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4096xf16>, %[[DST:.*]]: tensor<4096xf16>)
//      CHECK:   %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}} : tensor<64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
//      CHECK:   %[[RESULT:.*]] = rock.blockwise_load_ptr %[[PTRS]][%[[MASK]]] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
//      CHECK:   rock.blockwise_store_ptr %[[RESULT]] -> %{{.*}}({{.*}}) by  set
//  CHECK-NOT:   rock.blockwise_load
//  CHECK-NOT:   rock.blockwise_store
func.func @test_no_indices(%arg0: tensor<4096xf16>, %dst: tensor<4096xf16>) -> tensor<4096xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf16> to tensor<64x64xf16>
  %1 = rock.blockwise_load %0 {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xf16> -> tensor<64x64xf16>

  %dstView = rock.transform %dst by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf16> to tensor<64x64xf16>
  %2 = rock.blockwise_store %1 -> %dstView by set : tensor<64x64xf16> -> tensor<64x64xf16> -> tensor<4096xf16>

  return %2 : tensor<4096xf16>
}

// -----

// blockwise_store without extra indices
// CHECK-LABEL: @test_store_no_indices
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x64xf32>, %[[ARG1:.*]]: tensor<4096xf32>)
//      CHECK:   %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}} : tensor<64x64xf32> -> tensor<64x64xi32>, tensor<64x64xi1>
//      CHECK:   rock.blockwise_store_ptr %[[ARG0]] -> %[[PTRS]](%[[MASK]]) by  set
//  CHECK-NOT:   rock.blockwise_store
func.func @test_store_no_indices(%arg0: tensor<64x64xf32>, %arg1: tensor<4096xf32>) -> tensor<4096xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.transform %arg1 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf32> to tensor<64x64xf32>
  %1 = rock.blockwise_store %arg0 -> %0 by set : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>

  return %1 : tensor<4096xf32>
}

// -----

// Verifies that ReturnOpRewritePattern, when triggered by a return whose
// operand is produced by a blockwise_store, also clears the parent function's
// result attributes (the function becomes void).
// CHECK-LABEL: @test_return_clears_res_attrs
// CHECK-SAME: (%{{.*}}: tensor<64x64xf32>, %{{.*}}: tensor<4096xf32>)
// CHECK-SAME: attributes
//      CHECK:   rock.blockwise_store_ptr
//      CHECK:   return{{$}}
//  CHECK-NOT:   res_attrs
//  CHECK-NOT:   rock.prefill
func.func @test_return_clears_res_attrs(%arg0: tensor<64x64xf32>, %arg1: tensor<4096xf32>)
    -> (tensor<4096xf32> {rock.prefill = 0.000000e+00 : f32})
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.transform %arg1 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf32> to tensor<64x64xf32>
  %1 = rock.blockwise_store %arg0 -> %0 by set : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>

  return %1 : tensor<4096xf32>
}

// -----

// Functions without rock.kernel must be left untouched: signature is
// preserved, blockwise_load/store are not lowered, and the return value
// is not stripped.
// CHECK-LABEL: @test_non_kernel_func_untouched
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x64xf32>, %[[ARG1:.*]]: tensor<4096xf32>) -> tensor<4096xf32>
//  CHECK-NOT:   rock.transforms_to_ptr
//  CHECK-NOT:   rock.blockwise_store_ptr
//      CHECK:   %[[RES:.*]] = rock.blockwise_store
//      CHECK:   return %[[RES]] : tensor<4096xf32>
func.func @test_non_kernel_func_untouched(%arg0: tensor<64x64xf32>, %arg1: tensor<4096xf32>) -> tensor<4096xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg1 by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf32> to tensor<64x64xf32>
  %1 = rock.blockwise_store %arg0 -> %0 by set : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>

  return %1 : tensor<4096xf32>
}
