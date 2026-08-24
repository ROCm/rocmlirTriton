// Unit tests for the rocmlirTriton rock-add-triton-metadata pass.
//
// The pass runs right after rock-lower-stores. For each rock.blockwise_gemm
// consumed by a rock.blockwise_store it inspects the store destination's
// vectorization (getMaxVectorization on the fast M and N output dimensions) and
// records a discardable rock.o_transposed attribute: true when M vectorizes
// strictly better than N (column-major / transposed output), false otherwise.

// RUN: rocmlir-opt -rock-add-triton-metadata --mlir-print-local-scope --split-input-file %s 2>&1 | FileCheck %s --implicit-check-not="Unexpected op"

// Row-major destination (last dim contiguous): N vectorizes better than M, so
// the output is not transposed.

// CHECK-LABEL: @row_major
//      CHECK:   rock.blockwise_gemm
// CHECK-SAME:     rock.o_transposed = #rock.o_transposed<false>
func.func @row_major(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>, %dest: tensor<64x64xf32>) attributes {rock.kernel} {
  %g = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %r = rock.blockwise_store %g -> %dest by set
    : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<64x64xf32>
  return
}

// -----

// Transposed destination view: the last logical dim (N) maps to a strided
// coordinate of the raw buffer while M is contiguous, so M vectorizes better
// and the output is transposed (column-major).

#tmapT = #rock.transform_map<affine_map<(m, n) -> (n * 64 + m)> by [<Unmerge{64, 64} ["n", "m"] at [1, 0] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]>

// CHECK-LABEL: @col_major
//      CHECK:   rock.blockwise_gemm
// CHECK-SAME:     rock.o_transposed = #rock.o_transposed<true>
func.func @col_major(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>, %dest_raw: tensor<4096xf32>) attributes {rock.kernel} {
  %g = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %dest = rock.transform %dest_raw by #tmapT : tensor<4096xf32> to tensor<64x64xf32>
  %r = rock.blockwise_store %g -> %dest by set
    : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>
  return
}

// -----

// The store can be reached through epilogue fusion ops (here an arith.addf);
// the consuming store is still found and the metadata attached.

// CHECK-LABEL: @through_fusion
//      CHECK:   rock.blockwise_gemm
// CHECK-SAME:     rock.o_transposed = #rock.o_transposed<false>
func.func @through_fusion(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>, %bias: tensor<64x64xf32>, %dest: tensor<64x64xf32>) attributes {rock.kernel} {
  %g = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %add = arith.addf %g, %bias : tensor<64x64xf32>
  %r = rock.blockwise_store %add -> %dest by set
    : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<64x64xf32>
  return
}

// -----

// A GEMM with no reachable store (its result is returned, e.g. a chained-dot
// head) is left untouched so the downstream default layout is kept.

// CHECK-LABEL: @no_store
//      CHECK:   rock.blockwise_gemm
//  CHECK-NOT:     rock.o_transposed
func.func @no_store(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.kernel} {
  %g = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %g : tensor<64x64xf32>
}

// -----

// A single GEMM result can be written by several stores. The pass measures each
// store's underlying kernel-output buffer and uses the largest one to decide the
// layout. Here the large store (4096 elems) is column-major while the small
// store (2048 elems, a 64x32 slice) is row-major, so the transposed layout of
// the larger store wins.

#multi_big_t = #rock.transform_map<affine_map<(m, n) -> (n * 64 + m)> by [<Unmerge{64, 64} ["n", "m"] at [1, 0] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]>
#multi_slice = #rock.transform_map<affine_map<(m, n) -> (m, n)> by [<PassThrough ["m"] at [0] -> ["m"] at [0]>, <Slice{0, 32} ["n"] at [1] -> ["n"] at [1]>] bounds = [64, 32] -> [64, 64]>
#multi_small_row = #rock.transform_map<affine_map<(m, n) -> (m * 32 + n)> by [<Unmerge{64, 32} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 32] -> [2048]>

// CHECK-LABEL: @multi_store_big_transposed
//      CHECK:   rock.blockwise_gemm
// CHECK-SAME:     rock.o_transposed = #rock.o_transposed<true>
func.func @multi_store_big_transposed(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>, %dest_big_raw: tensor<4096xf32>, %dest_small_raw: tensor<2048xf32>) attributes {rock.kernel} {
  %g = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %dest_big = rock.transform %dest_big_raw by #multi_big_t
    : tensor<4096xf32> to tensor<64x64xf32>
  %r_big = rock.blockwise_store %g -> %dest_big by set
    : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>
  %g_slice = rock.transform %g by #multi_slice
    : tensor<64x64xf32> to tensor<64x32xf32>
  %dest_small = rock.transform %dest_small_raw by #multi_small_row
    : tensor<2048xf32> to tensor<64x32xf32>
  %r_small = rock.blockwise_store %g_slice -> %dest_small by set
    : tensor<64x32xf32> -> tensor<64x32xf32> -> tensor<2048xf32>
  return
}

// -----

// Same multi-store case, layouts swapped: the large store (4096 elems) is row-major and
// the small store (2048 elems) is column-major. The larger store still wins, so
// the recorded layout is not transposed -- proving the choice is driven by store
// size rather than which layout appears first.

#multi_big_row = #rock.transform_map<affine_map<(m, n) -> (m * 64 + n)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]>
#multi_slice2 = #rock.transform_map<affine_map<(m, n) -> (m, n)> by [<PassThrough ["m"] at [0] -> ["m"] at [0]>, <Slice{0, 32} ["n"] at [1] -> ["n"] at [1]>] bounds = [64, 32] -> [64, 64]>
#multi_small_t = #rock.transform_map<affine_map<(m, n) -> (n * 64 + m)> by [<Unmerge{32, 64} ["n", "m"] at [1, 0] -> ["raw"] at [0]>] bounds = [64, 32] -> [2048]>

// CHECK-LABEL: @multi_store_big_row_major
//      CHECK:   rock.blockwise_gemm
// CHECK-SAME:     rock.o_transposed = #rock.o_transposed<false>
func.func @multi_store_big_row_major(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>, %dest_big_raw: tensor<4096xf32>, %dest_small_raw: tensor<2048xf32>) attributes {rock.kernel} {
  %g = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %dest_big = rock.transform %dest_big_raw by #multi_big_row
    : tensor<4096xf32> to tensor<64x64xf32>
  %r_big = rock.blockwise_store %g -> %dest_big by set
    : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>
  %g_slice = rock.transform %g by #multi_slice2
    : tensor<64x64xf32> to tensor<64x32xf32>
  %dest_small = rock.transform %dest_small_raw by #multi_small_t
    : tensor<2048xf32> to tensor<64x32xf32>
  %r_small = rock.blockwise_store %g_slice -> %dest_small by set
    : tensor<64x32xf32> -> tensor<64x32xf32> -> tensor<2048xf32>
  return
}

// -----

// The gemm lives inside an scf.for loop body. Its result is accumulated into the
// loop-carried value and surfaces as the loop result via scf.yield; the store
// happens after the loop. The walk starts at the in-loop gemm, crosses the loop
// boundary (yield -> parent result), finds the transposed store, and tags the
// gemm.

#loop_tmapT = #rock.transform_map<affine_map<(m, n) -> (n * 64 + m)> by [<Unmerge{64, 64} ["n", "m"] at [1, 0] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]>

// CHECK-LABEL: @through_loop
//      CHECK:   scf.for
//      CHECK:     rock.blockwise_gemm
// CHECK-SAME:       rock.o_transposed = #rock.o_transposed<true>
func.func @through_loop(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>, %init: tensor<64x64xf32>, %dest_raw: tensor<4096xf32>) attributes {rock.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %loop = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init)
      -> (tensor<64x64xf32>) {
    %g = rock.blockwise_gemm(%a, %b, %c)
      : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
    %sum = arith.addf %acc, %g : tensor<64x64xf32>
    scf.yield %sum : tensor<64x64xf32>
  }
  %dest = rock.transform %dest_raw by #loop_tmapT
    : tensor<4096xf32> to tensor<64x64xf32>
  %r = rock.blockwise_store %loop -> %dest by set
    : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>
  return
}

// -----

// The OptimizeEpilogue automatic policy marks 16-bit stores when the exposed
// dwordx4 store burst is at least twice the enclosing reduction depth. This
// 64x64 f16 tile with block_size=64 and kDepth=4 hits E=2 exactly, so the pass
// asks Triton's OptimizeEpilogue to retain the paced LDS epilogue.

// CHECK-LABEL: @prefer_lds_epilogue_at_exposure_two
// CHECK-SAME:    rock.prefer_lds_epilogue
func.func @prefer_lds_epilogue_at_exposure_two(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf16>, %dest: tensor<64x64xf16>) attributes {rock.kernel, rock.block_size = 64 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %loop = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c)
      -> (tensor<64x64xf16>) {
    %g = rock.blockwise_gemm(%a, %b, %acc)
      : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf16>
    scf.yield %g : tensor<64x64xf16>
  }
  %r = rock.blockwise_store %loop -> %dest by set
    : tensor<64x64xf16> -> tensor<64x64xf16> -> tensor<64x64xf16>
  return
}

// -----

// Increasing kDepth to 8 drops exposure to E=1, so the marker is absent and
// Triton's OptimizeEpilogue may keep its register bypass.

// CHECK-LABEL: @keep_bypass_when_exposure_low
// CHECK-NOT:     rock.prefer_lds_epilogue
// CHECK:       return
func.func @keep_bypass_when_exposure_low(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf16>, %dest: tensor<64x64xf16>) attributes {rock.kernel, rock.block_size = 64 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %loop = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %c)
      -> (tensor<64x64xf16>) {
    %g = rock.blockwise_gemm(%a, %b, %acc)
      : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf16>
    scf.yield %g : tensor<64x64xf16>
  }
  %r = rock.blockwise_store %loop -> %dest by set
    : tensor<64x64xf16> -> tensor<64x64xf16> -> tensor<64x64xf16>
  return
}

// -----

// The two pieces of metadata are read off different stores when the destination
// ranking and the per-store burst disagree. Both stores below write the same
// f16 gemm result with block_size=64 and kDepth=4:
//
//   * `%big_dest` covers 8192 elements but only receives a 64x16 slice, so its
//     burst is 1024 elems -> E=0.5. It is the layout representative because it
//     is the largest destination, and its view is row-major -> o_transposed is
//     false.
//   * `%small_dest_raw` covers 4096 elements and receives the whole 64x64 tile,
//     so it hits E=2 and wants the LDS epilogue. Its view is column-major, so
//     it would have reported o_transposed = true had it been the representative.
//
// Scoring only the representative would leave the marker off, so the presence
// of rock.prefer_lds_epilogue together with o_transposed<false> shows the
// layout still follows the largest destination while the exposure follows the
// worst burst.

// Row-major 64x16 window of the larger (8192-element) destination.
#worst_burst_big_slice = #rock.transform_map<affine_map<(m, n) -> (m, n)> by [<PassThrough ["m"] at [0] -> ["m"] at [0]>, <Slice{0, 16} ["n"] at [1] -> ["n"] at [1]>] bounds = [64, 16] -> [64, 128]>
// Matching 64x16 slice of the gemm result feeding that store.
#worst_burst_src_slice = #rock.transform_map<affine_map<(m, n) -> (m, n)> by [<PassThrough ["m"] at [0] -> ["m"] at [0]>, <Slice{0, 16} ["n"] at [1] -> ["n"] at [1]>] bounds = [64, 16] -> [64, 64]>
// Column-major view of the smaller (4096-element) destination.
#worst_burst_small_t = #rock.transform_map<affine_map<(m, n) -> (n * 64 + m)> by [<Unmerge{64, 64} ["n", "m"] at [1, 0] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]>

// CHECK-LABEL: @worst_burst_store_decides_epilogue
// CHECK-SAME:    rock.prefer_lds_epilogue
//      CHECK:   rock.blockwise_gemm
// CHECK-SAME:     rock.o_transposed = #rock.o_transposed<false>
func.func @worst_burst_store_decides_epilogue(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf16>, %big_dest: tensor<64x128xf16>, %small_dest_raw: tensor<4096xf16>) attributes {rock.kernel, rock.block_size = 64 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %loop = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c)
      -> (tensor<64x64xf16>) {
    %g = rock.blockwise_gemm(%a, %b, %acc)
      : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf16>
    scf.yield %g : tensor<64x64xf16>
  }
  // Largest destination, smallest burst: wins the layout, loses the exposure.
  %big_view = rock.transform %big_dest by #worst_burst_big_slice
    : tensor<64x128xf16> to tensor<64x16xf16>
  %loop_slice = rock.transform %loop by #worst_burst_src_slice
    : tensor<64x64xf16> to tensor<64x16xf16>
  %r_big = rock.blockwise_store %loop_slice -> %big_view by set
    : tensor<64x16xf16> -> tensor<64x16xf16> -> tensor<64x128xf16>
  // Smaller destination, full-tile burst: decides the epilogue marker.
  %small_view = rock.transform %small_dest_raw by #worst_burst_small_t
    : tensor<4096xf16> to tensor<64x64xf16>
  %r_small = rock.blockwise_store %loop -> %small_view by set
    : tensor<64x64xf16> -> tensor<64x64xf16> -> tensor<4096xf16>
  return
}

// -----

// Explicitly forcing OptimizeEpilogue on disables the automatic LDS-retention
// policy. The temporary tuning attribute is consumed and the marker is not set,
// even though this shape would otherwise hit E=2.

// CHECK-LABEL: @explicit_optimize_epilogue_keeps_bypass
// CHECK-NOT:     rock.use_optimize_epilogue
// CHECK-NOT:     rock.prefer_lds_epilogue
// CHECK:       return
func.func @explicit_optimize_epilogue_keeps_bypass(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf16>, %dest: tensor<64x64xf16>) attributes {rock.kernel, rock.block_size = 64 : i32, rock.use_optimize_epilogue = 1 : i64} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %loop = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c)
      -> (tensor<64x64xf16>) {
    %g = rock.blockwise_gemm(%a, %b, %acc)
      : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16> -> tensor<64x64xf16>
    scf.yield %g : tensor<64x64xf16>
  }
  %r = rock.blockwise_store %loop -> %dest by set
    : tensor<64x64xf16> -> tensor<64x64xf16> -> tensor<64x64xf16>
  return
}

// -----

// Chained GEMMs: the first (head) gemm feeds the second one as an operand, so
// it is an intermediate; the walk bails at the consuming gemm and leaves the
// head untagged. Only the second gemm, whose result reaches the transposed
// output store, is tagged.

#chain_tmapT = #rock.transform_map<affine_map<(m, n) -> (n * 64 + m)> by [<Unmerge{64, 64} ["n", "m"] at [1, 0] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]>

// CHECK-LABEL: @chained_gemm
// The head gemm has no reachable store, so it stays untagged (no attribute on
// the same line, up to the truncf separating the two gemms).
//      CHECK:   rock.blockwise_gemm
//  CHECK-NOT:     rock.o_transposed
//      CHECK:   arith.truncf
// The second gemm reaches the transposed store and is tagged.
//      CHECK:   rock.blockwise_gemm
// CHECK-SAME:     rock.o_transposed = #rock.o_transposed<true>
func.func @chained_gemm(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>, %b2: tensor<64x64xf16>, %c2: tensor<64x64xf32>, %dest_raw: tensor<4096xf32>) attributes {rock.kernel} {
  %g1 = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %g1f16 = arith.truncf %g1 : tensor<64x64xf32> to tensor<64x64xf16>
  %g2 = rock.blockwise_gemm(%g1f16, %b2, %c2)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %dest = rock.transform %dest_raw by #chain_tmapT
    : tensor<4096xf32> to tensor<64x64xf32>
  %r = rock.blockwise_store %g2 -> %dest by set
    : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>
  return
}

// -----

// Chained GEMMs where the head's result ALSO reaches a store through a side
// path (here arith.addf %g1, %g2) in addition to feeding the second gemm. This
// mirrors attention, where the scores gemm feeds the second gemm via P yet also
// reaches the O store through the softmax statistics. The head must still be
// left untagged: because its result feeds a gemm it is an intermediate with no
// meaningful output layout, so the walk bails regardless of the reachable store.

#chain_side_tmapT = #rock.transform_map<affine_map<(m, n) -> (n * 64 + m)> by [<Unmerge{64, 64} ["n", "m"] at [1, 0] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]>

// CHECK-LABEL: @chained_gemm_head_reaches_store_via_side_path
// The head gemm feeds gemm2, so it stays untagged even though %g1 also flows
// into the store through arith.addf.
//      CHECK:   rock.blockwise_gemm
//  CHECK-NOT:     rock.o_transposed
//      CHECK:   arith.truncf
// The second gemm reaches the transposed store and is tagged.
//      CHECK:   rock.blockwise_gemm
// CHECK-SAME:     rock.o_transposed = #rock.o_transposed<true>
func.func @chained_gemm_head_reaches_store_via_side_path(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>, %b2: tensor<64x64xf16>, %c2: tensor<64x64xf32>, %dest_raw: tensor<4096xf32>) attributes {rock.kernel} {
  %g1 = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %g1f16 = arith.truncf %g1 : tensor<64x64xf32> to tensor<64x64xf16>
  %g2 = rock.blockwise_gemm(%g1f16, %b2, %c2)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %side = arith.addf %g1, %g2 : tensor<64x64xf32>
  %dest = rock.transform %dest_raw by #chain_side_tmapT
    : tensor<4096xf32> to tensor<64x64xf32>
  %r = rock.blockwise_store %side -> %dest by set
    : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>
  return
}

// -----

// Sequential stores to the same output thread the previous rock.blockwise_store
// result through resultAlias while keeping the destination view fixed.

#seq_store_row = #rock.transform_map<affine_map<(m, n) -> (m * 64 + n)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]>

// CHECK-LABEL: @sequential_stores_same_output
//      CHECK:   rock.blockwise_gemm
// CHECK-SAME:     rock.o_transposed = #rock.o_transposed<false>
//      CHECK:   rock.blockwise_gemm
// CHECK-SAME:     rock.o_transposed = #rock.o_transposed<false>
func.func @sequential_stores_same_output(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>, %dest_raw: tensor<4096xf32>) attributes {rock.kernel} {
  %g1 = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %dest1 = rock.transform %dest_raw by #seq_store_row
    : tensor<4096xf32> to tensor<64x64xf32>
  %r1 = rock.blockwise_store %g1 -> %dest1 by atomic_add
    : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<64x64xf32>
  %g2 = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %r2 = rock.blockwise_store %g2 -> %dest1 alias %r1 by atomic_add
    : tensor<64x64xf32> -> tensor<64x64xf32> alias tensor<64x64xf32> -> tensor<64x64xf32>
  return
}

// -----

// Non-kernel functions are skipped entirely.

// CHECK-LABEL: @non_kernel
//      CHECK:   rock.blockwise_gemm
//  CHECK-NOT:     rock.o_transposed
func.func @non_kernel(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>, %dest: tensor<64x64xf32>) {
  %g = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %r = rock.blockwise_store %g -> %dest by set
    : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<64x64xf32>
  return
}
