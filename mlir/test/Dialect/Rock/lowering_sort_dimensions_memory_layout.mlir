// RUN: rocmlir-opt --rock-sort-dimensions-memory-layout --split-input-file %s | FileCheck %s

// -----

// Non-kernel function: pass should skip it entirely.
// CHECK-LABEL: func.func @not_a_kernel
// CHECK-NEXT: rock.gemm %arg0 * %arg1
func.func @not_a_kernel(%arg0: tensor<1x8x4xf16>, %arg1: tensor<1x4x8xf16>) -> tensor<1x8x8xf16> {
  %0 = rock.gemm %arg0 * %arg1 : tensor<1x8x4xf16> * tensor<1x4x8xf16> -> tensor<1x8x8xf16>
  return %0 : tensor<1x8x8xf16>
}

// -----

// Gemm with direct block arg inputs (no transform chain).
// traceToBlockArgs finds block args immediately, transformsList is empty → no-op.
// CHECK-LABEL: func.func @gemm_no_transforms
// CHECK: rock.gemm %arg0 * %arg1 :
func.func @gemm_no_transforms(%arg0: tensor<1x8x4xf16>, %arg1: tensor<1x4x8xf16>) -> tensor<1x8x8xf16> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<1x8x4xf16> * tensor<1x4x8xf16> -> tensor<1x8x8xf16>
  return %0 : tensor<1x8x8xf16>
}

// -----

// Gemm with Unmerge transforms that are already in sorted stride order.
// A: flat → [1, 8, 4] (G, M, K) strides [1, 4, 1]
// B: flat → [1, 4, 8] (G, K, N) strides [1, 8, 1]
// sortByMemoryLayout reorders by stride but reorderBatch restores G-first,
// giving the same final layout → pattern returns failure() (no-op for gemm).
// CHECK-LABEL: func.func @gemm_already_sorted
// CHECK: rock.gemm %{{.*}} * %{{.*}} :

#map_gmk_sorted = affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
#sorted_a = #rock.transform_map<#map_gmk_sorted by [
  <AddDim{1} ["g"] at [0] -> [] at []>,
  <Unmerge{8, 4} ["m", "k"] at [1, 2] -> ["raw"] at [0]>
] bounds = [1, 8, 4] -> [32]>

#map_gkn_sorted = affine_map<(d0, d1, d2) -> (d1 * 8 + d2)>
#sorted_b = #rock.transform_map<#map_gkn_sorted by [
  <AddDim{1} ["g"] at [0] -> [] at []>,
  <Unmerge{4, 8} ["k", "n"] at [1, 2] -> ["raw"] at [0]>
] bounds = [1, 4, 8] -> [32]>

func.func @gemm_already_sorted(%arg0: tensor<32xf16>, %arg1: tensor<32xf16>) -> tensor<1x8x8xf16> attributes {rock.kernel} {
  %a = rock.transform %arg0 by #sorted_a : tensor<32xf16> to tensor<1x8x4xf16>
  %b = rock.transform %arg1 by #sorted_b : tensor<32xf16> to tensor<1x4x8xf16>
  %0 = rock.gemm %a * %b : tensor<1x8x4xf16> * tensor<1x4x8xf16> -> tensor<1x8x8xf16>
  return %0 : tensor<1x8x8xf16>
}

// -----

// Gemm where B has non-sorted strides (physical memory is GNK, not GKN).
// B: flat → Unmerge to [1, 8, 4] (G, N, K) → PassThrough swap to [1, 4, 8] (G, K, N)
// Effective strides for [G, K, N]: [1, 1, 4] → N > K, needs sort.
// After sort + reorderBatch: layout becomes [G, N, K], B is marked transposed.
// CHECK-LABEL: func.func @gemm_b_needs_sort
// CHECK: rock.gemm %{{.*}} * tr %{{.*}} :

#map_gnk_phys = affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
#unmerge_gnk = #rock.transform_map<#map_gnk_phys by [
  <AddDim{1} ["g"] at [0] -> [] at []>,
  <Unmerge{8, 4} ["n", "k"] at [1, 2] -> ["raw"] at [0]>
] bounds = [1, 8, 4] -> [32]>

#map_swap_nk = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#swap_to_gkn = #rock.transform_map<#map_swap_nk by [
  <PassThrough ["d0", "d2", "d1"] at [0, 1, 2] -> ["d0", "d2", "d1"] at [0, 2, 1]>
] bounds = [1, 4, 8] -> [1, 8, 4]>

#map_gmk_b = affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
#sorted_a_b = #rock.transform_map<#map_gmk_b by [
  <AddDim{1} ["g"] at [0] -> [] at []>,
  <Unmerge{8, 4} ["m", "k"] at [1, 2] -> ["raw"] at [0]>
] bounds = [1, 8, 4] -> [32]>

func.func @gemm_b_needs_sort(%arg0: tensor<32xf16>, %arg1: tensor<32xf16>) -> tensor<1x8x8xf16> attributes {rock.kernel} {
  %a = rock.transform %arg0 by #sorted_a_b : tensor<32xf16> to tensor<1x8x4xf16>
  %b_phys = rock.transform %arg1 by #unmerge_gnk : tensor<32xf16> to tensor<1x8x4xf16>
  %b = rock.transform %b_phys by #swap_to_gkn : tensor<1x8x4xf16> to tensor<1x4x8xf16>
  %0 = rock.gemm %a * %b : tensor<1x8x4xf16> * tensor<1x4x8xf16> -> tensor<1x8x8xf16>
  return %0 : tensor<1x8x8xf16>
}

// -----

// Gemm where A has non-sorted strides (physical memory is GKM, not GMK).
// A: flat → Unmerge to [1, 4, 8] (G, K, M) → PassThrough swap to [1, 8, 4] (G, M, K)
// Effective strides for [G, M, K]: [1, 1, 8] → K > M, needs sort.
// After sort + reorderBatch: layout becomes [G, K, M], A is marked transposed.
// CHECK-LABEL: func.func @gemm_a_needs_sort
// CHECK: rock.gemm tr %{{.*}} * %{{.*}} :

#map_gkm_phys = affine_map<(d0, d1, d2) -> (d1 * 8 + d2)>
#unmerge_gkm = #rock.transform_map<#map_gkm_phys by [
  <AddDim{1} ["g"] at [0] -> [] at []>,
  <Unmerge{4, 8} ["k", "m"] at [1, 2] -> ["raw"] at [0]>
] bounds = [1, 4, 8] -> [32]>

#map_swap_km = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#swap_to_gmk = #rock.transform_map<#map_swap_km by [
  <PassThrough ["d0", "d2", "d1"] at [0, 1, 2] -> ["d0", "d2", "d1"] at [0, 2, 1]>
] bounds = [1, 8, 4] -> [1, 4, 8]>

#map_gkn_a = affine_map<(d0, d1, d2) -> (d1 * 8 + d2)>
#sorted_b_a = #rock.transform_map<#map_gkn_a by [
  <AddDim{1} ["g"] at [0] -> [] at []>,
  <Unmerge{4, 8} ["k", "n"] at [1, 2] -> ["raw"] at [0]>
] bounds = [1, 4, 8] -> [32]>

func.func @gemm_a_needs_sort(%arg0: tensor<32xf16>, %arg1: tensor<32xf16>) -> tensor<1x8x8xf16> attributes {rock.kernel} {
  %a_phys = rock.transform %arg0 by #unmerge_gkm : tensor<32xf16> to tensor<1x4x8xf16>
  %a = rock.transform %a_phys by #swap_to_gmk : tensor<1x4x8xf16> to tensor<1x8x4xf16>
  %b = rock.transform %arg1 by #sorted_b_a : tensor<32xf16> to tensor<1x4x8xf16>
  %0 = rock.gemm %a * %b : tensor<1x8x4xf16> * tensor<1x4x8xf16> -> tensor<1x8x8xf16>
  return %0 : tensor<1x8x8xf16>
}

// -----

// 2D gemm (no batch dim). Rank-2 inputs trigger the layout drop-G path
// (layout.size()==3, tensorRank==2 → layout = {M, K} / {K, N}).
// Both A and B have sorted strides → no change.
// CHECK-LABEL: func.func @gemm_2d_no_batch
// CHECK: rock.gemm %{{.*}} * %{{.*}} :

#map_mk = affine_map<(d0, d1) -> (d0 * 4 + d1)>
#unmerge_mk = #rock.transform_map<#map_mk by [
  <Unmerge{8, 4} ["m", "k"] at [0, 1] -> ["raw"] at [0]>
] bounds = [8, 4] -> [32]>

#map_kn = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#unmerge_kn = #rock.transform_map<#map_kn by [
  <Unmerge{4, 8} ["k", "n"] at [0, 1] -> ["raw"] at [0]>
] bounds = [4, 8] -> [32]>

func.func @gemm_2d_no_batch(%arg0: tensor<32xf16>, %arg1: tensor<32xf16>) -> tensor<8x8xf16> attributes {rock.kernel} {
  %a = rock.transform %arg0 by #unmerge_mk : tensor<32xf16> to tensor<8x4xf16>
  %b = rock.transform %arg1 by #unmerge_kn : tensor<32xf16> to tensor<4x8xf16>
  %0 = rock.gemm %a * %b : tensor<8x4xf16> * tensor<4x8xf16> -> tensor<8x8xf16>
  return %0 : tensor<8x8xf16>
}

// -----

// Tracing through elementwise ops: gemm A input comes through arith.mulf.
// The pass traces through the shape-preserving mulf to find the block arg's
// transforms. Both mulf operands have the same transform chain, so it resolves.
// Transforms are already sorted → gemm is unchanged.
// CHECK-LABEL: func.func @gemm_trace_through_elementwise
// CHECK: arith.mulf
// CHECK: rock.gemm %{{.*}} * %{{.*}} :
#map_gmk_ew = affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
#unmerge_gmk_ew = #rock.transform_map<#map_gmk_ew by [
  <AddDim{1} ["g"] at [0] -> [] at []>,
  <Unmerge{8, 4} ["m", "k"] at [1, 2] -> ["raw"] at [0]>
] bounds = [1, 8, 4] -> [32]>

#map_gkn_ew = affine_map<(d0, d1, d2) -> (d1 * 8 + d2)>
#sorted_b_ew = #rock.transform_map<#map_gkn_ew by [
  <AddDim{1} ["g"] at [0] -> [] at []>,
  <Unmerge{4, 8} ["k", "n"] at [1, 2] -> ["raw"] at [0]>
] bounds = [1, 4, 8] -> [32]>

func.func @gemm_trace_through_elementwise(%arg0: tensor<32xf16>, %arg1: tensor<32xf16>, %arg2: tensor<32xf16>) -> tensor<1x8x8xf16> attributes {rock.kernel} {
  %a_raw = rock.transform %arg0 by #unmerge_gmk_ew : tensor<32xf16> to tensor<1x8x4xf16>
  %a_scale = rock.transform %arg2 by #unmerge_gmk_ew : tensor<32xf16> to tensor<1x8x4xf16>
  %a = arith.mulf %a_raw, %a_scale : tensor<1x8x4xf16>
  %b = rock.transform %arg1 by #sorted_b_ew : tensor<32xf16> to tensor<1x4x8xf16>
  %0 = rock.gemm %a * %b : tensor<1x8x4xf16> * tensor<1x4x8xf16> -> tensor<1x8x8xf16>
  return %0 : tensor<1x8x8xf16>
}

// -----

// Tracing through elementwise with non-sorted B to verify the pass can both
// trace through elementwise ops on A AND sort B.
// CHECK-LABEL: func.func @gemm_elementwise_and_sort
// CHECK: arith.mulf
// CHECK: rock.gemm %{{.*}} * tr %{{.*}} :

#map_gmk_es = affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
#unmerge_gmk_es = #rock.transform_map<#map_gmk_es by [
  <AddDim{1} ["g"] at [0] -> [] at []>,
  <Unmerge{8, 4} ["m", "k"] at [1, 2] -> ["raw"] at [0]>
] bounds = [1, 8, 4] -> [32]>

#map_gnk_es = affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
#unmerge_gnk_es = #rock.transform_map<#map_gnk_es by [
  <AddDim{1} ["g"] at [0] -> [] at []>,
  <Unmerge{8, 4} ["n", "k"] at [1, 2] -> ["raw"] at [0]>
] bounds = [1, 8, 4] -> [32]>

#map_swap_es = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#swap_to_gkn_es = #rock.transform_map<#map_swap_es by [
  <PassThrough ["d0", "d2", "d1"] at [0, 1, 2] -> ["d0", "d2", "d1"] at [0, 2, 1]>
] bounds = [1, 4, 8] -> [1, 8, 4]>

func.func @gemm_elementwise_and_sort(%arg0: tensor<32xf16>, %arg1: tensor<32xf16>, %arg2: tensor<32xf16>) -> tensor<1x8x8xf16> attributes {rock.kernel} {
  %a_raw = rock.transform %arg0 by #unmerge_gmk_es : tensor<32xf16> to tensor<1x8x4xf16>
  %a_scale = rock.transform %arg2 by #unmerge_gmk_es : tensor<32xf16> to tensor<1x8x4xf16>
  %a = arith.mulf %a_raw, %a_scale : tensor<1x8x4xf16>
  %b_phys = rock.transform %arg1 by #unmerge_gnk_es : tensor<32xf16> to tensor<1x8x4xf16>
  %b = rock.transform %b_phys by #swap_to_gkn_es : tensor<1x8x4xf16> to tensor<1x4x8xf16>
  %0 = rock.gemm %a * %b : tensor<1x8x4xf16> * tensor<1x4x8xf16> -> tensor<1x8x8xf16>
  return %0 : tensor<1x8x8xf16>
}

// -----

// Conv with filter layout ["g", "c", "k", "0", "1"] that needs sorting.
// Filter: flat → Unmerge [1, 8, 4, 1, 1] (G, K, C, Y, X) → PassThrough swap
//         to [1, 4, 8, 1, 1] with layout ["g", "c", "k", "0", "1"].
// Effective strides: G=1, C=1(was dim2), K=4(was dim1), Y=1, X=1.
// Sort puts K first (highest stride) → filter_layout = ["k", "g", "c", "0", "1"].
// Input is a direct block arg (no transforms) → input is unchanged.
// CHECK-LABEL: func.func @conv_filter_needs_sort
// CHECK: rock.conv
// CHECK-SAME: filter_layout = ["k", "g", "c", "0", "1"]

#map_filter_phys = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 4 + d2)>
#unmerge_filter_phys = #rock.transform_map<#map_filter_phys by [
  <AddDim{1} ["g"] at [0] -> [] at []>,
  <Unmerge{8, 4} ["k", "c"] at [1, 2] -> ["raw"] at [0]>,
  <AddDim{1} ["y"] at [3] -> [] at []>,
  <AddDim{1} ["x"] at [4] -> [] at []>
] bounds = [1, 8, 4, 1, 1] -> [32]>

#map_filter_swap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d3, d4)>
#swap_ck = #rock.transform_map<#map_filter_swap by [
  <PassThrough ["d0", "d2", "d1", "d3", "d4"] at [0, 1, 2, 3, 4] -> ["d0", "d2", "d1", "d3", "d4"] at [0, 2, 1, 3, 4]>
] bounds = [1, 4, 8, 1, 1] -> [1, 8, 4, 1, 1]>

func.func @conv_filter_needs_sort(%filt: tensor<32xf16>, %input: tensor<2x2x2x1x4xf16>) -> tensor<2x2x2x1x8xf16> attributes {rock.kernel} {
  %f_phys = rock.transform %filt by #unmerge_filter_phys : tensor<32xf16> to tensor<1x8x4x1x1xf16>
  %f = rock.transform %f_phys by #swap_ck : tensor<1x8x4x1x1xf16> to tensor<1x4x8x1x1xf16>
  %0 = rock.conv(%f, %input) {
    filter_layout = ["g", "c", "k", "0", "1"],
    input_layout = ["ni", "0i", "1i", "gi", "ci"],
    output_layout = ["no", "0o", "1o", "go", "ko"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index],
    dilations = [1 : index, 1 : index]
  } : tensor<1x4x8x1x1xf16>, tensor<2x2x2x1x4xf16> -> tensor<2x2x2x1x8xf16>
  return %0 : tensor<2x2x2x1x8xf16>
}

// -----

// Attention without LSE result. Verifies no crash on the optional lse null check.
// The inputs are direct block args (no transforms) → pass is a no-op.
// CHECK-LABEL: func.func @attention_no_lse
// CHECK: rock.attention
// CHECK: -> tensor<1x8x4xf16>
func.func @attention_no_lse(%q: tensor<1x8x4xf16>, %k: tensor<1x4x8xf16>, %v: tensor<1x8x4xf16>) -> tensor<1x8x4xf16> attributes {rock.kernel} {
  %0 = rock.attention{
    qk = %q * %k : tensor<1x8x4xf16>, tensor<1x4x8xf16>
    softmax(qk) * %v : tensor<1x8x4xf16>
  } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 1 : i32} -> tensor<1x8x4xf16>
  return %0 : tensor<1x8x4xf16>
}

// -----

// Attention with LSE result. Verifies the pass preserves both result types.
// CHECK-LABEL: func.func @attention_with_lse
// CHECK: rock.attention
// CHECK: -> tensor<1x8x4xf16>, tensor<1x8xf16>
func.func @attention_with_lse(%q: tensor<1x8x4xf16>, %k: tensor<1x4x8xf16>, %v: tensor<1x8x4xf16>) -> (tensor<1x8x4xf16>, tensor<1x8xf16>) attributes {rock.kernel} {
  %0:2 = rock.attention{
    qk = %q * %k : tensor<1x8x4xf16>, tensor<1x4x8xf16>
    softmax(qk) * %v : tensor<1x8x4xf16>
  } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 1 : i32} -> tensor<1x8x4xf16>, tensor<1x8xf16>
  return %0#0, %0#1 : tensor<1x8x4xf16>, tensor<1x8xf16>
}

// -----

// Attention where Q has non-sorted strides. The pass should sort Q and mark
// it transposed.
// CHECK-LABEL: func.func @attention_q_needs_sort
// CHECK: rock.attention
// CHECK: tr %{{.*}} * %{{.*}}

#map_gkm_attn = affine_map<(d0, d1, d2) -> (d1 * 8 + d2)>
#unmerge_gkm_attn = #rock.transform_map<#map_gkm_attn by [
  <AddDim{1} ["g"] at [0] -> [] at []>,
  <Unmerge{4, 8} ["k", "m"] at [1, 2] -> ["raw"] at [0]>
] bounds = [1, 4, 8] -> [32]>

#map_swap_attn = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#swap_to_gmk_attn = #rock.transform_map<#map_swap_attn by [
  <PassThrough ["d0", "d2", "d1"] at [0, 1, 2] -> ["d0", "d2", "d1"] at [0, 2, 1]>
] bounds = [1, 8, 4] -> [1, 4, 8]>

func.func @attention_q_needs_sort(%arg0: tensor<32xf16>, %k: tensor<1x4x8xf16>, %v: tensor<1x8x4xf16>) -> tensor<1x8x4xf16> attributes {rock.kernel} {
  %q_phys = rock.transform %arg0 by #unmerge_gkm_attn : tensor<32xf16> to tensor<1x4x8xf16>
  %q = rock.transform %q_phys by #swap_to_gmk_attn : tensor<1x4x8xf16> to tensor<1x8x4xf16>
  %0 = rock.attention{
    qk = %q * %k : tensor<1x8x4xf16>, tensor<1x4x8xf16>
    softmax(qk) * %v : tensor<1x8x4xf16>
  } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 1 : i32} -> tensor<1x8x4xf16>
  return %0 : tensor<1x8x4xf16>
}

// -----

// CHECK-LABEL: func.func @test_conv
// CHECK: %[[b:.*]] = rock.transform %{{.*}} : tensor<2x1x16x160x160xf16> to tensor<2x160x160x1x16xf16>
// CHECK: rock.conv(%{{.*}}, %[[b]])
func.func @test_conv(%arg0: tensor<2304xf16>, %arg1: tensor<1638400xf16>) -> tensor<2x1x16x160x160xf16> attributes {rock.kernel} {
  %3 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 3 + d1) * 3 + d2) * 16 + d3)> by [<Unmerge{16, 3, 3, 16} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [16, 3, 3, 16] -> [2304]> : tensor<2304xf16> to tensor<16x3x3x16xf16>
  %4 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((((((d0 * 2 + d1) * 5 + d2) * 5 + d3) * 32 + d4) * 32 + d5) * 16 + d6)> by [<Unmerge{2, 2, 5, 5, 32, 32, 16} ["exp0", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6"] at [0, 1, 2, 3, 4, 5, 6] -> ["dim0"] at [0]>] bounds = [2, 2, 5, 5, 32, 32, 16] -> [1638400]> : tensor<1638400xf16> to tensor<2x2x5x5x32x32x16xf16>
  %5 = rock.transform %4 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d6, d3, d5, d2)> by [<PassThrough ["dim0", "dim1", "dim6", "dim4", "dim2", "dim5", "dim3"] at [0, 1, 2, 3, 4, 5, 6] -> ["dim0", "dim1", "dim6", "dim4", "dim2", "dim5", "dim3"] at [0, 1, 6, 4, 2, 5, 3]>] bounds = [2, 2, 16, 32, 5, 32, 5] -> [2, 2, 5, 5, 32, 32, 16]> : tensor<2x2x5x5x32x32x16xf16> to tensor<2x2x16x32x5x32x5xf16>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + 1, d2, d3, d4, d5, d6)> by [<Slice{0, 2, 1, 2, 0, 16, 0, 32, 0, 5, 0, 32, 0, 5} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced", "dim4_sliced", "dim5_sliced", "dim6_sliced"] at [0, 1, 2, 3, 4, 5, 6] -> ["dim0", "dim1", "dim2", "dim3", "dim4", "dim5", "dim6"] at [0, 1, 2, 3, 4, 5, 6]>] bounds = [2, 1, 16, 32, 5, 32, 5] -> [2, 2, 16, 32, 5, 32, 5]> : tensor<2x2x16x32x5x32x5xf16> to tensor<2x1x16x32x5x32x5xf16>
  %7 = rock.transform %6 by <affine_map<(d0, d1, d2, d3) -> (d0, 0, d1, d2 floordiv 5, d2 mod 5, d3 floordiv 5, d3 mod 5)> by [<Merge{2, 1} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <Merge{32, 5} ["dim2"] at [2] -> ["col3", "col4"] at [3, 4]>, <Merge{32, 5} ["dim3"] at [3] -> ["col5", "col6"] at [5, 6]>] bounds = [2, 16, 160, 160] -> [2, 1, 16, 32, 5, 32, 5]> : tensor<2x1x16x32x5x32x5xf16> to tensor<2x16x160x160xf16>
  %8 = rock.transform %7 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 16 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 16} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [2, 1, 16, 160, 160] -> [2, 16, 160, 160]> : tensor<2x16x160x160xf16> to tensor<2x1x16x160x160xf16>
  %9 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 16 + d1, d2, d3, d4)> by [<PassThrough ["y", "x", "c"] at [2, 3, 4] -> ["y", "x", "c"] at [1, 2, 3]>, <Unmerge{1, 16} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 16, 3, 3, 16] -> [16, 3, 3, 16]> : tensor<16x3x3x16xf16> to tensor<1x16x3x3x16xf16>
  %10 = rock.conv(%9, %8) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "y", "x", "c"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    padding = [1 : index, 1 : index, 1 : index, 1 : index],
    strides = [1 : index, 1 : index]
  } : tensor<1x16x3x3x16xf16>, tensor<2x1x16x160x160xf16> -> tensor<2x1x16x160x160xf16>
  return %10 : tensor<2x1x16x160x160xf16>
}

// -----

// CHECK-LABEL: func.func @test_attention
// CHECK: %[[a:.*]] = rock.transform %{{.*}} : tensor<16x1x32xf16> to tensor<1x16x32xf16>
// CHECK: %[[b:.*]] = rock.transform %{{.*}} : tensor<64x1x16xf16> to tensor<1x64x16xf16>
// CHECK: rock.attention
// CHECK-NEXT: qk = tr %[[a]] * tr %[[b]]
// CHECK: perf_config = "attn:v1:128,128,16,1,1,4,0,4,1,0,0"
func.func @test_attention(%arg0: tensor<1024xf16>, %arg1: tensor<1024xf16>, %arg2: tensor<512xf16>) -> tensor<1x32x8xf16> attributes {rock.kernel} {
  %0 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 8 + d1) * 64 + d2)> by [<Unmerge{1, 8, 64} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 8, 64] -> [512]> : tensor<512xf16> to tensor<1x8x64xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [1, 64, 8] -> [1, 8, 64]> : tensor<1x8x64xf16> to tensor<1x64x8xf16>
  %2 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 16 + d2)> by [<Unmerge{1, 64, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 64, 16] -> [1024]> : tensor<1024xf16> to tensor<1x64x16xf16>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [1, 16, 64] -> [1, 64, 16]> : tensor<1x64x16xf16> to tensor<1x16x64xf16>
  %4 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 32 + d2)> by [<Unmerge{1, 32, 32} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 32, 32] -> [1024]> : tensor<1024xf16> to tensor<1x32x32xf16>
  %5 = rock.transform %4 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [1, 32, 32] -> [1, 32, 32]> : tensor<1x32x32xf16> to tensor<1x32x32xf16>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<Slice{0, 1, 0, 32, 0, 16} ["dim0_sliced", "dim1_sliced", "dim2_sliced"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [1, 32, 16] -> [1, 32, 32]> : tensor<1x32x32xf16> to tensor<1x32x16xf16>
  %7 = rock.attention{
    qk = %6 * %3 : tensor<1x32x16xf16>, tensor<1x16x64xf16>
    qk = elementwise {
  ^bb0(%arg3: tensor<1x32x64xf16>, %arg4: tensor<1x32x64xf16>):
    rock.yield
  }
    softmax(qk) * %1 : tensor<1x64x8xf16>
  } {firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, perf_config = "attn:v1:128,128,16,1,1,4,0,4,1,0,0"} -> tensor<1x32x8xf16>
  return %7 : tensor<1x32x8xf16>
}

// -----

// CHECK-LABEL: func.func @test_gemm
// CHECK: %[[a:.*]] = rock.transform %{{.*}} : tensor<2x320x4096xf16> to tensor<2x4096x320xf16>
// CHECK: rock.gemm %[[a]] * %{{.*}}
func.func @test_gemm(%arg0: tensor<5242880xf16>, %arg1: tensor<409600xf16>) -> tensor<2x4096x640xf16> attributes {rock.kernel} {
  %3 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 320 + d1) * 640 + d2)> by [<Unmerge{2, 320, 640} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 320, 640] -> [409600]> : tensor<409600xf16> to tensor<2x320x640xf16>
  %4 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 4096 + d1) * 640 + d2)> by [<Unmerge{2, 4096, 640} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 4096, 640] -> [5242880]> : tensor<5242880xf16> to tensor<2x4096x640xf16>
  %5 = rock.transform %4 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [2, 640, 4096] -> [2, 4096, 640]> : tensor<2x4096x640xf16> to tensor<2x640x4096xf16>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<Slice{0, 2, 0, 320, 0, 4096} ["dim0_sliced", "dim1_sliced", "dim2_sliced"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [2, 320, 4096] -> [2, 640, 4096]> : tensor<2x640x4096xf16> to tensor<2x320x4096xf16>
  %7 = rock.gemm tr %6 * %3 : tensor<2x320x4096xf16> * tensor<2x320x640xf16> -> tensor<2x4096x640xf16>
  return %7 : tensor<2x4096x640xf16>
}

// -----

// CHECK-LABEL: func.func @test_mlir_slice_sigmoid_mul_convolution
// CHECK: rock.conv
func.func @test_mlir_slice_sigmoid_mul_convolution(%arg0: tensor<1638400xf16>, %arg1: tensor<147456xf16>) -> tensor<1x1x128x80x80xf16> attributes {rock.kernel} {
  %cst = arith.constant 1.000000e+00 : f16
  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 3 + d1) * 3 + d2) * 128 + d3)> by [<Unmerge{128, 3, 3, 128} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [128, 3, 3, 128] -> [147456]> : tensor<147456xf16> to tensor<128x3x3x128xf16>
  %1 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> ((d1 * 80 + d2) * 256 + d3)> by [<Unmerge{80, 80, 256} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 80, 80, 256] -> [1638400]> : tensor<1638400xf16> to tensor<1x80x80x256xf16>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)> by [<PassThrough ["dim0", "dim3", "dim1", "dim2"] at [0, 1, 2, 3] -> ["dim0", "dim3", "dim1", "dim2"] at [0, 3, 1, 2]>] bounds = [1, 256, 80, 80] -> [1, 80, 80, 256]> : tensor<1x80x80x256xf16> to tensor<1x256x80x80xf16>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 + 128, d2, d3)> by [<Slice{0, 1, 128, 256, 0, 80, 0, 80} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3]>] bounds = [1, 128, 80, 80] -> [1, 256, 80, 80]> : tensor<1x256x80x80xf16> to tensor<1x128x80x80xf16>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 128} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [128, 80, 80] -> [1, 128, 80, 80]> : tensor<1x128x80x80xf16> to tensor<128x80x80xf16>
  %init = tensor.empty() : tensor<128x80x80xf16>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<128x80x80xf16>) outs(%init : tensor<128x80x80xf16>) {
  ^bb0(%in: f16, %out: f16):
    %16 = arith.negf %in : f16
    %17 = math.exp %16 : f16
    %18 = arith.addf %17, %cst : f16
    %19 = arith.divf %cst, %18 : f16
    %20 = arith.mulf %in, %19 : f16
    linalg.yield %20 : f16
  } -> tensor<128x80x80xf16>
  %t5 = rock.transform %5 by <affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)> by [<Unmerge{128} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 128, 80, 80] -> [128, 80, 80]> : tensor<128x80x80xf16> to tensor<1x128x80x80xf16>
  %6 = rock.transform %t5 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 128 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 128} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [1, 1, 128, 80, 80] -> [1, 128, 80, 80]> : tensor<1x128x80x80xf16> to tensor<1x1x128x80x80xf16>
  %7 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 128 + d1, d2, d3, d4)> by [<PassThrough ["y", "x", "c"] at [2, 3, 4] -> ["y", "x", "c"] at [1, 2, 3]>, <Unmerge{1, 128} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 128, 3, 3, 128] -> [128, 3, 3, 128]> : tensor<128x3x3x128xf16> to tensor<1x128x3x3x128xf16>
  %10 = rock.conv(%7, %6) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "y", "x", "c"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    padding = [1 : index, 1 : index, 1 : index, 1 : index],
    strides = [1 : index, 1 : index]
  } : tensor<1x128x3x3x128xf16>, tensor<1x1x128x80x80xf16> -> tensor<1x1x128x80x80xf16>
  return %10 : tensor<1x1x128x80x80xf16>
}

// -----

// CHECK-LABEL: func.func @test_mlir_slice_add_convolution
// CHECK: rock.conv
func.func @test_mlir_slice_add_convolution(%arg0: tensor<1638400xf16>, %arg1: tensor<1638400xf16>, %arg2: tensor<147456xf16>) -> tensor<1x1x128x80x80xf16> attributes {rock.kernel} {
  %0 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 3 + d1) * 3 + d2) * 128 + d3)> by [<Unmerge{128, 3, 3, 128} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [128, 3, 3, 128] -> [147456]> : tensor<147456xf16> to tensor<128x3x3x128xf16>
  %1 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> ((d1 * 80 + d2) * 256 + d3)> by [<Unmerge{80, 80, 256} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 80, 80, 256] -> [1638400]> : tensor<1638400xf16> to tensor<1x80x80x256xf16>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)> by [<PassThrough ["dim0", "dim3", "dim1", "dim2"] at [0, 1, 2, 3] -> ["dim0", "dim3", "dim1", "dim2"] at [0, 3, 1, 2]>] bounds = [1, 256, 80, 80] -> [1, 80, 80, 256]> : tensor<1x80x80x256xf16> to tensor<1x256x80x80xf16>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 + 128, d2, d3)> by [<Slice{0, 1, 128, 256, 0, 80, 0, 80} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3]>] bounds = [1, 128, 80, 80] -> [1, 256, 80, 80]> : tensor<1x256x80x80xf16> to tensor<1x128x80x80xf16>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 128} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [128, 80, 80] -> [1, 128, 80, 80]> : tensor<1x128x80x80xf16> to tensor<128x80x80xf16>
  %5 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3) -> ((d1 * 80 + d2) * 256 + d3)> by [<Unmerge{80, 80, 256} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 80, 80, 256] -> [1638400]> : tensor<1638400xf16> to tensor<1x80x80x256xf16>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)> by [<PassThrough ["dim0", "dim3", "dim1", "dim2"] at [0, 1, 2, 3] -> ["dim0", "dim3", "dim1", "dim2"] at [0, 3, 1, 2]>] bounds = [1, 256, 80, 80] -> [1, 80, 80, 256]> : tensor<1x80x80x256xf16> to tensor<1x256x80x80xf16>
  %7 = rock.transform %6 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 + 128, d2, d3)> by [<Slice{0, 1, 128, 256, 0, 80, 0, 80} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3]>] bounds = [1, 128, 80, 80] -> [1, 256, 80, 80]> : tensor<1x256x80x80xf16> to tensor<1x128x80x80xf16>
  %8 = rock.transform %7 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 128} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [128, 80, 80] -> [1, 128, 80, 80]> : tensor<1x128x80x80xf16> to tensor<128x80x80xf16>
  %init = tensor.empty() : tensor<128x80x80xf16>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %8 : tensor<128x80x80xf16>, tensor<128x80x80xf16>) outs(%init : tensor<128x80x80xf16>) {
  ^bb0(%in_1: f16, %in_2: f16, %out: f16):
    %r = arith.addf %in_1, %in_2 : f16
    linalg.yield %r : f16
  } -> tensor<128x80x80xf16>
  %10 = rock.transform %9 by <affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)> by [<Unmerge{128} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 128, 80, 80] -> [128, 80, 80]> : tensor<128x80x80xf16> to tensor<1x128x80x80xf16>
  %11 = rock.transform %10 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 128 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 128} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [1, 1, 128, 80, 80] -> [1, 128, 80, 80]> : tensor<1x128x80x80xf16> to tensor<1x1x128x80x80xf16>
  %12 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 128 + d1, d2, d3, d4)> by [<PassThrough ["y", "x", "c"] at [2, 3, 4] -> ["y", "x", "c"] at [1, 2, 3]>, <Unmerge{1, 128} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 128, 3, 3, 128] -> [128, 3, 3, 128]> : tensor<128x3x3x128xf16> to tensor<1x128x3x3x128xf16>
  %13 = rock.conv(%12, %11) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "y", "x", "c"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    padding = [1 : index, 1 : index, 1 : index, 1 : index],
    strides = [1 : index, 1 : index]
  } : tensor<1x128x3x3x128xf16>, tensor<1x1x128x80x80xf16> -> tensor<1x1x128x80x80xf16>
  return %13 : tensor<1x1x128x80x80xf16>
}

// -----

// CHECK-LABEL: func.func @test_mlir_slice_literal_add_convolution
// CHECK: rock.conv
func.func @test_mlir_slice_literal_add_convolution(%arg0: tensor<1638400xf16>, %arg1: tensor<147456xf16>) -> tensor<1x1x128x80x80xf16> attributes {rock.kernel} {
  %cst = arith.constant 1.000000e+00 : f16
  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 3 + d1) * 3 + d2) * 128 + d3)> by [<Unmerge{128, 3, 3, 128} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [128, 3, 3, 128] -> [147456]> : tensor<147456xf16> to tensor<128x3x3x128xf16>
  %1 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> ((d1 * 80 + d2) * 256 + d3)> by [<Unmerge{80, 80, 256} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 80, 80, 256] -> [1638400]> : tensor<1638400xf16> to tensor<1x80x80x256xf16>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)> by [<PassThrough ["dim0", "dim3", "dim1", "dim2"] at [0, 1, 2, 3] -> ["dim0", "dim3", "dim1", "dim2"] at [0, 3, 1, 2]>] bounds = [1, 256, 80, 80] -> [1, 80, 80, 256]> : tensor<1x80x80x256xf16> to tensor<1x256x80x80xf16>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 + 128, d2, d3)> by [<Slice{0, 1, 128, 256, 0, 80, 0, 80} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3]>] bounds = [1, 128, 80, 80] -> [1, 256, 80, 80]> : tensor<1x256x80x80xf16> to tensor<1x128x80x80xf16>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 128} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [128, 80, 80] -> [1, 128, 80, 80]> : tensor<1x128x80x80xf16> to tensor<128x80x80xf16>
  %init_lit = tensor.empty() : tensor<128x80x80xf16>
  %alloc_literal = linalg.fill ins(%cst : f16) outs(%init_lit : tensor<128x80x80xf16>) -> tensor<128x80x80xf16>
  %init = tensor.empty() : tensor<128x80x80xf16>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %alloc_literal : tensor<128x80x80xf16>, tensor<128x80x80xf16>) outs(%init : tensor<128x80x80xf16>) {
  ^bb0(%in: f16, %in_1 : f16, %out: f16):
    %11 = arith.addf %in, %in_1 : f16
    linalg.yield %11 : f16
  } -> tensor<128x80x80xf16>
  %t5 = rock.transform %5 by <affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)> by [<Unmerge{128} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 128, 80, 80] -> [128, 80, 80]> : tensor<128x80x80xf16> to tensor<1x128x80x80xf16>
  %6 = rock.transform %t5 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 128 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 128} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [1, 1, 128, 80, 80] -> [1, 128, 80, 80]> : tensor<1x128x80x80xf16> to tensor<1x1x128x80x80xf16>
  %7 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 128 + d1, d2, d3, d4)> by [<PassThrough ["y", "x", "c"] at [2, 3, 4] -> ["y", "x", "c"] at [1, 2, 3]>, <Unmerge{1, 128} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 128, 3, 3, 128] -> [128, 3, 3, 128]> : tensor<128x3x3x128xf16> to tensor<1x128x3x3x128xf16>
  %10 = rock.conv(%7, %6) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "y", "x", "c"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    padding = [1 : index, 1 : index, 1 : index, 1 : index],
    strides = [1 : index, 1 : index]
  } : tensor<1x128x3x3x128xf16>, tensor<1x1x128x80x80xf16> -> tensor<1x1x128x80x80xf16>
  return %10 : tensor<1x1x128x80x80xf16>
}

// -----

// CHECK-LABEL: func.func @test_mlir_slice_add_literal_weights_convolution
// CHECK: rock.conv
func.func @test_mlir_slice_add_literal_weights_convolution(%arg0: tensor<1638400xf16>) -> tensor<1x1x128x80x80xf16> attributes {rock.kernel} {
  %cst = arith.constant 1.000000e+00 : f16
  %init_weights = tensor.empty() : tensor<128x3x3x128xf16>
  %alloc_weights = linalg.fill ins(%cst : f16) outs(%init_weights : tensor<128x3x3x128xf16>) -> tensor<128x3x3x128xf16>
  %1 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> ((d1 * 80 + d2) * 256 + d3)> by [<Unmerge{80, 80, 256} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 80, 80, 256] -> [1638400]> : tensor<1638400xf16> to tensor<1x80x80x256xf16>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)> by [<PassThrough ["dim0", "dim3", "dim1", "dim2"] at [0, 1, 2, 3] -> ["dim0", "dim3", "dim1", "dim2"] at [0, 3, 1, 2]>] bounds = [1, 256, 80, 80] -> [1, 80, 80, 256]> : tensor<1x80x80x256xf16> to tensor<1x256x80x80xf16>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 + 128, d2, d3)> by [<Slice{0, 1, 128, 256, 0, 80, 0, 80} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3]>] bounds = [1, 128, 80, 80] -> [1, 256, 80, 80]> : tensor<1x256x80x80xf16> to tensor<1x128x80x80xf16>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 128} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [128, 80, 80] -> [1, 128, 80, 80]> : tensor<1x128x80x80xf16> to tensor<128x80x80xf16>
  %init_lit = tensor.empty() : tensor<128x80x80xf16>
  %alloc_literal = linalg.fill ins(%cst : f16) outs(%init_lit : tensor<128x80x80xf16>) -> tensor<128x80x80xf16>
  %init = tensor.empty() : tensor<128x80x80xf16>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %alloc_literal : tensor<128x80x80xf16>, tensor<128x80x80xf16>) outs(%init : tensor<128x80x80xf16>) {
  ^bb0(%in: f16, %in_1 : f16, %out: f16):
    %11 = arith.addf %in, %in_1 : f16
    linalg.yield %11 : f16
  } -> tensor<128x80x80xf16>
  %t5 = rock.transform %5 by <affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)> by [<Unmerge{128} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 128, 80, 80] -> [128, 80, 80]> : tensor<128x80x80xf16> to tensor<1x128x80x80xf16>
  %6 = rock.transform %t5 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 128 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 128} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [1, 1, 128, 80, 80] -> [1, 128, 80, 80]> : tensor<1x128x80x80xf16> to tensor<1x1x128x80x80xf16>
  %7 = rock.transform %alloc_weights by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 128 + d1, d2, d3, d4)> by [<PassThrough ["y", "x", "c"] at [2, 3, 4] -> ["y", "x", "c"] at [1, 2, 3]>, <Unmerge{1, 128} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 128, 3, 3, 128] -> [128, 3, 3, 128]> : tensor<128x3x3x128xf16> to tensor<1x128x3x3x128xf16>
  %10 = rock.conv(%7, %6) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "y", "x", "c"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    padding = [1 : index, 1 : index, 1 : index, 1 : index],
    strides = [1 : index, 1 : index]
  } : tensor<1x128x3x3x128xf16>, tensor<1x1x128x80x80xf16> -> tensor<1x1x128x80x80xf16>
  return %10 : tensor<1x1x128x80x80xf16>
}

// -----

// CHECK-LABEL: func.func @test_gemm_gemm
// CHECK: %[[a:.*]] = rock.transform %{{.*}} : tensor<16x1x32xf16> to tensor<1x16x32xf16>
// CHECK: %[[b:.*]] = rock.transform %{{.*}} : tensor<64x1x16xf16> to tensor<1x64x16xf16>
// CHECK: rock.gemm_elementwise_gemm
// CHECK-NEXT: ab = tr %[[a]] * tr %[[b]]
// CHECK: perf_config = "attn:v1:128,128,16,1,1,4,0,4,1,0,0"
func.func @test_gemm_gemm(%arg0: tensor<1024xf16>, %arg1: tensor<1024xf16>, %arg2: tensor<512xf16>) -> tensor<1x32x8xf16> attributes {rock.kernel} {
  %0 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 8 + d1) * 64 + d2)> by [<Unmerge{1, 8, 64} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 8, 64] -> [512]> : tensor<512xf16> to tensor<1x8x64xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [1, 64, 8] -> [1, 8, 64]> : tensor<1x8x64xf16> to tensor<1x64x8xf16>
  %2 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 16 + d2)> by [<Unmerge{1, 64, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 64, 16] -> [1024]> : tensor<1024xf16> to tensor<1x64x16xf16>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [1, 16, 64] -> [1, 64, 16]> : tensor<1x64x16xf16> to tensor<1x16x64xf16>
  %4 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 32 + d2)> by [<Unmerge{1, 32, 32} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 32, 32] -> [1024]> : tensor<1024xf16> to tensor<1x32x32xf16>
  %5 = rock.transform %4 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [1, 32, 32] -> [1, 32, 32]> : tensor<1x32x32xf16> to tensor<1x32x32xf16>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<Slice{0, 1, 0, 32, 0, 16} ["dim0_sliced", "dim1_sliced", "dim2_sliced"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [1, 32, 16] -> [1, 32, 32]> : tensor<1x32x32xf16> to tensor<1x32x16xf16>
  %7 = rock.gemm_elementwise_gemm{
    ab = %6 * %3 : tensor<1x32x16xf16>, tensor<1x16x64xf16>
    out = ab * %1 : tensor<1x64x8xf16>
  } {firstGemmIndices = array<i64: 0>, perf_config = "attn:v1:128,128,16,1,1,4,0,4,1,0,0"} -> tensor<1x32x8xf16>
  return %7 : tensor<1x32x8xf16>
}

// -----

// CHECK-LABEL: func.func @test_conv_gemm
// CHECK: rock.conv_elementwise_gemm
// CHECK: ab = conv(%{{.*}}, %{{.*}}) : tensor<256x3x3x
// CHECK: perf_config = "attn:v1:128,128,16,1,1,4,0,4,1,0,0"
func.func @test_conv_gemm(%arg0: tensor<147456xf32>, %arg1: tensor<802816xf32>, %arg2: tensor<65536xf32>) -> tensor<1x9216x256xf32> attributes {rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 3 + d2) * 3 + d3) * 64 + d4)> by [<Unmerge{256, 3, 3, 64} ["k", "0", "1", "c"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 3, 3, 64] -> [147456]> : tensor<147456xf32> to tensor<1x256x3x3x64xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2) * 64 + d4)> by [<Unmerge{64, 14, 14, 64} ["n", "0", "1", "c"] at [0, 1, 2, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [3] -> [] at []>] bounds = [64, 14, 14, 1, 64] -> [802816]> : tensor<802816xf32> to tensor<64x14x14x1x64xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 256 + d2)> by [<Unmerge{256, 256} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 256] -> [65536]> : tensor<65536xf32> to tensor<1x256x256xf32>
  %3 = rock.conv_elementwise_gemm{
    ab = conv(%0, %1) : tensor<1x256x3x3x64xf32>, tensor<64x14x14x1x64xf32>
    out = ab * %2 : tensor<1x256x256xf32>
  } {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], firstGemmIndices = array<i64: 0>, input_layout = ["ni", "0i", "1i", "gi", "ci"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index], perf_config = "attn:v1:128,128,16,1,1,4,0,4,1,0,0"} -> tensor<1x9216x256xf32>
  return %3 : tensor<1x9216x256xf32>
}

// -----

// CHECK-LABEL: func.func @test_scaled_gemm
// CHECK: rock.transform
// CHECK: rock.transform
// CHECK: rock.gemm{{.*}}scaled by{{.*}}scaled by
func.func @test_scaled_gemm(%arg0: tensor<512xf4E2M1FN>, %arg1: tensor<512xf4E2M1FN>, %arg2: tensor<64xf8E8M0FNU>, %arg3: tensor<64xf8E8M0FNU>) -> tensor<1x16x16xf32> attributes {rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 32 + d2)> by [<Unmerge{1, 16, 32} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 16, 32] -> [512]> : tensor<512xf4E2M1FN> to tensor<1x16x32xf4E2M1FN>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 16 + d2)> by [<Unmerge{1, 32, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 32, 16] -> [512]> : tensor<512xf4E2M1FN> to tensor<1x32x16xf4E2M1FN>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 4 + d2)> by [<Unmerge{1, 16, 4} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 16, 4] -> [64]> : tensor<64xf8E8M0FNU> to tensor<1x16x4xf8E8M0FNU>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 4 + d2)> by [<Unmerge{1, 16, 4} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 16, 4] -> [64]> : tensor<64xf8E8M0FNU> to tensor<1x16x4xf8E8M0FNU>
  %4 = rock.gemm %0 scaled by %2 * %1 scaled by %3 {quantBlockSize = 8 : i64} : tensor<1x16x32xf4E2M1FN> scaled by tensor<1x16x4xf8E8M0FNU> * tensor<1x32x16xf4E2M1FN> scaled by tensor<1x16x4xf8E8M0FNU> -> tensor<1x16x16xf32>
  return %4 : tensor<1x16x16xf32>
}

// -----

// CHECK-LABEL: func.func @test_scaled_gemm_tr_scale_a
// CHECK: rock.gemm %{{.*}} scaled by tr %{{.*}} * tr %{{.*}} scaled by
func.func @test_scaled_gemm_tr_scale_a(%arg0: tensor<512xf4E2M1FN>, %arg1: tensor<512xf4E2M1FN>, %arg2: tensor<64xf8E8M0FNU>, %arg3: tensor<64xf8E8M0FNU>) -> tensor<1x16x16xf32> attributes {rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 32 + d2)> by [<Unmerge{1, 16, 32} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 16, 32] -> [512]> : tensor<512xf4E2M1FN> to tensor<1x16x32xf4E2M1FN>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 32 + d2)> by [<Unmerge{1, 16, 32} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 16, 32] -> [512]> : tensor<512xf4E2M1FN> to tensor<1x16x32xf4E2M1FN>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 4 + d1) * 16 + d2)> by [<Unmerge{1, 4, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 4, 16] -> [64]> : tensor<64xf8E8M0FNU> to tensor<1x4x16xf8E8M0FNU>
  %scale_tr = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["G", "M", "K"] at [0, 1, 2] -> ["g", "k", "m"] at [0, 2, 1]>] bounds = [1, 16, 4] -> [1, 4, 16]> : tensor<1x4x16xf8E8M0FNU> to tensor<1x16x4xf8E8M0FNU>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 4 + d2)> by [<Unmerge{1, 16, 4} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 16, 4] -> [64]> : tensor<64xf8E8M0FNU> to tensor<1x16x4xf8E8M0FNU>
  %4 = rock.gemm %0 scaled by %scale_tr * tr %1 scaled by %3 {quantBlockSize = 8 : i64} : tensor<1x16x32xf4E2M1FN> scaled by tensor<1x16x4xf8E8M0FNU> * tensor<1x16x32xf4E2M1FN> scaled by tensor<1x16x4xf8E8M0FNU> -> tensor<1x16x16xf32>
  return %4 : tensor<1x16x16xf32>
}

// -----

// CHECK-LABEL: func.func @test_alloc_to_gemm
// CHECK: rock.gemm{{.*}}scaled by{{.*}}scaled by
func.func @test_alloc_to_gemm(%arg0: tensor<196608xf4E2M1FN>, %arg1: tensor<196608xf4E2M1FN>, %arg2: tensor<6144xf8E8M0FNU>, %arg3: tensor<6144xf8E8M0FNU>) -> tensor<12x256x256xf32> attributes {rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> ((d1 * 12 + d2) * 64 + d3)> by [<Unmerge{256, 12, 64} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 256, 12, 64] -> [196608]> : tensor<196608xf4E2M1FN> to tensor<1x256x12x64xf4E2M1FN>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)> by [<PassThrough ["dim2", "dim0", "dim1", "dim3"] at [0, 1, 2, 3] -> ["dim2", "dim0", "dim1", "dim3"] at [2, 0, 1, 3]>] bounds = [12, 1, 256, 64] -> [1, 256, 12, 64]> : tensor<1x256x12x64xf4E2M1FN> to tensor<12x1x256x64xf4E2M1FN>
  %a = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)> by [<Merge{12, 1} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [12, 256, 64] -> [12, 1, 256, 64]> : tensor<12x1x256x64xf4E2M1FN> to tensor<12x256x64xf4E2M1FN>
  %3 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3) -> ((d1 * 12 + d2) * 64 + d3)> by [<Unmerge{256, 12, 64} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 256, 12, 64] -> [196608]> : tensor<196608xf4E2M1FN> to tensor<1x256x12x64xf4E2M1FN>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)> by [<PassThrough ["dim2", "dim3", "dim0", "dim1"] at [0, 1, 2, 3] -> ["dim2", "dim3", "dim0", "dim1"] at [2, 3, 0, 1]>] bounds = [12, 64, 1, 256] -> [1, 256, 12, 64]> : tensor<1x256x12x64xf4E2M1FN> to tensor<12x64x1x256xf4E2M1FN>
  %b = rock.transform %4 by <affine_map<(d0, d1, d2) -> (d0, d1, 0, d2)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Merge{64, 1} ["dim1"] at [1] -> ["col1", "col2"] at [1, 2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [12, 64, 256] -> [12, 64, 1, 256]> : tensor<12x64x1x256xf4E2M1FN> to tensor<12x64x256xf4E2M1FN>
  %6 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> ((d1 * 12 + d3) * 64 + d4)> by [<Unmerge{8, 12, 64} ["exp1", "exp3", "exp4"] at [1, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [1, 8, 1, 12, 64] -> [6144]> : tensor<6144xf8E8M0FNU> to tensor<1x8x1x12x64xf8E8M0FNU>
  %7 = rock.transform %6 by <affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d0, d4)> by [<PassThrough ["dim3", "dim0", "dim1", "dim2", "dim4"] at [0, 1, 2, 3, 4] -> ["dim3", "dim0", "dim1", "dim2", "dim4"] at [3, 0, 1, 2, 4]>] bounds = [12, 1, 8, 1, 64] -> [1, 8, 1, 12, 64]> : tensor<1x8x1x12x64xf8E8M0FNU> to tensor<12x1x8x1x64xf8E8M0FNU>
  %8 = rock.transform %7 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [12, 1, 8, 32, 64] -> [12, 1, 8, 1, 64]> : tensor<12x1x8x1x64xf8E8M0FNU> to tensor<12x1x8x32x64xf8E8M0FNU>
  %scaleA = rock.transform %8 by <affine_map<(d0, d1, d2) -> (d0, 0, d1 floordiv 32, d1 mod 32, d2)> by [<Merge{12, 1} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <Merge{8, 32} ["dim1"] at [1] -> ["col2", "col3"] at [2, 3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [12, 256, 64] -> [12, 1, 8, 32, 64]> : tensor<12x1x8x32x64xf8E8M0FNU> to tensor<12x256x64xf8E8M0FNU>
  %10 = rock.transform %arg3 by <affine_map<(d0, d1, d2, d3, d4) -> ((d1 * 12 + d3) * 64 + d4)> by [<Unmerge{8, 12, 64} ["exp1", "exp3", "exp4"] at [1, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [1, 8, 1, 12, 64] -> [6144]> : tensor<6144xf8E8M0FNU> to tensor<1x8x1x12x64xf8E8M0FNU>
  %11 = rock.transform %10 by <affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d0, d4)> by [<PassThrough ["dim3", "dim0", "dim1", "dim2", "dim4"] at [0, 1, 2, 3, 4] -> ["dim3", "dim0", "dim1", "dim2", "dim4"] at [3, 0, 1, 2, 4]>] bounds = [12, 1, 8, 1, 64] -> [1, 8, 1, 12, 64]> : tensor<1x8x1x12x64xf8E8M0FNU> to tensor<12x1x8x1x64xf8E8M0FNU>
  %12 = rock.transform %11 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [12, 1, 8, 32, 64] -> [12, 1, 8, 1, 64]> : tensor<12x1x8x1x64xf8E8M0FNU> to tensor<12x1x8x32x64xf8E8M0FNU>
  %scaleB = rock.transform %12 by <affine_map<(d0, d1, d2) -> (d0, 0, d1 floordiv 32, d1 mod 32, d2)> by [<Merge{12, 1} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <Merge{8, 32} ["dim1"] at [1] -> ["col2", "col3"] at [2, 3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [12, 256, 64] -> [12, 1, 8, 32, 64]> : tensor<12x1x8x32x64xf8E8M0FNU> to tensor<12x256x64xf8E8M0FNU>
  %result = rock.gemm %a scaled by %scaleA * %b scaled by %scaleB {quantBlockSize = 1 : i64} : tensor<12x256x64xf4E2M1FN> scaled by tensor<12x256x64xf8E8M0FNU> * tensor<12x64x256xf4E2M1FN> scaled by tensor<12x256x64xf8E8M0FNU> -> tensor<12x256x256xf32>
  return %result : tensor<12x256x256xf32>
}
