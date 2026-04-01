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
