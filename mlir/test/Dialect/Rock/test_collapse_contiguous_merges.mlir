// RUN: rocmlir-opt -rock-collapse-contiguous-merges-test \
// RUN: -allow-unregistered-dialect -split-input-file %s \
// RUN: | FileCheck --enable-var-scope %s

// collapseContiguousMerges() needs to see, for a given Merge, a matching
// Unmerge further down the chain that recombines the same dimensions: that is
// what proves the merged dimensions are genuinely adjacent in the 1D memory.
// Every test therefore starts from a 1D kernel argument that an Unmerge splits
// into logical dimensions before a Merge folds them back together. The pass
// builds the collapsed chain fresh and rewires the marker onto it, leaving the
// original (now dead) transforms in place, so the checks track SSA values
// rather than relying on adjacency.

// A Merge{4, 3, 2} whose dims are split contiguously out of memory by the
// trailing Unmerge collapses to Merge{1, 1, 24}; the Unmerge widens to match.
// CHECK: [[FLAT:#.+]] = #rock.transform_map<{{.*}}Unmerge{1, 1, 24, 5} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["raw"] at [0]{{.*}}bounds = [1, 1, 24, 5] -> [120]>
// CHECK: [[MERGE:#.+]] = #rock.transform_map<{{.*}}Merge{1, 1, 24} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]{{.*}}bounds = [24, 5] -> [1, 1, 24, 5]>
// CHECK: func @test_basic_unmerge
// CHECK-SAME: ([[ARG0:%.+]]: tensor<120xf32>)
// CHECK: [[U:%.+]] = rock.transform [[ARG0]] by [[FLAT]] : tensor<120xf32> to tensor<1x1x24x5xf32>
// CHECK: [[M:%.+]] = rock.transform [[U]] by [[MERGE]] : tensor<1x1x24x5xf32> to tensor<24x5xf32>
// CHECK: "collapse_merges"([[M]])
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (((d0 * 3 + d1) * 2 + d2) * 5 + d3)>
  by [<Unmerge{4, 3, 2, 5} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["raw"] at [0]>]
  bounds = [4, 3, 2, 5] -> [120]>
#merge = #rock.transform_map<
  affine_map<(d0, d1) -> (d0 floordiv 6, (d0 floordiv 2) mod 3, d0 mod 2, d1)>
  by [<PassThrough ["a"] at [1] -> ["a"] at [3]>,
    <Merge{4, 3, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>]
  bounds = [24, 5] -> [4, 3, 2, 5]>

func.func @test_basic_unmerge(%arg0: tensor<120xf32>) {
  %0 = rock.transform %arg0 by #flatten : tensor<120xf32> to tensor<4x3x2x5xf32>
  %1 = rock.transform %0 by #merge : tensor<4x3x2x5xf32> to tensor<24x5xf32>
  "collapse_merges"(%1) : (tensor<24x5xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-test_partial_merge_conv2gemm"
"pre-split-mark-test_partial_merge_conv2gemm"() : () -> ()
// -----

// Conv-to-gemm style merge of {n, 0, 1}. In memory `n` is separated from the
// spatial dims {0, 1} by `k`, so only {0, 1} are contiguous: they collapse to
// Merge{4, 1, 6} while `n` is left alone.
// CHECK: [[FLAT:#.+]] = #rock.transform_map<{{.*}}Unmerge{4, 5, 1, 6} ["n", "k", "0", "1"] at [0, 1, 2, 3] -> ["raw"] at [0]{{.*}}bounds = [4, 5, 1, 6] -> [120]>
// CHECK: [[MERGE:#.+]] = #rock.transform_map<{{.*}}Merge{4, 1, 6} ["gemmN"] at [1] -> ["n", "0", "1"] at [0, 2, 3]{{.*}}bounds = [5, 24] -> [4, 5, 1, 6]>
// CHECK: func @test_partial_merge_conv2gemm
// CHECK-SAME: ([[ARG0:%.+]]: tensor<120xf32>)
// CHECK: [[U:%.+]] = rock.transform [[ARG0]] by [[FLAT]] : tensor<120xf32> to tensor<4x5x1x6xf32>
// CHECK: [[M:%.+]] = rock.transform [[U]] by [[MERGE]] : tensor<4x5x1x6xf32> to tensor<5x24xf32>
// CHECK: "collapse_merges"([[M]])
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (((d0 * 5 + d1) * 3 + d2) * 2 + d3)>
  by [<Unmerge{4, 5, 3, 2} ["n", "k", "0", "1"] at [0, 1, 2, 3] -> ["raw"] at [0]>]
  bounds = [4, 5, 3, 2] -> [120]>
#conv2gemm = #rock.transform_map<
  affine_map<(d0, d1) -> (d1 floordiv 6, d0, (d1 floordiv 2) mod 3, d1 mod 2)>
  by [<PassThrough ["gemmM"] at [0] -> ["k"] at [1]>,
    <Merge{4, 3, 2} ["gemmN"] at [1] -> ["n", "0", "1"] at [0, 2, 3]>]
  bounds = [5, 24] -> [4, 5, 3, 2]>

func.func @test_partial_merge_conv2gemm(%arg0: tensor<120xf32>) {
  %0 = rock.transform %arg0 by #flatten : tensor<120xf32> to tensor<4x5x3x2xf32>
  %1 = rock.transform %0 by #conv2gemm : tensor<4x5x3x2xf32> to tensor<5x24xf32>
  "collapse_merges"(%1) : (tensor<5x24xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-test_batch_transpose_bug_1407"
"pre-split-mark-test_batch_transpose_bug_1407"() : () -> ()
// -----

// A transpose sits between the trailing Unmerge and the conv2gemm Merge. The
// merged {0, 1, n} run is still contiguous through the transpose, collapsing to
// Merge{1, 65536, 1}.
// CHECK: [[FLAT:#.+]] = #rock.transform_map<{{.*}}Unmerge{1, 512, 1, 65536} ["n", "k", "0", "1"] at [0, 1, 2, 3] -> ["raw"] at [0]{{.*}}bounds = [1, 512, 1, 65536] -> [33554432]>
// CHECK: [[PERM:#.+]] = #rock.transform_map<{{.*}}PassThrough ["n", "k", "0", "1"] at [3, 1, 0, 2] -> ["n", "k", "0", "1"] at [0, 1, 2, 3]{{.*}}bounds = [1, 512, 65536, 1] -> [1, 512, 1, 65536]>
// CHECK: [[MERGE:#.+]] = #rock.transform_map<{{.*}}Merge{1, 65536, 1} ["gemmN"] at [1] -> ["0", "1", "n"] at [0, 2, 3]{{.*}}bounds = [512, 65536] -> [1, 512, 65536, 1]>
// CHECK: func @test_batch_transpose_bug_1407
// CHECK-SAME: ([[ARG0:%.+]]: tensor<33554432xf32>)
// CHECK: [[U:%.+]] = rock.transform [[ARG0]] by [[FLAT]] : tensor<33554432xf32> to tensor<1x512x1x65536xf32>
// CHECK: [[P:%.+]] = rock.transform [[U]] by [[PERM]] : tensor<1x512x1x65536xf32> to tensor<1x512x65536x1xf32>
// CHECK: [[M:%.+]] = rock.transform [[P]] by [[MERGE]] : tensor<1x512x65536x1xf32> to tensor<512x65536xf32>
// CHECK: "collapse_merges"([[M]])
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (((d0 * 512 + d1) * 256 + d2) * 256 + d3)>
  by [<Unmerge{1, 512, 256, 256} ["n", "k", "0", "1"] at [0, 1, 2, 3] -> ["raw"] at [0]>]
  bounds = [1, 512, 256, 256] -> [33554432]>
#perm = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (d3, d1, d0, d2)>
  by [<PassThrough ["n", "k", "0", "1"] at [3, 1, 0, 2]-> ["n", "k", "0", "1"] at [0, 1, 2, 3]>]
  bounds = [256, 512, 256, 1] -> [1, 512, 256, 256]>
#conv2gemm = #rock.transform_map<
  affine_map<(d0, d1) -> (d1 floordiv 256, d0, d1 mod 256, 0)>
  by [<PassThrough ["gemmM"] at [0] -> ["k"] at [1]>,
    <Merge{256, 256, 1} ["gemmN"] at [1] -> ["0", "1", "n"] at [0, 2, 3]>]
  bounds = [512, 65536] -> [256, 512, 256, 1]>

func.func @test_batch_transpose_bug_1407(%arg0: tensor<33554432xf32>) {
  %0 = rock.transform %arg0 by #flatten : tensor<33554432xf32> to tensor<1x512x256x256xf32>
  %1 = rock.transform %0 by #perm : tensor<1x512x256x256xf32> to tensor<256x512x256x1xf32>
  %2 = rock.transform %1 by #conv2gemm : tensor<256x512x256x1xf32> to tensor<512x65536xf32>
  "collapse_merges"(%2) : (tensor<512x65536xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-test_need_for_clone"
"pre-split-mark-test_need_for_clone"() : () -> ()
// -----

// The chain has a second user ("use"), so collapseContiguousMerges must clone
// the chain instead of editing it in place: the marker ends up on the collapsed
// Merge{1, 1, 24} while "use" keeps the original Merge{4, 3, 2}.
// CHECK: [[OLDMERGE:#.+]] = #rock.transform_map<{{.*}}Merge{4, 3, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]{{.*}}bounds = [24, 5] -> [4, 3, 2, 5]>
// CHECK: [[PERM:#.+]] = #rock.transform_map<{{.*}}PassThrough ["1", "a"] at [1, 0] -> ["1", "a"] at [0, 1]{{.*}}bounds = [5, 24] -> [24, 5]>
// CHECK: [[NEWFLAT:#.+]] = #rock.transform_map<{{.*}}Unmerge{1, 1, 24, 5} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["raw"] at [0]{{.*}}bounds = [1, 1, 24, 5] -> [120]>
// CHECK: [[NEWMERGE:#.+]] = #rock.transform_map<{{.*}}Merge{1, 1, 24} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]{{.*}}bounds = [24, 5] -> [1, 1, 24, 5]>
// CHECK: func @test_need_for_clone
// CHECK-SAME: ([[ARG0:%.+]]: tensor<120xf32>)
// The "use" keeps the original, uncollapsed Merge{4, 3, 2} chain.
// CHECK: [[OM:%.+]] = rock.transform %{{.+}} by [[OLDMERGE]] : tensor<4x3x2x5xf32> to tensor<24x5xf32>
// CHECK: [[OP:%.+]] = rock.transform [[OM]] by [[PERM]] : tensor<24x5xf32> to tensor<5x24xf32>
// The marker gets the freshly built, collapsed chain.
// CHECK: [[NU:%.+]] = rock.transform [[ARG0]] by [[NEWFLAT]] : tensor<120xf32> to tensor<1x1x24x5xf32>
// CHECK: [[NM:%.+]] = rock.transform [[NU]] by [[NEWMERGE]] : tensor<1x1x24x5xf32> to tensor<24x5xf32>
// CHECK: [[NP:%.+]] = rock.transform [[NM]] by [[PERM]] : tensor<24x5xf32> to tensor<5x24xf32>
// CHECK: "collapse_merges"([[NP]])
// CHECK: "use"([[OP]])
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (((d0 * 3 + d1) * 2 + d2) * 5 + d3)>
  by [<Unmerge{4, 3, 2, 5} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["raw"] at [0]>]
  bounds = [4, 3, 2, 5] -> [120]>
#merge = #rock.transform_map<
  affine_map<(d0, d1) -> (d0 floordiv 6, (d0 floordiv 2) mod 3, d0 mod 2, d1)>
  by [<PassThrough ["a"] at [1] -> ["a"] at [3]>,
    <Merge{4, 3, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>]
  bounds = [24, 5] -> [4, 3, 2, 5]>
#perm = #rock.transform_map<
  affine_map<(d0, d1) -> (d1, d0)>
  by [<PassThrough ["1", "a"] at [1, 0] -> ["1", "a"] at [0, 1]>]
  bounds = [5, 24] -> [24, 5]>

func.func @test_need_for_clone(%arg0: tensor<120xf32>) {
  %0 = rock.transform %arg0 by #flatten : tensor<120xf32> to tensor<4x3x2x5xf32>
  %1 = rock.transform %0 by #merge : tensor<4x3x2x5xf32> to tensor<24x5xf32>
  %2 = rock.transform %1 by #perm : tensor<24x5xf32> to tensor<5x24xf32>
  "collapse_merges"(%2) : (tensor<5x24xf32>) -> ()
  "use"(%2) : (tensor<5x24xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-negative-test_reuse_dim"
"pre-split-mark-negative-test_reuse_dim"() : () -> ()
// Two contiguous groups exist in the unmerge, {3,5} and {4,6}, and {3,5} has a
// singleton dimension between its members. Folding that singleton into the
// group would manufacture a bogus single {3,4,5,6} group, so once a dimension
// is consumed by one group it must not be reused by another. The pass must
// therefore leave both Merges untouched.
// CHECK: [[MERGEMAP:#.+]] = #rock.transform_map<{{.*}}Merge{1, 1, 2, 16} ["tid"] at [3] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [3, 4, 5, 6]>, <Merge{4, 4, 8} ["item"] at [4] -> ["rep_i", "rep_j", "item_i"] at [7, 8, 9]>{{.*}}>
// CHECK: func @negative_test_reuse_dim
// CHECK: "collapse_merges"
// CHECK: "use"
// -----
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2) -> ((d0 * 128 + d1) * 128 + d2)>
  by [<Unmerge{1, 128, 128} ["g", "gemmM", "gemmN"] at [0, 1, 2] -> ["raw"] at [0]>]
  bounds = [1, 128, 128] -> [16384]>
#unmerge_map = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, ((d1 * 4 + d7 + d3) * 2 + d5) * 8 + d9, (d2 * 4 + d8 + d4) * 16 + d6)>
               by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>,
                   <Unmerge{2, 4, 1, 2, 8} ["m_block", "rep_i", "wave_m", "m_tid", "item_i"] at [1, 7, 3, 5, 9] -> ["gemmM"] at [1]>,
                   <Unmerge{2, 4, 1, 16} ["n_block", "rep_j", "wave_n", "n_tid"] at [2, 8, 4, 6] -> ["gemmN"] at [2]>] bounds = [1, 2, 2, 1, 1, 2, 16, 4, 4, 8] -> [1, 128, 128]>
#merge_map = #rock.transform_map<
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, 0, d3 floordiv 16, d3 mod 16, d4 floordiv 32, (d4 floordiv 8) mod 4, d4 mod 8)>
         by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>,
             <Merge{1, 1, 2, 16} ["tid"] at [3] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [3, 4, 5, 6]>,
             <Merge{4, 4, 8} ["item"] at [4] -> ["rep_i", "rep_j", "item_i"] at [7, 8, 9]>] bounds = [1, 2, 2, 32, 128] -> [1, 2, 2, 1, 1, 2, 16, 4, 4, 8]>

func.func @negative_test_reuse_dim(%arg0: tensor<16384xf32>) {
  %flat = rock.transform %arg0 by #flatten : tensor<16384xf32> to tensor<1x128x128xf32>
  %0 = rock.transform %flat by #unmerge_map : tensor<1x128x128xf32> to tensor<1x2x2x1x1x2x16x4x4x8xf32>
  %1 = rock.transform %0 by #merge_map : tensor<1x2x2x1x1x2x16x4x4x8xf32> to tensor<1x2x2x32x128xf32>
  "collapse_merges"(%1) : (tensor<1x2x2x32x128xf32>) -> ()
  "use"(%1) : (tensor<1x2x2x32x128xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-test_slice_between_aborts"
"pre-split-mark-test_slice_between_aborts"() : () -> ()
// -----

// A no-op Slice sits between the Unmerge and the Merge. The {b, c, d} group is
// still contiguous in memory, so the analysis groups it, but the resize trace
// cannot travel through a Slice and aborts: nothing collapses and the marker
// keeps the original Merge{4, 3, 2} chain.
// CHECK: [[MERGE:#.+]] = #rock.transform_map<{{.*}}Merge{4, 3, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]{{.*}}bounds = [24, 5] -> [4, 3, 2, 5]>
// CHECK: func @test_slice_between_aborts
// CHECK-SAME: ([[ARG0:%.+]]: tensor<120xf32>)
// CHECK: [[U:%.+]] = rock.transform [[ARG0]] by #{{.+}} : tensor<120xf32> to tensor<4x3x2x5xf32>
// CHECK: [[S:%.+]] = rock.transform [[U]] by #{{.+}} : tensor<4x3x2x5xf32> to tensor<4x3x2x5xf32>
// CHECK: [[M:%.+]] = rock.transform [[S]] by [[MERGE]] : tensor<4x3x2x5xf32> to tensor<24x5xf32>
// CHECK: "collapse_merges"([[M]])
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (((d0 * 3 + d1) * 2 + d2) * 5 + d3)>
  by [<Unmerge{4, 3, 2, 5} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["raw"] at [0]>]
  bounds = [4, 3, 2, 5] -> [120]>
#slice = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  by [<Slice{0, 4, 0, 3, 0, 2, 0, 5} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["b", "c", "d", "a"] at [0, 1, 2, 3]>]
  bounds = [4, 3, 2, 5] -> [4, 3, 2, 5]>
#merge = #rock.transform_map<
  affine_map<(d0, d1) -> (d0 floordiv 6, (d0 floordiv 2) mod 3, d0 mod 2, d1)>
  by [<PassThrough ["a"] at [1] -> ["a"] at [3]>,
    <Merge{4, 3, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>]
  bounds = [24, 5] -> [4, 3, 2, 5]>

func.func @test_slice_between_aborts(%arg0: tensor<120xf32>) {
  %0 = rock.transform %arg0 by #flatten : tensor<120xf32> to tensor<4x3x2x5xf32>
  %1 = rock.transform %0 by #slice : tensor<4x3x2x5xf32> to tensor<4x3x2x5xf32>
  %2 = rock.transform %1 by #merge : tensor<4x3x2x5xf32> to tensor<24x5xf32>
  "collapse_merges"(%2) : (tensor<24x5xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-test_collapse_through_zero_pad"
"pre-split-mark-test_collapse_through_zero_pad"() : () -> ()
// -----

// A *zero* Pad is size-preserving, so the resize trace travels through it
// (unlike the Slice case) and the {b, c, d} group still collapses to
// Merge{1, 1, 24}. The Pad is rebuilt on the widened spaces.
// CHECK: [[FLAT:#.+]] = #rock.transform_map<{{.*}}Unmerge{1, 1, 24, 5} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["raw"] at [0]{{.*}}bounds = [1, 1, 24, 5] -> [120]>
// CHECK: [[PAD:#.+]] = #rock.transform_map<{{.*}}Pad{0, 0, 0, 0, 0, 0, 0, 0} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["b", "c", "d", "a"] at [0, 1, 2, 3]{{.*}}bounds = [1, 1, 24, 5] -> [1, 1, 24, 5]>
// CHECK: [[MERGE:#.+]] = #rock.transform_map<{{.*}}Merge{1, 1, 24} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]{{.*}}bounds = [24, 5] -> [1, 1, 24, 5]>
// CHECK: func @test_collapse_through_zero_pad
// CHECK-SAME: ([[ARG0:%.+]]: tensor<120xf32>)
// CHECK: [[U:%.+]] = rock.transform [[ARG0]] by [[FLAT]] : tensor<120xf32> to tensor<1x1x24x5xf32>
// CHECK: [[P:%.+]] = rock.transform [[U]] by [[PAD]] : tensor<1x1x24x5xf32> to tensor<1x1x24x5xf32>
// CHECK: [[M:%.+]] = rock.transform [[P]] by [[MERGE]] : tensor<1x1x24x5xf32> to tensor<24x5xf32>
// CHECK: "collapse_merges"([[M]])
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (((d0 * 3 + d1) * 2 + d2) * 5 + d3)>
  by [<Unmerge{4, 3, 2, 5} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["raw"] at [0]>]
  bounds = [4, 3, 2, 5] -> [120]>
#pad = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  by [<Pad{0, 0, 0, 0, 0, 0, 0, 0} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["b", "c", "d", "a"] at [0, 1, 2, 3]>]
  bounds = [4, 3, 2, 5] -> [4, 3, 2, 5]>
#merge = #rock.transform_map<
  affine_map<(d0, d1) -> (d0 floordiv 6, (d0 floordiv 2) mod 3, d0 mod 2, d1)>
  by [<PassThrough ["a"] at [1] -> ["a"] at [3]>,
    <Merge{4, 3, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>]
  bounds = [24, 5] -> [4, 3, 2, 5]>

func.func @test_collapse_through_zero_pad(%arg0: tensor<120xf32>) {
  %0 = rock.transform %arg0 by #flatten : tensor<120xf32> to tensor<4x3x2x5xf32>
  %1 = rock.transform %0 by #pad : tensor<4x3x2x5xf32> to tensor<4x3x2x5xf32>
  %2 = rock.transform %1 by #merge : tensor<4x3x2x5xf32> to tensor<24x5xf32>
  "collapse_merges"(%2) : (tensor<24x5xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-test_nonzero_pad_not_contiguous"
"pre-split-mark-test_nonzero_pad_not_contiguous"() : () -> ()
// -----

// Non-zero Pad on the *middle* member (c) breaks every adjacency in the merged
// run, so no contiguous pair survives and nothing collapses: the marker keeps
// the original Merge{4, 5, 2} chain.
// CHECK: [[MERGE:#.+]] = #rock.transform_map<{{.*}}Merge{4, 5, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]{{.*}}bounds = [40, 5] -> [4, 5, 2, 5]>
// CHECK: func @test_nonzero_pad_not_contiguous
// CHECK-SAME: ([[ARG0:%.+]]: tensor<120xf32>)
// CHECK: [[U:%.+]] = rock.transform [[ARG0]] by #{{.+}} : tensor<120xf32> to tensor<4x3x2x5xf32>
// CHECK: [[P:%.+]] = rock.transform [[U]] by #{{.+}} : tensor<4x3x2x5xf32> to tensor<4x5x2x5xf32>
// CHECK: [[M:%.+]] = rock.transform [[P]] by [[MERGE]] : tensor<4x5x2x5xf32> to tensor<40x5xf32>
// CHECK: "collapse_merges"([[M]])
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (((d0 * 3 + d1) * 2 + d2) * 5 + d3)>
  by [<Unmerge{4, 3, 2, 5} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["raw"] at [0]>]
  bounds = [4, 3, 2, 5] -> [120]>
#pad = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (d0, d1 - 1, d2, d3)>
  by [<PassThrough ["b"] at [0] -> ["b"] at [0]>,
      <Pad{1, 1} ["c"] at [1] -> ["c"] at [1]>,
      <PassThrough ["d"] at [2] -> ["d"] at [2]>,
      <PassThrough ["a"] at [3] -> ["a"] at [3]>]
  bounds = [4, 5, 2, 5] -> [4, 3, 2, 5]>
#merge = #rock.transform_map<
  affine_map<(d0, d1) -> (d0 floordiv 10, (d0 floordiv 2) mod 5, d0 mod 2, d1)>
  by [<PassThrough ["a"] at [1] -> ["a"] at [3]>,
    <Merge{4, 5, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>]
  bounds = [40, 5] -> [4, 5, 2, 5]>

func.func @test_nonzero_pad_not_contiguous(%arg0: tensor<120xf32>) {
  %0 = rock.transform %arg0 by #flatten : tensor<120xf32> to tensor<4x3x2x5xf32>
  %1 = rock.transform %0 by #pad : tensor<4x3x2x5xf32> to tensor<4x5x2x5xf32>
  %2 = rock.transform %1 by #merge : tensor<4x5x2x5xf32> to tensor<40x5xf32>
  "collapse_merges"(%2) : (tensor<40x5xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-test_nonzero_pad_excludes_member"
"pre-split-mark-test_nonzero_pad_excludes_member"() : () -> ()
// -----

// Non-zero Pad on the *fastest* member (d) only excludes that member from the
// contiguous run; the remaining {b, c} prefix is still contiguous and collapses
// to Merge{1, 12, 4}, leaving the padded d as a standalone dimension.
// CHECK: [[FLAT:#.+]] = #rock.transform_map<{{.*}}Unmerge{1, 12, 2, 5} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["raw"] at [0]{{.*}}bounds = [1, 12, 2, 5] -> [120]>
// CHECK: [[PAD:#.+]] = #rock.transform_map<{{.*}}Pad{1, 1} ["d"] at [2] -> ["d"] at [2]{{.*}}bounds = [1, 12, 4, 5] -> [1, 12, 2, 5]>
// CHECK: [[MERGE:#.+]] = #rock.transform_map<{{.*}}Merge{1, 12, 4} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]{{.*}}bounds = [48, 5] -> [1, 12, 4, 5]>
// CHECK: func @test_nonzero_pad_excludes_member
// CHECK-SAME: ([[ARG0:%.+]]: tensor<120xf32>)
// CHECK: [[U:%.+]] = rock.transform [[ARG0]] by [[FLAT]] : tensor<120xf32> to tensor<1x12x2x5xf32>
// CHECK: [[P:%.+]] = rock.transform [[U]] by [[PAD]] : tensor<1x12x2x5xf32> to tensor<1x12x4x5xf32>
// CHECK: [[M:%.+]] = rock.transform [[P]] by [[MERGE]] : tensor<1x12x4x5xf32> to tensor<48x5xf32>
// CHECK: "collapse_merges"([[M]])
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (((d0 * 3 + d1) * 2 + d2) * 5 + d3)>
  by [<Unmerge{4, 3, 2, 5} ["b", "c", "d", "a"] at [0, 1, 2, 3] -> ["raw"] at [0]>]
  bounds = [4, 3, 2, 5] -> [120]>
#pad = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 - 1, d3)>
  by [<PassThrough ["b"] at [0] -> ["b"] at [0]>,
      <PassThrough ["c"] at [1] -> ["c"] at [1]>,
      <Pad{1, 1} ["d"] at [2] -> ["d"] at [2]>,
      <PassThrough ["a"] at [3] -> ["a"] at [3]>]
  bounds = [4, 3, 4, 5] -> [4, 3, 2, 5]>
#merge = #rock.transform_map<
  affine_map<(d0, d1) -> (d0 floordiv 12, (d0 floordiv 4) mod 3, d0 mod 4, d1)>
  by [<PassThrough ["a"] at [1] -> ["a"] at [3]>,
    <Merge{4, 3, 4} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>]
  bounds = [48, 5] -> [4, 3, 4, 5]>

func.func @test_nonzero_pad_excludes_member(%arg0: tensor<120xf32>) {
  %0 = rock.transform %arg0 by #flatten : tensor<120xf32> to tensor<4x3x2x5xf32>
  %1 = rock.transform %0 by #pad : tensor<4x3x2x5xf32> to tensor<4x3x4x5xf32>
  %2 = rock.transform %1 by #merge : tensor<4x3x4x5xf32> to tensor<48x5xf32>
  "collapse_merges"(%2) : (tensor<48x5xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-no_test_yet"
"pre-split-mark-no_test_yet"() : () -> ()
// -----
