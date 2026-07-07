// Tests for the rock-fuse-sibling-loops pass.

// RUN: rocmlir-opt -rock-fuse-sibling-loops -split-input-file %s | FileCheck %s

// Two independent sibling loops with identical (shared) bounds are fused into a
// single loop carrying both accumulators.

// CHECK-LABEL: func.func @fuse_two_siblings
// CHECK: %[[FUSED:.*]]:2 = scf.for
// CHECK-NOT: scf.for
// CHECK: return
func.func @fuse_two_siblings(%init0: f32, %init1: f32) -> (f32, f32) attributes {rock.kernel} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init0) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init1) -> (f32) {
    %v = arith.mulf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r0, %r1 : f32, f32
}

// -----

// Three independent sibling loops with identical bounds collapse into a single
// loop carrying all three accumulators.

// CHECK-LABEL: func.func @fuse_three_siblings
// CHECK: %[[FUSED:.*]]:3 = scf.for
// CHECK-NOT: scf.for
// CHECK: return
func.func @fuse_three_siblings(%init0: f32, %init1: f32, %init2: f32) -> (f32, f32, f32) attributes {rock.kernel} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init0) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init1) -> (f32) {
    %v = arith.mulf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r2 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init2) -> (f32) {
    %v = arith.subf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r0, %r1, %r2 : f32, f32, f32
}

// -----

// Four independent sibling loops with identical bounds collapse into a single
// loop carrying all four accumulators.

// CHECK-LABEL: func.func @fuse_four_siblings
// CHECK: %[[FUSED:.*]]:4 = scf.for
// CHECK-NOT: scf.for
// CHECK: return
func.func @fuse_four_siblings(%init0: f32, %init1: f32, %init2: f32, %init3: f32) -> (f32, f32, f32, f32) attributes {rock.kernel} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init0) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init1) -> (f32) {
    %v = arith.mulf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r2 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init2) -> (f32) {
    %v = arith.subf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r3 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init3) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r0, %r1, %r2, %r3 : f32, f32, f32, f32
}

// -----

// Different upper bounds: not fused.

// CHECK-LABEL: func.func @different_bounds
// CHECK: scf.for
// CHECK: scf.for
func.func @different_bounds(%init0: f32, %init1: f32) -> (f32, f32) attributes {rock.kernel} {
  %lb = arith.constant 0 : index
  %ub0 = arith.constant 8 : index
  %ub1 = arith.constant 16 : index
  %step = arith.constant 1 : index
  %r0 = scf.for %i = %lb to %ub0 step %step iter_args(%acc = %init0) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r1 = scf.for %i = %lb to %ub1 step %step iter_args(%acc = %init1) -> (f32) {
    %v = arith.mulf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r0, %r1 : f32, f32
}

// -----

// Second loop depends on the first loop's result: not fused.

// CHECK-LABEL: func.func @dependent_loops
// CHECK: scf.for
// CHECK: scf.for
func.func @dependent_loops(%init0: f32) -> (f32) attributes {rock.kernel} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init0) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%acc = %r0) -> (f32) {
    %v = arith.mulf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r1 : f32
}

// -----

// An op sitting between the two loops feeds the second loop but transitively
// depends on the first loop's result. It is pure (so it would normally be
// hoisted above the anchor to satisfy dominance), but hoisting recurses into
// its operands and hits the anchor's own result, which cannot move. Fusion is
// therefore correctly rejected and both loops are left in place.

// CHECK-LABEL: func.func @dependent_loops_via_intermediate
// CHECK: scf.for
// CHECK: scf.for
func.func @dependent_loops_via_intermediate(%init0: f32) -> (f32) attributes {rock.kernel} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init0) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  %mid = arith.mulf %r0, %r0 : f32
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%acc = %mid) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r1 : f32
}

// -----

// A kernel whose gridwise lowering produced no scf.for loops at all: there is
// nothing to fuse, so the pass must run cleanly and leave the body unchanged.

// CHECK-LABEL: func.func @no_loops
// CHECK-NOT: scf.for
// CHECK: return
func.func @no_loops(%arg0: f32) -> f32 attributes {rock.kernel} {
  %v = arith.addf %arg0, %arg0 : f32
  return %v : f32
}

// -----

// Bounds with the same value but distinct constant ops still fuse (bounds are
// compared by constant value, not SSA identity).

// CHECK-LABEL: func.func @fuse_distinct_bound_constants
// CHECK: scf.for
// CHECK-NOT: scf.for
func.func @fuse_distinct_bound_constants(%init0: f32, %init1: f32) -> (f32, f32) attributes {rock.kernel} {
  %lb0 = arith.constant 0 : index
  %ub0 = arith.constant 8 : index
  %step0 = arith.constant 1 : index
  %r0 = scf.for %i = %lb0 to %ub0 step %step0 iter_args(%acc = %init0) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  %lb1 = arith.constant 0 : index
  %ub1 = arith.constant 8 : index
  %step1 = arith.constant 1 : index
  %r1 = scf.for %i = %lb1 to %ub1 step %step1 iter_args(%acc = %init1) -> (f32) {
    %v = arith.mulf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r0, %r1 : f32, f32
}

// -----

// Dynamic bounds that are the *same SSA value* fuse: the shared %ub guarantees
// an identical iteration space (bounds match by SSA identity, no constant
// needed).

// CHECK-LABEL: func.func @fuse_same_dynamic_bound
// CHECK: scf.for
// CHECK-NOT: scf.for
func.func @fuse_same_dynamic_bound(%ub: index, %init0: f32, %init1: f32) -> (f32, f32) attributes {rock.kernel} {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init0) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init1) -> (f32) {
    %v = arith.mulf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r0, %r1 : f32, f32
}

// -----

// Distinct dynamic bounds are not fused: %ub0 and %ub1 are different SSA values
// and not constants, so the iteration spaces cannot be proven equal.

// CHECK-LABEL: func.func @distinct_dynamic_bounds
// CHECK: scf.for
// CHECK: scf.for
func.func @distinct_dynamic_bounds(%ub0: index, %ub1: index, %init0: f32, %init1: f32) -> (f32, f32) attributes {rock.kernel} {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %r0 = scf.for %i = %lb to %ub0 step %step iter_args(%acc = %init0) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r1 = scf.for %i = %lb to %ub1 step %step iter_args(%acc = %init1) -> (f32) {
    %v = arith.mulf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r0, %r1 : f32, f32
}

// -----

// Same numeric bounds but different induction-variable types (index vs i32):
// not fused. The bounds are equal as integers, but scf.for ties the IV type to
// the bound types, so the loops do not share an iteration space.

// CHECK-LABEL: func.func @different_iv_types
// CHECK: scf.for
// CHECK: scf.for
func.func @different_iv_types(%init0: f32, %init1: f32) -> (f32, f32) attributes {rock.kernel} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  %lbi = arith.constant 0 : i32
  %ubi = arith.constant 8 : i32
  %stepi = arith.constant 1 : i32
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init0) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r1 = scf.for %i = %lbi to %ubi step %stepi iter_args(%acc = %init1) -> (f32) : i32 {
    %v = arith.mulf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r0, %r1 : f32, f32
}

// -----

// Single-iteration loops (0..1, all constant) still fuse at the pass level.
// (In the full pipeline the DCE step before this pass inlines single-trip loops
// into straight-line code, so this case is only reachable when running the pass
// in isolation.)

// CHECK-LABEL: func.func @fuse_single_iteration
// CHECK: scf.for
// CHECK-NOT: scf.for
func.func @fuse_single_iteration(%init0: f32, %init1: f32) -> (f32, f32) attributes {rock.kernel} {
  %lb = arith.constant 0 : index
  %ub = arith.constant 1 : index
  %step = arith.constant 1 : index
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init0) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init1) -> (f32) {
    %v = arith.mulf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r0, %r1 : f32, f32
}

// -----

// A function without the rock.kernel attribute is skipped entirely.

// CHECK-LABEL: func.func @not_a_kernel
// CHECK: scf.for
// CHECK: scf.for
func.func @not_a_kernel(%init0: f32, %init1: f32) -> (f32, f32) {
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init0) -> (f32) {
    %v = arith.addf %acc, %acc : f32
    scf.yield %v : f32
  }
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%acc = %init1) -> (f32) {
    %v = arith.mulf %acc, %acc : f32
    scf.yield %v : f32
  }
  return %r0, %r1 : f32, f32
}

// -----

// Representative example: the two K-loops produced when an mPerBlock=80 GEMM is
// decomposed into pow2 (64 + 16) M-segment sub-gemms by
// rock-decompose-nonpow2-tiles + rock-gridwise-gemm-to-blockwise. The loops
// share the B tile (tensor<32x128xf16>) and differ only in their A M-segment.
// The pass must hoist the second loop's accumulator-init, grid-coordinate and
// sliced A-view transform chain (all defined between the loops) above the
// anchor, then fuse into a single loop carrying both accumulators. Both
// blockwise GEMMs survive; CSE later collapses the duplicated B load.

// CHECK-LABEL: func.func @decomposed_m_split
// Both segments' operands are hoisted above the anchor and the two K-loops fuse
// into a single loop carrying both accumulators; both blockwise GEMMs survive.
// CHECK: %{{.*}}:2 = scf.for
// CHECK-COUNT-2: rock.blockwise_gemm
// CHECK-NOT: scf.for
// CHECK: return
func.func @decomposed_m_split(%arg0: tensor<59136xf16>, %arg1: tensor<28016640xf16>, %arg2: tensor<2808960xf16>) -> tensor<2808960xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 285 : i32, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 96 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{77, 768} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 77, 768] -> [59136]> : tensor<59136xf16> to tensor<1x77x768xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 3} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <PassThrough ["gemmK"] at [2] -> ["gemmK"] at [2]>] bounds = [1, 80, 768] -> [1, 77, 768]> : tensor<1x77x768xf16> to tensor<1x80x768xf16>
  %2 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 36480 + d2)> by [<Unmerge{768, 36480} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 768, 36480] -> [28016640]> : tensor<28016640xf16> to tensor<1x768x36480xf16>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 80 + d2, d3)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{1, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <PassThrough ["d2"] at [3] -> ["d2"] at [2]>] bounds = [1, 1, 80, 768] -> [1, 80, 768]> : tensor<1x80x768xf16> to tensor<1x1x80x768xf16>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> by [<PassThrough ["d0", "d1b", "d2"] at [0, 1, 3] -> ["d0", "d1b", "d2"] at [0, 1, 3]>, <Slice{0, 64} ["d1i"] at [2] -> ["d1i"] at [2]>] bounds = [1, 1, 64, 768] -> [1, 1, 80, 768]> : tensor<1x1x80x768xf16> to tensor<1x1x64x768xf16>
  %5 = rock.transform %4 by <affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{1, 64} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <PassThrough ["d2"] at [2] -> ["d2"] at [3]>] bounds = [1, 64, 768] -> [1, 1, 64, 768]> : tensor<1x1x64x768xf16> to tensor<1x64x768xf16>
  %6 = tt.get_program_id x : i32
  %c10_i32 = arith.constant 10 : i32
  %c2850_i32 = arith.constant 2850 : i32
  %c1_i32 = arith.constant 1 : i32
  %c285_i32 = arith.constant 285 : i32
  %7 = arith.divui %6, %c285_i32 : i32
  %8 = arith.remui %6, %c285_i32 : i32
  %9 = arith.divui %8, %c2850_i32 : i32
  %10 = arith.muli %9, %c10_i32 : i32
  %11 = arith.subi %c1_i32, %10 : i32
  %12 = arith.minui %11, %c10_i32 : i32
  %13 = arith.remui %8, %12 : i32
  %14 = arith.addi %10, %13 : i32
  %15 = arith.remui %8, %c2850_i32 : i32
  %16 = arith.divui %15, %12 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
  %c24_i32 = arith.constant 24 : i32
  %c1_i32_0 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %17 = scf.for %arg3 = %c0_i32 to %c24_i32 step %c1_i32_0 iter_args(%arg4 = %cst) -> (tensor<64x128xf32>)  : i32 {
    %49 = rock.load_marker %2 views [#rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 32 + d4, d3 * 128 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{24, 32} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{285, 128} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [24, 1, 1, 285, 32, 128] -> [1, 768, 36480]>][%arg3, %7, %14, %16] {cacheModifier = #rock<CacheModifier none>} : tensor<1x768x36480xf16> -> tensor<32x128xf16>
    %50 = rock.load_marker %5 views [#rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 32 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{24, 32} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{285} ["n_block"] at [3] -> [] at []>] bounds = [24, 1, 1, 285, 64, 32] -> [1, 64, 768]>][%arg3, %7, %14, %16] {cacheModifier = #rock<CacheModifier none>} : tensor<1x64x768xf16> -> tensor<64x32xf16>
    %51 = rock.blockwise_gemm(%50, %49, %arg4) : tensor<64x32xf16>, tensor<32x128xf16>, tensor<64x128xf32> -> tensor<64x128xf32>
    scf.yield %51 : tensor<64x128xf32>
  }
  %18 = arith.truncf %17 : tensor<64x128xf32> to tensor<64x128xf16>
  %19 = rock.store_marker %18 views [#rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d3, d2 * 128 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{285, 128} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 285, 64, 128] -> [1, 64, 36480]>][%7, %14, %16] : tensor<64x128xf16> -> tensor<1x64x36480xf16>
  %20 = rock.transform %1 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 80 + d2, d3)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{1, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <PassThrough ["d2"] at [3] -> ["d2"] at [2]>] bounds = [1, 1, 80, 768] -> [1, 80, 768]> : tensor<1x80x768xf16> to tensor<1x1x80x768xf16>
  %21 = rock.transform %20 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 + 64, d3)> by [<PassThrough ["d0", "d1b", "d2"] at [0, 1, 3] -> ["d0", "d1b", "d2"] at [0, 1, 3]>, <Slice{64, 80} ["d1i"] at [2] -> ["d1i"] at [2]>] bounds = [1, 1, 16, 768] -> [1, 1, 80, 768]> : tensor<1x1x80x768xf16> to tensor<1x1x16x768xf16>
  %22 = rock.transform %21 by <affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{1, 16} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <PassThrough ["d2"] at [2] -> ["d2"] at [3]>] bounds = [1, 16, 768] -> [1, 1, 16, 768]> : tensor<1x1x16x768xf16> to tensor<1x16x768xf16>
  %23 = tt.get_program_id x : i32
  %c10_i32_1 = arith.constant 10 : i32
  %c2850_i32_2 = arith.constant 2850 : i32
  %c1_i32_3 = arith.constant 1 : i32
  %c285_i32_4 = arith.constant 285 : i32
  %24 = arith.divui %23, %c285_i32_4 : i32
  %25 = arith.remui %23, %c285_i32_4 : i32
  %26 = arith.divui %25, %c2850_i32_2 : i32
  %27 = arith.muli %26, %c10_i32_1 : i32
  %28 = arith.subi %c1_i32_3, %27 : i32
  %29 = arith.minui %28, %c10_i32_1 : i32
  %30 = arith.remui %25, %29 : i32
  %31 = arith.addi %27, %30 : i32
  %32 = arith.remui %25, %c2850_i32_2 : i32
  %33 = arith.divui %32, %29 : i32
  %cst_5 = arith.constant dense<0.000000e+00> : tensor<16x128xf32>
  %c24_i32_6 = arith.constant 24 : i32
  %c1_i32_7 = arith.constant 1 : i32
  %c0_i32_8 = arith.constant 0 : i32
  %34 = scf.for %arg3 = %c0_i32_8 to %c24_i32_6 step %c1_i32_7 iter_args(%arg4 = %cst_5) -> (tensor<16x128xf32>)  : i32 {
    %49 = rock.load_marker %2 views [#rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 32 + d4, d3 * 128 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{24, 32} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{285, 128} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [24, 1, 1, 285, 32, 128] -> [1, 768, 36480]>][%arg3, %24, %31, %33] {cacheModifier = #rock<CacheModifier none>} : tensor<1x768x36480xf16> -> tensor<32x128xf16>
    %50 = rock.load_marker %22 views [#rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 16 + d4, d0 * 32 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{24, 32} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{1, 16} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{285} ["n_block"] at [3] -> [] at []>] bounds = [24, 1, 1, 285, 16, 32] -> [1, 16, 768]>][%arg3, %24, %31, %33] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x768xf16> -> tensor<16x32xf16>
    %51 = rock.blockwise_gemm(%50, %49, %arg4) : tensor<16x32xf16>, tensor<32x128xf16>, tensor<16x128xf32> -> tensor<16x128xf32>
    scf.yield %51 : tensor<16x128xf32>
  }
  %35 = arith.truncf %34 : tensor<16x128xf32> to tensor<16x128xf16>
  %36 = rock.store_marker %35 views [#rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 16 + d3, d2 * 128 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 16} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{285, 128} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 285, 16, 128] -> [1, 16, 36480]>][%24, %31, %33] : tensor<16x128xf16> -> tensor<1x16x36480xf16>
  %37 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 36480 + d2)> by [<Unmerge{77, 36480} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 77, 36480] -> [2808960]> : tensor<2808960xf16> to tensor<1x77x36480xf16>
  %38 = rock.transform %37 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 3} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 80, 36480] -> [1, 77, 36480]> : tensor<1x77x36480xf16> to tensor<1x80x36480xf16>
  %39 = rock.transform %38 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 80 + d2, d3)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{1, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <PassThrough ["d2"] at [3] -> ["d2"] at [2]>] bounds = [1, 1, 80, 36480] -> [1, 80, 36480]> : tensor<1x80x36480xf16> to tensor<1x1x80x36480xf16>
  %40 = rock.transform %39 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> by [<PassThrough ["d0", "d1b", "d2"] at [0, 1, 3] -> ["d0", "d1b", "d2"] at [0, 1, 3]>, <Slice{0, 64} ["d1i"] at [2] -> ["d1i"] at [2]>] bounds = [1, 1, 64, 36480] -> [1, 1, 80, 36480]> : tensor<1x1x80x36480xf16> to tensor<1x1x64x36480xf16>
  %41 = rock.transform %40 by <affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{1, 64} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <PassThrough ["d2"] at [2] -> ["d2"] at [3]>] bounds = [1, 64, 36480] -> [1, 1, 64, 36480]> : tensor<1x1x64x36480xf16> to tensor<1x64x36480xf16>
  %42 = rock.store %19 to %41 alias %arg2 by set : tensor<1x64x36480xf16> -> tensor<2808960xf16> to tensor<1x64x36480xf16> alias tensor<2808960xf16>
  %43 = rock.transform %42 by <affine_map<(d0, d1, d2) -> (d1 * 36480 + d2)> by [<Unmerge{77, 36480} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 77, 36480] -> [2808960]> : tensor<2808960xf16> to tensor<1x77x36480xf16>
  %44 = rock.transform %43 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 3} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 80, 36480] -> [1, 77, 36480]> : tensor<1x77x36480xf16> to tensor<1x80x36480xf16>
  %45 = rock.transform %44 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 80 + d2, d3)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{1, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <PassThrough ["d2"] at [3] -> ["d2"] at [2]>] bounds = [1, 1, 80, 36480] -> [1, 80, 36480]> : tensor<1x80x36480xf16> to tensor<1x1x80x36480xf16>
  %46 = rock.transform %45 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 + 64, d3)> by [<PassThrough ["d0", "d1b", "d2"] at [0, 1, 3] -> ["d0", "d1b", "d2"] at [0, 1, 3]>, <Slice{64, 80} ["d1i"] at [2] -> ["d1i"] at [2]>] bounds = [1, 1, 16, 36480] -> [1, 1, 80, 36480]> : tensor<1x1x80x36480xf16> to tensor<1x1x16x36480xf16>
  %47 = rock.transform %46 by <affine_map<(d0, d1, d2) -> (d0, 0, d1, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{1, 16} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <PassThrough ["d2"] at [2] -> ["d2"] at [3]>] bounds = [1, 16, 36480] -> [1, 1, 16, 36480]> : tensor<1x1x16x36480xf16> to tensor<1x16x36480xf16>
  %48 = rock.store %36 to %47 alias %42 by set : tensor<1x16x36480xf16> -> tensor<2808960xf16> to tensor<1x16x36480xf16> alias tensor<2808960xf16>
  return %48 : tensor<2808960xf16>
}
