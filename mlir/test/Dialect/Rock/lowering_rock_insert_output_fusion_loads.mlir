// Unit tests for rock-insert-output-fusion-loads pass.
// Tests that extra fusion inputs (operands of fusion ops NOT from the GEMM chain)
// get rock.load_marker + rock.untile inserted, while chain-only fusions and
// non-kernel functions remain unchanged.

// RUN: rocmlir-opt -rock-insert-output-fusion-loads -mlir-print-local-scope %s | FileCheck %s

#tmap = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 16 + d3, d2 * 16 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 16} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{1, 16} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 1, 16, 16] -> [1, 16, 16]>

module {

  // ============================================================
  // ADDF with external bias tensor.
  // The bias is an extra input → gets load_marker + untile.
  // ============================================================

  // The pass streams epilogue output-fusion inputs (cs), since they are read
  // once per output tile with no reuse.
  // CHECK-LABEL: func.func @test_addf_bias
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: rock.transform %{{.*}} by
  // CHECK: %[[LM:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: {cacheModifier = #rock<CacheModifier cs>}
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[LM]] : tensor<16x16xf32> -> tensor<1x16x16xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[SM]], %[[UT]] : tensor<1x16x16xf32>
  // CHECK: rock.store %[[ADD]]
  func.func @test_addf_bias(%tile: tensor<16x16xf32>, %bias: tensor<1x16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %marker = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %fused = arith.addf %marker, %bias : tensor<1x16x16xf32>
    %r = rock.store %fused to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // SUBF with external bias tensor.
  // ============================================================

  // CHECK-LABEL: func.func @test_subf_bias
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[LM:.*]] = rock.load_marker %{{.*}} views
  // CHECK: %[[UT:.*]] = rock.untile %[[LM]] : tensor<16x16xf32> -> tensor<1x16x16xf32>
  // CHECK: %[[SUB:.*]] = arith.subf %[[SM]], %[[UT]] : tensor<1x16x16xf32>
  // CHECK: rock.store %[[SUB]]
  func.func @test_subf_bias(%tile: tensor<16x16xf32>, %bias: tensor<1x16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %marker = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %fused = arith.subf %marker, %bias : tensor<1x16x16xf32>
    %r = rock.store %fused to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // TRUNCF only: single-operand chain fusion → no extra inputs.
  // Pass should NOT insert any load_marker or untile.
  // ============================================================

  // CHECK-LABEL: func.func @test_truncf_only
  // CHECK: rock.store_marker
  // CHECK-NOT: rock.load_marker
  // CHECK-NOT: rock.untile
  // CHECK: arith.truncf
  // CHECK: rock.store
  func.func @test_truncf_only(%tile: tensor<16x16xf32>, %dest: tensor<1x16x16xf16>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf16> attributes {rock.kernel} {
    %marker = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %trunc = arith.truncf %marker : tensor<1x16x16xf32> to tensor<1x16x16xf16>
    %r = rock.store %trunc to %dest by set : tensor<1x16x16xf16> -> tensor<1x16x16xf16> to tensor<1x16x16xf16>
    return %r : tensor<1x16x16xf16>
  }

  // ============================================================
  // No fusion: store_marker → store directly.
  // Pass should NOT insert anything.
  // ============================================================

  // CHECK-LABEL: func.func @test_no_fusion
  // CHECK: rock.store_marker
  // CHECK-NOT: rock.load_marker
  // CHECK-NOT: rock.untile
  // CHECK: rock.store
  func.func @test_no_fusion(%tile: tensor<16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %marker = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %r = rock.store %marker to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Two extra inputs: addf(marker, bias) then mulf(result, scale).
  // Both bias and scale are extra inputs → each gets load_marker + untile.
  // ============================================================

  // CHECK-LABEL: func.func @test_two_extra_inputs
  // CHECK: %[[SM:.*]] = rock.store_marker
  // Two load_marker + untile pairs (order may vary due to DenseMap iteration)
  // CHECK-DAG: %[[LM1:.*]] = rock.load_marker %{{.*}} views
  // CHECK-DAG: %[[UT1:.*]] = rock.untile %{{.*}} : tensor<16x16xf32> -> tensor<1x16x16xf32>
  // CHECK-DAG: %[[LM2:.*]] = rock.load_marker %{{.*}} views
  // CHECK-DAG: %[[UT2:.*]] = rock.untile %{{.*}} : tensor<16x16xf32> -> tensor<1x16x16xf32>
  // Fusion ops consume the untiled values
  // CHECK: %[[ADD:.*]] = arith.addf %[[SM]], %{{.*}} : tensor<1x16x16xf32>
  // CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %{{.*}} : tensor<1x16x16xf32>
  // CHECK: rock.store %[[MUL]]
  func.func @test_two_extra_inputs(%tile: tensor<16x16xf32>, %bias: tensor<1x16x16xf32>, %scale: tensor<1x16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %marker = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %marker, %bias : tensor<1x16x16xf32>
    %mul = arith.mulf %add, %scale : tensor<1x16x16xf32>
    %r = rock.store %mul to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Bias goes through rock.transform before fusion.
  // The transformed bias is the extra input (load_marker sources from
  // the transform result, not the raw argument).
  // ============================================================

  // CHECK-LABEL: func.func @test_bias_through_transform
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[BIAS_T:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<16xf32> to tensor<1x16x16xf32>
  // CHECK: rock.transform %[[BIAS_T]] by
  // CHECK: %[[LM:.*]] = rock.load_marker %[[BIAS_T]] views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[LM]] : tensor<16x16xf32> -> tensor<1x16x16xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[SM]], %[[UT]] : tensor<1x16x16xf32>
  // CHECK: rock.store %[[ADD]]
  func.func @test_bias_through_transform(%tile: tensor<16x16xf32>, %bias_raw: tensor<16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %marker = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %bias_3d = rock.transform %bias_raw by <affine_map<(d0, d1, d2) -> (d2)> by [<AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{16} ["m"] at [1] -> [] at []>, <PassThrough ["n"] at [2] -> ["n"] at [0]>] bounds = [1, 16, 16] -> [16]> : tensor<16xf32> to tensor<1x16x16xf32>
    %add = arith.addf %marker, %bias_3d : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Shared extra input: same bias used twice in the chain.
  // Only one load_marker + untile is created; both addf ops
  // reference the same untile result.
  // ============================================================

  // CHECK-LABEL: func.func @test_shared_extra_input
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[LM:.*]] = rock.load_marker %{{.*}} views
  // CHECK: %[[UT:.*]] = rock.untile %[[LM]] : tensor<16x16xf32> -> tensor<1x16x16xf32>
  // CHECK-NOT: rock.load_marker
  // CHECK: %[[ADD1:.*]] = arith.addf %[[SM]], %[[UT]] : tensor<1x16x16xf32>
  // CHECK: %[[ADD2:.*]] = arith.addf %[[ADD1]], %[[UT]] : tensor<1x16x16xf32>
  // CHECK: rock.store %[[ADD2]]
  func.func @test_shared_extra_input(%tile: tensor<16x16xf32>, %bias: tensor<1x16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %marker = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add1 = arith.addf %marker, %bias : tensor<1x16x16xf32>
    %add2 = arith.addf %add1, %bias : tensor<1x16x16xf32>
    %r = rock.store %add2 to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Type conversion in chain: store_marker(f16) → extf(f32) → addf(f32 bias).
  // The f32 bias is the extra input.
  // ============================================================

  // CHECK-LABEL: func.func @test_extf_then_addf
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[LM:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[LM]] : tensor<16x16xf32> -> tensor<1x16x16xf32>
  // CHECK: %[[EXT:.*]] = arith.extf %[[SM]] : tensor<1x16x16xf16> to tensor<1x16x16xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[EXT]], %[[UT]] : tensor<1x16x16xf32>
  // CHECK: rock.store %[[ADD]]
  func.func @test_extf_then_addf(%tile: tensor<16x16xf16>, %bias: tensor<1x16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %marker = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf16> -> tensor<1x16x16xf16>
    %ext = arith.extf %marker : tensor<1x16x16xf16> to tensor<1x16x16xf32>
    %add = arith.addf %ext, %bias : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Constant output fusion: subf(marker, constant_splat).
  // The constant is an extra input → gets load_marker + untile.
  // ============================================================

  // CHECK-LABEL: func.func @test_subf_constant
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<1x16x16xf32>
  // CHECK: %[[LM:.*]] = rock.load_marker %[[CST]] views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[LM]] : tensor<16x16xf32> -> tensor<1x16x16xf32>
  // CHECK: %[[SUB:.*]] = arith.subf %[[SM]], %[[UT]] : tensor<1x16x16xf32>
  // CHECK: rock.store %[[SUB]]
  func.func @test_subf_constant(%tile: tensor<16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %marker = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %cst = arith.constant dense<1.000000e+00> : tensor<1x16x16xf32>
    %fused = arith.subf %marker, %cst : tensor<1x16x16xf32>
    %r = rock.store %fused to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Multi-tensor add: addf(t1, t2) is outside chain, its result
  // is the extra input to addf(marker, sum).
  // ============================================================

  // CHECK-LABEL: func.func @test_add_tensors
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[SUM:.*]] = arith.addf %{{.*}}, %{{.*}} : tensor<1x16x16xf16>
  // CHECK: %[[LM:.*]] = rock.load_marker %[[SUM]] views
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // CHECK: %[[UT:.*]] = rock.untile %[[LM]] : tensor<16x16xf16> -> tensor<1x16x16xf16>
  // CHECK: %[[ADD:.*]] = arith.addf %[[SM]], %[[UT]] : tensor<1x16x16xf16>
  // CHECK: rock.store %[[ADD]]
  func.func @test_add_tensors(%tile: tensor<16x16xf16>,
      %t1: tensor<1x16x16xf16>, %t2: tensor<1x16x16xf16>,
      %dest: tensor<1x16x16xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf16> attributes {rock.kernel} {
    %marker = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf16> -> tensor<1x16x16xf16>
    %sum = arith.addf %t1, %t2 : tensor<1x16x16xf16>
    %fused = arith.addf %marker, %sum : tensor<1x16x16xf16>
    %r = rock.store %fused to %dest by set : tensor<1x16x16xf16> -> tensor<1x16x16xf16> to tensor<1x16x16xf16>
    return %r : tensor<1x16x16xf16>
  }

  // ============================================================
  // Non-kernel function: pass skips (no rock.kernel attribute).
  // Nothing should be modified.
  // ============================================================

  // CHECK-LABEL: func.func @test_non_kernel
  // CHECK-NOT: rock.load_marker
  // CHECK-NOT: rock.untile
  // CHECK: rock.store_marker
  // CHECK: arith.addf
  // CHECK: rock.store
  func.func @test_non_kernel(%tile: tensor<16x16xf32>, %bias: tensor<1x16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> {
    %marker = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %fused = arith.addf %marker, %bias : tensor<1x16x16xf32>
    %r = rock.store %fused to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }
}
