// Unit tests for rock-fusion-splitk-regularization pass.
// Tests that addf/subf with an external operand get that operand divided by
// splitKFactor, while other fusion ops (mulf, extf, etc.) are unchanged.

// RUN: rocmlir-opt -rock-fusion-splitk-regularization -mlir-print-local-scope %s | FileCheck %s

// Split-K introduces `arith.divf` to scale the fused bias/other term. When
// `rock-allow-fast-math-flags` runs after (same order as the kernel
// pipeline), those divisions pick up `fastmath<nsz,arcp,afn>`; in
// particular, `arcp` allows lowering to treat them as
// multiply-by-reciprocal.

// RUN: rocmlir-opt -rock-fusion-splitk-regularization -rock-allow-fast-math-flags -mlir-print-local-scope %s | FileCheck %s --check-prefix=RECIP

module {

  // ============================================================
  // NO-OP: splitKFactor = 1 means no modification at all.
  // ============================================================

  // CHECK-LABEL: func.func @test_noop_splitk1
  // CHECK-NOT: arith.divf
  // CHECK-NOT: arith.constant dense<
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[F:.*]] = arith.addf %[[G]], %{{.*}} : tensor<1x4x4xf16>
  // CHECK: rock.store %[[F]]
  // CHECK-NOT: arith.divf
  func.func @test_noop_splitk1(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %fused = arith.addf %gemm, %ext : tensor<1x4x4xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }

  // ============================================================
  // ADDF: gemm + ext → ext is divided by splitKFactor before add.
  // Input:  arith.addf %gemm, %ext
  // Expect: arith.divf %ext, splat(4.0); arith.addf %gemm, %divf
  // ============================================================

  // CHECK-LABEL: func.func @test_addf
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[CST:.*]] = arith.constant dense<4.000000e+00> : tensor<1x4x4xf16>
  // CHECK: %[[DIV:.*]] = arith.divf %{{.*}}, %[[CST]] : tensor<1x4x4xf16>
  // CHECK: %[[F:.*]] = arith.addf %[[G]], %[[DIV]] : tensor<1x4x4xf16>
  // CHECK: rock.store %[[F]]
  func.func @test_addf(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %fused = arith.addf %gemm, %ext : tensor<1x4x4xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }

  // ============================================================
  // ADDF reversed: ext + gemm → ext still gets divided,
  // but operand order becomes addf gemm, divf(ext).
  // ============================================================

  // CHECK-LABEL: func.func @test_addf_reversed
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[CST:.*]] = arith.constant dense<4.000000e+00> : tensor<1x4x4xf16>
  // CHECK: %[[DIV:.*]] = arith.divf %{{.*}}, %[[CST]] : tensor<1x4x4xf16>
  // CHECK: %[[F:.*]] = arith.addf %[[G]], %[[DIV]] : tensor<1x4x4xf16>
  // CHECK: rock.store %[[F]]
  func.func @test_addf_reversed(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %fused = arith.addf %ext, %gemm : tensor<1x4x4xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }

  // ============================================================
  // SUBF: gemm - ext → ext divided, operand order preserved.
  // ============================================================

  // CHECK-LABEL: func.func @test_subf
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[CST:.*]] = arith.constant dense<4.000000e+00> : tensor<1x4x4xf16>
  // CHECK: %[[DIV:.*]] = arith.divf %{{.*}}, %[[CST]] : tensor<1x4x4xf16>
  // CHECK: %[[F:.*]] = arith.subf %[[G]], %[[DIV]] : tensor<1x4x4xf16>
  // CHECK: rock.store %[[F]]
  func.func @test_subf(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %fused = arith.subf %gemm, %ext : tensor<1x4x4xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }

  // ============================================================
  // SUBF reversed: ext - gemm → ext divided, subf(divf(ext), gemm).
  // ============================================================

  // CHECK-LABEL: func.func @test_subf_reversed
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[CST:.*]] = arith.constant dense<4.000000e+00> : tensor<1x4x4xf16>
  // CHECK: %[[DIV:.*]] = arith.divf %{{.*}}, %[[CST]] : tensor<1x4x4xf16>
  // CHECK: %[[F:.*]] = arith.subf %[[DIV]], %[[G]] : tensor<1x4x4xf16>
  // CHECK: rock.store %[[F]]
  func.func @test_subf_reversed(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %fused = arith.subf %ext, %gemm : tensor<1x4x4xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }

  // ============================================================
  // ADDF same: both operands from gemm → no division needed.
  // ============================================================

  // CHECK-LABEL: func.func @test_addf_same
  // CHECK-NOT: arith.divf
  // CHECK-NOT: arith.constant dense<
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[F:.*]] = arith.addf %[[G]], %[[G]] : tensor<1x4x4xf16>
  // CHECK: rock.store %[[F]]
  // CHECK-NOT: arith.divf
  func.func @test_addf_same(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %fused = arith.addf %gemm, %gemm : tensor<1x4x4xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }

  // ============================================================
  // SUBF same: both operands from gemm → no division needed.
  // ============================================================

  // CHECK-LABEL: func.func @test_subf_same
  // CHECK-NOT: arith.divf
  // CHECK-NOT: arith.constant dense<
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[F:.*]] = arith.subf %[[G]], %[[G]] : tensor<1x4x4xf16>
  // CHECK: rock.store %[[F]]
  // CHECK-NOT: arith.divf
  func.func @test_subf_same(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %fused = arith.subf %gemm, %gemm : tensor<1x4x4xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }

  // ============================================================
  // MULF: not addf/subf → pass leaves it unchanged.
  // ============================================================

  // CHECK-LABEL: func.func @test_mulf
  // CHECK-NOT: arith.divf
  // CHECK-NOT: arith.constant dense<
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[F:.*]] = arith.mulf %[[G]], %{{.*}} : tensor<1x4x4xf16>
  // CHECK: rock.store %[[F]]
  // CHECK-NOT: arith.divf
  func.func @test_mulf(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %fused = arith.mulf %gemm, %ext : tensor<1x4x4xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }

  // ============================================================
  // Cascaded: addf + mulf chain. Only addf has external operand,
  // so only addf's external operand gets divided.
  // ============================================================

  // CHECK-LABEL: func.func @test_cascaded
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[CST:.*]] = arith.constant dense<4.000000e+00> : tensor<1x4x4xf16>
  // CHECK: %[[DIV:.*]] = arith.divf %{{.*}}, %[[CST]] : tensor<1x4x4xf16>
  // CHECK: %[[ADD:.*]] = arith.addf %[[G]], %[[DIV]] : tensor<1x4x4xf16>
  // mulf is unchanged (not addf/subf)
  // CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %{{.*}} : tensor<1x4x4xf16>
  // CHECK: rock.store %[[MUL]]
  func.func @test_cascaded(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext1: tensor<1x4x4xf16>, %ext2: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %f1 = arith.addf %gemm, %ext1 : tensor<1x4x4xf16>
    %f2 = arith.mulf %f1, %ext2 : tensor<1x4x4xf16>
    %r = rock.store %f2 to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }

  // ============================================================
  // Cascaded extf + addf: gemm → extf → addf with f32 external.
  // The addf's external operand is f32, so constant and divf are f32.
  // ============================================================

  // CHECK-LABEL: func.func @test_extf_then_addf
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[EXT:.*]] = arith.extf %[[G]] : tensor<1x4x4xf16> to tensor<1x4x4xf32>
  // CHECK: %[[CST:.*]] = arith.constant dense<4.000000e+00> : tensor<1x4x4xf32>
  // CHECK: %[[DIV:.*]] = arith.divf %{{.*}}, %[[CST]] : tensor<1x4x4xf32>
  // CHECK: %[[F:.*]] = arith.addf %[[EXT]], %[[DIV]] : tensor<1x4x4xf32>
  // CHECK: rock.store %[[F]]
  func.func @test_extf_then_addf(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext: tensor<1x4x4xf32>, %dest: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %ext_gemm = arith.extf %gemm : tensor<1x4x4xf16> to tensor<1x4x4xf32>
    %fused = arith.addf %ext_gemm, %ext : tensor<1x4x4xf32>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf32> -> tensor<1x4x4xf32> to tensor<1x4x4xf32>
    return %r : tensor<1x4x4xf32>
  }

  // ============================================================
  // Multiple addf in chain: both should get their external
  // operands divided independently.
  // ============================================================

  // CHECK-LABEL: func.func @test_two_adds
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[CST1:.*]] = arith.constant dense<4.000000e+00> : tensor<1x4x4xf16>
  // CHECK: %[[DIV1:.*]] = arith.divf %{{.*}}, %[[CST1]] : tensor<1x4x4xf16>
  // CHECK: %[[ADD1:.*]] = arith.addf %[[G]], %[[DIV1]] : tensor<1x4x4xf16>
  // CHECK: %[[CST2:.*]] = arith.constant dense<4.000000e+00> : tensor<1x4x4xf16>
  // CHECK: %[[DIV2:.*]] = arith.divf %{{.*}}, %[[CST2]] : tensor<1x4x4xf16>
  // CHECK: %[[ADD2:.*]] = arith.addf %[[ADD1]], %[[DIV2]] : tensor<1x4x4xf16>
  // CHECK: rock.store %[[ADD2]]
  func.func @test_two_adds(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext1: tensor<1x4x4xf16>, %ext2: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %f1 = arith.addf %gemm, %ext1 : tensor<1x4x4xf16>
    %f2 = arith.addf %f1, %ext2 : tensor<1x4x4xf16>
    %r = rock.store %f2 to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }

  // ============================================================
  // Different splitKFactor: verify the constant matches the factor.
  // ============================================================

  // CHECK-LABEL: func.func @test_addf_splitk2
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<1x4x4xf16>
  // CHECK: %[[DIV:.*]] = arith.divf %{{.*}}, %[[CST]] : tensor<1x4x4xf16>
  // CHECK: %[[F:.*]] = arith.addf %[[G]], %[[DIV]] : tensor<1x4x4xf16>
  // CHECK: rock.store %[[F]]
  func.func @test_addf_splitk2(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 2, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %fused = arith.addf %gemm, %ext : tensor<1x4x4xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }

  // ============================================================
  // ADDF with splat constant: constant(10.0)/4 gets folded to 2.5.
  // ============================================================

  // CHECK-LABEL: func.func @test_addf_constant
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: arith.constant dense<2.500000e+00> : tensor<1x4x4xf32>
  // CHECK-NOT: arith.constant dense<1.000000e+01>
  // CHECK: %[[F:.*]] = arith.addf %[[G]], %{{.*}} : tensor<1x4x4xf32>
  // CHECK: rock.store %[[F]]
  func.func @test_addf_constant(%a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %dest: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> attributes {rock.kernel} {
    %cst = arith.constant dense<1.000000e+01> : tensor<1x4x4xf32>
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf32> * tensor<1x4x4xf32> -> tensor<1x4x4xf32>
    %fused = arith.addf %gemm, %cst : tensor<1x4x4xf32>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf32> -> tensor<1x4x4xf32> to tensor<1x4x4xf32>
    return %r : tensor<1x4x4xf32>
  }

  // ============================================================
  // SUBF with splat constant: constant(10.0)/4 folded to 2.5.
  // ============================================================

  // CHECK-LABEL: func.func @test_subf_constant
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: arith.constant dense<2.500000e+00> : tensor<1x4x4xf32>
  // CHECK-NOT: arith.constant dense<1.000000e+01>
  // CHECK: %[[F:.*]] = arith.subf %[[G]], %{{.*}} : tensor<1x4x4xf32>
  // CHECK: rock.store %[[F]]
  func.func @test_subf_constant(%a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %dest: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> attributes {rock.kernel} {
    %cst = arith.constant dense<1.000000e+01> : tensor<1x4x4xf32>
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf32> * tensor<1x4x4xf32> -> tensor<1x4x4xf32>
    %fused = arith.subf %gemm, %cst : tensor<1x4x4xf32>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf32> -> tensor<1x4x4xf32> to tensor<1x4x4xf32>
    return %r : tensor<1x4x4xf32>
  }

  // ============================================================
  // Multiple fusion ops: addf(gemm,ext1) + extf(ext2) + mulf(gemm,ext2_f32)
  // + addf + truncf. First addf gets ext1 divided; second addf has both
  // operands from chain so is left alone.
  // ============================================================

  // CHECK-LABEL: func.func @test_multiple_ops
  // CHECK: %[[G:.*]] = rock.gemm
  // First addf: ext1 is divided
  // CHECK: %[[CST:.*]] = arith.constant dense<4.000000e+00> : tensor<1x4x4xf32>
  // CHECK: %[[DIV:.*]] = arith.divf %{{.*}}, %[[CST]] : tensor<1x4x4xf32>
  // CHECK: %[[ADD1:.*]] = arith.addf %[[G]], %[[DIV]] : tensor<1x4x4xf32>
  // extf and mulf are untouched
  // CHECK: %[[EXTF:.*]] = arith.extf %{{.*}} : tensor<1x4x4xf16> to tensor<1x4x4xf32>
  // CHECK: %[[MUL:.*]] = arith.mulf %[[G]], %[[EXTF]] : tensor<1x4x4xf32>
  // Second addf: both operands from chain → no divf
  // CHECK: %[[ADD2:.*]] = arith.addf %[[ADD1]], %[[MUL]] : tensor<1x4x4xf32>
  // CHECK: %[[TR:.*]] = arith.truncf %[[ADD2]] : tensor<1x4x4xf32> to tensor<1x4x4xf16>
  // CHECK: rock.store %[[TR]]
  func.func @test_multiple_ops(%a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %ext1: tensor<1x4x4xf32>, %ext2: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf32> * tensor<1x4x4xf32> -> tensor<1x4x4xf32>
    %add1 = arith.addf %gemm, %ext1 : tensor<1x4x4xf32>
    %ext2_f32 = arith.extf %ext2 : tensor<1x4x4xf16> to tensor<1x4x4xf32>
    %mul = arith.mulf %gemm, %ext2_f32 : tensor<1x4x4xf32>
    %add2 = arith.addf %add1, %mul : tensor<1x4x4xf32>
    %trunc = arith.truncf %add2 : tensor<1x4x4xf32> to tensor<1x4x4xf16>
    %r = rock.store %trunc to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }

  // ============================================================
  // Multi-output: gemm → mulf → two separate addf paths, each
  // adding a different constant to a different output.
  // Each constant gets independently divided: 1.0→0.25, 2.0→0.5.
  // ============================================================

  // CHECK-LABEL: func.func @test_multi_output
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[MUL:.*]] = arith.mulf %[[G]], %{{.*}} : tensor<1x4x4xf32>
  // First output: 1.0/4 = 0.25
  // CHECK: arith.constant dense<2.500000e-01> : tensor<1x4x4xf32>
  // CHECK: %[[OUT1:.*]] = arith.addf %[[MUL]], %{{.*}} : tensor<1x4x4xf32>
  // Second output: 2.0/4 = 0.5
  // CHECK: arith.constant dense<5.000000e-01> : tensor<1x4x4xf32>
  // CHECK: %[[OUT2:.*]] = arith.addf %[[MUL]], %{{.*}} : tensor<1x4x4xf32>
  // CHECK: rock.store %[[OUT1]]
  // CHECK: rock.store %[[OUT2]]
  func.func @test_multi_output(%a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %dest1: tensor<1x4x4xf32>, %dest2: tensor<1x4x4xf32>) -> (tensor<1x4x4xf32>, tensor<1x4x4xf32>) attributes {rock.kernel} {
    %cst1 = arith.constant dense<1.000000e+00> : tensor<1x4x4xf32>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<1x4x4xf32>
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf32> * tensor<1x4x4xf32> -> tensor<1x4x4xf32>
    %scale = arith.mulf %gemm, %cst1 : tensor<1x4x4xf32>
    %out1 = arith.addf %scale, %cst1 : tensor<1x4x4xf32>
    %out2 = arith.addf %scale, %cst2 : tensor<1x4x4xf32>
    %r1 = rock.store %out1 to %dest1 by set : tensor<1x4x4xf32> -> tensor<1x4x4xf32> to tensor<1x4x4xf32>
    %r2 = rock.store %out2 to %dest2 by set : tensor<1x4x4xf32> -> tensor<1x4x4xf32> to tensor<1x4x4xf32>
    return %r1, %r2 : tensor<1x4x4xf32>, tensor<1x4x4xf32>
  }

  // ============================================================
  // Add twice to same output: gemm → mulf → addf(cst1) → addf(cst2).
  // Both addf get their constants divided: 1.0→0.25, 3.0→0.75.
  // ============================================================

  // CHECK-LABEL: func.func @test_add_twice_same_output
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[MUL:.*]] = arith.mulf %[[G]], %{{.*}} : tensor<1x4x4xf32>
  // First add: 1.0/4 = 0.25
  // CHECK: arith.constant dense<2.500000e-01> : tensor<1x4x4xf32>
  // CHECK: %[[ADD1:.*]] = arith.addf %[[MUL]], %{{.*}} : tensor<1x4x4xf32>
  // Second add: 3.0/4 = 0.75
  // CHECK: arith.constant dense<7.500000e-01> : tensor<1x4x4xf32>
  // CHECK: %[[ADD2:.*]] = arith.addf %[[ADD1]], %{{.*}} : tensor<1x4x4xf32>
  // CHECK: rock.store %[[ADD2]]
  func.func @test_add_twice_same_output(%a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %dest: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> attributes {rock.kernel} {
    %cst1 = arith.constant dense<1.000000e+00> : tensor<1x4x4xf32>
    %cst2 = arith.constant dense<3.000000e+00> : tensor<1x4x4xf32>
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf32> * tensor<1x4x4xf32> -> tensor<1x4x4xf32>
    %scale = arith.mulf %gemm, %cst1 : tensor<1x4x4xf32>
    %add1 = arith.addf %scale, %cst1 : tensor<1x4x4xf32>
    %add2 = arith.addf %add1, %cst2 : tensor<1x4x4xf32>
    %r = rock.store %add2 to %dest by set : tensor<1x4x4xf32> -> tensor<1x4x4xf32> to tensor<1x4x4xf32>
    return %r : tensor<1x4x4xf32>
  }

  // ============================================================
  // Multiply-with-reciprocal path (pipeline): split-k inserts divf on the
  // fused operand; rock-allow-fast-math-flags tags those divfs with arcp.
  // Checked only by the second RUN line (RECIP prefix) above.
  // ============================================================

  // RECIP-LABEL: func.func @test_multiply_with_reciprocal_addf
  // RECIP-DAG: %[[CST:.*]] = arith.constant dense<4.000000e+00> : tensor<1x4x4xf16>
  // RECIP-DAG: %[[G:.*]] = rock.gemm
  // RECIP: %[[DIV:.*]] = arith.divf %{{.*}}, %[[CST]] fastmath<nsz,arcp,afn> : tensor<1x4x4xf16>
  // RECIP: %[[F:.*]] = arith.addf %[[G]], %[[DIV]] fastmath<nsz,contract> : tensor<1x4x4xf16>
  // RECIP: rock.store %[[F]]
  func.func @test_multiply_with_reciprocal_addf(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext: tensor<1x4x4xf16>, %dest: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %fused = arith.addf %gemm, %ext : tensor<1x4x4xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4xf16> -> tensor<1x4x4xf16> to tensor<1x4x4xf16>
    return %r : tensor<1x4x4xf16>
  }
}
