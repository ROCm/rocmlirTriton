// Tests that `sumErrorTolerance` uses different per-accumulation-step bounds
// for fixed vs random input data. For f32:
//   fixed  (seed = "fixed"): sumErrTol = 1e-6
//   random (seed != "fixed"): sumErrTol = 1e-5
//
// Both cases use the same GEMM with K=64 and f32, so the only variable is
// the per-step bound. The observable is the `atol` constant emitted before
// the `mcpuVerifyFloatAllclose` call:
//   atol = baseAtol + K_eff * sumErrTol
//        = 1e-5    + 64    * sumErrTol

func.func private @gemm_fut(%arg0: tensor<1x256x64xf32>, %arg1: tensor<1x64x128xf32>) -> tensor<1x256x128xf32> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp {acc_type = f32} : (tensor<1x256x64xf32>, tensor<1x64x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x256x128xf32>
  return %0 : tensor<1x256x128xf32>
}

// ============================================================================
// (1) Random data (`-rand 1`). sumErrTol(f32, random) = 1e-5.
// atol = 1e-5 + 64 * 1e-5 = 6.5e-4.
// ============================================================================

// RUN: rocmlir-gen -fut gemm_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=highlevel -host-pipeline=highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut gemm_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=RANDOM --enable-var-scope

// RANDOM:      arith.constant 6.5{{[0-9]*}}e-04 : f32
// RANDOM-NEXT: arith.constant 1.300000e-06 : f32
// RANDOM:      call @mcpuVerifyFloatAllclose

// ============================================================================
// (2) Fixed data (`-rand fixed`). sumErrTol(f32, fixed) = 1e-6.
// atol = 1e-5 + 64 * 1e-6 = 7.4e-5.
// ============================================================================

// RUN: rocmlir-gen -fut gemm_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=highlevel -host-pipeline=highlevel \
// RUN:   | rocmlir-gen -ph -rand fixed -rand_type float -fut gemm_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=FIXED --enable-var-scope

// The random-data atol (6.5e-4) must NOT appear -- proves the isRandom flag
// selects a tighter per-step bound for fixed data.
// FIXED-NOT:  arith.constant 6.5{{[0-9]*}}e-04
// FIXED:      arith.constant 7.{{[0-9]+}}E-5 : f32
// FIXED-NEXT: arith.constant 1.300000e-06 : f32
// FIXED:      call @mcpuVerifyFloatAllclose

// ============================================================================
// (3) No-data mode (`-rand none`). Same as fixed: sumErrTol(f32, fixed) = 1e-6.
// atol = 1e-5 + 64 * 1e-6 = 7.4e-5.
// ============================================================================

// RUN: rocmlir-gen -fut gemm_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=highlevel -host-pipeline=highlevel \
// RUN:   | rocmlir-gen -ph -rand none -rand_type float -fut gemm_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=NONE --enable-var-scope

// NONE:      arith.constant 7.{{[0-9]+}}E-5 : f32
// NONE-NEXT: arith.constant 1.300000e-06 : f32
// NONE:      call @mcpuVerifyFloatAllclose
