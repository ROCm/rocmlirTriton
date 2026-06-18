// Tests that `-atol` / `-rtol` replace (not augment) the K-scaled
// defaults. 

func.func private @gemm_fut(%arg0: tensor<1x256x64xf32>, %arg1: tensor<1x64x128xf32>) -> tensor<1x256x128xf32> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp {acc_type = f32} : (tensor<1x256x64xf32>, tensor<1x64x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x256x128xf32>
  return %0 : tensor<1x256x128xf32>
}

// ============================================================================
// (1) `-atol=0.015625` overrides the K-scaled default. Setting `-atol` also
// implies `--comparator=allclose`, so `mcpuVerifyFloatAllclose` is emitted.
// atol must be exactly the user-supplied value (not `default + override`);
// rtol falls back to the fp32 baseline `1.3e-6`.
// ============================================================================

// RUN: rocmlir-gen -fut gemm_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=highlevel -host-pipeline=highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut gemm_fut --verifier clone -atol=0.015625 - \
// RUN:   | FileCheck %s --check-prefix=ATOL_OVERRIDE --enable-var-scope

// `-atol` must imply allclose; the legacy three-threshold call must not appear.
// ATOL_OVERRIDE-NOT: call @mcpuVerifyFloat(
// The K-scaled default `6.5e-4` (= 1e-5 + 64*1e-5) must NOT appear -- override replaces it.
// ATOL_OVERRIDE-NOT: arith.constant 6.5{{[0-9]*}}e-04
// The user's atol is emitted verbatim (= 1.5625e-2, exactly representable).
// ATOL_OVERRIDE:     arith.constant 1.562500e-02 : f32
// rtol falls back to the fp32 PyTorch baseline.
// ATOL_OVERRIDE-NEXT: arith.constant 1.300000e-06 : f32
// ATOL_OVERRIDE:     call @mcpuVerifyFloatAllclose

// ============================================================================
// (2) `-rtol=0.0625` overrides the rtol baseline. atol falls back to the
// K-scaled default (6.5e-4 = 1e-5 + 64*1e-5 for K=64 fp32) since `-atol`
// was not set.
// ============================================================================

// RUN: rocmlir-gen -fut gemm_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=highlevel -host-pipeline=highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut gemm_fut --verifier clone -rtol=0.0625 - \
// RUN:   | FileCheck %s --check-prefix=RTOL_OVERRIDE --enable-var-scope

// `-rtol` also implies allclose.
// RTOL_OVERRIDE-NOT: call @mcpuVerifyFloat(
// The fp32 rtol baseline `1.3e-6` must NOT appear -- override replaces it.
// RTOL_OVERRIDE-NOT: arith.constant 1.300000e-06
// atol stays at the K-scaled default for K=64 fp32.
// RTOL_OVERRIDE:     arith.constant 6.5{{[0-9]*}}e-04 : f32
// RTOL_OVERRIDE-NEXT: arith.constant 6.250000e-02 : f32
// RTOL_OVERRIDE:     call @mcpuVerifyFloatAllclose

// ============================================================================
// (3) Both `-atol` and `-rtol` set: each component is overridden verbatim.
// Neither default appears anywhere in the output.
// ============================================================================

// RUN: rocmlir-gen -fut gemm_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=highlevel -host-pipeline=highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut gemm_fut --verifier clone -atol=0.015625 -rtol=0.0625 - \
// RUN:   | FileCheck %s --check-prefix=BOTH_OVERRIDE --enable-var-scope

// BOTH_OVERRIDE-NOT: call @mcpuVerifyFloat(
// Neither the K-scaled default (6.5e-4) nor the rtol baseline (1.3e-6) appears.
// BOTH_OVERRIDE-NOT: arith.constant 6.5{{[0-9]*}}e-04
// BOTH_OVERRIDE-NOT: arith.constant 1.300000e-06
// BOTH_OVERRIDE:     arith.constant 1.562500e-02 : f32
// BOTH_OVERRIDE-NEXT: arith.constant 6.250000e-02 : f32
// BOTH_OVERRIDE:     call @mcpuVerifyFloatAllclose
