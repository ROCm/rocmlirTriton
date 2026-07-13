// Verifies the BF16 precision difference between GPU and CPU for the
// `tosa.matmul + tosa.reduce_sum` fusion (the same pattern exercised by
// mlir/test/fusion/pr-e2e/reductions/atomic_add_bf16/tosa-gemm-reduce-sum-case1-bf16.e2e.mlir).
//
// On the GPU (gfx950, the only target with native bf16 atomic-add hardware
// support), the reduce_sum lowers to `buffer_atomic_pk_add_bf16`, so the
// reduction accumulates in bf16 precision.
//
// On the CPU (host reference produced by --clone-harness), the matmul
// accumulator is bf16 but the reduce_sum is explicitly promoted to f32
// (`arith.extf bf16 to f32` + `arith.addf : f32` + `arith.truncf f32 to
// bf16`), so the reduction accumulates in f32.  This precision asymmetry is
// why the e2e test needs the allclose atomic-add rtol boost (sqrt(N)*eps)
// to be reliably non-flaky.
//
// gfx942 is intentionally excluded: it does not have hardware support for
// bf16 atomic add, so the GPU lowering does not use
// `buffer_atomic_pk_add_bf16` there.

// --- GPU assembly: gfx950 ---
// RUN: rocmlir-gen -fut dot_add --arch gfx950 --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline highlevel -host-pipeline highlevel \
// RUN:   | rocmlir-gen -ph -fut dot_add --verifier clone - \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c -arch gfx950 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GFX950

// --- CPU LLVM IR (host reference) ---
// RUN: rocmlir-gen -fut dot_add --arch gfx950 --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline highlevel -host-pipeline highlevel \
// RUN:   | rocmlir-gen -ph -fut dot_add --verifier clone - \
// RUN:   | rocmlir-driver -c -arch gfx950 \
// RUN:   | FileCheck %s --check-prefix=CPU

// GPU performs the reduce_sum in bf16 via the packed bf16 atomic-add.
// GFX950: buffer_atomic_pk_add_bf16

// CPU host reference promotes the reduce_sum to f32: each bf16 input is
// extended to f32, the accumulation is performed in f32, and the result is
// truncated back to bf16.
// CPU-LABEL: llvm.func @dot_add_cpu_host
// CPU: llvm.fpext {{.*}} : bf16 to f32
// CPU: llvm.fadd {{.*}} : f32
// CPU: llvm.fptrunc {{.*}} : f32 to bf16

func.func private @dot_add(%arg0: tensor<1x128x64xbf16>, %arg1: tensor<1x64x256xbf16>) -> tensor<1x128x1xbf16> attributes {rock.kernel} {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xbf16>}> : () -> tensor<1xbf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xbf16>}> : () -> tensor<1xbf16>
  %0 = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) {acc_type = f32, perf_config = "gemm:v1:128,64,32,1,1,4,0,1,2,0,0"} : (tensor<1x128x64xbf16>, tensor<1x64x256xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x128x256xbf16>
  %1 = "tosa.reduce_sum"(%0) {axis = 2 : i32} : (tensor<1x128x256xbf16>) -> tensor<1x128x1xbf16>
  return %1 : tensor<1x128x1xbf16>
}
