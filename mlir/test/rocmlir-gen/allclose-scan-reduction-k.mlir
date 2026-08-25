// Tests for `scanModuleForReductionK` in rocmlir-gen.cpp. The scanner is the
// fallback path for selecting the K-scaled allclose `atol` when the user does
// not pass `-operation` (e.g. `--clone-harness` / `--verifier=clone`). The
// observable is the `atol` constant emitted right before the
// `mcpuVerifyFloatAllclose` call, which is built as:
//
//   atol_eff = baseAtol + K_eff * sumErrTol(elemType)
//

// ============================================================================
// (1) Plain `rock.gemm`. K=64 -> atol = 1e-5 + 64*1e-5 = 6.5e-4.
// ============================================================================

// RUN: rocmlir-gen -fut gemm_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=highlevel -host-pipeline=highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut gemm_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=GEMM --enable-var-scope

func.func private @gemm_fut(%arg0: tensor<1x256x64xf32>, %arg1: tensor<1x64x128xf32>) -> tensor<1x256x128xf32> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp {acc_type = f32} : (tensor<1x256x64xf32>, tensor<1x64x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x256x128xf32>
  return %0 : tensor<1x256x128xf32>
}

// The pipeline must produce a `rock.gemm` for the scanner to see.
// GEMM:        rock.gemm
// The K-scaled atol (matching K=64): 1e-5 + 64*1e-5 = 6.5e-4.
// GEMM:        arith.constant 6.5{{[0-9]*}}e-04 : f32
// The PyTorch f32 rtol is unchanged.
// GEMM-NEXT:   arith.constant 1.300000e-06 : f32
// GEMM:        call @mcpuVerifyFloatAllclose

// ============================================================================
// (2) Chained `rock.reduce` on `rock.gemm`. The scanner walks the reduce input
// back to the matmul and adds the reduce axis extent to K_gemm (one extra
// reduction phase). For K_gemm=64 and reduce axis extent 128,
// K_eff = 64 + 128 = 192 and atol = 1e-5 + 192*1e-5 = 1.93e-3.
// ============================================================================

// RUN: rocmlir-gen -fut gemm_reduce_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=highlevel -host-pipeline=highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut gemm_reduce_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=GEMM_REDUCE --enable-var-scope

func.func private @gemm_reduce_fut(%arg0: tensor<1x256x64xf32>, %arg1: tensor<1x64x128xf32>) -> tensor<1x256x1xf32> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp {acc_type = f32} : (tensor<1x256x64xf32>, tensor<1x64x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x256x128xf32>
  %1 = tosa.reduce_sum %0 {axis = 2 : i32} : (tensor<1x256x128xf32>) -> tensor<1x256x1xf32>
  return %1 : tensor<1x256x1xf32>
}

// Both ops must reach the IR the scanner walks.
// GEMM_REDUCE:        rock.gemm
// GEMM_REDUCE:        rock.reduce
// Additive K_eff = 64 + 128 = 192; atol = 1e-5 + 192*1e-5 = 1.93e-3.
// GEMM_REDUCE:        arith.constant 0.00192{{[0-9]*}} : f32
// rtol is boosted by the atomic-add heuristic: base 1.3e-6 + sqrt(128)*eps(f32) ~ 2.65e-6.
// GEMM_REDUCE-NEXT:   arith.constant 2.{{[0-9]+}}E-6 : f32
// GEMM_REDUCE:        call @mcpuVerifyFloatAllclose

// ============================================================================
// (3) Convolution from a MIGraphX kernel. K_eff = Cin * product(filter_spatial).
// Cin=1, filter=3x3 gives K=9 and atol = 1e-5 + 9*1e-5 = 1e-4.
// ============================================================================

// RUN: rocmlir-gen -fut conv_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut conv_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=CONV --enable-var-scope

func.func private @conv_fut(%arg0: !migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1>, %arg1: !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>) -> (!migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1>) {
  %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x1x32x32xf32, 1024x1024x32x1>, <1x1x3x3xf32, 9x9x3x1> -> <1x1x32x32xf32, 1024x1024x32x1>
  return %0 : !migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1>
}

// The migraphx pipeline lowers the convolution to a `rock.conv` op.
// CONV:        rock.conv
// atol = 1e-5 + 9*1e-5 = 1e-4 (full-precision: 1.00000005E-4).
// CONV:        arith.constant 1.{{[0-9]+}}E-4 : f32
// CONV-NEXT:   arith.constant 1.300000e-06 : f32
// CONV:        call @mcpuVerifyFloatAllclose

// ============================================================================
// (4) No reduction op in the module. The scanner returns nullopt, K_eff
// defaults to 1, and atol = 1e-5 + 1*1e-5 = 2e-5 -- the PyTorch element-wise
// default plus one unit of accumulation slack.
// ============================================================================

// RUN: rocmlir-gen -fut elemwise_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=highlevel -host-pipeline=highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut elemwise_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=ELEMWISE --enable-var-scope

func.func private @elemwise_fut(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
  return %0 : tensor<256xf32>
}

// ELEMWISE-NOT:  rock.gemm
// ELEMWISE-NOT:  rock.conv
// atol = 1e-5 + 1*1e-5 = 2e-5.
// ELEMWISE:      arith.constant 2.0{{[0-9]*}}e-05 : f32
// ELEMWISE-NEXT: arith.constant 1.300000e-06 : f32
// ELEMWISE:      call @mcpuVerifyFloatAllclose

// ============================================================================
// (5) MIGraphX `migraphx.dot`. Same shape as case (1); confirms the MIGraphX
// pipeline lowers `migraphx.dot` to `rock.gemm` with the same K=64 that the
// scanner picks up. Expected atol matches case (1): 6.5e-4.
// ============================================================================

// RUN: rocmlir-gen -fut mx_dot_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut mx_dot_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=MX_DOT --enable-var-scope

func.func private @mx_dot_fut(%arg0: !migraphx.shaped<1x256x64xf32, 16384x64x1>, %arg1: !migraphx.shaped<1x64x128xf32, 8192x128x1>) -> !migraphx.shaped<1x256x128xf32, 32768x128x1> {
  %0 = migraphx.dot %arg0, %arg1 : <1x256x64xf32, 16384x64x1>, <1x64x128xf32, 8192x128x1> -> <1x256x128xf32, 32768x128x1>
  return %0 : !migraphx.shaped<1x256x128xf32, 32768x128x1>
}

// MX_DOT:        rock.gemm
// atol = 1e-5 + 64*1e-5 = 6.5e-4.
// MX_DOT:        arith.constant 6.5{{[0-9]*}}e-04 : f32
// MX_DOT-NEXT:   arith.constant 1.300000e-06 : f32
// MX_DOT:        call @mcpuVerifyFloatAllclose

// ============================================================================
// (6) MIGraphX `migraphx.dot` followed by `migraphx.reduce_sum`. Same shape as
// case (2); confirms the additive K_eff = K_gemm + reduce_axis_extent rule
// fires for the MIGraphX -> rock lowering too (rock.reduce reaches the
// scanner with `axis = 2 : index` and extent 128). Expected atol = 1.93e-3.
// ============================================================================

// RUN: rocmlir-gen -fut mx_dot_reduce_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut mx_dot_reduce_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=MX_DOT_REDUCE --enable-var-scope

func.func private @mx_dot_reduce_fut(%arg0: !migraphx.shaped<1x256x64xf32, 16384x64x1>, %arg1: !migraphx.shaped<1x64x128xf32, 8192x128x1>) -> !migraphx.shaped<1x256x1xf32, 256x1x1> {
  %0 = migraphx.dot %arg0, %arg1 : <1x256x64xf32, 16384x64x1>, <1x64x128xf32, 8192x128x1> -> <1x256x128xf32, 32768x128x1>
  %1 = migraphx.reduce_sum %0 {axes = [2]} : <1x256x128xf32, 32768x128x1> -> <1x256x1xf32, 256x1x1>
  return %1 : !migraphx.shaped<1x256x1xf32, 256x1x1>
}

// MX_DOT_REDUCE:        rock.gemm
// MX_DOT_REDUCE:        rock.reduce
// Additive K_eff = 64 + 128 = 192; atol = 1e-5 + 192*1e-5 = 1.93e-3.
// MX_DOT_REDUCE:        arith.constant 0.00192{{[0-9]*}} : f32
// rtol is boosted by the atomic-add heuristic: base 1.3e-6 + sqrt(128)*eps(f32) ~ 2.65e-6.
// MX_DOT_REDUCE-NEXT:   arith.constant 2.{{[0-9]+}}E-6 : f32
// MX_DOT_REDUCE:        call @mcpuVerifyFloatAllclose

// ============================================================================
// (7) Multi-output element-wise kernel. The kernel returns two tensors of the
// same dtype; the harness must emit one `_verify` function per output (here
// `_verify0` and `_verify1`) with its own `(atol, rtol)` constants. Because
// both outputs are f32, the tolerances are the same.
// ============================================================================

// RUN: rocmlir-gen -fut multi_out_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=highlevel -host-pipeline=highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut multi_out_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=MULTI_OUT --enable-var-scope

func.func private @multi_out_fut(%arg0: tensor<1x64x64xf32>, %arg1: tensor<1x64x64xf32>) -> (tensor<1x64x64xf32>, tensor<1x64x64xf32>) {
  %0 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %1 = tosa.matmul %arg0, %arg1, %0, %0 {acc_type = f32} : (tensor<1x64x64xf32>, tensor<1x64x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x64x64xf32>
  %2 = tosa.add %arg0, %arg1 : (tensor<1x64x64xf32>, tensor<1x64x64xf32>) -> tensor<1x64x64xf32>
  %3 = tosa.sub %arg0, %arg1 : (tensor<1x64x64xf32>, tensor<1x64x64xf32>) -> tensor<1x64x64xf32>
  %4 = tosa.add %1, %2 : (tensor<1x64x64xf32>, tensor<1x64x64xf32>) -> tensor<1x64x64xf32>
  %5 = tosa.sub %1, %3 : (tensor<1x64x64xf32>, tensor<1x64x64xf32>) -> tensor<1x64x64xf32>
  return %4, %5 : tensor<1x64x64xf32>, tensor<1x64x64xf32>
}

// Two separate verifier functions are emitted, one per output. Both share the
// K=64 matmul, so each gets atol = 1e-5 + 64*1e-5 = 6.5e-4.
// MULTI_OUT:      func.func @multi_out_fut_verify
// MULTI_OUT:      arith.constant 6.5{{[0-9]*}}e-04 : f32
// MULTI_OUT-NEXT: arith.constant 1.300000e-06 : f32
// MULTI_OUT:      call @mcpuVerifyFloatAllclose
// MULTI_OUT:      func.func @multi_out_fut_verify
// MULTI_OUT:      arith.constant 6.5{{[0-9]*}}e-04 : f32
// MULTI_OUT-NEXT: arith.constant 1.300000e-06 : f32
// MULTI_OUT:      call @mcpuVerifyFloatAllclose

// ============================================================================
// (8) Multi-output kernel with a shared matmul feeding both outputs, with the
// second output cast to f16. The harness emits one `_verify` function per
// output, each with its own per-output-dtype `(atol, rtol)` baseline
// (Section 2.6 of docs/allclose_comparator.md). The K_eff multiplier is
// module-wide, so both verifiers share K=64 from the matmul:
//   - f32 output: atol = 1e-5 + 64 * 1e-5         = 6.5e-4   ; rtol = 1.3e-6
//   - f16 output: atol = 1e-5 + 64 * (1/900)     ~ 7.11e-2  ; rtol = 1e-3
// ============================================================================

// RUN: rocmlir-gen -fut multi_out_mixed_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=highlevel -host-pipeline=highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut multi_out_mixed_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=MULTI_OUT_MIXED --enable-var-scope

func.func private @multi_out_mixed_fut(%arg0: tensor<1x64x64xf32>, %arg1: tensor<1x64x64xf32>) -> (tensor<1x64x64xf32>, tensor<1x64x64xf16>) {
  %0 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %1 = tosa.matmul %arg0, %arg1, %0, %0 {acc_type = f32} : (tensor<1x64x64xf32>, tensor<1x64x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x64x64xf32>
  %2 = tosa.add %arg0, %arg1 : (tensor<1x64x64xf32>, tensor<1x64x64xf32>) -> tensor<1x64x64xf32>
  %3 = tosa.sub %arg0, %arg1 : (tensor<1x64x64xf32>, tensor<1x64x64xf32>) -> tensor<1x64x64xf32>
  %4 = tosa.add %1, %2 : (tensor<1x64x64xf32>, tensor<1x64x64xf32>) -> tensor<1x64x64xf32>
  %5 = tosa.sub %1, %3 : (tensor<1x64x64xf32>, tensor<1x64x64xf32>) -> tensor<1x64x64xf32>
  %6 = tosa.cast %5 : (tensor<1x64x64xf32>) -> tensor<1x64x64xf16>
  return %4, %6 : tensor<1x64x64xf32>, tensor<1x64x64xf16>
}

// Two separate verifier functions are emitted, one per output. The f32 output
// keeps the fp32 baseline + K_eff*sumErrTol(f32) = 6.5e-4; the f16 output uses
// the fp16 baseline + K_eff*sumErrTol(f16) ~ 7.11e-2. The argument-memref
// element type distinguishes the two verifiers.
// MULTI_OUT_MIXED:      func.func @multi_out_mixed_fut_verify{{[0-9]+}}({{.*}}memref<{{.*}}xf32>{{.*}}memref<{{.*}}xf32>
// MULTI_OUT_MIXED:      arith.constant 6.5{{[0-9]*}}e-04 : f32
// MULTI_OUT_MIXED-NEXT: arith.constant 1.300000e-06 : f32
// MULTI_OUT_MIXED:      call @mcpuVerifyFloatAllclose
// MULTI_OUT_MIXED:      func.func @multi_out_mixed_fut_verify{{[0-9]+}}({{.*}}memref<{{.*}}xf16>{{.*}}memref<{{.*}}xf16>
// MULTI_OUT_MIXED:      arith.constant 0.0711{{[0-9]*}} : f32
// MULTI_OUT_MIXED-NEXT: arith.constant 1.000000e-03 : f32
// MULTI_OUT_MIXED:      call @mcpuVerifyFloatAllclose

// ============================================================================
// (9) Atomic-add rtol boost for f16 reduce(sum). The reduce axis extent (128)
// triggers the heuristic: rtol += sqrt(128) * eps(f16). For f16:
//   base rtol  = 1e-3
//   boost      = sqrt(128) * 9.765625e-4 ~ 1.104e-2
//   boosted    ~ 1.204e-2
// This is a much larger boost than f32 (case 2), demonstrating the heuristic
// is significant for low-precision types.
// ============================================================================

// RUN: rocmlir-gen -fut mx_dot_reduce_f16_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut mx_dot_reduce_f16_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=MX_DOT_REDUCE_F16 --enable-var-scope

func.func private @mx_dot_reduce_f16_fut(%arg0: !migraphx.shaped<1x256x64xf16, 16384x64x1>, %arg1: !migraphx.shaped<1x64x128xf16, 8192x128x1>) -> !migraphx.shaped<1x256x1xf16, 256x1x1> {
  %0 = migraphx.dot %arg0, %arg1 : <1x256x64xf16, 16384x64x1>, <1x64x128xf16, 8192x128x1> -> <1x256x128xf16, 32768x128x1>
  %1 = migraphx.reduce_sum %0 {axes = [2]} : <1x256x128xf16, 32768x128x1> -> <1x256x1xf16, 256x1x1>
  return %1 : !migraphx.shaped<1x256x1xf16, 256x1x1>
}

// MX_DOT_REDUCE_F16:        rock.gemm
// MX_DOT_REDUCE_F16:        rock.reduce
// K_eff = 64 + 128 = 192; f16 atol = 1e-5 + 192*(1/900) ~ 0.2133.
// MX_DOT_REDUCE_F16:        arith.constant 0.213{{[0-9]*}} : f32
// rtol boosted: 1e-3 + sqrt(128)*eps(f16) ~ 1.2e-2 (NOT the base 1e-3).
// MX_DOT_REDUCE_F16-NEXT:   arith.constant 0.012{{[0-9]*}} : f32
// MX_DOT_REDUCE_F16:        call @mcpuVerifyFloatAllclose

// ============================================================================
// (10) reduce(max) does NOT trigger atomic-add rtol boost. atomic_max is
// idempotent (picks the larger value), so there is no cumulative rounding
// error. rtol must stay at the base f32 value (1.3e-6).
// ============================================================================

// RUN: rocmlir-gen -fut mx_dot_reduce_max_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut mx_dot_reduce_max_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=MX_DOT_REDUCE_MAX --enable-var-scope

func.func private @mx_dot_reduce_max_fut(%arg0: !migraphx.shaped<1x256x64xf32, 16384x64x1>, %arg1: !migraphx.shaped<1x64x128xf32, 8192x128x1>) -> !migraphx.shaped<1x256x1xf32, 256x1x1> {
  %0 = migraphx.dot %arg0, %arg1 : <1x256x64xf32, 16384x64x1>, <1x64x128xf32, 8192x128x1> -> <1x256x128xf32, 32768x128x1>
  %1 = migraphx.reduce_max %0 {axes = [2]} : <1x256x128xf32, 32768x128x1> -> <1x256x1xf32, 256x1x1>
  return %1 : !migraphx.shaped<1x256x1xf32, 256x1x1>
}

// MX_DOT_REDUCE_MAX:        rock.gemm
// MX_DOT_REDUCE_MAX:        rock.reduce
// K_eff = 64 + 128 = 192; atol = 1e-5 + 192*1e-5 = 1.93e-3.
// MX_DOT_REDUCE_MAX:        arith.constant 0.00192{{[0-9]*}} : f32
// rtol is NOT boosted (reduce_max -> atomic_max is exact): stays at 1.3e-6.
// MX_DOT_REDUCE_MAX-NEXT:   arith.constant 1.300000e-06 : f32
// MX_DOT_REDUCE_MAX:        call @mcpuVerifyFloatAllclose

// ============================================================================
// (11) SplitK rtol boost from perf_config. When perf_config encodes
// splitKFactor > 1, the rtol is boosted by sqrt(splitK) * eps(dtype).
// perf_config attn:v1:32,64,64,1,1,4,0,3,1,0,0 has splitKFactor=3.
// For f16 gemm_gemm with K=128, gemmO=128:
//   K_eff = 128 + 64 = 192; atol = 1e-5 + 192*(1/900) ~ 0.2133
//   rtol  = 1e-3 + sqrt(3) * 9.765625e-4 ~ 2.691e-3
// This uses the `-pv` flow (not clone-harness) because splitK is encoded
// in the perf_config and processed by `createVerifierFunc` directly.
// ============================================================================

// RUN: rocmlir-gen --arch gfx942 --operation gemm_gemm -t f16 \
// RUN:   -m 64 -n 64 -k 128 -gemmO 128 \
// RUN:   -perf_config=attn:v1:32,64,64,1,1,4,0,3,1,0,0 \
// RUN:   -pv --comparator=allclose \
// RUN:   | FileCheck %s --check-prefix=SPLITK_F16

// atol ~ 0.2133 (K_eff=192, f16 sumErrorTol).
// SPLITK_F16:      arith.constant 0.213{{[0-9]*}} : f32
// rtol boosted: 1e-3 + sqrt(3)*eps(f16) ~ 2.69e-3 (NOT the base 1e-3).
// SPLITK_F16-NEXT: arith.constant 0.00269{{[0-9]*}} : f32
// SPLITK_F16:      call @mcpuVerifyFloatAllclose

// ============================================================================
// (12) Explicit `-rtol` suppresses the splitK boost. Same gemm_gemm as (11)
// but with `-rtol=0.005`; the user override takes precedence.
// ============================================================================

// RUN: rocmlir-gen --arch gfx942 --operation gemm_gemm -t f16 \
// RUN:   -m 64 -n 64 -k 128 -gemmO 128 \
// RUN:   -perf_config=attn:v1:32,64,64,1,1,4,0,3,1,0,0 \
// RUN:   -pv -rtol=0.005 \
// RUN:   | FileCheck %s --check-prefix=SPLITK_RTOL_OVERRIDE

// atol stays K-scaled (no -atol override).
// SPLITK_RTOL_OVERRIDE:      arith.constant 0.213{{[0-9]*}} : f32
// rtol is the user-supplied value, NOT boosted.
// SPLITK_RTOL_OVERRIDE-NEXT: arith.constant 5.000000e-03 : f32
// SPLITK_RTOL_OVERRIDE:      call @mcpuVerifyFloatAllclose

// ============================================================================
// (13) Conv backward-weight K-block rtol boost. The generated source IR does
// not have kBlocks affixed yet, so the verifier reproduces the affix-params
// calculation from the pinned tuning config. This gfx950 problem selects
// kBlocks=64:
//   K_eff = 256*28*28 = 200704
//   atol   = 1e-5 + 200704*(1/900) ~ 223.004
//   rtol   = 1e-3 + sqrt(64)*eps(f16) = 0.0088125
// ============================================================================

// RUN: rocmlir-gen --arch gfx950 --operation conv_bwd_weight -t f16 \
// RUN:   -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk \
// RUN:   -batchsize=256 -groupsize=1 -in_channels=128 -in_h=28 -in_w=28 \
// RUN:   -out_channels=512 -fil_h=1 -fil_w=1 -dilation_h=1 -dilation_w=1 \
// RUN:   -conv_stride_h=1 -conv_stride_w=1 -padding_h=0 -padding_w=0 \
// RUN:   -perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 \
// RUN:   -pv --comparator=allclose \
// RUN:   | FileCheck %s --check-prefix=WRW_KBLOCK_F16

// WRW_KBLOCK_F16:      arith.constant 223.004{{[0-9]*}} : f32
// WRW_KBLOCK_F16-NEXT: arith.constant 8.8125{{[0-9]*}}e-03 : f32
// WRW_KBLOCK_F16:      call @mcpuVerifyFloatAllclose

// ============================================================================
// (14) The backward-weight K-block tolerance path validates perf_config before
// computing kBlocks. Use an invalid numWaves value that is otherwise irrelevant
// to the K-block computation to ensure this path performs the full validation.
// ============================================================================

// RUN: not rocmlir-gen --arch gfx950 --operation conv_bwd_weight -t f16 \
// RUN:   -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk \
// RUN:   -batchsize=256 -groupsize=1 -in_channels=128 -in_h=28 -in_w=28 \
// RUN:   -out_channels=512 -fil_h=1 -fil_w=1 -dilation_h=1 -dilation_w=1 \
// RUN:   -conv_stride_h=1 -conv_stride_w=1 -padding_h=0 -padding_w=0 \
// RUN:   -perf_config=gemm:v1:64,64,64,1,1,3,16,1,2,0,0 \
// RUN:   -pv --comparator=allclose 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WRW_INVALID_PERF

// WRW_INVALID_PERF: error: numWaves=3 must be a positive power of two
