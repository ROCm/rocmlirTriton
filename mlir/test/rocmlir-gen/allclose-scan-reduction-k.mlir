// Tests for `scanModuleForReductionK` in rocmlir-gen.cpp. The scanner is the
// fallback path for selecting the K-scaled allclose `atol` when the user does
// not pass `-operation` (e.g. `--clone-harness` / `--verifier=clone`). The
// observable is the `atol` constant emitted right before the
// `mcpuVerifyFloatAllclose` call, which is built as:
//
//   atol_eff = baseAtol + K_eff * sumErrTol(elemType)
//

// ============================================================================
// (1) Plain `rock.gemm`. K=64 -> atol = 1e-5 + 64*1e-4 = 6.41e-3.
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
// The K-scaled atol (matching K=64).
// GEMM:        arith.constant 6.410000e-03 : f32
// The PyTorch f32 rtol is unchanged.
// GEMM-NEXT:   arith.constant 1.300000e-06 : f32
// GEMM:        call @mcpuVerifyFloatAllclose

// ============================================================================
// (2) Chained `rock.reduce` on `rock.gemm`. The scanner walks the reduce input
// back to the matmul and multiplies K_gemm by the reduce axis extent. For
// K_gemm=64 reduced along axis of extent 128, K_eff=64*128=8192 and
// atol = 1e-5 + 8192*1e-4 ~= 0.81921.
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
// Multiplicative K_eff = 64*128 = 8192; atol ~= 0.819209992.
// GEMM_REDUCE:        arith.constant 0.819209992 : f32
// GEMM_REDUCE-NEXT:   arith.constant 1.300000e-06 : f32
// GEMM_REDUCE:        call @mcpuVerifyFloatAllclose

// ============================================================================
// (3) Convolution from a MIGraphX kernel. K_eff = Cin * product(filter_spatial).
// Cin=1, filter=3x3 gives K=9 and atol = 1e-5 + 9*1e-4 = 9.1e-4.
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
// CONV:        arith.constant 9.100000e-04 : f32
// CONV-NEXT:   arith.constant 1.300000e-06 : f32
// CONV:        call @mcpuVerifyFloatAllclose

// ============================================================================
// (4) No reduction op in the module. The scanner returns nullopt, K_eff
// defaults to 1, and atol = 1e-5 + 1*1e-4 = 1.1e-4 -- the PyTorch element-wise
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
// ELEMWISE:      arith.constant 1.09999994E-4 : f32
// ELEMWISE-NEXT: arith.constant 1.300000e-06 : f32
// ELEMWISE:      call @mcpuVerifyFloatAllclose

// ============================================================================
// (5) MIGraphX `migraphx.dot`. Same shape as case (1); confirms the MIGraphX
// pipeline lowers `migraphx.dot` to `rock.gemm` with the same K=64 that the
// scanner picks up. Expected atol unchanged at 6.41e-3.
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
// MX_DOT:        arith.constant 6.410000e-03 : f32
// MX_DOT-NEXT:   arith.constant 1.300000e-06 : f32
// MX_DOT:        call @mcpuVerifyFloatAllclose

// ============================================================================
// (6) MIGraphX `migraphx.dot` followed by `migraphx.reduce_sum`. Same shape as
// case (2); confirms the multiplicative K_eff = K_gemm * reduce_axis_extent
// rule fires for the MIGraphX -> rock lowering too (rock.reduce reaches the
// scanner with `axis = 2 : index` and extent 128). Expected atol ~= 0.81921.
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
// MX_DOT_REDUCE:        arith.constant 0.819209992 : f32
// MX_DOT_REDUCE-NEXT:   arith.constant 1.300000e-06 : f32
// MX_DOT_REDUCE:        call @mcpuVerifyFloatAllclose

// ============================================================================
// (7) Multi-output element-wise kernel. The kernel returns two tensors of the
// same dtype; the harness must emit one `_verify` function per output (here
// `_verify0` and `_verify1`) with its own `(atol, rtol)` constants. Each call
// gets the per-output PyTorch f32 baseline scaled by K_eff=1, i.e.
// atol = 1e-5 + 1*1e-4 = 1.1e-4 and rtol = 1.3e-6.
//
// Note: this test exercises the per-output verification-emission path. It
// uses same-dtype outputs because today rocmlir-gen has a known limitation
// when assembling the host harness for tensor-result kernels with multiple
// outputs of *different* dtypes (the placeholder `outIndices` computation
// in `populateHostHarnessLogic` indexes into the input localVars). The
// observable that matters here -- two distinct `_verify` functions with
// matching `(atol, rtol)` constants -- is independent of that limitation.
// ============================================================================

// RUN: rocmlir-gen -fut multi_out_fut --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=highlevel -host-pipeline=highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut multi_out_fut --verifier clone --comparator=allclose - \
// RUN:   | FileCheck %s --check-prefix=MULTI_OUT --enable-var-scope

func.func private @multi_out_fut(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> (tensor<256xf32>, tensor<256xf32>) {
  %0 = tosa.add %arg0, %arg1 : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
  %1 = tosa.sub %arg0, %arg1 : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
  return %0, %1 : tensor<256xf32>, tensor<256xf32>
}

// Two separate verifier functions are emitted, one per output.
// MULTI_OUT:      func.func @multi_out_fut_verify
// MULTI_OUT:      arith.constant 1.09999994E-4 : f32
// MULTI_OUT-NEXT: arith.constant 1.300000e-06 : f32
// MULTI_OUT:      call @mcpuVerifyFloatAllclose
// MULTI_OUT:      func.func @multi_out_fut_verify
// MULTI_OUT:      arith.constant 1.09999994E-4 : f32
// MULTI_OUT-NEXT: arith.constant 1.300000e-06 : f32
// MULTI_OUT:      call @mcpuVerifyFloatAllclose
