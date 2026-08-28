// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-to-ttir --split-input-file -verify-diagnostics

// Test: matrixAOrigElemType set to a type not in ScaleDotElemType should error.

func.func @test_dot_scaled_unsupported_orig_type(
    %a: tensor<64x64xi8>, %b: tensor<64x64xi8>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+2 {{unsupported element type for tt.dot_scaled}}
  // expected-error @+1 {{failed to legalize operation 'rock.blockwise_gemm' that was explicitly marked illegal}}
  %result = rock.blockwise_gemm(%a, %b, %c)
    {matrixAOrigElemType = i4,
     matrixBOrigElemType = i4}
    : tensor<64x64xi8>, tensor<64x64xi8>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: directed rounding modes have no Triton equivalent and must not be
// silently dropped by Triton arith.truncf lowering.

func.func @test_truncf_f32_to_f16_downward(
    %arg0: tensor<64xf32>) -> tensor<64xf16>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+2 {{arith.truncf rounding mode downward cannot be honoured by Triton lowering}}
  // expected-error @+1 {{failed to legalize operation 'arith.truncf' that was explicitly marked illegal}}
  %0 = arith.truncf %arg0 downward : tensor<64xf32> to tensor<64xf16>
  return %0 : tensor<64xf16>
}

// -----

// Test: RTZ on a type pair tt.fp_to_fp cannot express must error rather than
// staying in arith where Triton lowering would ignore the attribute.

func.func @test_truncf_f64_to_f32_rtz(
    %arg0: tensor<64xf64>) -> tensor<64xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+2 {{arith.truncf rounding mode toward_zero cannot be honoured by Triton lowering}}
  // expected-error @+1 {{failed to legalize operation 'arith.truncf' that was explicitly marked illegal}}
  %0 = arith.truncf %arg0 toward_zero : tensor<64xf64> to tensor<64xf32>
  return %0 : tensor<64xf32>
}

// -----

// Test: FP8 destinations must go through tt.fp_to_fp, but unmapped rounding
// modes must fail rather than being silently replaced with RTNE.

func.func @test_truncf_f32_to_f8E4M3FN_downward(
    %arg0: tensor<4xf32>) -> tensor<4xf8E4M3FN>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+2 {{arith.truncf rounding mode downward cannot be honoured by Triton lowering}}
  // expected-error @+1 {{failed to legalize operation 'arith.truncf' that was explicitly marked illegal}}
  %0 = arith.truncf %arg0 downward : tensor<4xf32> to tensor<4xf8E4M3FN>
  return %0 : tensor<4xf8E4M3FN>
}

// -----

// Test: Triton AMD fp_to_fp has no f32 -> f8E4M3FN RTZ lowering (only RTNE),
// so toward_zero must be rejected here rather than emitting tt.fp_to_fp that
// fails later in TritonAMDGPUToLLVM.

func.func @test_truncf_f32_to_f8E4M3FN_rtz(
    %arg0: tensor<4xf32>) -> tensor<4xf8E4M3FN>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+2 {{arith.truncf rounding mode toward_zero cannot be honoured by Triton lowering}}
  // expected-error @+1 {{failed to legalize operation 'arith.truncf' that was explicitly marked illegal}}
  %0 = arith.truncf %arg0 toward_zero : tensor<4xf32> to tensor<4xf8E4M3FN>
  return %0 : tensor<4xf8E4M3FN>
}
