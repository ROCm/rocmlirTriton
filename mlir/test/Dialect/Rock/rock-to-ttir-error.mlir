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

func.func @test_atomic_max_f16_unsupported(
    %arg0: tensor<64x64xf16>, %arg1: tensor<64x64xi32>,
    %arg2: tensor<64x64xi1>) -> tensor<64x64xf16>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+2 {{atomic_max on floating point requires f32; got 'f16'}}
  // expected-error @+1 {{failed to legalize operation 'rock.blockwise_store_ptr' that was explicitly marked illegal}}
  %0 = rock.blockwise_store_ptr %arg0 -> %arg1(%arg2) by atomic_max
    : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

func.func @test_atomic_max_bf16_unsupported(
    %arg0: tensor<64x64xbf16>, %arg1: tensor<64x64xi32>,
    %arg2: tensor<64x64xi1>) -> tensor<64x64xbf16>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @+2 {{atomic_max on floating point requires f32; got 'bf16'}}
  // expected-error @+1 {{failed to legalize operation 'rock.blockwise_store_ptr' that was explicitly marked illegal}}
  %0 = rock.blockwise_store_ptr %arg0 -> %arg1(%arg2) by atomic_max
    : tensor<64x64xbf16> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
}
