// RUN: rocmlir-opt -rock-verify-ttir --split-input-file -verify-diagnostics %s

// The architecture is spelled out rather than taken from %arch because what
// counts as an exact lowering differs between CDNA and RDNA, and this pass only
// reads the arch string, so no GPU is needed to pin either behaviour down.

// Test: on CDNA, i8 x i8 -> i32 with K=2 has neither a matrix-core lowering
// (every MFMA kDim exceeds 2, and chooseMfmaInstruction refuses a kDim that
// does not divide K) nor a v_dot4, so Triton would accumulate it in f32. This
// is the shape the AMD min_dot_size stub, which returns (1, 1, 1), fails to
// reject.

func.func @dot_i8_k_not_multiple_of_four(
    %a: tensor<64x2xi8>, %b: tensor<2x64xi8>,
    %c: tensor<64x64xi32>) -> tensor<64x64xi32>
    attributes {rock.arch = "gfx950", rock.kernel} {
  // expected-error @+1 {{K=2 leaves this 'i8' x 'i8' -> 'i32' dot without a matrix-core or v_dot lowering}}
  %result = tt.dot %a, %b, %c
    : tensor<64x2xi8> * tensor<2x64xi8> -> tensor<64x64xi32>
  return %result : tensor<64x64xi32>
}

// -----

// Test: RDNA tolerates a WMMA kDim that does not divide K, but BlockedToWMMA
// still refuses K=1, which leaves an integer dot on the lossy FMA path.

func.func @dot_i8_k_one_on_rdna(
    %a: tensor<64x1xi8>, %b: tensor<1x64xi8>,
    %c: tensor<64x64xi32>) -> tensor<64x64xi32>
    attributes {rock.arch = "gfx1200", rock.kernel} {
  // expected-error @+1 {{K=1 leaves this 'i8' x 'i8' -> 'i32' dot without a matrix-core or v_dot lowering}}
  %result = tt.dot %a, %b, %c
    : tensor<64x1xi8> * tensor<1x64xi8> -> tensor<64x64xi32>
  return %result : tensor<64x64xi32>
}

// -----

// Test: operands must share an element type unless both are fp8
// (semantic.py:1441).

func.func @dot_mismatched_operand_types(
    %a: tensor<64x64xf16>, %b: tensor<64x64xbf16>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "gfx950", rock.kernel} {
  // expected-error @+1 {{operands A and B must have the same element type, got 'f16' and 'bf16'}}
  %result = tt.dot %a, %b, %c
    : tensor<64x64xf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: an fp8 operand paired with a non-fp8 one is not the "all fp8 x fp8
// combinations" case, so the whitelist at semantic.py:1437-1440 applies.

func.func @dot_fp8_paired_with_int(
    %a: tensor<64x64xi8>, %b: tensor<64x64xf8E4M3FN>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "gfx950", rock.kernel} {
  // expected-error @+1 {{unsupported element type for operand B: 'f8E4M3FN'}}
  %result = tt.dot %a, %b, %c
    : tensor<64x64xi8> * tensor<64x64xf8E4M3FN> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: integer dots must accumulate into i32 (semantic.py:1483-1486).

func.func @dot_i8_float_accumulator(
    %a: tensor<64x64xi8>, %b: tensor<64x64xi8>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "gfx950", rock.kernel} {
  // expected-error @+1 {{integer dot must accumulate into i32, got 'f32'}}
  %result = tt.dot %a, %b, %c
    : tensor<64x64xi8> * tensor<64x64xi8> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: bf16 accumulators are rejected outright (semantic.py:1487-1490).

func.func @dot_bf16_accumulator(
    %a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
    %c: tensor<64x64xbf16>) -> tensor<64x64xbf16>
    attributes {rock.arch = "gfx950", rock.kernel} {
  // expected-error @+1 {{bf16 accumulator is unsupported}}
  %result = tt.dot %a, %b, %c
    : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xbf16>
  return %result : tensor<64x64xbf16>
}

// -----

// Test: bf16 operands accumulate in f32 (semantic.py:1491-1493).

func.func @dot_bf16_operands_f16_accumulator(
    %a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>,
    %c: tensor<64x64xf16>) -> tensor<64x64xf16>
    attributes {rock.arch = "gfx950", rock.kernel} {
  // expected-error @+1 {{dot with 'bf16' operands must accumulate into f32, got 'f16'}}
  %result = tt.dot %a, %b, %c
    : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf16>
  return %result : tensor<64x64xf16>
}
