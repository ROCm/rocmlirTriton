// RUN: rocmlir-opt -rock-verify-ttir --split-input-file %s | FileCheck %s

// Legal dots must pass through untouched. The architecture is spelled out
// rather than taken from %arch because what counts as an exact lowering differs
// between CDNA and RDNA, and this pass only reads the arch string, so no GPU is
// needed to pin either behaviour down.

// CHECK-LABEL: @verify_dot_f16_f32
// CHECK: tt.dot
func.func @verify_dot_f16_f32(
    %a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "gfx950", rock.kernel} {
  %result = tt.dot %a, %b, %c
    : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: @verify_dot_i8_i32
// CHECK: tt.dot
func.func @verify_dot_i8_i32(
    %a: tensor<64x64xi8>, %b: tensor<64x64xi8>,
    %c: tensor<64x64xi32>) -> tensor<64x64xi32>
    attributes {rock.arch = "gfx950", rock.kernel} {
  %result = tt.dot %a, %b, %c
    : tensor<64x64xi8> * tensor<64x64xi8> -> tensor<64x64xi32>
  return %result : tensor<64x64xi32>
}

// -----

// A K too short for any MFMA intrinsic still reaches v_dot4 as long as it is a
// multiple of 4.

// CHECK-LABEL: @verify_dot_i8_k_four
// CHECK: tt.dot
func.func @verify_dot_i8_k_four(
    %a: tensor<64x4xi8>, %b: tensor<4x64xi8>,
    %c: tensor<64x64xi32>) -> tensor<64x64xi32>
    attributes {rock.arch = "gfx950", rock.kernel} {
  %result = tt.dot %a, %b, %c
    : tensor<64x4xi8> * tensor<4x64xi8> -> tensor<64x64xi32>
  return %result : tensor<64x64xi32>
}

// -----

// The same K=2 integer dot this pass rejects on CDNA is fine on RDNA:
// chooseWmmaInstruction has no counterpart to the MFMA rule that a kDim must
// divide K, so WMMA serves it by duplicating data.

// CHECK-LABEL: @verify_dot_i8_k_two_on_rdna
// CHECK: tt.dot
func.func @verify_dot_i8_k_two_on_rdna(
    %a: tensor<64x2xi8>, %b: tensor<2x64xi8>,
    %c: tensor<64x64xi32>) -> tensor<64x64xi32>
    attributes {rock.arch = "gfx1200", rock.kernel} {
  %result = tt.dot %a, %b, %c
    : tensor<64x2xi8> * tensor<2x64xi8> -> tensor<64x64xi32>
  return %result : tensor<64x64xi32>
}

// -----

// Mixed fp8 operand types are legal: semantic.py permits all supported
// fp8 x fp8 combinations without requiring the two to agree.

// CHECK-LABEL: @verify_dot_f8_mixed
// CHECK: tt.dot
func.func @verify_dot_f8_mixed(
    %a: tensor<64x64xf8E4M3FNUZ>, %b: tensor<64x64xf8E5M2FNUZ>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "gfx942", rock.kernel} {
  %result = tt.dot %a, %b, %c
    : tensor<64x64xf8E4M3FNUZ> * tensor<64x64xf8E5M2FNUZ> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// f32 operands fall back to a scalar fmuladd FMA when no matrix-core intrinsic
// applies, which is exact, so a K that divides nothing in particular is fine.

// CHECK-LABEL: @verify_dot_f32_small_k
// CHECK: tt.dot
func.func @verify_dot_f32_small_k(
    %a: tensor<64x2xf32>, %b: tensor<2x64xf32>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "gfx950", rock.kernel} {
  %result = tt.dot %a, %b, %c
    : tensor<64x2xf32> * tensor<2x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Functions without rock.kernel are not ours to check.

// CHECK-LABEL: @verify_dot_not_a_kernel
// CHECK: tt.dot
func.func @verify_dot_not_a_kernel(
    %a: tensor<64x2xi8>, %b: tensor<2x64xi8>,
    %c: tensor<64x64xi32>) -> tensor<64x64xi32> {
  %result = tt.dot %a, %b, %c
    : tensor<64x2xi8> * tensor<2x64xi8> -> tensor<64x64xi32>
  return %result : tensor<64x64xi32>
}
