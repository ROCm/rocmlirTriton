// Negative tests for rock-tosa-to-elementwise pass.
//
// Plain tosa.cast reports unsupported source float types gracefully, while
// the pre-existing custom fp_to_int_cast source-type guard aborts the process.
// Keep the fatal case last so FileCheck observes the earlier diagnostics.

// RUN: not rocmlir-opt --rock-tosa-to-elementwise --split-input-file %s 2>&1 | FileCheck %s

// The clamped conversion helper needs zero, infinity, and a signed
// representation. Diagnose unsupported types before invoking the helper.
// CHECK: error: {{.*}}floating-point to integer cast requires a source type with representable zero, signed representation, and infinity; promote the source to a wider floating-point type first
func.func @cast_f8e4m3fn_to_i8_rejected(%arg0: tensor<16xf8E4M3FN>) -> tensor<16xi8>
    attributes {rock.kernel} {
  %0 = tosa.cast %arg0 : (tensor<16xf8E4M3FN>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// CHECK: error: {{.*}}floating-point to integer cast requires a source type with representable zero, signed representation, and infinity; promote the source to a wider floating-point type first
func.func @cast_f8e8m0fnu_to_i32_rejected(%arg0: tensor<16xf8E8M0FNU>) -> tensor<16xi32>
    attributes {rock.kernel} {
  %0 = tosa.cast %arg0 : (tensor<16xf8E8M0FNU>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// `fp_to_int_cast` lowers via rock::createClampedFPToInt, which requires
// the source float type to have representable zero, representable
// infinity, AND a signed representation (so it can materialize -inf for
// the case-1 overflow check). Source types that violate any of these are
// rejected via llvm::reportFatalUsageError, which aborts the process.
//
// Today the only LLVM/MLIR float type without a sign bit is F8E8M0FNU,
// which also lacks zero, so this case short-circuits on the first
// sub-condition rather than the new signed-repr one. We still test it
// because the guard is the unit of behaviour; the signed-repr branch is
// added defensively against a future MX/OCP-style unsigned-with-infinity
// format.
// CHECK: rock::createClampedFPToInt: source float type lacks a representable zero, signed representation, or infinity
func.func @fp_to_int_cast_unsigned_float_rejected(%arg0: tensor<16xf8E8M0FNU>) -> tensor<16xi32>
    attributes {rock.kernel} {
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<16xf8E8M0FNU>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}
