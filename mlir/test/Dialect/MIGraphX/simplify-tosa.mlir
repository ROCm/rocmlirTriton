// RUN: rocmlir-opt %s -migraphx-tosa-simplify | FileCheck %s

// Test 1: Eliminate redundant cast (same input and output types)
// CHECK-LABEL: @eliminate_redundant_cast
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NOT: tosa.cast
// CHECK: return %[[ARG0]] : tensor<4xf32>
func.func @eliminate_redundant_cast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// ----

// Test 2: Eliminate cast chain that results in same type
// CHECK-LABEL: @eliminate_cast_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NOT: tosa.cast
// CHECK: return %[[ARG0]] : tensor<4xf32>
func.func @eliminate_cast_chain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf16>
  %1 = tosa.cast %0 : (tensor<4xf16>) -> tensor<4xi32>
  %2 = tosa.cast %1 : (tensor<4xi32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

// ----

// Test 3: Keep necessary cast (different types)
// CHECK-LABEL: @keep_necessary_cast
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf16>
// CHECK: %[[CAST:.*]] = tosa.cast %[[ARG0]] : (tensor<4xf32>) -> tensor<4xf16>
// CHECK: return %[[CAST]] : tensor<4xf16>
func.func @keep_necessary_cast(%arg0: tensor<4xf32>) -> tensor<4xf16> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// ----

// Test 4: Partial cast chain elimination
// CHECK-LABEL: @partial_cast_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf16>
// CHECK: %[[CAST:.*]] = tosa.cast %[[ARG0]] : (tensor<4xf32>) -> tensor<4xf16>
// CHECK: return %[[CAST]] : tensor<4xf16>
func.func @partial_cast_chain(%arg0: tensor<4xf32>) -> tensor<4xf16> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %1 = tosa.cast %0 : (tensor<4xf32>) -> tensor<4xf16>
  return %1 : tensor<4xf16>
}

// ----

// Test 5: Multiple independent casts
// CHECK-LABEL: @multiple_independent_casts
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>)
// CHECK-NOT: tosa.cast
// CHECK: return %[[ARG0]], %[[ARG1]] : tensor<4xf32>, tensor<4xi32>
func.func @multiple_independent_casts(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %1 = tosa.cast %arg1 : (tensor<4xi32>) -> tensor<4xi32>
  return %0, %1 : tensor<4xf32>, tensor<4xi32>
}

// ----

// Test 6: Complex cast chain with intermediate operations
// CHECK-LABEL: @complex_cast_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[CAST0:.*]] = tosa.cast %[[ARG0]] : (tensor<4xf32>) -> tensor<4xf16>
// CHECK: %[[ABS:.*]] = tosa.abs %[[CAST0]] : (tensor<4xf16>) -> tensor<4xf16>
// CHECK: %[[CAST1:.*]] = tosa.cast %[[ABS]] : (tensor<4xf16>) -> tensor<4xf32>
// CHECK-NOT: tosa.cast %[[CAST1]]
// CHECK: return %[[CAST1]] : tensor<4xf32>
func.func @complex_cast_chain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf16>
  %1 = tosa.abs %0 : (tensor<4xf16>) -> tensor<4xf16>
  %2 = tosa.cast %1 : (tensor<4xf16>) -> tensor<4xf32>
  %3 = tosa.cast %2 : (tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}

// ----

// Test 7: Long chain of same-type casts
// CHECK-LABEL: @long_redundant_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NOT: tosa.cast
// CHECK: return %[[ARG0]] : tensor<4xf32>
func.func @long_redundant_chain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %1 = tosa.cast %0 : (tensor<4xf32>) -> tensor<4xf32>
  %2 = tosa.cast %1 : (tensor<4xf32>) -> tensor<4xf32>
  %3 = tosa.cast %2 : (tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}

// ----

// Test 8: Mixed necessary and unnecessary casts
// CHECK-LABEL: @mixed_casts
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xi8>
// CHECK: %[[CAST0:.*]] = tosa.cast %[[ARG0]] : (tensor<4xf32>) -> tensor<4xi32>
// CHECK: %[[CAST1:.*]] = tosa.cast %[[CAST0]] : (tensor<4xi32>) -> tensor<4xi8>
// CHECK: return %[[CAST1]] : tensor<4xi8>
func.func @mixed_casts(%arg0: tensor<4xf32>) -> tensor<4xi8> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>  // redundant
  %1 = tosa.cast %0 : (tensor<4xf32>) -> tensor<4xi32>     // necessary
  %2 = tosa.cast %1 : (tensor<4xi32>) -> tensor<4xi8>      // necessary
  return %2 : tensor<4xi8>
}

// ----

// Test 9: Cast with multiple uses
// CHECK-LABEL: @cast_multiple_uses
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-NOT: tosa.cast
// CHECK: return %[[ARG0]], %[[ARG0]] : tensor<4xf32>, tensor<4xf32>
func.func @cast_multiple_uses(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0, %0 : tensor<4xf32>, tensor<4xf32>
}

// ----

// Test 10: Complex cast chain with multiple uses
// CHECK-LABEL: @complex_cast_chain_multiple_uses
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf16>)
// CHECK: %[[CAST0:.*]] = tosa.cast %[[ARG0]] : (tensor<4xf32>) -> tensor<4xf16>
// CHECK-NOT: tosa.cast %[[CAST0]]
// CHECK: %[[ABS:.*]] = tosa.abs %[[CAST0]] : (tensor<4xf16>) -> tensor<4xf16>
// CHECK: return %[[ARG0]], %[[ABS]] : tensor<4xf32>, tensor<4xf16>
func.func @complex_cast_chain_multiple_uses(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf16>) {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf16>
  %1 = tosa.cast %0 : (tensor<4xf16>) -> tensor<4xf32>
  %2 = tosa.cast %1 : (tensor<4xf32>) -> tensor<4xf32>
  %3 = tosa.abs %0 : (tensor<4xf16>) -> tensor<4xf16>
  return %2, %3: tensor<4xf32>, tensor<4xf16>
}

// ----

// Test 11: Preserve cast chain involving Float8E8M0FNUType
// CHECK-LABEL: @preserve_f8e8m0fnu_cast_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[CAST0:.*]] = tosa.cast %[[ARG0]] : (tensor<4xf32>) -> tensor<4xf8E8M0FNU>
// CHECK: %[[CAST1:.*]] = tosa.cast %[[CAST0]] : (tensor<4xf8E8M0FNU>) -> tensor<4xf32>
// CHECK: return %[[CAST1]] : tensor<4xf32>
func.func @preserve_f8e8m0fnu_cast_chain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xf8E8M0FNU>
  %1 = tosa.cast %0 : (tensor<4xf8E8M0FNU>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// ----

// Test 12: Remove the cast -> fp_to_int_cast -> cast chain in front of a
// select, so that the select takes the tosa.greater as input instead of the
// useless casts
// CHECK-LABEL: @boolean_fp_to_int_round_trip
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[PRED:.*]] = tosa.greater %[[ARG0]], %[[ARG1]]
// CHECK-NOT: tosa.custom
// CHECK-NOT: tosa.cast
// CHECK: tosa.select %[[PRED]], %[[ARG0]], %[[ARG1]]
func.func @boolean_fp_to_int_round_trip(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %pred = tosa.greater %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %asFloat = tosa.cast %pred : (tensor<4xi1>) -> tensor<4xf32>
  %asInt = tosa.custom %asFloat {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<4xf32>) -> tensor<4xi8>
  %cond = tosa.cast %asInt : (tensor<4xi8>) -> tensor<4xi1>
  %0 = tosa.select %cond, %arg0, %arg1 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// ----

// Test 13: A boolean-sourced fp_to_int_cast becomes a zero extension of the
// boolean, skipping the float domain, when the integer result is used as an
// integer
// CHECK-LABEL: @boolean_fp_to_int_cast
// CHECK: %[[PRED:.*]] = tosa.greater
// CHECK-NOT: tosa.custom
// CHECK: tosa.cast %[[PRED]] : (tensor<4xi1>) -> tensor<4xi8>
func.func @boolean_fp_to_int_cast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi8> {
  %pred = tosa.greater %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %asFloat = tosa.cast %pred : (tensor<4xi1>) -> tensor<4xf32>
  %asInt = tosa.custom %asFloat {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<4xf32>) -> tensor<4xi8>
  return %asInt : tensor<4xi8>
}

// ----

// Test 14: Keep the fp_to_int_cast when the float operand isn't a boolean
// CHECK-LABEL: @preserve_fp_to_int_cast
// CHECK: tosa.custom{{.*}}"fp_to_int_cast"{{.*}}(tensor<4xf32>) -> tensor<4xi8>
func.func @preserve_fp_to_int_cast(%arg0: tensor<4xf32>) -> tensor<4xi8> {
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<4xf32>) -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// ----

// Test 15: Keep the fp_to_int_cast when the float operand is a widened integer
// that isn't i1, since those values can overflow the destination
// CHECK-LABEL: @preserve_fp_to_int_cast_of_widened_int
// CHECK: tosa.custom{{.*}}"fp_to_int_cast"{{.*}}(tensor<4xf32>) -> tensor<4xi8>
func.func @preserve_fp_to_int_cast_of_widened_int(%arg0: tensor<4xi32>) -> tensor<4xi8> {
  %asFloat = tosa.cast %arg0 : (tensor<4xi32>) -> tensor<4xf32>
  %0 = tosa.custom %asFloat {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<4xf32>) -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// ----

// Test 16: Simplify the fp_to_int_cast even when the widening cast feeding it
// has other uses, in which case the widening cast has to stay behind
// CHECK-LABEL: @boolean_fp_to_int_cast_shared_widening
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xf32>)
// CHECK: %[[PRED:.*]] = tosa.greater %[[ARG0]], %[[ARG1]]
// CHECK: %[[FLOAT:.*]] = tosa.cast %[[PRED]] : (tensor<4xi1>) -> tensor<4xf32>
// CHECK-NOT: tosa.custom
// CHECK: %[[INT:.*]] = tosa.cast %[[PRED]] : (tensor<4xi1>) -> tensor<4xi8>
// CHECK: return %[[INT]], %[[FLOAT]]
func.func @boolean_fp_to_int_cast_shared_widening(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xi8>, tensor<4xf32>) {
  %pred = tosa.greater %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %asFloat = tosa.cast %pred : (tensor<4xi1>) -> tensor<4xf32>
  %asInt = tosa.custom %asFloat {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<4xf32>) -> tensor<4xi8>
  return %asInt, %asFloat : tensor<4xi8>, tensor<4xf32>
}

// ----

// Test 17: Keep the fp_to_int_cast of a boolean when the destination is a
// signed 1-bit integer.
// CHECK-LABEL: @preserve_boolean_fp_to_int_cast_to_i1
// CHECK: tosa.custom{{.*}}"fp_to_int_cast"{{.*}}(tensor<4xf32>) -> tensor<4xi1>
func.func @preserve_boolean_fp_to_int_cast_to_i1(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  %pred = tosa.greater %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %asFloat = tosa.cast %pred : (tensor<4xi1>) -> tensor<4xf32>
  %0 = tosa.custom %asFloat {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// ----

// Test 18: Keep the fp_to_int_cast of a boolean when the destination isn't an
// integer at all.
// CHECK-LABEL: @preserve_boolean_fp_to_int_cast_to_float
// CHECK: tosa.custom{{.*}}"fp_to_int_cast"{{.*}}(tensor<4xf32>) -> tensor<4xf16>
func.func @preserve_boolean_fp_to_int_cast_to_float(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf16> {
  %pred = tosa.greater %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %asFloat = tosa.cast %pred : (tensor<4xi1>) -> tensor<4xf32>
  %0 = tosa.custom %asFloat {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<4xf32>) -> tensor<4xf16>
  return %0 : tensor<4xf16>
}
