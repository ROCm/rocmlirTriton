// RUN: rocmlir-opt --rock-tosa-to-elementwise --split-input-file %s | FileCheck %s
// RUN: rocmlir-opt --rock-tosa-to-elementwise="disable-fast-math=true" --split-input-file %s | FileCheck %s --check-prefix=IEEE

// The saturating float-to-int casts below are the only lowerings in this pass
// that depend on `disable-fast-math`, so they are the only ones carrying IEEE
// check lines. By default the kernel path assumes no NaN occurs anywhere in its
// floating-point dataflow and drops the NaN -> 0 sanitize pair that otherwise
// guards `arith.fptosi`/`fptoui`; `-disable-fast-math` restores it. That
// assumption covers intermediates, not just kernel arguments: the value being
// converted is typically already the result of arithmetic (a
// `migraphx.quantizelinear` divides by a scale first), so a manufactured NaN
// breaks the assumption as surely as an incoming one would. Without the pair a
// NaN no longer maps to 0; what it maps to instead depends on which of the
// three range cases the (float, int) pair takes, and the `nnan` added later
// leaves it unspecified anyway.

// CHECK-LABEL: @add_f32
// CHECK-NOT:   tosa.add
// CHECK:       arith.addf %arg0, %arg1 : tensor<64x128xf32>
func.func @add_f32(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> attributes {rock.kernel} {
  %0 = tosa.add %arg0, %arg1 : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// -----

// CHECK-LABEL: @add_i32
// CHECK-NOT:   tosa.add
// CHECK:       arith.addi %arg0, %arg1 : tensor<32x64xi32>
func.func @add_i32(%arg0: tensor<32x64xi32>, %arg1: tensor<32x64xi32>) -> tensor<32x64xi32> attributes {rock.kernel} {
  %0 = tosa.add %arg0, %arg1 : (tensor<32x64xi32>, tensor<32x64xi32>) -> tensor<32x64xi32>
  return %0 : tensor<32x64xi32>
}

// -----

// CHECK-LABEL: @sub_f16
// CHECK-NOT:   tosa.sub
// CHECK:       arith.subf %arg0, %arg1 : tensor<8x16xf16>
func.func @sub_f16(%arg0: tensor<8x16xf16>, %arg1: tensor<8x16xf16>) -> tensor<8x16xf16> attributes {rock.kernel} {
  %0 = tosa.sub %arg0, %arg1 : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
  return %0 : tensor<8x16xf16>
}

// -----

// CHECK-LABEL: @maximum_f32
// CHECK-NOT:   tosa.maximum
// CHECK:       arith.maximumf %arg0, %arg1 : tensor<4x8xf32>
func.func @maximum_f32(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {rock.kernel} {
  %0 = tosa.maximum %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @maximum_f32_ignore_nan
// CHECK-NOT:   tosa.maximum
// CHECK:       arith.maxnumf %arg0, %arg1 : tensor<4x8xf32>
func.func @maximum_f32_ignore_nan(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {rock.kernel} {
  %0 = tosa.maximum %arg0, %arg1 {nan_mode = IGNORE} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @maximum_f32_propagate_nan
// CHECK-NOT:   tosa.maximum
// CHECK:       arith.maximumf %arg0, %arg1 : tensor<4x8xf32>
func.func @maximum_f32_propagate_nan(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {rock.kernel} {
  %0 = tosa.maximum %arg0, %arg1 {nan_mode = PROPAGATE} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @maximum_i32
// CHECK-NOT:   tosa.maximum
// CHECK:       arith.maxsi %arg0, %arg1 : tensor<4x8xi32>
func.func @maximum_i32(%arg0: tensor<4x8xi32>, %arg1: tensor<4x8xi32>) -> tensor<4x8xi32> attributes {rock.kernel} {
  %0 = tosa.maximum %arg0, %arg1 : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}

// -----

// CHECK-LABEL: @minimum_f32
// CHECK-NOT:   tosa.minimum
// CHECK:       arith.minimumf %arg0, %arg1 : tensor<4x8xf32>
func.func @minimum_f32(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {rock.kernel} {
  %0 = tosa.minimum %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @minimum_f32_ignore_nan
// CHECK-NOT:   tosa.minimum
// CHECK:       arith.minnumf %arg0, %arg1 : tensor<4x8xf32>
func.func @minimum_f32_ignore_nan(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {rock.kernel} {
  %0 = tosa.minimum %arg0, %arg1 {nan_mode = IGNORE} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @minimum_f32_propagate_nan
// CHECK-NOT:   tosa.minimum
// CHECK:       arith.minimumf %arg0, %arg1 : tensor<4x8xf32>
func.func @minimum_f32_propagate_nan(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {rock.kernel} {
  %0 = tosa.minimum %arg0, %arg1 {nan_mode = PROPAGATE} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @minimum_i32
// CHECK-NOT:   tosa.minimum
// CHECK:       arith.minsi %arg0, %arg1 : tensor<4x8xi32>
func.func @minimum_i32(%arg0: tensor<4x8xi32>, %arg1: tensor<4x8xi32>) -> tensor<4x8xi32> attributes {rock.kernel} {
  %0 = tosa.minimum %arg0, %arg1 {nan_mode = IGNORE} : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}

// -----

// CHECK-LABEL: @exp_f32
// CHECK-NOT:   tosa.exp
// CHECK:       math.exp %arg0 : tensor<64xf32>
func.func @exp_f32(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel} {
  %0 = tosa.exp %arg0 : (tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// -----

// CHECK-LABEL: @log_f32
// CHECK-NOT:   tosa.log
// CHECK:       math.log %arg0 : tensor<32xf32>
func.func @log_f32(%arg0: tensor<32xf32>) -> tensor<32xf32> attributes {rock.kernel} {
  %0 = tosa.log %arg0 : (tensor<32xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

// CHECK-LABEL: @rsqrt_f32
// CHECK-NOT:   tosa.rsqrt
// CHECK:       math.rsqrt %arg0 : tensor<16xf32>
func.func @rsqrt_f32(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.rsqrt %arg0 : (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @sqrt_via_reciprocal_rsqrt
// CHECK-NOT:   tosa.rsqrt
// CHECK-NOT:   tosa.reciprocal
// CHECK:       math.sqrt %arg0 : tensor<16xf32>
func.func @sqrt_via_reciprocal_rsqrt(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.rsqrt %arg0 : (tensor<16xf32>) -> tensor<16xf32>
  %1 = tosa.reciprocal %0 : (tensor<16xf32>) -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @ceil_f32
// CHECK-NOT:   tosa.ceil
// CHECK:       math.ceil %arg0 : tensor<16xf32>
func.func @ceil_f32(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.ceil %arg0 : (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @floor_f32
// CHECK-NOT:   tosa.floor
// CHECK:       math.floor %arg0 : tensor<16xf32>
func.func @floor_f32(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.floor %arg0 : (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// Tanh is left as math.tanh; rock-legalize-math-for-triton picks its lowering
// once the target is known.
// CHECK-LABEL: @tanh_f32
// CHECK-NOT:   tosa.tanh
// CHECK:       math.tanh %arg0 : tensor<64xf32>
func.func @tanh_f32(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel} {
  %0 = tosa.tanh %arg0 : (tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// -----

// CHECK-LABEL: @erf_f32
// CHECK-NOT:   tosa.erf
// CHECK:       math.erf %arg0 : tensor<64xf32>
func.func @erf_f32(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel} {
  %0 = tosa.erf %arg0 : (tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// -----

// CHECK-LABEL: @abs_f32
// CHECK-NOT:   tosa.abs
// CHECK:       math.absf %arg0 : tensor<32xf32>
func.func @abs_f32(%arg0: tensor<32xf32>) -> tensor<32xf32> attributes {rock.kernel} {
  %0 = tosa.abs %arg0 : (tensor<32xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

// CHECK-LABEL: @abs_i32
// CHECK-NOT:   tosa.abs
// CHECK:       %[[ZERO:.*]] = arith.constant dense<0> : tensor<32xi32>
// CHECK:       %[[NEG:.*]] = arith.subi %[[ZERO]], %arg0 : tensor<32xi32>
// CHECK:       arith.maxsi %arg0, %[[NEG]] : tensor<32xi32>
func.func @abs_i32(%arg0: tensor<32xi32>) -> tensor<32xi32> attributes {rock.kernel} {
  %0 = tosa.abs %arg0 : (tensor<32xi32>) -> tensor<32xi32>
  return %0 : tensor<32xi32>
}

// -----

// CHECK-LABEL: @negate_f32
// CHECK-NOT:   tosa.negate
// CHECK:       arith.negf %arg0 : tensor<16xf32>
func.func @negate_f32(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %in_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %out_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = tosa.negate %arg0, %in_zp, %out_zp : (tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @negate_i32
// CHECK-NOT:   tosa.negate
// CHECK:       %[[ZERO:.*]] = arith.constant dense<0> : tensor<16xi32>
// CHECK:       arith.subi %[[ZERO]], %arg0 : tensor<16xi32>
func.func @negate_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> attributes {rock.kernel} {
  %in_zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %out_zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = tosa.negate %arg0, %in_zp, %out_zp : (tensor<16xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// CHECK-LABEL: @mul_f32
// CHECK-NOT:   tosa.mul
// CHECK:       arith.mulf %arg0, %arg1 : tensor<4x8xf32>
func.func @mul_f32(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {rock.kernel} {
  %shift = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = "tosa.mul"(%arg0, %arg1, %shift) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<1xi8>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @mul_i32
// CHECK-NOT:   tosa.mul
// CHECK:       arith.muli %arg0, %arg1 : tensor<4x8xi32>
func.func @mul_i32(%arg0: tensor<4x8xi32>, %arg1: tensor<4x8xi32>) -> tensor<4x8xi32> attributes {rock.kernel} {
  %shift = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = "tosa.mul"(%arg0, %arg1, %shift) : (tensor<4x8xi32>, tensor<4x8xi32>, tensor<1xi8>) -> tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}

// -----

// CHECK-LABEL: @reciprocal_f32
// CHECK-NOT:   tosa.reciprocal
// CHECK:       %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK:       arith.divf %[[ONE]], %arg0 : tensor<32xf32>
func.func @reciprocal_f32(%arg0: tensor<32xf32>) -> tensor<32xf32> attributes {rock.kernel} {
  %0 = tosa.reciprocal %arg0 : (tensor<32xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

// CHECK-LABEL: @sigmoid_f32
// CHECK-NOT:   tosa.sigmoid
// sigmoid(x) = 1 / (1 + exp(-x)), where -x is computed as (0 - x) following
// Triton's approach. See: triton/python/triton/language/semantic.py (minus)
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK-DAG:   %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<16xf32>
// CHECK:       %[[NEG:.*]] = arith.subf %[[ZERO]], %arg0 : tensor<16xf32>
// CHECK:       %[[EXP:.*]] = math.exp %[[NEG]] : tensor<16xf32>
// CHECK:       %[[DENOM:.*]] = arith.addf %[[ONE]], %[[EXP]] : tensor<16xf32>
// CHECK:       arith.divf %[[ONE]], %[[DENOM]] : tensor<16xf32>
func.func @sigmoid_f32(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.sigmoid %arg0 : (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @select_op
// CHECK-NOT:   tosa.select
// CHECK:       arith.select %arg0, %arg1, %arg2 : tensor<4x8xi1>, tensor<4x8xf32>
func.func @select_op(%arg0: tensor<4x8xi1>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {rock.kernel} {
  %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @clamp_f32
// CHECK-NOT:   tosa.clamp
// CHECK-DAG:   %[[MIN:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK-DAG:   %[[MAX:.*]] = arith.constant dense<6.000000e+00> : tensor<16xf32>
// CHECK:       %[[CLAMPED:.*]] = arith.minimumf %arg0, %[[MAX]] : tensor<16xf32>
// CHECK:       arith.maximumf %[[CLAMPED]], %[[MIN]] : tensor<16xf32>
func.func @clamp_f32(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.clamp %arg0 {min_val = 0.0 : f32, max_val = 6.0 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// An explicit PROPAGATE is the default spelling and keeps the NaN-propagating
// ops.

// CHECK-LABEL: @clamp_f32_propagate
// CHECK-NOT:   tosa.clamp
// CHECK:       %[[CLAMPED:.*]] = arith.minimumf %arg0, %{{.*}} : tensor<16xf32>
// CHECK:       arith.maximumf %[[CLAMPED]], %{{.*}} : tensor<16xf32>
func.func @clamp_f32_propagate(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.clamp %arg0 {min_val = 0.0 : f32, max_val = 6.0 : f32, nan_mode = PROPAGATE} : (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// IGNORE selects the non-propagating ops. RockAllowFastMathFlags later marks
// them nnan on kernels that promised no NaN, which is what lets RockToTTIR
// fold the pair into a single tt.clampf.

// CHECK-LABEL: @clamp_f32_ignore
// CHECK-NOT:   tosa.clamp
// CHECK-DAG:   %[[MIN:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK-DAG:   %[[MAX:.*]] = arith.constant dense<6.000000e+00> : tensor<16xf32>
// CHECK:       %[[CLAMPED:.*]] = arith.minnumf %arg0, %[[MAX]] : tensor<16xf32>
// CHECK:       arith.maxnumf %[[CLAMPED]], %[[MIN]] : tensor<16xf32>
func.func @clamp_f32_ignore(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.clamp %arg0 {min_val = 0.0 : f32, max_val = 6.0 : f32, nan_mode = IGNORE} : (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// The integer clamp has no NaN to reason about, so IGNORE changes nothing.

// CHECK-LABEL: @clamp_i32_ignore
// CHECK-NOT:   tosa.clamp
// CHECK:       %[[CLAMPED:.*]] = arith.minsi %arg0, %{{.*}} : tensor<16xi32>
// CHECK:       arith.maxsi %[[CLAMPED]], %{{.*}} : tensor<16xi32>
func.func @clamp_i32_ignore(%arg0: tensor<16xi32>) -> tensor<16xi32> attributes {rock.kernel} {
  %0 = tosa.clamp %arg0 {min_val = -128 : i32, max_val = 127 : i32, nan_mode = IGNORE} : (tensor<16xi32>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// CHECK-LABEL: @clamp_i32
// CHECK-NOT:   tosa.clamp
// CHECK-DAG:   %[[MIN:.*]] = arith.constant dense<-128> : tensor<16xi32>
// CHECK-DAG:   %[[MAX:.*]] = arith.constant dense<127> : tensor<16xi32>
// CHECK:       %[[CLAMPED:.*]] = arith.minsi %arg0, %[[MAX]] : tensor<16xi32>
// CHECK:       arith.maxsi %[[CLAMPED]], %[[MIN]] : tensor<16xi32>
func.func @clamp_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> attributes {rock.kernel} {
  %0 = tosa.clamp %arg0 {min_val = -128 : i32, max_val = 127 : i32} : (tensor<16xi32>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// CHECK-LABEL: @cast_f16_to_f32
// CHECK-NOT:   tosa.cast
// CHECK:       arith.extf %arg0 : tensor<32xf16> to tensor<32xf32>
func.func @cast_f16_to_f32(%arg0: tensor<32xf16>) -> tensor<32xf32> attributes {rock.kernel} {
  %0 = tosa.cast %arg0 : (tensor<32xf16>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

// CHECK-LABEL: @cast_f32_to_f16
// CHECK-NOT:   tosa.cast
// CHECK:       arith.truncf %arg0 : tensor<32xf32> to tensor<32xf16>
func.func @cast_f32_to_f16(%arg0: tensor<32xf32>) -> tensor<32xf16> attributes {rock.kernel} {
  %0 = tosa.cast %arg0 : (tensor<32xf32>) -> tensor<32xf16>
  return %0 : tensor<32xf16>
}

// -----

// CHECK-LABEL: @cast_i32_to_f32
// CHECK-NOT:   tosa.cast
// CHECK:       arith.sitofp %arg0 : tensor<16xi32> to tensor<16xf32>
func.func @cast_i32_to_f32(%arg0: tensor<16xi32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.cast %arg0 : (tensor<16xi32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// Float-to-int via plain `tosa.cast` is intentionally rejected by this pass:
// the MIGraphX frontend must emit `tosa.custom "fp_to_int_cast"` instead so
// the saturating-truncation semantics are preserved (see CustomOpConverter
// and the @fp_to_int_cast_* tests below for the lowered IR).

// Float-to-bool: non-zero is true.
// CHECK-LABEL: @cast_f32_to_i1
// CHECK-NOT:   tosa.cast
// CHECK:       %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK:       arith.cmpf une, %arg0, %[[ZERO]] : tensor<16xf32>
func.func @cast_f32_to_i1(%arg0: tensor<16xf32>) -> tensor<16xi1> attributes {rock.kernel} {
  %0 = tosa.cast %arg0 : (tensor<16xf32>) -> tensor<16xi1>
  return %0 : tensor<16xi1>
}

// -----

// CHECK-LABEL: @cast_identity
// CHECK-NOT:   tosa.cast
// CHECK:       return %arg0 : tensor<16xf32>
func.func @cast_identity(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.cast %arg0 : (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @cast_i8_to_i32
// CHECK-NOT:   tosa.cast
// CHECK:       arith.extsi %arg0 : tensor<16xi8> to tensor<16xi32>
func.func @cast_i8_to_i32(%arg0: tensor<16xi8>) -> tensor<16xi32> attributes {rock.kernel} {
  %0 = tosa.cast %arg0 : (tensor<16xi8>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// i1 is treated as unsigned: sign-extending i1 would give 0/-1 instead of 0/1.
// CHECK-LABEL: @cast_i1_to_i32
// CHECK-NOT:   tosa.cast
// CHECK:       arith.extui %arg0 : tensor<16xi1> to tensor<16xi32>
func.func @cast_i1_to_i32(%arg0: tensor<16xi1>) -> tensor<16xi32> attributes {rock.kernel} {
  %0 = tosa.cast %arg0 : (tensor<16xi1>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// i1 -> float should also zero-extend (uitofp gives 0.0/1.0).
// CHECK-LABEL: @cast_i1_to_f32
// CHECK-NOT:   tosa.cast
// CHECK:       arith.uitofp %arg0 : tensor<16xi1> to tensor<16xf32>
func.func @cast_i1_to_f32(%arg0: tensor<16xi1>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.cast %arg0 : (tensor<16xi1>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @cast_i32_to_i8
// CHECK-NOT:   tosa.cast
// CHECK:       arith.trunci %arg0 : tensor<16xi32> to tensor<16xi8>
func.func @cast_i32_to_i8(%arg0: tensor<16xi32>) -> tensor<16xi8> attributes {rock.kernel} {
  %0 = tosa.cast %arg0 : (tensor<16xi32>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// Int-to-bool: any nonzero value is true. Must use cmpne, not trunci (which
// would only look at the LSB, giving wrong results for even nonzero values).
// CHECK-LABEL: @cast_i32_to_i1
// CHECK-NOT:   tosa.cast
// CHECK-NOT:   arith.trunci
// CHECK:       %[[ZERO:.*]] = arith.constant dense<0> : tensor<16xi32>
// CHECK:       arith.cmpi ne, %arg0, %[[ZERO]] : tensor<16xi32>
func.func @cast_i32_to_i1(%arg0: tensor<16xi32>) -> tensor<16xi1> attributes {rock.kernel} {
  %0 = tosa.cast %arg0 : (tensor<16xi32>) -> tensor<16xi1>
  return %0 : tensor<16xi1>
}

// -----

// CHECK-LABEL: @greater_f32
// CHECK-NOT:   tosa.greater
// CHECK:       arith.cmpf ogt, %arg0, %arg1 : tensor<4x8xf32>
func.func @greater_f32(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xi1> attributes {rock.kernel} {
  %0 = tosa.greater %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xi1>
  return %0 : tensor<4x8xi1>
}

// -----

// CHECK-LABEL: @greater_equal_f32
// CHECK-NOT:   tosa.greater_equal
// CHECK:       arith.cmpf oge, %arg0, %arg1 : tensor<4x8xf32>
func.func @greater_equal_f32(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xi1> attributes {rock.kernel} {
  %0 = tosa.greater_equal %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xi1>
  return %0 : tensor<4x8xi1>
}

// -----

// CHECK-LABEL: @equal_f32
// CHECK-NOT:   tosa.equal
// CHECK:       arith.cmpf oeq, %arg0, %arg1 : tensor<4x8xf32>
func.func @equal_f32(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xi1> attributes {rock.kernel} {
  %0 = tosa.equal %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xi1>
  return %0 : tensor<4x8xi1>
}

// -----

// CHECK-LABEL: @bitwise_and_i32
// CHECK-NOT:   tosa.bitwise_and
// CHECK:       arith.andi %arg0, %arg1 : tensor<8xi32>
func.func @bitwise_and_i32(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> attributes {rock.kernel} {
  %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: @bitwise_or_i32
// CHECK-NOT:   tosa.bitwise_or
// CHECK:       arith.ori %arg0, %arg1 : tensor<8xi32>
func.func @bitwise_or_i32(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> attributes {rock.kernel} {
  %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: @logical_left_shift
// CHECK-NOT:   tosa.logical_left_shift
// CHECK:       arith.shli %arg0, %arg1 : tensor<8xi32>
func.func @logical_left_shift(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> attributes {rock.kernel} {
  %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: @const_legal
// CHECK:       tosa.const
// CHECK:       arith.addf
func.func @const_legal(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {rock.kernel} {
  %cst = "tosa.const"() {values = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = tosa.add %arg0, %cst : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @non_kernel_untouched
// CHECK:       tosa.add
func.func @non_kernel_untouched(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// -----

// CHECK-LABEL: @chained_ops
// CHECK-NOT:   tosa.
// CHECK-DAG:   %[[ADD:.*]] = arith.addf %arg0, %arg1 : tensor<8xf32>
// CHECK-DAG:   %[[EXP:.*]] = math.exp %[[ADD]] : tensor<8xf32>
// CHECK:       return %[[EXP]]
func.func @chained_ops(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> attributes {rock.kernel} {
  %0 = tosa.add %arg0, %arg1 : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  %1 = tosa.exp %0 : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @sin_f32
// CHECK-NOT:   tosa.sin
// CHECK:       math.sin %arg0 : tensor<16xf32>
func.func @sin_f32(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.sin %arg0 : (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @cos_f32
// CHECK-NOT:   tosa.cos
// CHECK:       math.cos %arg0 : tensor<16xf32>
func.func @cos_f32(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.cos %arg0 : (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @intdiv
// CHECK-NOT:   tosa.intdiv
// CHECK:       arith.divsi %arg0, %arg1 : tensor<8xi32>
func.func @intdiv(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> attributes {rock.kernel} {
  %0 = tosa.intdiv %arg0, %arg1 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: @add_f32_broadcast
// CHECK-NOT:   tosa.add
// CHECK:       %[[B:.*]] = rock.transform %arg1
// CHECK:       arith.addf %arg0, %[[B]] : tensor<4x8xf32>
func.func @add_f32_broadcast(%arg0: tensor<4x8xf32>, %arg1: tensor<1x8xf32>) -> tensor<4x8xf32> attributes {rock.kernel} {
  %0 = tosa.add %arg0, %arg1 : (tensor<4x8xf32>, tensor<1x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// unsigned_cast float->uint: f32 -> i32 hits case 3 (mantissa too narrow for
// uint max). Clamp the lower bound to 0 in the float domain, convert, then fix
// up overflow with a select against (uintMax + 1) = 2^32.
// CHECK-LABEL: @unsigned_cast_fptoui
// CHECK-NOT:   tosa.custom
// CHECK-NOT:   arith.cmpf uno
// CHECK:       %[[MINCLAMP:.*]] = arith.maxnumf %arg0, {{.*}} : tensor<16xf32>
// CHECK:       %[[CONV:.*]] = arith.fptoui %[[MINCLAMP]] : tensor<16xf32> to tensor<16xi32>
// CHECK:       %[[OVF:.*]] = arith.cmpf uge, %arg0, {{.*}} : tensor<16xf32>
// CHECK:       arith.select %[[OVF]], {{.*}}, %[[CONV]] : tensor<16xi1>, tensor<16xi32>

// IEEE-LABEL: @unsigned_cast_fptoui
// IEEE-NOT:   tosa.custom
// IEEE:       %[[NAN:.*]] = arith.cmpf uno, %arg0, %arg0 : tensor<16xf32>
// IEEE:       %[[SAN:.*]] = arith.select %[[NAN]], {{.*}}, %arg0 : tensor<16xi1>, tensor<16xf32>
// IEEE:       %[[MINCLAMP:.*]] = arith.maxnumf %[[SAN]], {{.*}} : tensor<16xf32>
// IEEE:       %[[CONV:.*]] = arith.fptoui %[[MINCLAMP]] : tensor<16xf32> to tensor<16xi32>
// IEEE:       %[[OVF:.*]] = arith.cmpf uge, %[[SAN]], {{.*}} : tensor<16xf32>
// IEEE:       arith.select %[[OVF]], {{.*}}, %[[CONV]] : tensor<16xi1>, tensor<16xi32>
func.func @unsigned_cast_fptoui(%arg0: tensor<16xf32>) -> tensor<16xi32> attributes {rock.kernel} {
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xf32>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// f32 -> i8 hits case 2 (both 0 and 255 are exactly representable), so the
// whole cast is a two-sided clamp. This is the shape that matters for codegen:
// with nothing to sanitize, the pair also earns an nnan from
// RockAllowFastMathFlags and folds to a single tt.clampf (one v_med3).
// CHECK-LABEL: @unsigned_cast_fptoui_f32_to_i8
// CHECK-NOT:   tosa.custom
// CHECK-NOT:   arith.cmpf
// CHECK-NOT:   arith.select
// CHECK:       %[[HI:.*]] = arith.minnumf %arg0, {{.*}} : tensor<16xf32>
// CHECK:       %[[CLAMPED:.*]] = arith.maxnumf %[[HI]], {{.*}} : tensor<16xf32>
// CHECK:       arith.fptoui %[[CLAMPED]] : tensor<16xf32> to tensor<16xi8>

// IEEE-LABEL: @unsigned_cast_fptoui_f32_to_i8
// IEEE-NOT:   tosa.custom
// IEEE:       %[[NAN:.*]] = arith.cmpf uno, %arg0, %arg0 : tensor<16xf32>
// IEEE:       %[[SAN:.*]] = arith.select %[[NAN]], {{.*}}, %arg0 : tensor<16xi1>, tensor<16xf32>
// IEEE:       %[[HI:.*]] = arith.minnumf %[[SAN]], {{.*}} : tensor<16xf32>
// IEEE:       %[[CLAMPED:.*]] = arith.maxnumf %[[HI]], {{.*}} : tensor<16xf32>
// IEEE:       arith.fptoui %[[CLAMPED]] : tensor<16xf32> to tensor<16xi8>
func.func @unsigned_cast_fptoui_f32_to_i8(%arg0: tensor<16xf32>) -> tensor<16xi8> attributes {rock.kernel} {
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xf32>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// f16 -> uint32 is the only one of the three cases where the destination range
// swallows the whole float exponent range, so no finite input can overflow and
// the cast needs a clamp on one side only: negatives (including -inf) go to 0
// before converting, and +inf is patched to uintMax afterwards.
//
// Only the sanitization is conditional. The +inf fix-up stays either way,
// because the no-NaN assumption says nothing about infinities, and running its
// UEQ compare on the raw input is safe for the same reason it was safe on the
// sanitized value.
// CHECK-LABEL: @unsigned_cast_fptoui_f16_to_i32
// CHECK-NOT:   tosa.custom
// CHECK-NOT:   arith.cmpf uno
// CHECK:       %[[OVF:.*]] = arith.cmpf ueq, %arg0, {{.*}} : tensor<16xf16>
// CHECK:       %[[POS:.*]] = arith.maxnumf %arg0, {{.*}} : tensor<16xf16>
// CHECK:       %[[CONV:.*]] = arith.fptoui %[[POS]] : tensor<16xf16> to tensor<16xi32>
// CHECK:       arith.select %[[OVF]], {{.*}}, %[[CONV]] : tensor<16xi1>, tensor<16xi32>

// IEEE-LABEL: @unsigned_cast_fptoui_f16_to_i32
// IEEE-NOT:   tosa.custom
// IEEE:       %[[NAN:.*]] = arith.cmpf uno, %arg0, %arg0 : tensor<16xf16>
// IEEE:       %[[SAN:.*]] = arith.select %[[NAN]], {{.*}}, %arg0 : tensor<16xi1>, tensor<16xf16>
// IEEE:       %[[OVF:.*]] = arith.cmpf ueq, %[[SAN]], {{.*}} : tensor<16xf16>
// IEEE:       %[[POS:.*]] = arith.maxnumf %[[SAN]], {{.*}} : tensor<16xf16>
// IEEE:       %[[CONV:.*]] = arith.fptoui %[[POS]] : tensor<16xf16> to tensor<16xi32>
// IEEE:       arith.select %[[OVF]], {{.*}}, %[[CONV]] : tensor<16xi1>, tensor<16xi32>
func.func @unsigned_cast_fptoui_f16_to_i32(%arg0: tensor<16xf16>) -> tensor<16xi32> attributes {rock.kernel} {
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xf16>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// CHECK-LABEL: @add_f32_broadcast_both
// CHECK-NOT:   tosa.add
// CHECK-DAG:   %[[B0:.*]] = rock.transform %arg0
// CHECK-DAG:   %[[B1:.*]] = rock.transform %arg1
// CHECK:       arith.addf %[[B0]], %[[B1]] : tensor<4x8xf32>
func.func @add_f32_broadcast_both(%arg0: tensor<4x1xf32>, %arg1: tensor<1x8xf32>) -> tensor<4x8xf32> attributes {rock.kernel} {
  %0 = tosa.add %arg0, %arg1 : (tensor<4x1xf32>, tensor<1x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @unsigned_cast_uitofp
// CHECK-NOT:   tosa.custom
// CHECK:       arith.uitofp %arg0 : tensor<16xi32> to tensor<16xf32>
func.func @unsigned_cast_uitofp(%arg0: tensor<16xi32>) -> tensor<16xf32> attributes {rock.kernel} {
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xi32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @mul_f32_broadcast
// CHECK-NOT:   tosa.mul
// CHECK:       %[[B:.*]] = rock.transform %arg1
// CHECK:       arith.mulf %arg0, %[[B]] : tensor<4x8xf32>
func.func @mul_f32_broadcast(%arg0: tensor<4x8xf32>, %arg1: tensor<1x8xf32>) -> tensor<4x8xf32> attributes {rock.kernel} {
  %shift = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = "tosa.mul"(%arg0, %arg1, %shift) : (tensor<4x8xf32>, tensor<1x8xf32>, tensor<1xi8>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @unsigned_cast_extui
// CHECK-NOT:   tosa.custom
// CHECK:       arith.extui %arg0 : tensor<16xi8> to tensor<16xi32>
func.func @unsigned_cast_extui(%arg0: tensor<16xi8>) -> tensor<16xi32> attributes {rock.kernel} {
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xi8>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// CHECK-LABEL: @greater_f32_broadcast
// CHECK-NOT:   tosa.greater
// CHECK:       %[[B:.*]] = rock.transform %arg1
// CHECK:       arith.cmpf ogt, %arg0, %[[B]] : tensor<4x8xf32>
func.func @greater_f32_broadcast(%arg0: tensor<4x8xf32>, %arg1: tensor<1x8xf32>) -> tensor<4x8xi1> attributes {rock.kernel} {
  %0 = tosa.greater %arg0, %arg1 : (tensor<4x8xf32>, tensor<1x8xf32>) -> tensor<4x8xi1>
  return %0 : tensor<4x8xi1>
}

// -----

// CHECK-LABEL: @unsigned_cast_trunci
// CHECK-NOT:   tosa.custom
// CHECK:       arith.trunci %arg0 : tensor<16xi32> to tensor<16xi8>
func.func @unsigned_cast_trunci(%arg0: tensor<16xi32>) -> tensor<16xi8> attributes {rock.kernel} {
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xi32>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// CHECK-LABEL: @bitwise_and_broadcast
// CHECK-NOT:   tosa.bitwise_and
// CHECK:       %[[B:.*]] = rock.transform %arg1
// CHECK:       arith.andi %arg0, %[[B]] : tensor<8x4xi32>
func.func @bitwise_and_broadcast(%arg0: tensor<8x4xi32>, %arg1: tensor<1x4xi32>) -> tensor<8x4xi32> attributes {rock.kernel} {
  %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<8x4xi32>, tensor<1x4xi32>) -> tensor<8x4xi32>
  return %0 : tensor<8x4xi32>
}

// -----

// CHECK-LABEL: @unsigned_div
// CHECK-NOT:   tosa.custom
// CHECK:       arith.divui %arg0, %arg1 : tensor<8xi32>
func.func @unsigned_div(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> attributes {rock.kernel} {
  %0 = tosa.custom %arg0, %arg1 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_div"} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: @unsigned_max
// CHECK-NOT:   tosa.custom
// CHECK-NOT:   arith.maxsi
// CHECK:       arith.maxui %arg0, %arg1 : tensor<8xi32>
func.func @unsigned_max(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> attributes {rock.kernel} {
  %0 = tosa.custom %arg0, %arg1 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_max"} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// fp_to_int_cast: float-to-signed-int with saturation, matching MIGraphX
// convert semantics. Lowers via rock::createClampedFPToInt. This is the
// only path through which fp->int casts reach this pass; plain `tosa.cast`
// fp->int is rejected (see CastConverter), so the MIGraphX frontend must
// emit this custom op for any fp->int conversion.
// f32 -> i32 is Case 3 of createClampedFPToInt: f32 mantissa (24) is too
// narrow to represent i32 max (2^31-1) exactly, so we clamp the lower
// bound, fptosi, then fix up overflow with a select against int-max-plus-one.
// CHECK-LABEL: @fp_to_int_cast_f32_to_i32
// CHECK-NOT:   tosa.custom
// CHECK-NOT:   math.roundeven
// CHECK-NOT:   arith.cmpf uno
// CHECK:       %[[MINCLAMP:.*]] = arith.maxnumf %arg0, {{.*}} : tensor<16xf32>
// CHECK:       %[[CONV:.*]] = arith.fptosi %[[MINCLAMP]] : tensor<16xf32> to tensor<16xi32>
// CHECK:       %[[OVF:.*]] = arith.cmpf uge, %arg0, {{.*}} : tensor<16xf32>
// CHECK:       arith.select %[[OVF]], {{.*}}, %[[CONV]] : tensor<16xi1>, tensor<16xi32>

// IEEE-LABEL: @fp_to_int_cast_f32_to_i32
// IEEE-NOT:   tosa.custom
// IEEE-NOT:   math.roundeven
// IEEE:       %[[NAN:.*]] = arith.cmpf uno, %arg0, %arg0 : tensor<16xf32>
// IEEE:       %[[SAN:.*]] = arith.select %[[NAN]], {{.*}}, %arg0 : tensor<16xi1>, tensor<16xf32>
// IEEE:       %[[MINCLAMP:.*]] = arith.maxnumf %[[SAN]], {{.*}} : tensor<16xf32>
// IEEE:       %[[CONV:.*]] = arith.fptosi %[[MINCLAMP]] : tensor<16xf32> to tensor<16xi32>
// IEEE:       %[[OVF:.*]] = arith.cmpf uge, %[[SAN]], {{.*}} : tensor<16xf32>
// IEEE:       arith.select %[[OVF]], {{.*}}, %[[CONV]] : tensor<16xi1>, tensor<16xi32>
func.func @fp_to_int_cast_f32_to_i32(%arg0: tensor<16xf32>) -> tensor<16xi32> attributes {rock.kernel} {
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<16xf32>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// f32 -> i8 (signed): Case 2 of rock::createClampedFPToInt (f32 mantissa
// (24) >= dstWidth-1 (7), so both -128 and 127 are exactly representable), so
// the cast is a two-sided in-float clamp followed by fptosi. With no sanitize
// pair in front, the clamp also earns an nnan from RockAllowFastMathFlags and
// folds to a single tt.clampf (one v_med3).
// CHECK-LABEL: @fp_to_int_cast_f32_to_i8
// CHECK-NOT:   tosa.custom
// CHECK-NOT:   arith.cmpf
// CHECK-NOT:   arith.select
// CHECK:       %[[HI:.*]] = arith.minnumf %arg0, {{.*}} : tensor<16xf32>
// CHECK:       %[[CLAMPED:.*]] = arith.maxnumf %[[HI]], {{.*}} : tensor<16xf32>
// CHECK:       arith.fptosi %[[CLAMPED]] : tensor<16xf32> to tensor<16xi8>

// IEEE-LABEL: @fp_to_int_cast_f32_to_i8
// IEEE-NOT:   tosa.custom
// IEEE:       %[[NAN:.*]] = arith.cmpf uno, %arg0, %arg0 : tensor<16xf32>
// IEEE:       %[[SAN:.*]] = arith.select %[[NAN]], {{.*}}, %arg0 : tensor<16xi1>, tensor<16xf32>
// IEEE:       %[[HI:.*]] = arith.minnumf %[[SAN]], {{.*}} : tensor<16xf32>
// IEEE:       %[[CLAMPED:.*]] = arith.maxnumf %[[HI]], {{.*}} : tensor<16xf32>
// IEEE:       arith.fptosi %[[CLAMPED]] : tensor<16xf32> to tensor<16xi8>
func.func @fp_to_int_cast_f32_to_i8(%arg0: tensor<16xf32>) -> tensor<16xi8> attributes {rock.kernel} {
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<16xf32>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// CHECK-LABEL: @select_broadcast
// CHECK-NOT:   tosa.select
// CHECK-DAG:   %[[BP:.*]] = rock.transform %arg0 {{.*}} : tensor<1x8xi1> to tensor<4x8xi1>
// CHECK-DAG:   %[[BF:.*]] = rock.transform %arg2 {{.*}} : tensor<4x1xf32> to tensor<4x8xf32>
// CHECK:       arith.select %[[BP]], %arg1, %[[BF]] : tensor<4x8xi1>, tensor<4x8xf32>
func.func @select_broadcast(%arg0: tensor<1x8xi1>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x1xf32>) -> tensor<4x8xf32> attributes {rock.kernel} {
  %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<1x8xi1>, tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// Non-rocmlir custom ops should be left untouched.
// CHECK-LABEL: @custom_other_domain
// CHECK:       tosa.custom
func.func @custom_other_domain(%arg0: tensor<8xf32>) -> tensor<8xf32> attributes {rock.kernel} {
  %0 = tosa.custom %arg0 {domain_name = "other", implementation_attrs = "", operator_name = "some_op"} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// Rocmlir custom ops in a non-kernel function should be left untouched
// (the pass early-returns without rock.kernel).
// CHECK-LABEL: @unsigned_cast_non_kernel
// CHECK:       tosa.custom
// CHECK-NOT:   arith.fptoui
func.func @unsigned_cast_non_kernel(%arg0: tensor<16xf32>) -> tensor<16xi32> {
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<16xf32>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// CHECK-LABEL: @tanh_f16
// CHECK-NOT:   tosa.tanh
// CHECK:       math.tanh %arg0 : tensor<64xf16>
func.func @tanh_f16(%arg0: tensor<64xf16>) -> tensor<64xf16> attributes {rock.kernel} {
  %0 = tosa.tanh %arg0 : (tensor<64xf16>) -> tensor<64xf16>
  return %0 : tensor<64xf16>
}

// -----

// CHECK-LABEL: @tanh_direct
// CHECK:       math.tanh %arg0 : tensor<32xf32>
func.func @tanh_direct(%arg0: tensor<32xf32>) -> tensor<32xf32> attributes {rock.kernel} {
  %0 = math.tanh %arg0 : tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

// arith.negf is in the op subset Triton converts, and it lowers to an LLVM
// fneg, i.e. a free source modifier, so it is left alone on tensors as well as
// on scalars.
// CHECK-LABEL: @negf_direct
// CHECK:       arith.negf %arg0 : tensor<32xf32>
func.func @negf_direct(%arg0: tensor<32xf32>) -> tensor<32xf32> attributes {rock.kernel} {
  %0 = arith.negf %arg0 : tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

// CHECK-LABEL: @negf_scalar_preserved
// CHECK:       arith.negf %arg0 : f32
func.func @negf_scalar_preserved(%arg0: f32) -> f32 attributes {rock.kernel} {
  %0 = arith.negf %arg0 : f32
  return %0 : f32
}

// -----

// Like tanh, pow is left as math.powf; rock-legalize-math-for-triton turns it
// into the OCML call once the target is known.
// CHECK-LABEL: @pow_f32
// CHECK-NOT:   tosa.pow
// CHECK:       math.powf %arg0, %arg1 : tensor<64xf32>
func.func @pow_f32(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel} {
  %0 = tosa.pow %arg0, %arg1 : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// -----

// CHECK-LABEL: @powf_direct
// CHECK:       math.powf %arg0, %arg1 : tensor<32xf32>
func.func @powf_direct(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32> attributes {rock.kernel} {
  %0 = math.powf %arg0, %arg1 : tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

// Scalar roundeven is preserved (math dialect ops are legal).
// CHECK-LABEL: @roundeven_scalar_preserved
// CHECK:       math.roundeven %arg0 : f32
func.func @roundeven_scalar_preserved(%arg0: f32) -> f32 attributes {rock.kernel} {
  %0 = math.roundeven %arg0 : f32
  return %0 : f32
}

// -----

// TanhTritonWorkaround only applies to shaped types; scalar tanh is preserved.
// CHECK-LABEL: @tanh_scalar_preserved
// CHECK:       math.tanh %arg0 : f32
func.func @tanh_scalar_preserved(%arg0: f32) -> f32 attributes {rock.kernel} {
  %0 = math.tanh %arg0 : f32
  return %0 : f32
}

// -----

// PowFTritonWorkaround only applies to shaped types; scalar powf is preserved.
// CHECK-LABEL: @powf_scalar_preserved
// CHECK:       math.powf %arg0, %arg1 : f32
func.func @powf_scalar_preserved(%arg0: f32, %arg1: f32) -> f32 attributes {rock.kernel} {
  %0 = math.powf %arg0, %arg1 : f32
  return %0 : f32
}
