// RUN: rocmlir-opt --rocmlir-custom-tosa-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: @integers_i4_to_i8
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xi4>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xi8>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xi4>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xi8>)
// CHECK-NEXT: %[[in:.+]]: i4
// CHECK-NEXT: %[[res:.+]] = arith.extui %[[in]] : i4 to i8
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<8x8x2xi8>
// CHECK-NEXT: return %[[ret]]
func.func @integers_i4_to_i8(%arg0: tensor<8x8x2xi4>) -> tensor<8x8x2xi8> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xi4>) -> tensor<8x8x2xi8>
  func.return %out : tensor<8x8x2xi8>
}

// CHECK-LABEL: @integers_i8_to_i4
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xi8>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xi4>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xi8>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xi4>)
// CHECK-NEXT: %[[in:.+]]: i8
// CHECK-NEXT: %[[res:.+]] = arith.trunci %[[in]] : i8 to i4
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<8x8x2xi4>
// CHECK-NEXT: return %[[ret]]
func.func @integers_i8_to_i4(%arg0: tensor<8x8x2xi8>) -> tensor<8x8x2xi4> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xi8>) -> tensor<8x8x2xi4>
  func.return %out : tensor<8x8x2xi4>
}

// -----

// CHECK-LABEL: @floats_i4_to_f16
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xi4>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xf16>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xi4>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xf16>)
// CHECK-NEXT: %[[in:.+]]: i4
// CHECK-NEXT: %[[res:.+]] = arith.uitofp %[[in]] : i4 to f16
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<8x8x2xf16>
// CHECK-NEXT: return %[[ret]]
func.func @floats_i4_to_f16(%arg0: tensor<8x8x2xi4>) -> tensor<8x8x2xf16> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xi4>) -> tensor<8x8x2xf16>
  func.return %out : tensor<8x8x2xf16>
}

// CHECK-LABEL: @floats_i4_to_f32
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xi4>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xf32>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xi4>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xf32>)
// CHECK-NEXT: %[[in:.+]]: i4
// CHECK-NEXT: %[[res:.+]] = arith.uitofp %[[in]] : i4 to f32
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<8x8x2xf32>
// CHECK-NEXT: return %[[ret]]
func.func @floats_i4_to_f32(%arg0: tensor<8x8x2xi4>) -> tensor<8x8x2xf32> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xi4>) -> tensor<8x8x2xf32>
  func.return %out : tensor<8x8x2xf32>
}

// f16 -> i8 (unsigned): hits Case 2 of rock::createClampedFPToInt (mantissa
// 11 >= dstWidth 8, so both 0 and 255 are exactly representable in f16).
// First a NaN-sanitization prologue (cmpf uno + select to 0), then the
// in-float-domain min/max clamp, then the truncating fptoui. The min comes
// before the max because the helper emits `minnumf(in, intMax)` first and
// `maxnumf(hi, intMin)` second; this differs from the GPU/MIGraphX path
// only in being inside a linalg.generic body. The clamp uses the
// non-propagating num forms because the select above has already removed any
// NaN, leaving nothing for the IEEE-2019 spelling to propagate.
// CHECK-LABEL: @floats_f16_to_i8
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xf16>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xi8>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xf16>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xi8>)
// CHECK-NEXT: %[[in:.+]]: f16
// CHECK:      %[[isnan:.+]] = arith.cmpf uno, %[[in]], %[[in]] : f16
// CHECK:      %[[san:.+]] = arith.select %[[isnan]], {{.*}}, %[[in]] : f16
// CHECK:      %[[hi:.+]] = arith.minnumf %[[san]], {{.*}} : f16
// CHECK:      %[[clamped:.+]] = arith.maxnumf %[[hi]], {{.*}} : f16
// CHECK:      %[[res:.+]] = arith.fptoui %[[clamped]] : f16 to i8
// CHECK:      linalg.yield %[[res]]
// CHECK:      -> tensor<8x8x2xi8>
// CHECK:      return %[[ret]]
func.func @floats_f16_to_i8(%arg0: tensor<8x8x2xf16>) -> tensor<8x8x2xi8> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xf16>) -> tensor<8x8x2xi8>
  func.return %out : tensor<8x8x2xi8>
}

// -----

// fp_to_int_cast lowers to rock::createClampedFPToInt, which implements the
// MIGraphX saturating + truncating convert semantics (NaN -> 0, out-of-range
// -> INT_MIN/MAX, no round-to-nearest-even). The exact case picked depends
// on the (source-float, dest-int) pair.

// f32 -> i32 (signed): Case 3, because f32 mantissa (24) is too narrow to
// represent INT32_MAX (2^31 - 1) exactly. Strategy: NaN-sanitize, clamp the
// lower bound in-float, fptosi, then patch up overflow against the exactly
// representable (INT32_MAX + 1) = 2^31 with a select.
// CHECK-LABEL: @fp_to_int_cast_f32_to_i32
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xf32>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xi32>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xf32>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xi32>)
// CHECK-NEXT: %[[in:.+]]: f32
// CHECK:      %[[isnan:.+]] = arith.cmpf uno, %[[in]], %[[in]] : f32
// CHECK:      %[[san:.+]] = arith.select %[[isnan]], {{.*}}, %[[in]] : f32
// CHECK:      %[[clamped:.+]] = arith.maxnumf %[[san]], {{.*}} : f32
// CHECK:      %[[conv:.+]] = arith.fptosi %[[clamped]] : f32 to i32
// CHECK:      %[[ovf:.+]] = arith.cmpf uge, %[[san]], {{.*}} : f32
// CHECK:      %[[res:.+]] = arith.select %[[ovf]], {{.*}}, %[[conv]] : i32
// CHECK:      linalg.yield %[[res]]
// CHECK:      -> tensor<8x8x2xi32>
// CHECK:      return %[[ret]]
func.func @fp_to_int_cast_f32_to_i32(%arg0: tensor<8x8x2xf32>) -> tensor<8x8x2xi32> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<8x8x2xf32>) -> tensor<8x8x2xi32>
  func.return %out : tensor<8x8x2xi32>
}

// f16 -> i8 (signed): Case 2, because f16 mantissa (11) >= dstWidth-1 (7),
// so both -128 and 127 are exactly representable. Strategy: NaN-sanitize,
// then fully clamp in-float (min then max), then fptosi.
// CHECK-LABEL: @fp_to_int_cast_f16_to_i8
// CHECK-SAME: (%[[arg0:.+]]: tensor<16xf16>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<16xi8>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<16xf16>)
// CHECK-SAME: outs(%[[empty]] : tensor<16xi8>)
// CHECK-NEXT: %[[in:.+]]: f16
// CHECK:      %[[isnan:.+]] = arith.cmpf uno, %[[in]], %[[in]] : f16
// CHECK:      %[[san:.+]] = arith.select %[[isnan]], {{.*}}, %[[in]] : f16
// CHECK:      %[[hi:.+]] = arith.minnumf %[[san]], {{.*}} : f16
// CHECK:      %[[clamped:.+]] = arith.maxnumf %[[hi]], {{.*}} : f16
// CHECK:      %[[res:.+]] = arith.fptosi %[[clamped]] : f16 to i8
// CHECK:      linalg.yield %[[res]]
// CHECK:      -> tensor<16xi8>
// CHECK:      return %[[ret]]
func.func @fp_to_int_cast_f16_to_i8(%arg0: tensor<16xf16>) -> tensor<16xi8> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<16xf16>) -> tensor<16xi8>
  func.return %out : tensor<16xi8>
}

// -----

// Basic acc_type promotion: f16 output with f32 acc_type gets promoted.

// CHECK-LABEL: @matmul_acc_promotion_basic
// CHECK: %[[MATMUL:.+]] = tosa.matmul
// CHECK-SAME: -> tensor<1x4x4xf32>
// CHECK: %[[CAST:.+]] = tosa.cast %[[MATMUL]] : (tensor<1x4x4xf32>) -> tensor<1x4x4xf16>
// CHECK: return %[[CAST]]
func.func @matmul_acc_promotion_basic(%arg0: tensor<1x4x8xf16>, %arg1: tensor<1x8x4xf16>) -> tensor<1x4x4xf16> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %0 = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) {acc_type = f32} : (tensor<1x4x8xf16>, tensor<1x8x4xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x4x4xf16>
  func.return %0 : tensor<1x4x4xf16>
}

// -----

// No-op: acc_type matches the output element type, pattern should not fire.

// CHECK-LABEL: @matmul_acc_type_matches_output
// CHECK: tosa.matmul
// CHECK-SAME: -> tensor<1x4x4xf32>
// CHECK-NOT: tosa.cast
// CHECK: return
func.func @matmul_acc_type_matches_output(%arg0: tensor<1x4x8xf16>, %arg1: tensor<1x8x4xf16>) -> tensor<1x4x4xf32> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %0 = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) {acc_type = f32} : (tensor<1x4x8xf16>, tensor<1x8x4xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x4x4xf32>
  func.return %0 : tensor<1x4x4xf32>
}

// -----

// No-op: no acc_type attribute, pattern should not fire.

// CHECK-LABEL: @matmul_no_acc_type
// CHECK: tosa.matmul
// CHECK-SAME: -> tensor<1x4x4xf16>
// CHECK-NOT: tosa.cast
// CHECK: return
func.func @matmul_no_acc_type(%arg0: tensor<1x4x8xf16>, %arg1: tensor<1x8x4xf16>) -> tensor<1x4x4xf16> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %0 = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) : (tensor<1x4x8xf16>, tensor<1x8x4xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x4x4xf16>
  func.return %0 : tensor<1x4x4xf16>
}

// -----

// Verify that the acc_type promotion pattern preserves discardable attributes
// such as perf_config.

// CHECK-LABEL: @matmul_acc_promotion_preserves_attrs
// CHECK: %[[MATMUL:.+]] = tosa.matmul
// CHECK-SAME: acc_type = f32
// CHECK-SAME: perf_config = "some_config"
// CHECK: tosa.cast %[[MATMUL]]
func.func @matmul_acc_promotion_preserves_attrs(%arg0: tensor<1x4x8xf16>, %arg1: tensor<1x8x4xf16>) -> tensor<1x4x4xf16> {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %0 = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) {acc_type = f32, perf_config = "some_config"} : (tensor<1x4x8xf16>, tensor<1x8x4xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x4x4xf16>
  func.return %0 : tensor<1x4x4xf16>
}

// -----

// CHECK-LABEL: @unsigned_div
// CHECK-SAME: (%[[arg0:.+]]: tensor<1x36x384x64xi32>, %[[arg1:.+]]: tensor<1x36x384x64xi32>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<1x36x384x64xi32>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]], %[[arg1]] : tensor<1x36x384x64xi32>, tensor<1x36x384x64xi32>)
// CHECK-SAME: outs(%[[empty]] : tensor<1x36x384x64xi32>)
// CHECK-NEXT: %[[in:.+]]: i32, %[[in1:.+]]: i32, %[[out:.+]]: i32
// CHECK-NEXT: %[[res:.+]] = arith.divui %[[in]], %[[in1]] : i32
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<1x36x384x64xi32>
// CHECK-NEXT: return %[[ret]]
func.func @unsigned_div(%arg0: tensor<1x36x384x64xi32>, %arg1: tensor<1x36x384x64xi32>) -> tensor<1x36x384x64xi32> {
  %out = tosa.custom %arg0, %arg1 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_div"} : (tensor<1x36x384x64xi32>, tensor<1x36x384x64xi32>) -> tensor<1x36x384x64xi32>
  func.return %out : tensor<1x36x384x64xi32>
}

// -----

// CHECK-LABEL: @unsigned_max
// CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>, %[[arg1:.+]]: tensor<4xi32>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<4xi32>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]], %[[arg1]] : tensor<4xi32>, tensor<4xi32>)
// CHECK-SAME: outs(%[[empty]] : tensor<4xi32>)
// CHECK-NEXT: %[[in:.+]]: i32, %[[in1:.+]]: i32, %[[out:.+]]: i32
// CHECK-NEXT: %[[res:.+]] = arith.maxui %[[in]], %[[in1]] : i32
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<4xi32>
// CHECK-NEXT: return %[[ret]]
func.func @unsigned_max(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %out = tosa.custom %arg0, %arg1 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_max"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %out : tensor<4xi32>
}
