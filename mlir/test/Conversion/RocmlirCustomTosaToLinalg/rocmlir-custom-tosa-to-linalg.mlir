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

// CHECK-LABEL: @floats_f16_to_i8
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xf16>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xi8>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xf16>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xi8>)
// CHECK-NEXT: %[[in:.+]]: f16
// CHECK-NEXT: %[[res:.+]] = arith.fptoui %[[in]] : f16 to i8
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<8x8x2xi8>
// CHECK-NEXT: return %[[ret]]
func.func @floats_f16_to_i8(%arg0: tensor<8x8x2xf16>) -> tensor<8x8x2xi8> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xf16>) -> tensor<8x8x2xi8>
  func.return %out : tensor<8x8x2xi8>
}

// -----

// fp_to_int_cast: float-to-int using direct truncation (arith.fptosi),
// bypassing upstream tosa-to-linalg round-to-nearest-even behavior.
// CHECK-LABEL: @fp_to_int_cast_f32_to_i32
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xf32>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xi32>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xf32>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xi32>)
// CHECK-NEXT: %[[in:.+]]: f32
// CHECK-NEXT: %[[res:.+]] = arith.fptosi %[[in]] : f32 to i32
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<8x8x2xi32>
// CHECK-NEXT: return %[[ret]]
func.func @fp_to_int_cast_f32_to_i32(%arg0: tensor<8x8x2xf32>) -> tensor<8x8x2xi32> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<8x8x2xf32>) -> tensor<8x8x2xi32>
  func.return %out : tensor<8x8x2xi32>
}

// CHECK-LABEL: @fp_to_int_cast_f16_to_i8
// CHECK-SAME: (%[[arg0:.+]]: tensor<16xf16>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<16xi8>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<16xf16>)
// CHECK-SAME: outs(%[[empty]] : tensor<16xi8>)
// CHECK-NEXT: %[[in:.+]]: f16
// CHECK-NEXT: %[[res:.+]] = arith.fptosi %[[in]] : f16 to i8
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<16xi8>
// CHECK-NEXT: return %[[ret]]
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
