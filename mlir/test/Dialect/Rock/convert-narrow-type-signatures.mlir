// Unit tests for rock-convert-narrow-type-signatures pass.
// Verifies that i4 memref types in function signatures are converted to packed i8.

// RUN: rocmlir-opt -rock-convert-narrow-type-signatures -mlir-print-local-scope %s | FileCheck %s

// Test 1: Basic i4 arg is converted, body gets unrealized_conversion_cast.

// CHECK-LABEL: func.func @test_i4_load
// CHECK-SAME: (%[[MEM:.*]]: memref<16xi8>)
// CHECK:      unrealized_conversion_cast %[[MEM]] : memref<16xi8> to memref<32xi4>
// CHECK:      memref.load
func.func @test_i4_load(%mem: memref<32xi4>) -> i4 {
  %idx = arith.constant 5 : index
  %val = memref.load %mem[%idx] : memref<32xi4>
  return %val : i4
}

// Test 2: Store arg is converted.

// CHECK-LABEL: func.func @test_i4_store
// CHECK-SAME: (%[[MEM:.*]]: memref<16xi8>, %[[VAL:.*]]: i4)
// CHECK:      unrealized_conversion_cast %[[MEM]] : memref<16xi8> to memref<32xi4>
func.func @test_i4_store(%mem: memref<32xi4>, %val: i4) {
  %idx = arith.constant 3 : index
  memref.store %val, %mem[%idx] : memref<32xi4>
  return
}

// Test 3: Non-4-bit types are unchanged.

// CHECK-LABEL: func.func @test_no_i4_unchanged
// CHECK-SAME: memref<32xi8>
// CHECK-NOT:  unrealized_conversion_cast
func.func @test_no_i4_unchanged(%mem: memref<32xi8>) -> i8 {
  %idx = arith.constant 0 : index
  %val = memref.load %mem[%idx] : memref<32xi8>
  return %val : i8
}

// Test 4: 2D memref arg is linearized.

// CHECK-LABEL: func.func @test_i4_load_2d
// CHECK-SAME: (%[[MEM:.*]]: memref<16xi8>)
// CHECK:      unrealized_conversion_cast %[[MEM]] : memref<16xi8> to memref<8x4xi4>
func.func @test_i4_load_2d(%mem: memref<8x4xi4>) -> i4 {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %val = memref.load %mem[%c2, %c3] : memref<8x4xi4>
  return %val : i4
}

// Test 5: Mixed args: only i4 is converted.

// CHECK-LABEL: func.func @test_mixed_args
// CHECK-SAME: (%[[I4MEM:.*]]: memref<8xi8>, %[[I8MEM:.*]]: memref<32xi8>)
// CHECK:      unrealized_conversion_cast %[[I4MEM]] : memref<8xi8> to memref<16xi4>
// CHECK-NOT:  unrealized_conversion_cast %[[I8MEM]]
func.func @test_mixed_args(%i4mem: memref<16xi4>, %i8mem: memref<32xi8>) -> i8 {
  %idx = arith.constant 0 : index
  %v4 = memref.load %i4mem[%idx] : memref<16xi4>
  %v8 = memref.load %i8mem[%idx] : memref<32xi8>
  return %v8 : i8
}

// Test 6: Caller whose own signature has no i4 types, but calls a callee
// with i4 args. The call operands must be converted to match the callee's
// new packed-i8 signature.

func.func private @callee_i4(%mem: memref<32xi4>, %out: memref<64xf32>)

// CHECK-LABEL: func.func @test_caller_no_i4_signature
// CHECK-SAME: ()
// CHECK:      %[[ALLOC:.*]] = memref.alloc() : memref<32xi4>
// CHECK:      %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]] : memref<32xi4> to memref<16xi8>
// CHECK:      call @callee_i4(%[[CAST]], %{{.*}}) : (memref<16xi8>, memref<64xf32>) -> ()
func.func @test_caller_no_i4_signature() {
  %mem = memref.alloc() : memref<32xi4>
  %out = memref.alloc() : memref<64xf32>
  call @callee_i4(%mem, %out) : (memref<32xi4>, memref<64xf32>) -> ()
  memref.dealloc %mem : memref<32xi4>
  memref.dealloc %out : memref<64xf32>
  return
}
