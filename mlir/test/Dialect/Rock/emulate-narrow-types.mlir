// Unit tests for the rock-emulate-narrow-types pass.
// Input IR assumes RockConvertNarrowTypeSignaturesPass has already run:
// function args are packed i8 with unrealized_conversion_cast back to i4.

// RUN: rocmlir-opt -rock-emulate-narrow-types -mlir-print-local-scope %s | FileCheck %s

// Test 1: Basic i4 load is emulated via packed i8.
// The unrealized_conversion_cast feeds the memref.load; after emulation,
// the load is rewritten to operate on the i8 arg directly.

// CHECK-LABEL: func.func @test_i4_load
// CHECK-SAME: (%[[MEM:.*]]: memref<16xi8>)
// CHECK:      memref.load %[[MEM]]
// CHECK:      arith.shrsi
// CHECK:      arith.trunci
func.func @test_i4_load(%mem: memref<16xi8>) -> i4 {
  %cast = builtin.unrealized_conversion_cast %mem : memref<16xi8> to memref<32xi4>
  %idx = arith.constant 5 : index
  %val = memref.load %cast[%idx] : memref<32xi4>
  return %val : i4
}

// Test 2: Basic i4 store is emulated via atomic read-modify-write on i8.

// CHECK-LABEL: func.func @test_i4_store
// CHECK-SAME: (%[[MEM:.*]]: memref<16xi8>, %[[VAL:.*]]: i4)
// CHECK:      arith.extui %[[VAL]] : i4 to i8
// CHECK:      memref.atomic_rmw andi
// CHECK:      memref.atomic_rmw ori
func.func @test_i4_store(%mem: memref<16xi8>, %val: i4) {
  %cast = builtin.unrealized_conversion_cast %mem : memref<16xi8> to memref<32xi4>
  %idx = arith.constant 3 : index
  memref.store %val, %cast[%idx] : memref<32xi4>
  return
}

// Test 3: Non-4-bit types are unchanged.

// CHECK-LABEL: func.func @test_no_i4_unchanged
// CHECK-SAME: memref<32xi8>
// CHECK-NOT: arith.shrsi
func.func @test_no_i4_unchanged(%mem: memref<32xi8>) -> i8 {
  %idx = arith.constant 0 : index
  %val = memref.load %mem[%idx] : memref<32xi8>
  return %val : i8
}

// Test 4: 2D memref<8x4xi4> (linearized to 16xi8 by signature conversion).

// CHECK-LABEL: func.func @test_i4_load_2d
// CHECK-SAME: (%[[MEM:.*]]: memref<16xi8>)
// CHECK:      memref.load %[[MEM]]
// CHECK:      arith.shrsi
func.func @test_i4_load_2d(%mem: memref<16xi8>) -> i4 {
  %cast = builtin.unrealized_conversion_cast %mem : memref<16xi8> to memref<8x4xi4>
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %val = memref.load %cast[%c2, %c3] : memref<8x4xi4>
  return %val : i4
}

// Test 5: Mixed args: only the i4 path is emulated.

// CHECK-LABEL: func.func @test_mixed_args
// CHECK-SAME: (%[[I4MEM:.*]]: memref<8xi8>, %[[I8MEM:.*]]: memref<32xi8>)
// CHECK:      memref.load %[[I4MEM]]
// CHECK:      memref.load %[[I8MEM]]
func.func @test_mixed_args(%i4mem: memref<8xi8>, %i8mem: memref<32xi8>) -> i8 {
  %cast = builtin.unrealized_conversion_cast %i4mem : memref<8xi8> to memref<16xi4>
  %idx = arith.constant 0 : index
  %v4 = memref.load %cast[%idx] : memref<16xi4>
  %v8 = memref.load %i8mem[%idx] : memref<32xi8>
  return %v8 : i8
}

// Test 6: memref.alloc of i4 is converted to smaller i8 allocation.

// CHECK-LABEL: func.func @test_i4_alloc
// CHECK:      memref.alloc() : memref<10xi8>
// CHECK:      memref.atomic_rmw andi
// CHECK:      memref.atomic_rmw ori
// CHECK:      memref.dealloc
func.func @test_i4_alloc(%val: i4) {
  %mem = memref.alloc() : memref<20xi4>
  %idx = arith.constant 7 : index
  memref.store %val, %mem[%idx] : memref<20xi4>
  memref.dealloc %mem : memref<20xi4>
  return
}
