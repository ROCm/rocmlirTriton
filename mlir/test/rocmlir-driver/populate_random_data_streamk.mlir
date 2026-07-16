// Random-data population with a stream-K perf_config (v5, streamKMultiple = 1;
// splitKFactor stays 1). Like split-k, the stream-K output is a zero-prefilled
// atomic_add accumulator, so only the filter and input are populated as read
// operands while the output is zero-filled (not randomized). This checks a
// stream-K perf_config flows through rocmlir-gen for the -rand modes and that
// the output buffer stays zero-initialized.

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -perf_config="gemm:v5:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1,-1,-1" -ph -p -rand=none | rocmlir-opt -canonicalize | FileCheck %s --check-prefix=NONE

// NONE-NOT: call @seedRandomValues

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -perf_config="gemm:v5:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1,-1,-1" -ph -p -rand 1 | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,RAND1,RAND2
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -perf_config="gemm:v5:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1,-1,-1" -ph -p -rand 1 -rand_side filter | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,RAND1,FIXED2
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -perf_config="gemm:v5:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1,-1,-1" -ph -p -rand 1 -rand_side input | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,FIXED1,RAND2

// CHECK-LABEL: @main
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[min:.*]] = arith.constant -5 : i16
// CHECK-DAG: %[[max:.*]] = arith.constant 5 : i16
// CHECK-DAG: %[[one:.*]] = arith.constant 1 : i32
// HASFIXED-DAG: %[[one_i16:.*]] = arith.constant 1 : i16
// CHECK: call @seedRandomValues(%[[one]])

// CHECK: memref.alloc
// CHECK: affine.for
// RAND1-NEXT: %[[val1:.*]] = func.call @randomIntegerValue(%[[min]], %[[max]])
// FIXED1-NEXT: %[[val1:.*]] = func.call @randomIntegerValue(%[[one_i16]], %[[one_i16]])
// RAND1-NEXT: memref.store %[[val1]]
// FIXED1-NEXT: memref.store %[[val1]]
// CHECK: memref.alloc
// CHECK-NEXT: affine.for
// RAND2-NEXT: %[[val2:.*]] = func.call @randomIntegerValue(%[[min]], %[[max]])
// FIXED2-NEXT: %[[val2:.*]] = func.call @randomIntegerValue(%[[one_i16]], %[[one_i16]])
// RAND2-NEXT: memref.store %[[val2]]
// FIXED2-NEXT: memref.store %[[val2]]
// The stream-K output accumulator is zero-filled (like split-k), not randomized.
// CHECK: memref.alloc
// CHECK-NEXT: affine.for
// CHECK-NEXT: memref.store %[[zero]]
