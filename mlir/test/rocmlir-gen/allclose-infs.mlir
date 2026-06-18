// End-to-end test of the `mcpuVerifyFloatAllclose` runtime's inf handling.
//
// Two scenarios in a single test:
//   (a) Inf-mismatch: GPU produces +inf while CPU produces a finite value.
//       The verifier must flag this as an unconditional failure —
//       no tolerance (atol, rtol) can mask an inf-mismatch.
//   (b) Matching inf: both GPU and CPU produce +inf.
//       The verifier must treat this as an exact match (pass).
//
// Test strategy: hand-build two `memref<4xf32>`s:
//   index 0: exact match    (1.0  vs 1.0)
//   index 1: inf-mismatch   (+inf vs 2.0)  — must fail
//   index 2: matching inf   (+inf vs +inf) — must pass
//   index 3: exact match    (3.0  vs 3.0)
// Call `mcpuVerifyFloatAllclose` directly. No kernel is involved.

// RUN: rocmlir-driver -c -arch %arch %s \
// RUN:   | mlir-runner --shared-libs=%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void \
// RUN:   | FileCheck %s

func.func private @mcpuVerifyFloatAllclose(memref<?xf32>, memref<?xf32>, f32, f32, i8, i1)

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  // f32 +infinity bit pattern (sign=0, exponent=all 1s, mantissa=0).
  %inf = arith.constant 0x7F800000 : f32

  // GPU side: [1, +inf, +inf, 3].
  %gpu = memref.alloc() : memref<4xf32>
  memref.store %f1,  %gpu[%c0] : memref<4xf32>
  memref.store %inf, %gpu[%c1] : memref<4xf32>
  memref.store %inf, %gpu[%c2] : memref<4xf32>
  memref.store %f3,  %gpu[%c3] : memref<4xf32>

  // CPU/validation side: [1, 2, +inf, 3].
  %val = memref.alloc() : memref<4xf32>
  memref.store %f1,  %val[%c0] : memref<4xf32>
  memref.store %f2,  %val[%c1] : memref<4xf32>
  memref.store %inf, %val[%c2] : memref<4xf32>
  memref.store %f3,  %val[%c3] : memref<4xf32>

  %gpuD = memref.cast %gpu : memref<4xf32> to memref<?xf32>
  %valD = memref.cast %val : memref<4xf32> to memref<?xf32>

  %atol = arith.constant 1.0e-2 : f32
  %rtol = arith.constant 1.0e-2 : f32
  // printDebug = 1 (Summary) -> print stats only on failure.
  %dbg = arith.constant 1 : i8
  %fp32 = arith.constant true

  func.call @mcpuVerifyFloatAllclose(%gpuD, %valD, %atol, %rtol, %dbg, %fp32)
    : (memref<?xf32>, memref<?xf32>, f32, f32, i8, i1) -> ()

  memref.dealloc %gpu : memref<4xf32>
  memref.dealloc %val : memref<4xf32>
  return
}

// 1 failing element out of 4 (the inf-mismatch at index 1).
// Index 2 (+inf vs +inf) must NOT count as a failure.
// CHECK:     allclose(atol={{.*}}, rtol={{.*}}): 1/4 failing element(s)
// CHECK:     allclose statistics:
// CHECK:       1 inf-mismatch(es); no tolerance can fix these
// CHECK:     histogram of absDiff/tolerance:
// CHECK:     [0 0 0]
