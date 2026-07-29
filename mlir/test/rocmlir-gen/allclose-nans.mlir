// End-to-end test of the `mcpuVerifyFloatAllclose` runtime entry point's NaN
// handling. 
// The diagnostic output for a NaN-mismatch must 
// (a) flag failure in the `[%d %d %d]` triple
// (b) surface "NaN-mismatch" in the worst-element line
// (c) skip the calibration hints (no atol/rtol can mask a NaN)
// (d) report a `ratio == nan` row in the histogram.
//
// Test strategy: hand-build two `memref<3xf32>`s, insert a NaN at index 1
// on the validation side, and call `mcpuVerifyFloatAllclose` directly. 
// No kernel is involved, so the kernel pipeline is a no-op.

// RUN: rocmlir-driver -c -arch %arch %s \
// RUN:   | mlir-runner --shared-libs=%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext --entry-point-result=void \
// RUN:   | FileCheck %s

// `mcpuVerifyFloatAllclose` C signature:
//   (gpuMemref, valMemref, atol: f32, rtol: f32, printDebug: i8, isFP32: i1)
func.func private @mcpuVerifyFloatAllclose(memref<?xf32>, memref<?xf32>, f32, f32, i8, i1)

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  // f32 quiet NaN bit pattern (sign=0, exponent=all 1s, mantissa=MSB set).
  %nan = arith.constant 0x7FC00000 : f32

  // GPU side: [1, 2, 3]. CPU/validation side: [1, NaN, 3]. Index 1 is the
  // NaN-mismatch under test; indices 0 and 2 are exact matches and must
  // not contribute to the failing-elements count.
  %gpu = memref.alloc() : memref<3xf32>
  memref.store %f1, %gpu[%c0] : memref<3xf32>
  memref.store %f2, %gpu[%c1] : memref<3xf32>
  memref.store %f3, %gpu[%c2] : memref<3xf32>

  %val = memref.alloc() : memref<3xf32>
  memref.store %f1,  %val[%c0] : memref<3xf32>
  memref.store %nan, %val[%c1] : memref<3xf32>
  memref.store %f3,  %val[%c2] : memref<3xf32>

  // The verifier signature takes `memref<?xf32>`; cast both buffers.
  %gpuD = memref.cast %gpu : memref<3xf32> to memref<?xf32>
  %valD = memref.cast %val : memref<3xf32> to memref<?xf32>

  // Generous tolerances so a misbehaving verifier could *not* claim the NaN
  // happens to pass: any sane pair of finite tolerances must still flag the
  // NaN as a failure. atol=rtol=1e-2 ~= PyTorch fp16 defaults.
  %atol = arith.constant 1.0e-2 : f32
  %rtol = arith.constant 1.0e-2 : f32
  // printDebug = 1 (Summary) -> print stats only on failure.
  %dbg = arith.constant 1 : i8
  // isFP32 = true -> match the runtime's float specialisation.
  %fp32 = arith.constant true

  func.call @mcpuVerifyFloatAllclose(%gpuD, %valD, %atol, %rtol, %dbg, %fp32)
    : (memref<?xf32>, memref<?xf32>, f32, f32, i8, i1) -> ()

  memref.dealloc %gpu : memref<3xf32>
  memref.dealloc %val : memref<3xf32>
  return
}

// Top-level summary: 1 failing element out of 3.
// The NaN must drive the worst-element line, not be overwritten by an
// inf-ratio or finite-ratio element.
// Calibration hints must NOT appear when a NaN is present.

// CHECK:     allclose(atol={{.*}}, rtol={{.*}}): 1/3 failing element(s)
// CHECK:     allclose statistics:
// CHECK:     worst element: valNum=nan gpuNum=2 (NaN-mismatch)
// CHECK-NOT: to pass with current rtol={{.*}}: atol >=
// CHECK-NOT: to pass with current atol={{.*}}: rtol >=
// CHECK:     no tolerance can mask a NaN-mismatch; fix the kernel
// CHECK:     histogram of absDiff/tolerance:
// CHECK:     0 < ratio <=   0.1 : 0/3 (0.0000%)
// CHECK:     ratio == nan   : 1/3 ({{.*}}%)  <-- failing
// CHECK:     note: 1 element(s) had NaN on either side
// CHECK:     [0 0 0]
