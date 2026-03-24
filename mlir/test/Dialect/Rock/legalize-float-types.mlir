// Unit tests for rock-legalize-float-types pass.
// Verifies that non-TT_Float types used in scaled GEMM kernels and wrappers
// are legalized to integer types.

// RUN: rocmlir-opt -rock-legalize-float-types -mlir-print-local-scope --split-input-file %s | FileCheck %s

// Test 1: f8E8M0FNU scales -> i8 (data types are TT_Float, no packing needed)

// CHECK-LABEL: func.func @test_f8_scale_to_i8
// CHECK-SAME: (%{{.*}}: tensor<64x64xf8E4M3FN>, %{{.*}}: tensor<64x64xf8E4M3FN>,
// CHECK-SAME:  %[[SA:.*]]: tensor<64x2xi8>, %[[SB:.*]]: tensor<64x2xi8>,
// CHECK-SAME:  %{{.*}}: tensor<64x64xf32>)
//      CHECK: rock.blockwise_gemm
// CHECK-SAME: scaled by %[[SA]]
// CHECK-SAME: scaled by %[[SB]]
// CHECK-SAME: tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>
// CHECK-SAME: tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>
// matrixA/B types (f8E4M3FN) are TT_Float, so no origElemType attr is set.
// CHECK-NOT: matrixAOrigElemType
// CHECK-NOT: matrixBOrigElemType
func.func @test_f8_scale_to_i8(
    %a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>,
    %scaleA: tensor<64x2xf8E8M0FNU>, %scaleB: tensor<64x2xf8E8M0FNU>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %result = rock.blockwise_gemm(%a scaled by %scaleA, %b scaled by %scaleB, %c)
    {quantBlockSize = 32 : i64}
    : tensor<64x64xf8E4M3FN> scaled by tensor<64x2xf8E8M0FNU>,
      tensor<64x64xf8E4M3FN> scaled by tensor<64x2xf8E8M0FNU>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test 2: Non-kernel functions (no rock.kernel) are unchanged

// CHECK-LABEL: func.func @test_non_kernel_unchanged
// CHECK-SAME: tensor<64x2xf8E8M0FNU>
func.func @test_non_kernel_unchanged(
    %a: tensor<64x64xf8E4M3FN>,
    %scaleA: tensor<64x2xf8E8M0FNU>) -> tensor<64x2xf8E8M0FNU> {
  return %scaleA : tensor<64x2xf8E8M0FNU>
}

// -----

// Test 3: Wrapper converts f8E8M0FNU memref via unrealized_conversion_cast

func.func @kernel_f8(%arg0: tensor<256xf8E8M0FNU>) -> tensor<256xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %cst = arith.constant dense<0.0> : tensor<256xf32>
  return %cst : tensor<256xf32>
}

// CHECK-LABEL: func.func @wrapper_f8_scale
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : memref<256xf8E8M0FNU> to memref<256xi8>
// CHECK: bufferization.to_tensor %[[CAST]]
func.func @wrapper_f8_scale(%mem: memref<256xf8E8M0FNU>) {
  %t = bufferization.to_tensor %mem : memref<256xf8E8M0FNU> to tensor<256xf8E8M0FNU>
  %r = func.call @kernel_f8(%t) : (tensor<256xf8E8M0FNU>) -> tensor<256xf32>
  return
}
