// RUN: rocmlir-opt -rock-analyze-memory-use %s | FileCheck %s

// Large tensor (>2GB): tt.pointer_range should NOT be set
// CHECK-LABEL: @large_tensor
// CHECK-SAME: llvm.dereferenceable = 8589934592 : i64
// CHECK-NOT: tt.pointer_range
func.func @large_tensor(%arg0: tensor<2147483648xf32>) attributes {rock.kernel} {
  return
}

// Small tensor (<2GB): tt.pointer_range = 32 should be set
// CHECK-LABEL: @small_tensor
// CHECK-SAME: tt.pointer_range = 32 : i32
func.func @small_tensor(%arg0: tensor<16xf32>) attributes {rock.kernel} {
  return
}

// Boundary case: exactly 2GB (2147483648 bytes = 536870912 x f32)
// This is NOT < 2GB, so tt.pointer_range should NOT be set
// CHECK-LABEL: @boundary_2gb
// CHECK-NOT: tt.pointer_range
func.func @boundary_2gb(%arg0: tensor<536870912xf32>) attributes {rock.kernel} {
  return
}

// Just under 2GB: should get tt.pointer_range
// CHECK-LABEL: @just_under_2gb
// CHECK-SAME: tt.pointer_range = 32 : i32
func.func @just_under_2gb(%arg0: tensor<536870911xf32>) attributes {rock.kernel} {
  return
}
