// RUN: rocmlir-opt -resolve-kernel-launch-params --split-input-file %s | FileCheck %s

// Verifies that @global_smem is converted from external [0 x i8] (dynamic LDS)
// to internal [N x i8] (static LDS) and ttg.shared is removed.
// CHECK-LABEL: module
// CHECK-NOT: ttg.shared
// CHECK: llvm.mlir.global internal @global_smem(#llvm.undef) {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<4096 x i8>
// CHECK: llvm.func @my_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
// CHECK-NOT: ptr<1>
module attributes {
    "ttg.shared" = 4096 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @my_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that unused trailing ptr<1> workspace args are removed from the
// kernel signature.
// CHECK-LABEL: module
// CHECK: llvm.func @kernel_with_ws(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
// CHECK-NOT: ptr<1>
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @kernel_with_ws(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>)
      attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that exactly 2 trailing ptr<1> workspace args are removed.
// The third ptr<1> from the end is kept because it is a real data argument.
// CHECK-LABEL: module
// CHECK: llvm.func @cap_at_two(%arg0: !llvm.ptr, %arg1: !llvm.ptr<1>)
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @cap_at_two(%arg0: !llvm.ptr, %extra: !llvm.ptr<1>, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>)
      attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that ttg.shared = 0 leaves @global_smem unchanged (no static alloc
// needed) but ttg.shared is still removed.
// CHECK-LABEL: module
// CHECK-NOT: ttg.shared
// CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// CHECK: llvm.func @zero_shared(%arg0: !llvm.ptr)
// CHECK-NOT: ptr<1>
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @zero_shared(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
    llvm.return
  }
}

