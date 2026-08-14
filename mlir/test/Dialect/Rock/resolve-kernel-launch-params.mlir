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

// -----

// Verifies that ttg.shared equal to the per-arch LDS limit compiles successfully
// for gfx942 (CDNA3) has 64 KiB = 65536 B.
// CHECK-LABEL: module
// CHECK-NOT: ttg.shared
// CHECK-NOT: rock.not_applicable
// CHECK: llvm.mlir.global internal @global_smem(#llvm.undef) {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<65536 x i8>
// CHECK: llvm.func @max_lds_gfx942(%arg0: !llvm.ptr)
// CHECK-NOT: ptr<1>
module attributes {
    "ttg.shared" = 65536 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @max_lds_gfx942(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that ttg.shared equal to the per-arch LDS limit compiles successfully
// for gfx950 (CDNA4), which has 160 KiB = 163840 B.
// CHECK-LABEL: module
// CHECK-NOT: ttg.shared
// CHECK-NOT: rock.not_applicable
// CHECK: llvm.mlir.global internal @global_smem(#llvm.undef) {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<163840 x i8>
// CHECK: llvm.func @max_lds_gfx950(%arg0: !llvm.ptr)
// CHECK-NOT: ptr<1>
module attributes {
    "ttg.shared" = 163840 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @max_lds_gfx950(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that ttg.shared equal to the per-arch LDS limit compiles successfully
// for gfx1250, which has 320 KiB = 327680 B.
// CHECK-LABEL: module
// CHECK-NOT: ttg.shared
// CHECK-NOT: rock.not_applicable
// CHECK: llvm.mlir.global internal @global_smem(#llvm.undef) {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<327680 x i8>
// CHECK: llvm.func @max_lds_gfx1250(%arg0: !llvm.ptr)
// CHECK-NOT: ptr<1>
module attributes {
    "ttg.shared" = 327680 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @max_lds_gfx1250(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1250", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that the largest grid representable by the dispatch packet remains
// applicable: 4294967295 workgroups * 1 warp * 1 thread = UINT32_MAX
// work-items.
// CHECK-LABEL: module
// CHECK-NOT: rock.not_applicable
// CHECK: llvm.func @max_grid_work_items(%arg0: !llvm.ptr)
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 1 : i32,
    "ttg.num-ctas" = 1 : i32,
    "rock.grid_size.max_grid_work_items" = 4294967295 : i64
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @max_grid_work_items(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
    llvm.return
  }
}
