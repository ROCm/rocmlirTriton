// RUN: rocmlir-opt -bake-kernel-launch-params --split-input-file %s -verify-diagnostics

// Verifies that a missing @global_smem triggers an error.
// expected-error @+1 {{@global_smem not found in module}}
module attributes {
    "ttg.shared" = 4096 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.func @my_kernel(%arg0: !llvm.ptr) attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a negative ttg.shared triggers an error.
// expected-error @+1 {{ttg.shared is negative (-1)}}
module attributes {
    "ttg.shared" = -1 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @my_kernel(%arg0: !llvm.ptr) attributes {rock.kernel} {
    llvm.return
  }
}
