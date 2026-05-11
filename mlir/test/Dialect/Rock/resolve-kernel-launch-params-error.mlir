// RUN: rocmlir-opt -resolve-kernel-launch-params --split-input-file %s -verify-diagnostics
// RUN: rocmlir-opt -resolve-kernel-launch-params --split-input-file %s -verify-diagnostics --mlir-print-ir-after-failure 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NA --implicit-check-not=rock.not_applicable

// Verifies that a missing ttg.shared triggers an error.
// expected-error @+1 {{ttg.shared attribute not found on module}}
module {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @my_kernel(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a missing @global_smem triggers an error.
// expected-error @+1 {{@global_smem not found in module}}
module attributes {
    "ttg.shared" = 4096 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.func @my_kernel(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
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

  llvm.func @my_kernel(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that ttg.shared exceeding the hardware LDS limit triggers an error
// AND that the pass marks the module with `rock.not_applicable` so that the
// tuning driver can classify the failure as "config doesn't fit" rather than
// a real compilation bug. gfx90a has 65536 bytes of LDS.
// expected-error @+2 {{ttg.shared (65537) exceeds LDS limit (65536) for amdgcn-amd-amdhsa:gfx90a}}
// NA: module attributes {rock.not_applicable, {{.*}}ttg.shared = 65537
module attributes {
    "ttg.shared" = 65537 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @my_kernel(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a missing rock.arch on the kernel triggers an error.
// expected-error @+1 {{rock.arch not found on kernel function or module}}
module attributes {
    "ttg.shared" = 4096 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @my_kernel(%arg0: !llvm.ptr) attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a kernel with fewer than 2 args triggers an error.
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // expected-error @+1 {{kernel 'tiny_kernel' has fewer than 2 arguments}}
  llvm.func @tiny_kernel(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that trailing args that aren't ptr<1> trigger an error.
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // expected-error @+1 {{expected trailing workspace arg %arg0 to be ptr<1>, got '!llvm.ptr'}}
  llvm.func @bad_trailing(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that used workspace args trigger an error.
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // expected-error @+1 {{workspace argument %arg1 has unexpected uses}}
  llvm.func @used_ws(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>)
      attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
    %0 = llvm.load %gs : !llvm.ptr<1> -> i64
    llvm.return
  }
}
