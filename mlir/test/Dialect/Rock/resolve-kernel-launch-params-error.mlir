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

// Verifies that ttg.shared within the architecture's LDS size but over a
// caller-supplied rock.max_lds is still rejected and marked not applicable,
// and that the diagnostic names the caller's ceiling as the binding constraint
// so a caller knows to raise its budget rather than to blame the hardware.
// gfx90a has 65536 bytes of LDS, so only the budget rejects this kernel.
// expected-error @+2 {{ttg.shared (32768) exceeds LDS limit (16384) for amdgcn-amd-amdhsa:gfx90a, capped by rock.max_lds = 16384 (architecture allows 65536)}}
// NA: module attributes {rock.not_applicable, {{.*}}ttg.shared = 32768
module attributes {
    "ttg.shared" = 32768 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @over_caller_budget(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel, rock.max_lds = 16384 : i64} {
    llvm.return
  }
}

// -----

// Verifies that when the architecture is the binding constraint the diagnostic
// does not blame the caller's (looser) ceiling.
// expected-error @+2 {{ttg.shared (65537) exceeds LDS limit (65536) for amdgcn-amd-amdhsa:gfx90a}}
// NA: module attributes {rock.not_applicable, {{.*}}ttg.shared = 65537
module attributes {
    "ttg.shared" = 65537 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @arch_binds_not_budget(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel, rock.max_lds = 1048576 : i64} {
    llvm.return
  }
}

// -----

// Verifies that a module sized for the loosest of several kernels is rejected:
// @global_smem is sized once for the whole module, so the tightest budget
// across the kernels has to bind.
// expected-error @+2 {{ttg.shared (8192) exceeds LDS limit (4096) for amdgcn-amd-amdhsa:gfx90a, capped by rock.max_lds = 4096 (architecture allows 65536)}}
// NA: module attributes {rock.not_applicable, {{.*}}ttg.shared = 8192
module attributes {
    "ttg.shared" = 8192 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @roomy(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel, rock.max_lds = 16384 : i64} {
    llvm.return
  }

  llvm.func @tight(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel, rock.max_lds = 4096 : i64} {
    llvm.return
  }
}

// -----

// Verifies that a non-positive rock.max_lds is a caller error rather than
// being silently read as "no ceiling", which would let an over-budget kernel
// through.
module attributes {
    "ttg.shared" = 4096 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // expected-error @+1 {{rock.max_lds must be greater than zero, got 0}}
  llvm.func @zero_budget(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel, rock.max_lds = 0 : i64} {
    llvm.return
  }
}

// -----

// Verifies that a grid larger than the runtime's uint32 grid-size limit is
// rejected and marked not applicable.
// NA: module attributes {rock.grid_size.oversized_grid_dimension = 4294967296 : i64, rock.not_applicable
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 1 : i32,
    "ttg.num-ctas" = 1 : i32,
    "rock.grid_size.oversized_grid_dimension" = 4294967296 : i64
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // expected-error @+1 {{grid size 4294967296 exceeds the runtime grid size limit of 4294967295}}
  llvm.func @oversized_grid_dimension(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a launch whose work-item count does not fit in the AMDGPU
// dispatch packet is rejected and marked not applicable. The grid contains
// 33554432 workgroups of 2 * 64 = 128 threads, for 2^32 work-items.
// NA: module attributes {rock.grid_size.oversized_grid = 33554432 : i32, rock.not_applicable
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 2 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "rock.grid_size.oversized_grid" = 33554432 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // expected-error @+1 {{launch dimensions (grid size 33554432, block size 128, cluster size 1) exceed the runtime limit of 4294967295 work-items}}
  llvm.func @oversized_grid(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that the cluster size is included in the dispatch-packet limit
// check. The grid and block product is 2^31 work-items, but multiplying by the
// cluster size produces 2^32 work-items.
// NA: module attributes {rock.grid_size.oversized_clustered_grid = 33554432 : i32, rock.not_applicable
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 2 : i32,
    "rock.grid_size.oversized_clustered_grid" = 33554432 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // expected-error @+1 {{launch dimensions (grid size 33554432, block size 64, cluster size 2) exceed the runtime limit of 4294967295 work-items}}
  llvm.func @oversized_clustered_grid(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a launch whose block size exceeds the hardware workgroup limit
// is rejected and marked not applicable.
// NA: module attributes {rock.grid_size.oversized_workgroup = 1 : i32, rock.not_applicable
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 17 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "rock.grid_size.oversized_workgroup" = 1 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // expected-error @+1 {{block size 1088 exceeds the AMDGPU workgroup size limit of 1024}}
  llvm.func @oversized_workgroup(%arg0: !llvm.ptr) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that a module carrying grid metadata but missing the Triton launch
// metadata that the grid check needs (here ttg.num-ctas) reports that kernel
// metadata collection failed instead of failing the pass without a diagnostic.
// The module is not marked not applicable: missing metadata is a pipeline bug,
// not a configuration a tuning run should silently skip.
// expected-error @+1 {{could not validate kernel launch dimensions because kernel metadata collection failed}}
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 2 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.missing_num_ctas" = 4 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @missing_num_ctas(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that collectKernelInfo's missing-grid diagnostic is accompanied by
// context about why kernel metadata was being collected.
// expected-error @+1 {{could not validate kernel launch dimensions because kernel metadata collection failed}}
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 2 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "rock.grid_size.with_grid" = 4 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @with_grid(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
    llvm.return
  }

  // expected-error @+1 {{'llvm.func' op missing rock.grid_size.without_grid module attribute}}
  llvm.func @without_grid(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies that collectKernelInfo's malformed-prefill diagnostic is
// accompanied by context about why kernel metadata was being collected.
// expected-error @+1 {{could not validate kernel launch dimensions because kernel metadata collection failed}}
module attributes {
    "ttg.shared" = 0 : i32,
    "ttg.num-warps" = 2 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "rock.grid_size.bad_prefill" = 4 : i32,
    "rock.prefill_args.bad_prefill" = [{value = 0 : i32}]
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // expected-error @+1 {{'llvm.func' op malformed rock.prefill_args.bad_prefill: entry missing 'index' IntegerAttr}}
  llvm.func @bad_prefill(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
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
