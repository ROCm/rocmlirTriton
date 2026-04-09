// RUN: rocmlir-opt -rock-restore-host-code="arch=gfx90a" --split-input-file %s | FileCheck %s

// Verifies gpu.binary is created with block_size and grid_size metadata
// CHECK: gpu.container_module
// CHECK: gpu.binary @rock_kernels
// CHECK-SAME: #gpu.kernel_metadata<"test_basic_kernel"
// CHECK-SAME: block_size = 256 : i64
// CHECK-SAME: grid_size = 4 : i64
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.test_basic_kernel" = 4 : i32,
    "triton.hsaco" = "DUMMY_HSACO"
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_basic_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies single prefill arg is embedded in gpu.binary kernel metadata
// CHECK: gpu.binary @rock_kernels
// CHECK-SAME: #gpu.kernel_metadata<"test_prefill_kernel"
// CHECK-SAME: block_size = 256 : i64
// CHECK-SAME: grid_size = 2 : i64
// CHECK-SAME: rock.prefill = [{index = 2 : i64, value = 0.000000e+00 : f32}]
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.test_prefill_kernel" = 2 : i32,
    "rock.prefill_args.test_prefill_kernel" = [{index = 2 : i64, value = 0.000000e+00 : f32}],
    "triton.hsaco" = "DUMMY_HSACO"
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_prefill_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies multiple prefill args of the same type
// CHECK: gpu.binary @rock_kernels
// CHECK-SAME: #gpu.kernel_metadata<"test_multi_prefill_same"
// CHECK-SAME: rock.prefill = [{index = 2 : i64, value = 0.000000e+00 : f32}, {index = 3 : i64, value = 0.000000e+00 : f32}]
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.test_multi_prefill_same" = 2 : i32,
    "rock.prefill_args.test_multi_prefill_same" = [
        {index = 2 : i64, value = 0.000000e+00 : f32},
        {index = 3 : i64, value = 0.000000e+00 : f32}
    ],
    "triton.hsaco" = "DUMMY_HSACO"
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_multi_prefill_same(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies multiple prefill args with different data types (f32 and f16)
// CHECK: gpu.binary @rock_kernels
// CHECK-SAME: #gpu.kernel_metadata<"test_multi_prefill_mixed"
// CHECK-SAME: rock.prefill = [{index = 2 : i64, value = 0.000000e+00 : f32}, {index = 3 : i64, value = 0.000000e+00 : f16}]
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.test_multi_prefill_mixed" = 2 : i32,
    "rock.prefill_args.test_multi_prefill_mixed" = [
        {index = 2 : i64, value = 0.000000e+00 : f32},
        {index = 3 : i64, value = 0.000000e+00 : f16}
    ],
    "triton.hsaco" = "DUMMY_HSACO"
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_multi_prefill_mixed(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies host functions are restored and kernel calls converted to gpu.launch_func
// CHECK: gpu.container_module
// CHECK: gpu.binary @rock_kernels
// CHECK: func.func @host_wrapper
// CHECK: gpu.launch_func @rock_kernels::@test_host_restore
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.test_host_restore" = 8 : i32,
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @host_wrapper(%arg0: tensor<4096xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<4096xf32>) -> tensor<4096xf32> {\n  %0 = func.call @test_host_restore(%arg0, %arg1, %arg2) : (tensor<4096xf32>, tensor<4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>\n  return %0 : tensor<4096xf32>\n}"
    ]
} {
  llvm.mlir.global internal @global_smem(#llvm.undef) {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<16384 x i8>
  llvm.func @test_host_restore(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies no dynamic_shared_memory_size is emitted — LDS is statically
// baked into the binary by ResolveKernelLaunchParams before RestoreHostCode runs.
// CHECK: gpu.binary @rock_kernels
// CHECK: func.func @host_with_lds
// CHECK: gpu.launch_func @rock_kernels::@test_lds_kernel
// CHECK-NOT: dynamic_shared_memory_size
// CHECK: return
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.test_lds_kernel" = 4 : i32,
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @host_with_lds(%arg0: tensor<4096xf32>, %arg1: tensor<4096xf32>) -> tensor<4096xf32> {\n  %0 = func.call @test_lds_kernel(%arg0, %arg1) : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>\n  return %0 : tensor<4096xf32>\n}"
    ]
} {
  llvm.mlir.global internal @global_smem(#llvm.undef) {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<16384 x i8>
  llvm.func @test_lds_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies ttg.total-num-warps takes precedence over ttg.num-warps for block size
// block_size = ttg.total-num-warps * ttg.threads-per-warp = 8 * 64 = 512
// CHECK: gpu.binary @rock_kernels
// CHECK-SAME: #gpu.kernel_metadata<"test_total_num_warps"
// CHECK-SAME: block_size = 512 : i64
// CHECK-SAME: grid_size = 4 : i64
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.total-num-warps" = 8 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.test_total_num_warps" = 4 : i32,
    "triton.hsaco" = "DUMMY_HSACO"
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_total_num_warps(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies multiple kernels in a single module both get metadata in gpu.binary
// CHECK: gpu.binary @rock_kernels
// CHECK-SAME: #gpu.kernel_metadata<"kernel_a"
// CHECK-SAME: block_size = 256 : i64
// CHECK-SAME: grid_size = 2 : i64
// CHECK-SAME: #gpu.kernel_metadata<"kernel_b"
// CHECK-SAME: block_size = 256 : i64
// CHECK-SAME: grid_size = 8 : i64
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.kernel_a" = 2 : i32,
    "rock.grid_size.kernel_b" = 8 : i32,
    "triton.hsaco" = "DUMMY_HSACO"
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @kernel_a(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
  llvm.func @kernel_b(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies kernel LLVM function is preserved when no host functions are present
// and that the arch from the pass option appears in the gpu.binary target
// CHECK: gpu.binary @rock_kernels
// CHECK-SAME: gfx90a
// CHECK: llvm.func @test_kernel_preserved
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.test_kernel_preserved" = 4 : i32,
    "triton.hsaco" = "DUMMY_HSACO"
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_kernel_preserved(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies kernel LLVM function is removed when host functions are restored,
// and rock.host_functions attribute is cleaned up
// CHECK-NOT: rock.host_functions
// CHECK: gpu.binary @rock_kernels
// CHECK: func.func @host_with_removal
// CHECK: gpu.launch_func @rock_kernels::@test_kernel_removed
// CHECK-NOT: llvm.func @test_kernel_removed
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.test_kernel_removed" = 4 : i32,
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @host_with_removal(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {\n  %0 = func.call @test_kernel_removed(%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>\n  return %0 : tensor<1024xf32>\n}"
    ]
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_kernel_removed(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies no dynamic_shared_memory_size with zero LDS
// CHECK: func.func @host_no_lds
// CHECK: gpu.launch_func @rock_kernels::@test_no_lds_kernel
// CHECK-NOT: dynamic_shared_memory_size
// CHECK: return
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.test_no_lds_kernel" = 4 : i32,
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @host_no_lds(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {\n  %0 = func.call @test_no_lds_kernel(%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>\n  return %0 : tensor<1024xf32>\n}"
    ]
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_no_lds_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies launch args match kernel signature 1:1 after ResolveKernelLaunchParams
// has stripped unused workspace arguments. No null pointer padding needed.
// CHECK: func.func @host_workspace
// CHECK-NOT: llvm.mlir.zero : !llvm.ptr
// CHECK: gpu.launch_func @rock_kernels::@test_workspace_kernel
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.test_workspace_kernel" = 4 : i32,
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @host_workspace(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024xf32>) -> tensor<1024xf32> {\n  %0 = func.call @test_workspace_kernel(%arg0, %arg1, %arg2) : (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>\n  return %0 : tensor<1024xf32>\n}"
    ]
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_workspace_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies void kernel call (no return value) is converted correctly
// CHECK: func.func @host_void
// CHECK: gpu.launch_func @rock_kernels::@test_void_kernel
// CHECK-NOT: llvm.func @test_void_kernel
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.test_void_kernel" = 4 : i32,
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @host_void(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) {\n  func.call @test_void_kernel(%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> ()\n  return\n}"
    ]
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @test_void_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies multiple host functions calling different kernels are all restored
// CHECK: gpu.binary @rock_kernels
// CHECK-SAME: #gpu.kernel_metadata<"kernel_x"
// CHECK-SAME: #gpu.kernel_metadata<"kernel_y"
// CHECK: func.func @host_x
// CHECK: gpu.launch_func @rock_kernels::@kernel_x
// CHECK: func.func @host_y
// CHECK: gpu.launch_func @rock_kernels::@kernel_y
// CHECK-NOT: llvm.func @kernel_x
// CHECK-NOT: llvm.func @kernel_y
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "rock.grid_size.kernel_x" = 4 : i32,
    "rock.grid_size.kernel_y" = 8 : i32,
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @host_x(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {\n  %0 = func.call @kernel_x(%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>\n  return %0 : tensor<1024xf32>\n}",
        "func.func @host_y(%arg0: tensor<512xf16>, %arg1: tensor<512xf16>, %arg2: tensor<512xf16>) -> tensor<512xf16> {\n  %0 = func.call @kernel_y(%arg0, %arg1, %arg2) : (tensor<512xf16>, tensor<512xf16>, tensor<512xf16>) -> tensor<512xf16>\n  return %0 : tensor<512xf16>\n}"
    ]
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @kernel_x(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
  llvm.func @kernel_y(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}
