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
    "ttg.shared" = 0 : i32,
    "rock.grid_size.test_basic_kernel" = 4 : i32,
    "triton.hsaco" = "DUMMY_HSACO"
} {
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
    "ttg.shared" = 0 : i32,
    "rock.grid_size.test_prefill_kernel" = 2 : i32,
    "rock.prefill_args.test_prefill_kernel" = [{index = 2 : i64, value = 0.000000e+00 : f32}],
    "triton.hsaco" = "DUMMY_HSACO"
} {
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
    "ttg.shared" = 0 : i32,
    "rock.grid_size.test_multi_prefill_same" = 2 : i32,
    "rock.prefill_args.test_multi_prefill_same" = [
        {index = 2 : i64, value = 0.000000e+00 : f32},
        {index = 3 : i64, value = 0.000000e+00 : f32}
    ],
    "triton.hsaco" = "DUMMY_HSACO"
} {
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
    "ttg.shared" = 0 : i32,
    "rock.grid_size.test_multi_prefill_mixed" = 2 : i32,
    "rock.prefill_args.test_multi_prefill_mixed" = [
        {index = 2 : i64, value = 0.000000e+00 : f32},
        {index = 3 : i64, value = 0.000000e+00 : f16}
    ],
    "triton.hsaco" = "DUMMY_HSACO"
} {
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
    "ttg.shared" = 16384 : i32,
    "rock.grid_size.test_host_restore" = 8 : i32,
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @host_wrapper(%arg0: tensor<4096xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<4096xf32>) -> tensor<4096xf32> {\n  %0 = func.call @test_host_restore(%arg0, %arg1, %arg2) : (tensor<4096xf32>, tensor<4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>\n  return %0 : tensor<4096xf32>\n}"
    ]
} {
  llvm.func @test_host_restore(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Verifies shared memory size is passed as dynamic_shared_memory_size
// CHECK: gpu.binary @rock_kernels
// CHECK: func.func @host_with_lds
// CHECK: gpu.launch_func @rock_kernels::@test_lds_kernel
// CHECK-SAME: dynamic_shared_memory_size
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.shared" = 16384 : i32,
    "rock.grid_size.test_lds_kernel" = 4 : i32,
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @host_with_lds(%arg0: tensor<4096xf32>, %arg1: tensor<4096xf32>) -> tensor<4096xf32> {\n  %0 = func.call @test_lds_kernel(%arg0, %arg1) : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>\n  return %0 : tensor<4096xf32>\n}"
    ]
} {
  llvm.func @test_lds_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
      attributes {rock.kernel} {
    llvm.return
  }
}
