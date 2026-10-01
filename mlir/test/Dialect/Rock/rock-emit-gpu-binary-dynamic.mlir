// RUN: rocmlir-opt -rock-emit-gpu-binary="arch=gfx90a" --split-input-file %s | FileCheck %s

// A dynamic kernel: the grid size is an expression over argument dimensions
// and the trailing i32 arguments are dimensions of the buffer arguments.
// The binary's metadata records both in builtin form, the launch reads each
// dimension from its memref and computes the grid from them, and the rock
// module attributes are removed.
// CHECK-NOT: rock.dim_args.dyn_gemm
// CHECK-NOT: rock.grid_size.dyn_gemm
// CHECK: gpu.binary @rock_kernels
// CHECK-SAME: #gpu.kernel_metadata<"dyn_gemm", (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
// CHECK-SAME: rock.block_size = 256 : i64
// CHECK-SAME: rock.dim_args = [array<i32: 0, 1>, array<i32: 1, 2>, array<i32: 2, 1>, array<i32: 2, 2>]
// CHECK-SAME: rock.grid_size = "#rock.arg_expr<(s0 ceildiv 128) * (s1 ceildiv 64), [arg(0, 1), arg(1, 2)]>"
// CHECK-LABEL: func.func @host_dyn
// CHECK-SAME: (%[[A:.*]]: tensor<1x?x64xf16>, %[[B:.*]]: tensor<1x64x?xf16>, %[[C:.*]]: tensor<1x?x?xf16>)
// CHECK-DAG: %[[AM:.*]] = bufferization.to_buffer %[[A]] : tensor<1x?x64xf16> to memref<1x?x64xf16>
// CHECK-DAG: %[[BM:.*]] = bufferization.to_buffer %[[B]] : tensor<1x64x?xf16> to memref<1x64x?xf16>
// CHECK-DAG: %[[CM:.*]] = bufferization.to_buffer %[[C]] : tensor<1x?x?xf16> to memref<1x?x?xf16>
// CHECK: %[[D0:.*]] = memref.dim %[[AM]], %{{.*}} : memref<1x?x64xf16>
// CHECK: %[[M:.*]] = arith.index_cast %[[D0]] : index to i32
// CHECK: %[[D1:.*]] = memref.dim %[[BM]], %{{.*}} : memref<1x64x?xf16>
// CHECK: %[[N:.*]] = arith.index_cast %[[D1]] : index to i32
// CHECK: %[[D2:.*]] = memref.dim %[[CM]], %{{.*}} : memref<1x?x?xf16>
// CHECK: %[[MC:.*]] = arith.index_cast %[[D2]] : index to i32
// CHECK: %[[D3:.*]] = memref.dim %[[CM]], %{{.*}} : memref<1x?x?xf16>
// CHECK: %[[NC:.*]] = arith.index_cast %[[D3]] : index to i32
// CHECK: %[[GM:.*]] = memref.dim %[[AM]], %{{.*}} : memref<1x?x64xf16>
// CHECK: %[[GMP:.*]] = arith.addi %[[GM]], %{{.*}} : index
// CHECK: %[[MBLOCKS:.*]] = arith.divui %[[GMP]], %{{.*}} : index
// CHECK: %[[GN:.*]] = memref.dim %[[BM]], %{{.*}} : memref<1x64x?xf16>
// CHECK: %[[GNP:.*]] = arith.addi %[[GN]], %{{.*}} : index
// CHECK: %[[NBLOCKS:.*]] = arith.divui %[[GNP]], %{{.*}} : index
// CHECK: %[[BLOCKS:.*]] = arith.muli %[[MBLOCKS]], %[[NBLOCKS]] : index
// CHECK: %[[GRID:.*]] = arith.muli %[[BLOCKS]], %{{.*}} : index
// CHECK: gpu.launch_func @rock_kernels::@dyn_gemm blocks in (%[[GRID]], %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}})
// CHECK-SAME: args(%{{[0-9]+}} : !llvm.ptr, %{{[0-9]+}} : !llvm.ptr, %{{[0-9]+}} : !llvm.ptr, %[[M]] : i32, %[[N]] : i32, %[[MC]] : i32, %[[NC]] : i32)
// CHECK: return %[[C]] : tensor<1x?x?xf16>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "rock.grid_size.dyn_gemm" = #rock.arg_expr<(s0 ceildiv 128) * (s1 ceildiv 64), [arg(0, 1), arg(1, 2)]>,
    "rock.dim_args.dyn_gemm" = [#rock.arg_dim<0, 1>, #rock.arg_dim<1, 2>, #rock.arg_dim<2, 1>, #rock.arg_dim<2, 2>],
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @host_dyn(%arg0: tensor<1x?x64xf16>, %arg1: tensor<1x64x?xf16>, %arg2: tensor<1x?x?xf16>) -> tensor<1x?x?xf16> {\n  %0 = func.call @dyn_gemm(%arg0, %arg1, %arg2) : (tensor<1x?x64xf16>, tensor<1x64x?xf16>, tensor<1x?x?xf16>) -> tensor<1x?x?xf16>\n  return %0 : tensor<1x?x?xf16>\n}"
    ]
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @dyn_gemm(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Dimension arguments with a static grid: the dimensions are passed, the grid
// stays a constant, and only rock.dim_args is removed from the module.
// CHECK: module attributes
// CHECK-SAME: rock.grid_size.dyn_dims_static_grid = 4 : i32
// CHECK-NOT: rock.dim_args.dyn_dims_static_grid
// CHECK: gpu.binary @rock_kernels
// CHECK-SAME: #gpu.kernel_metadata<"dyn_dims_static_grid"
// CHECK-SAME: rock.dim_args = [array<i32: 0, 0>]
// CHECK-SAME: rock.grid_size = 4 : i64
// CHECK-LABEL: func.func @host_static_grid
// CHECK: %[[D:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xf32>
// CHECK: %[[S:.*]] = arith.index_cast %[[D]] : index to i32
// CHECK: gpu.launch_func @rock_kernels::@dyn_dims_static_grid blocks in (%c4, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}})
// CHECK-SAME: args(%{{[0-9]+}} : !llvm.ptr, %{{[0-9]+}} : !llvm.ptr, %[[S]] : i32)
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "rock.grid_size.dyn_dims_static_grid" = 4 : i32,
    "rock.dim_args.dyn_dims_static_grid" = [#rock.arg_dim<0, 0>],
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @host_static_grid(%arg0: tensor<?xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {\n  %0 = func.call @dyn_dims_static_grid(%arg0, %arg1) : (tensor<?xf32>, tensor<1024xf32>) -> tensor<1024xf32>\n  return %0 : tensor<1024xf32>\n}"
    ]
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @dyn_dims_static_grid(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32)
      attributes {rock.kernel} {
    llvm.return
  }
}

// -----

// Without host functions the kernel is kept, and the metadata still carries
// the dynamic grid and the dimension arguments.
// CHECK: gpu.binary @rock_kernels
// CHECK-SAME: #gpu.kernel_metadata<"dyn_no_host"
// CHECK-SAME: rock.dim_args = [array<i32: 0, 0>, array<i32: 1, 1>]
// CHECK-SAME: rock.grid_size = "#rock.arg_expr<s0 * s1, [arg(0, 0), arg(1, 1)]>"
// CHECK: llvm.func @dyn_no_host
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "rock.grid_size.dyn_no_host" = #rock.arg_expr<s0 * s1, [arg(0, 0), arg(1, 1)]>,
    "rock.dim_args.dyn_no_host" = [#rock.arg_dim<0, 0>, #rock.arg_dim<1, 1>],
    "triton.hsaco" = "DUMMY_HSACO"
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @dyn_no_host(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: i32)
      attributes {rock.kernel} {
    llvm.return
  }
}
