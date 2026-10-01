// RUN: not rocmlir-opt -rock-emit-gpu-binary="arch=gfx90a" %s 2>&1 | FileCheck %s

// A dimension argument naming a launch operand that does not exist is
// diagnosed at the call in the restored host function. That function is
// re-parsed from a string, so the diagnostic has no location in this file and
// cannot be checked with -verify-diagnostics.
// CHECK: error: kernel dimension argument does not refer to a shaped launch operand
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32,
    "ttg.num-ctas" = 1 : i32,
    "ttg.shared" = 0 : i32,
    "rock.grid_size.test_dim_arg_out_of_range" = 4 : i32,
    "rock.dim_args.test_dim_arg_out_of_range" = [#rock.arg_dim<5, 0>],
    "triton.hsaco" = "DUMMY_HSACO",
    "rock.host_functions" = [
        "func.func @host(%arg0: tensor<?xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {\n  %0 = func.call @test_dim_arg_out_of_range(%arg0, %arg1) : (tensor<?xf32>, tensor<1024xf32>) -> tensor<1024xf32>\n  return %0 : tensor<1024xf32>\n}"
    ]
} {
  llvm.func @test_dim_arg_out_of_range(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32)
      attributes {rock.kernel} {
    llvm.return
  }
}
