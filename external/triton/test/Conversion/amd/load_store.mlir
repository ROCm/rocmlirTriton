// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec8
    tt.func @global_load_store_vec8(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    // Load 8 elements from A with two vectorized load instruction
    // CHECK-COUNT-2: llvm.load {{.*}} : !llvm.ptr<1> -> vector<4xf32>
    %9 = tt.load %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #blocked0>
    // Load 8 elements from B with two vectorized load instruction
    // CHECK-COUNT-2: llvm.load {{.*}} : !llvm.ptr<1> -> vector<4xf32>
    %10 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = true}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: global_store_mfma_vec16
  tt.func public @global_store_mfma_vec16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x32xf32, #mma>
    %1 = math.exp2 %0 : tensor<32x32xf32, #mma>
    %2 = arith.truncf %1 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
    %c32_i32 = arith.constant 32 : i32
    %100 = tt.get_program_id x : i32
    %101 = arith.muli %100, %c32_i32 : i32
    %102 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %300 = tt.expand_dims %102 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x32xi32, #mma>
    %200 = tt.broadcast %300 : tensor<1x32xi32, #mma> -> tensor<32x32xi32, #mma>
    %103 = tt.splat %101 : i32 -> tensor<32x32xi32, #mma>
    %104 = arith.addi %103, %200 : tensor<32x32xi32, #mma>
    %105 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #mma>
    %106 = tt.addptr %105, %104 : tensor<32x32x!tt.ptr<f16>, #mma>, tensor<32x32xi32, #mma>
    // Store 16 elements with four vectorized store instruction
    // CHECK-COUNT-4: llvm.store {{.*}} : vector<4xf16>, !llvm.ptr<1>
    tt.store %106, %2 : tensor<32x32x!tt.ptr<f16>, #mma>
    tt.return
  }
}

// -----

#gelu32 = #ttg.blocked<{sizePerThread = [32], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[UNROLL32:.+]] = #llvm.loop_unroll<disable = true>
  // CHECK-DAG: #[[LOOP32:.+]] = #llvm.loop_annotation<unroll = #[[UNROLL32]]>
  // CHECK-LABEL: looped_gelu_store_32_masked
  // CHECK: llvm.alloca
  // CHECK: llvm.cond_br
  // CHECK-COUNT-4: llvm.call @__ocml_erf_f32
  // CHECK: llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<1>
  // CHECK: llvm.br {{.*}} {loop_annotation = #[[LOOP32]]}
  tt.func @looped_gelu_store_32_masked(
      %value : tensor<1024xf32, #gelu32>,
      %base : !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %n : i32 {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #gelu32>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #gelu32>
    %ptrs = tt.addptr %base_tensor, %range : tensor<1024x!tt.ptr<f32>, #gelu32>, tensor<1024xi32, #gelu32>
    %n_tensor = tt.splat %n : i32 -> tensor<1024xi32, #gelu32>
    %mask = arith.cmpi slt, %range, %n_tensor : tensor<1024xi32, #gelu32>
    tt.store %ptrs, %value, %mask {
      amdg.looped_gelu = {
        bias = 1.000000e+00 : f32,
        bias_fastmath = #arith.fastmath<nnan>,
        core_fastmath = #arith.fastmath<reassoc>,
        core_is_lhs = true,
        erf_fastmath = #arith.fastmath<nnan>,
        erf_is_lhs = true,
        output_fastmath = #arith.fastmath<afn>,
        output_scale = 5.000000e-01 : f32,
        scale = 7.07106769E-1 : f32,
        scale_fastmath = #arith.fastmath<contract>,
        scaled_x_is_lhs = true,
        x_is_lhs = true
      }
    } : tensor<1024x!tt.ptr<f32>, #gelu32>
    tt.return
  }
}

// -----

#gelu64 = #ttg.blocked<{sizePerThread = [64], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[UNROLL64:.+]] = #llvm.loop_unroll<disable = true>
  // CHECK-DAG: #[[LOOP64:.+]] = #llvm.loop_annotation<unroll = #[[UNROLL64]]>
  // CHECK-LABEL: looped_gelu_store_64
  // CHECK-COUNT-4: llvm.call @__ocml_erf_f32
  // CHECK: llvm.store {{.*}} : vector<4xf32>, !llvm.ptr<1>
  // CHECK: llvm.br {{.*}} {loop_annotation = #[[LOOP64]]}
  tt.func @looped_gelu_store_64(
      %value : tensor<2048xf32, #gelu64>,
      %base : !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {start = 0 : i32, end = 2048 : i32} : tensor<2048xi32, #gelu64>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>, #gelu64>
    %ptrs = tt.addptr %base_tensor, %range : tensor<2048x!tt.ptr<f32>, #gelu64>, tensor<2048xi32, #gelu64>
    tt.store %ptrs, %value {
      amdg.looped_gelu = {
        bias = 1.000000e+00 : f32,
        core_is_lhs = true,
        erf_is_lhs = true,
        scale = 7.07106769E-1 : f32,
        scaled_x_is_lhs = true,
        x_is_lhs = true
      }
    } : tensor<2048x!tt.ptr<f32>, #gelu64>
    tt.return
  }
}
