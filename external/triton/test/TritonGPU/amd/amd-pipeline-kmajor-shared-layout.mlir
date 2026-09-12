// RUN: triton-opt %s -split-input-file -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline | FileCheck %s

// A matrix core dot operand reads its tile K-contiguous, but the shared order
// is inherited from the global load. When the load is not K-contiguous and its
// layout has no room across K for tritonamdgpu-in-thread-transpose to widen
// (K is fully consumed by lanes and warps), the buffer alone is re-oriented to
// the K-contiguous rotating layout.

// Positive case: WMMA B operand (opIdx = 1) on gfx1100, which has no LDS
// transpose load. K = 64 == threadsPerWarp[0] * warpsPerCTA[0] == 16 * 4, so
// there is no room across K.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 1, isTranspose = false, ctaLayout = {warp = [[1, 0], [2, 0]]}}>
// CHECK: #shared = #ttg.amd_rotating_shared<{{.*}}order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_operand_b_no_k_room_is_flipped
  tt.func @wmma_operand_b_no_k_room_is_flipped(
                %argB: tensor<64x16x!tt.ptr<f16>, #blocked>,
                %argA: tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<64x16xf16, #mma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    // CHECK-NOT: amdgpu.in_thread_transpose
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<64x16xf16, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<64x16xf16, #mma>) : i32 {
      %b = tt.load %argB : tensor<64x16x!tt.ptr<f16>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<64x16xf16, #mma>
      scf.yield %c : tensor<64x16xf16, #mma>
    }
    tt.return %result : tensor<64x16xf16, #mma>
  }
}

// -----

// Positive case: pre-gfx950 MFMA, which has no ds_read_tr for f16 either.
// K = 64 == threadsPerWarp[0] * warpsPerCTA[0] == 16 * 4. CDNA3 only pipelines
// loads whose vectorization it can prove, hence the explicit addressing.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
// CHECK: #shared = #ttg.amd_rotating_shared<{{.*}}order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_operand_b_no_k_room_is_flipped
  tt.func @mfma_operand_b_no_k_room_is_flipped(
                %base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                %argA: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<128x32xf16, #mma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    // CHECK-NOT: amdgpu.in_thread_transpose
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #mma>
    %cst_stride = arith.constant dense<32> : tensor<64x1xi32, #blocked>
    %rk = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rn = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %ek = tt.expand_dims %rk {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %en = tt.expand_dims %rn {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %row = arith.muli %ek, %cst_stride : tensor<64x1xi32, #blocked>
    %bk = tt.broadcast %row : tensor<64x1xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %bn = tt.broadcast %en : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %off = arith.addi %bk, %bn : tensor<64x32xi32, #blocked>
    %sp = tt.splat %base : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    %argB = tt.addptr %sp, %off : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<128x32xf16, #mma>) : i32 {
      %b = tt.load %argB : tensor<64x32x!tt.ptr<f16>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf16, #mma>
      scf.yield %c : tensor<128x32xf16, #mma>
    }
    tt.return %result : tensor<128x32xf16, #mma>
  }
}

// -----

// Negative case: the load layout has room across K (K = 128, lanes and warps
// claim 64), so tritonamdgpu-in-thread-transpose owns this operand and picks
// its own buffer. The pipeliner must leave the inherited order alone.

#blocked = #ttg.blocked<{sizePerThread = [2, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
// CHECK-NOT: #ttg.amd_rotating_shared
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_operand_b_k_room_left_to_in_thread_transpose
  tt.func @mfma_operand_b_k_room_left_to_in_thread_transpose(
                %base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                %argA: tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<128x32xf16, #mma> {
    // CHECK: ttg.local_alloc
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #mma>
    %cst_stride = arith.constant dense<32> : tensor<128x1xi32, #blocked>
    %rk = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rn = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %ek = tt.expand_dims %rk {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %en = tt.expand_dims %rn {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %row = arith.muli %ek, %cst_stride : tensor<128x1xi32, #blocked>
    %bk = tt.broadcast %row : tensor<128x1xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %bn = tt.broadcast %en : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %off = arith.addi %bk, %bn : tensor<128x32xi32, #blocked>
    %sp = tt.splat %base : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked>
    %argB = tt.addptr %sp, %off : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<128x32xf16, #mma>) : i32 {
      %b = tt.load %argB : tensor<128x32x!tt.ptr<f16>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf16, #mma>
      scf.yield %c : tensor<128x32xf16, #mma>
    }
    tt.return %result : tensor<128x32xf16, #mma>
  }
}

// -----

// Negative case: gfx950 reads a non-K-contiguous operand with ds_read_tr, so
// its buffer is already in the orientation the hardware wants.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 16], isTransposed = true}>
// CHECK-NOT: #ttg.amd_rotating_shared
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_operand_b_lds_transpose_load
  tt.func @mfma_operand_b_lds_transpose_load(
                %argB: tensor<16x32x!tt.ptr<f16>, #blocked>,
                %argA: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<16x32xf16, #mma> {
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<16x32xf16, #mma>) : i32 {
      %b = tt.load %argB : tensor<16x32x!tt.ptr<f16>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<16x32xf16, #blocked> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x32xf16, #mma>
      scf.yield %c : tensor<16x32xf16, #mma>
    }
    tt.return %result : tensor<16x32xf16, #mma>
  }
}

// -----

// Negative case: one element per lane per ds_read cannot be packed any
// further, so a K-major buffer would buy nothing on the read side.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = true}>
// CHECK-NOT: #ttg.amd_rotating_shared
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_operand_b_k_width_one
  tt.func @mfma_operand_b_k_width_one(
                %argB: tensor<16x16x!tt.ptr<f32>, #blocked>,
                %argA: tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<16x16xf32, #mma> {
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<16x16xf32, #mma>) : i32 {
      %b = tt.load %argB : tensor<16x16x!tt.ptr<f32>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x16xf32, #mma>
      scf.yield %c : tensor<16x16xf32, #mma>
    }
    tt.return %result : tensor<16x16xf32, #mma>
  }
}
