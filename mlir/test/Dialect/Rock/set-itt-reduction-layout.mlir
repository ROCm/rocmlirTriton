// Unit tests for the rocmlirTriton pass rock-set-itt-reduction-layout.
//
// The inputs look like the IR tritonamdgpu-in-thread-transpose leaves behind
// on a software-pipelined kernel: the gather is loaded once in the prologue
// and once per iteration, and both copies go through amdg.in_thread_transpose
// into the same shared-memory buffer. rock-incremental-pointer-arith marks the
// gather's loads rock.loop_variant_index_math when their index math keeps a
// non-power-of-two division in the loop or advances carried coordinates.
// RUN: rocmlir-opt -rock-set-itt-reduction-layout --mlir-print-local-scope --split-input-file %s | FileCheck %s
//
// The remove-layout-conversions run that follows in the pipeline carries the
// new layout back into the address computation and leaves no conversion.
// RUN: rocmlir-opt -rock-set-itt-reduction-layout -tritongpu-remove-layout-conversions --mlir-print-local-scope --split-input-file %s | FileCheck %s --check-prefix=PROP

// The gather's loads are marked, so both loads staged into the buffer move to
// warpsPerCTA = [4, 1], between convert_layout ops, and their
// in_thread_transpose #linear follows. The A operand is not staged through
// in_thread_transpose and keeps its layout.

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedA = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [4, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @marked_gather
  // CHECK:         tt.load {{.*}} : tensor<128x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>>
  // CHECK:         %[[PTR0:.*]] = ttg.convert_layout {{.*}} -> tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK:         tt.load %[[PTR0]] {rock.loop_variant_index_math} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK:         amdg.in_thread_transpose {{.*}} -> tensor<32x64xf16, #ttg.linear<{register = {{\[}}[1, 0], [2, 0], [0, 32], [16, 0]], lane = {{\[}}[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = {{\[}}[4, 0], [8, 0]], block = []}>>
  // CHECK:         scf.for
  // CHECK:           %[[PTR:.*]] = ttg.convert_layout {{.*}} -> tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK:           tt.load %[[PTR]] {rock.loop_variant_index_math} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK:           amdg.in_thread_transpose {{.*}} -> tensor<32x64xf16, #ttg.linear<{register = {{\[}}[1, 0], [2, 0], [0, 32], [16, 0]], lane = {{\[}}[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = {{\[}}[4, 0], [8, 0]], block = []}>>
  // PROP-LABEL: tt.func @marked_gather
  // PROP-NOT:     ttg.convert_layout {{.*}} : tensor<32x
  // PROP:         scf.for
  // PROP-NOT:       ttg.convert_layout {{.*}} : tensor<32x
  // PROP:           arith.divui {{.*}} : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // PROP-NOT:       ttg.convert_layout {{.*}} : tensor<32x
  // PROP:           tt.load {{.*}} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // PROP-NOT:     ttg.convert_layout {{.*}} : tensor<32x
  // PROP:         tt.return
  tt.func @marked_gather(%argA: !tt.ptr<f16>, %argB: !tt.ptr<f16>) -> tensor<128x64xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst9 = arith.constant dense<9> : tensor<32x1xi32, #blocked>
    %cstStride = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %acc0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %ptrA = tt.splat %argA : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blockedA>
    %ptrB = tt.splat %argB : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %rowRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows = tt.expand_dims %rowRange {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %colRange = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols = tt.expand_dims %colRange {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %colsB = tt.broadcast %cols : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %a = tt.load %ptrA : tensor<128x32x!tt.ptr<f16>, #blockedA>
    %adot = ttg.convert_layout %a : tensor<128x32xf16, #blockedA> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared, #smem, mutable>
    %slot = ttg.memdesc_index %buf[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %q0 = arith.divui %rows, %cst9 : tensor<32x1xi32, #blocked>
    %rowOff0 = arith.muli %q0, %cstStride : tensor<32x1xi32, #blocked>
    %rowOffB0 = tt.broadcast %rowOff0 : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %offB0 = arith.addi %rowOffB0, %colsB : tensor<32x64xi32, #blocked>
    %ptrB0 = tt.addptr %ptrB, %offB0 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
    %b0 = tt.load %ptrB0 {rock.loop_variant_index_math} : tensor<32x64x!tt.ptr<f16>, #blocked>
    %bt0 = amdg.in_thread_transpose %b0 : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #linear>
    ttg.local_store %bt0, %slot : tensor<32x64xf16, #linear> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %res:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %acc0, %cur = %slot) -> (tensor<128x64xf32, #mma>, !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) : i32 {
      %next = arith.addi %iv, %c1_i32 : i32
      %k = arith.muli %next, %c32_i32 : i32
      %kSplat = tt.splat %k : i32 -> tensor<32x1xi32, #blocked>
      %row = arith.addi %kSplat, %rows : tensor<32x1xi32, #blocked>
      %q = arith.divui %row, %cst9 : tensor<32x1xi32, #blocked>
      %rowOff = arith.muli %q, %cstStride : tensor<32x1xi32, #blocked>
      %rowOffB = tt.broadcast %rowOff : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
      %offB = arith.addi %rowOffB, %colsB : tensor<32x64xi32, #blocked>
      %ptrBk = tt.addptr %ptrB, %offB : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
      %b = tt.load %ptrBk {rock.loop_variant_index_math} : tensor<32x64x!tt.ptr<f16>, #blocked>
      %bdot = ttg.local_load %cur : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %adot, %bdot, %acc : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
      %bt = amdg.in_thread_transpose %b : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #linear>
      ttg.local_store %bt, %slot : tensor<32x64xf16, #linear> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
      scf.yield %d, %slot : tensor<128x64xf32, #mma>, !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    }
    %bLast = ttg.local_load %res#1 : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %out = tt.dot %adot, %bLast, %res#0 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
    ttg.local_dealloc %buf : !ttg.memdesc<1x32x64xf16, #shared, #smem, mutable>
    tt.return %out : tensor<128x64xf32, #mma>
  }
}

// -----

// Same kernel, division by 9 in the loop included, but the loads are not
// marked: the pass keys on the marker rather than on the index math, so
// nothing changes.

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedA = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [4, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @unmarked_gather
  // CHECK-NOT:     warpsPerCTA = [4, 1]
  // CHECK:         tt.return
  tt.func @unmarked_gather(%argA: !tt.ptr<f16>, %argB: !tt.ptr<f16>) -> tensor<128x64xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst9 = arith.constant dense<9> : tensor<32x1xi32, #blocked>
    %cstStride = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %acc0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %ptrA = tt.splat %argA : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blockedA>
    %ptrB = tt.splat %argB : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %rowRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows = tt.expand_dims %rowRange {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %colRange = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols = tt.expand_dims %colRange {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %colsB = tt.broadcast %cols : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %a = tt.load %ptrA : tensor<128x32x!tt.ptr<f16>, #blockedA>
    %adot = ttg.convert_layout %a : tensor<128x32xf16, #blockedA> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x32x64xf16, #shared, #smem, mutable>
    %slot = ttg.memdesc_index %buf[%c0_i32] : !ttg.memdesc<1x32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %q0 = arith.divui %rows, %cst9 : tensor<32x1xi32, #blocked>
    %rowOff0 = arith.muli %q0, %cstStride : tensor<32x1xi32, #blocked>
    %rowOffB0 = tt.broadcast %rowOff0 : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %offB0 = arith.addi %rowOffB0, %colsB : tensor<32x64xi32, #blocked>
    %ptrB0 = tt.addptr %ptrB, %offB0 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
    %b0 = tt.load %ptrB0 : tensor<32x64x!tt.ptr<f16>, #blocked>
    %bt0 = amdg.in_thread_transpose %b0 : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #linear>
    ttg.local_store %bt0, %slot : tensor<32x64xf16, #linear> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %res:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %acc0, %cur = %slot) -> (tensor<128x64xf32, #mma>, !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) : i32 {
      %next = arith.addi %iv, %c1_i32 : i32
      %k = arith.muli %next, %c32_i32 : i32
      %kSplat = tt.splat %k : i32 -> tensor<32x1xi32, #blocked>
      %row = arith.addi %kSplat, %rows : tensor<32x1xi32, #blocked>
      %q = arith.divui %row, %cst9 : tensor<32x1xi32, #blocked>
      %rowOff = arith.muli %q, %cstStride : tensor<32x1xi32, #blocked>
      %rowOffB = tt.broadcast %rowOff : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
      %offB = arith.addi %rowOffB, %colsB : tensor<32x64xi32, #blocked>
      %ptrBk = tt.addptr %ptrB, %offB : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
      %b = tt.load %ptrBk : tensor<32x64x!tt.ptr<f16>, #blocked>
      %bdot = ttg.local_load %cur : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %adot, %bdot, %acc : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
      %bt = amdg.in_thread_transpose %b : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #linear>
      ttg.local_store %bt, %slot : tensor<32x64xf16, #linear> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
      scf.yield %d, %slot : tensor<128x64xf32, #mma>, !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    }
    %bLast = ttg.local_load %res#1 : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %out = tt.dot %adot, %bLast, %res#0 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xf32, #mma>
    ttg.local_dealloc %buf : !ttg.memdesc<1x32x64xf16, #shared, #smem, mutable>
    tt.return %out : tensor<128x64xf32, #mma>
  }
}
