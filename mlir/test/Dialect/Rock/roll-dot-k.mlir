// RUN: rocmlir-opt --split-input-file --rock-roll-dot-k %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// A 128x64 by 64x64 f32 dot on an RDNA target. The result layout puts 128
// accumulators in each thread, so the unrolled dot is 8192 FMAs and the
// heuristic cuts K into sixteen segments of 4 to land on 512.

// CHECK-LABEL: tt.func @roll_f32_fma_dot
// CHECK-DAG:     %[[A:.*]] = ttg.memdesc_reinterpret %{{.*}} : !ttg.memdesc<128x64xf32, {{.*}}> -> !ttg.memdesc<16x128x4xf32, {{.*}}>
// CHECK-DAG:     %[[B:.*]] = ttg.memdesc_reinterpret %{{.*}} : !ttg.memdesc<64x64xf32, {{.*}}> -> !ttg.memdesc<16x4x64xf32, {{.*}}>
// CHECK:         %[[LOOP:.*]] = scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (tensor<128x64xf32, {{.*}}>)  : i32 {
// CHECK-DAG:       %[[AV:.*]] = ttg.memdesc_index %[[A]][%[[J]]] : {{.*}} -> !ttg.memdesc<128x4xf32, {{.*}}>
// CHECK-DAG:       %[[BV:.*]] = ttg.memdesc_index %[[B]][%[[J]]] : {{.*}} -> !ttg.memdesc<4x64xf32, {{.*}}>
// The operands load straight into the dot layout, so no convert_layout is left.
// CHECK-DAG:       %[[AL:.*]] = ttg.local_load %[[AV]] : {{.*}} -> tensor<128x4xf32, #ttg.dot_op<{opIdx = 0,{{.*}}>>
// CHECK-DAG:       %[[BL:.*]] = ttg.local_load %[[BV]] : {{.*}} -> tensor<4x64xf32, #ttg.dot_op<{opIdx = 1,{{.*}}>>
// CHECK:           %[[D:.*]] = tt.dot %[[AL]], %[[BL]], %[[ACC]]
// CHECK:           scf.yield %[[D]]
// CHECK:         tt.return %[[LOOP]]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @roll_f32_fma_dot(%acc: tensor<128x64xf32, #blocked2>) -> tensor<128x64xf32, #blocked2> {
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
    %al = ttg.local_load %a : !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> tensor<128x64xf32, #blocked1>
    %bl = ttg.local_load %b : !ttg.memdesc<64x64xf32, #shared1, #smem, mutable> -> tensor<64x64xf32, #blocked>
    %ac = ttg.convert_layout %al : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %bc = ttg.convert_layout %bl : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %d = tt.dot %ac, %bc, %acc : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x64xf32, #blocked2>
    tt.return %d : tensor<128x64xf32, #blocked2>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// A kernel pipelined to three stages: the allocation carries two buffers, but
// the pipeliner peels the buffer indexing into the prologue and rotates rank-2
// views through the loop, so the rolled view is rank 3 and says nothing about
// the number of buffers. Its trailing drain dot has to be rolled too, since
// that is another full tile of FMAs.

// CHECK-LABEL: tt.func @roll_multi_buffer
// The loop-body dot. The reinterpret is rank 3 whatever the buffer count is,
// because it applies to the rank-2 view the loop carries.
// CHECK:         ttg.memdesc_reinterpret %{{.*}} : !ttg.memdesc<128x64xf32, {{.*}}> -> !ttg.memdesc<16x128x4xf32, {{.*}}>
// CHECK:         scf.for
// CHECK:           tt.dot {{.*}} tensor<128x4xf32, {{.*}}> * tensor<4x64xf32, {{.*}}>
// The drain dot, in its own scf.if.
// CHECK:         scf.if
// CHECK:           ttg.memdesc_reinterpret %{{.*}} : !ttg.memdesc<128x64xf32, {{.*}}> -> !ttg.memdesc<16x128x4xf32, {{.*}}>
// CHECK:           scf.for
// CHECK:             tt.dot {{.*}} tensor<128x4xf32, {{.*}}> * tensor<4x64xf32, {{.*}}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @roll_multi_buffer(%acc: tensor<128x64xf32, #blocked2>, %n: i32, %cond: i1) -> tensor<128x64xf32, #blocked2> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf32, #shared, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xf32, #shared1, #smem, mutable>
    %a0 = ttg.memdesc_index %alloc_a[%c0] : !ttg.memdesc<2x128x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    %b0 = ttg.memdesc_index %alloc_b[%c0] : !ttg.memdesc<2x64x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
    %a1 = ttg.memdesc_index %alloc_a[%c1] : !ttg.memdesc<2x128x64xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    %b1 = ttg.memdesc_index %alloc_b[%c1] : !ttg.memdesc<2x64x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
    %loop:3 = scf.for %i = %c0 to %n step %c1 iter_args(%it = %acc, %ca = %a0, %cb = %b0) -> (tensor<128x64xf32, #blocked2>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>)  : i32 {
      %al = ttg.local_load %ca : !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> tensor<128x64xf32, #blocked1>
      %bl = ttg.local_load %cb : !ttg.memdesc<64x64xf32, #shared1, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %ac = ttg.convert_layout %al : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
      %bc = ttg.convert_layout %bl : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
      %d = tt.dot %ac, %bc, %it : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x64xf32, #blocked2>
      scf.yield %d, %a1, %b1 : tensor<128x64xf32, #blocked2>, !ttg.memdesc<128x64xf32, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
    }
    %drain = scf.if %cond -> (tensor<128x64xf32, #blocked2>) {
      %al = ttg.local_load %loop#1 : !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> tensor<128x64xf32, #blocked1>
      %bl = ttg.local_load %loop#2 : !ttg.memdesc<64x64xf32, #shared1, #smem, mutable> -> tensor<64x64xf32, #blocked>
      %ac = ttg.convert_layout %al : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
      %bc = ttg.convert_layout %bl : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
      %d = tt.dot %ac, %bc, %loop#0 : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x64xf32, #blocked2>
      scf.yield %d : tensor<128x64xf32, #blocked2>
    } else {
      scf.yield %loop#0 : tensor<128x64xf32, #blocked2>
    }
    tt.return %drain : tensor<128x64xf32, #blocked2>
  }
}

// -----

// 64-wide warps here, so these are not the layouts the RDNA sections use.
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// Same dot on a CDNA target: out of scope. Its 64 accumulators per thread
// would otherwise clear the heuristic, so the arch is what stops it.

// CHECK-LABEL: tt.func @no_roll_on_cdna
// CHECK-NOT:     memdesc_reinterpret
// CHECK-NOT:     scf.for
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @no_roll_on_cdna(%acc: tensor<128x64xf32, #blocked2>) -> tensor<128x64xf32, #blocked2> {
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
    %al = ttg.local_load %a : !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> tensor<128x64xf32, #blocked1>
    %bl = ttg.local_load %b : !ttg.memdesc<64x64xf32, #shared1, #smem, mutable> -> tensor<64x64xf32, #blocked>
    %ac = ttg.convert_layout %al : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %bc = ttg.convert_layout %bl : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %d = tt.dot %ac, %bc, %acc : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x64xf32, #blocked2>
    tt.return %d : tensor<128x64xf32, #blocked2>
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#wmma = #ttg.amd_wmma<{version = 1, isTranspose = false, ctaLayout = {warp = [[0, 1]]}}>

// A WMMA dot on the same Navi target computes a whole tile per instruction
// rather than expanding into scalar FMAs, so there is no oversized basic
// block to shrink. Its result encoding is what says so, which is also the
// test the backend uses to pick the lowering.

// CHECK-LABEL: tt.func @no_roll_wmma
// CHECK-NOT:     memdesc_reinterpret
// CHECK-NOT:     scf.for
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @no_roll_wmma(%acc: tensor<128x64xf32, #wmma>) -> tensor<128x64xf32, #wmma> {
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %al = ttg.local_load %a : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma, kWidth = 16}>>
    %bl = ttg.local_load %b : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma, kWidth = 16}>>
    %d = tt.dot %al, %bl, %acc : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma, kWidth = 16}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma, kWidth = 16}>> -> tensor<128x64xf32, #wmma>
    tt.return %d : tensor<128x64xf32, #wmma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
// A's shared encoding runs K fastest rather than slowest, so a K segment is
// strided in shared memory rather than contiguous and reinterpreting the
// buffer would read the wrong bytes. The linear-layout check catches it.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// CHECK-LABEL: tt.func @no_roll_k_not_slowest
// CHECK-NOT:     memdesc_reinterpret
// CHECK-NOT:     scf.for
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @no_roll_k_not_slowest(%acc: tensor<128x64xf32, #blocked2>) -> tensor<128x64xf32, #blocked2> {
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared1, #smem, mutable>
    %al = ttg.local_load %a : !ttg.memdesc<128x64xf32, #shared, #smem, mutable> -> tensor<128x64xf32, #blocked1>
    %bl = ttg.local_load %b : !ttg.memdesc<64x64xf32, #shared1, #smem, mutable> -> tensor<64x64xf32, #blocked>
    %ac = ttg.convert_layout %al : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %bc = ttg.convert_layout %bl : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %d = tt.dot %ac, %bc, %acc : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x64xf32, #blocked2>
    tt.return %d : tensor<128x64xf32, #blocked2>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// Only 16 accumulators per thread here, so even unrolled the dot is 256 FMAs
// and the block is nowhere near the size where the scheduler struggles.

// CHECK-LABEL: tt.func @no_roll_small_dot
// CHECK-NOT:     memdesc_reinterpret
// CHECK-NOT:     scf.for
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @no_roll_small_dot(%acc: tensor<16x32xf32, #blocked2>) -> tensor<16x32xf32, #blocked2> {
    %a = ttg.local_alloc : () -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<16x32xf32, #shared1, #smem, mutable>
    %al = ttg.local_load %a : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> tensor<16x16xf32, #blocked1>
    %bl = ttg.local_load %b : !ttg.memdesc<16x32xf32, #shared1, #smem, mutable> -> tensor<16x32xf32, #blocked>
    %ac = ttg.convert_layout %al : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %bc = ttg.convert_layout %bl : tensor<16x32xf32, #blocked> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %d = tt.dot %ac, %bc, %acc : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<16x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<16x32xf32, #blocked2>
    tt.return %d : tensor<16x32xf32, #blocked2>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// Two chained dots of 128 accumulators and a K of 4, so each one is exactly
// the 512-FMA target on its own and neither can shrink against it, yet the
// block they share holds 1024. Halving the first one's K puts its body under
// the target and moves it out of this block, which leaves 512 here and stops
// the second one from being touched.

// CHECK-LABEL: tt.func @roll_until_block_fits
// CHECK:         ttg.memdesc_reinterpret %{{.*}} : !ttg.memdesc<128x4xf32, {{.*}}> -> !ttg.memdesc<2x128x2xf32, {{.*}}>
// CHECK:         scf.for
// CHECK:           tt.dot {{.*}} tensor<128x2xf32, {{.*}}> * tensor<2x64xf32, {{.*}}>
// The second dot keeps its full K: what is left unrolled already fits.
// CHECK:         tt.dot {{.*}} tensor<128x4xf32, {{.*}}> * tensor<4x64xf32, {{.*}}>
// CHECK-NOT:     scf.for
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @roll_until_block_fits(%acc: tensor<128x64xf32, #blocked2>) -> tensor<128x64xf32, #blocked2> {
    %a0 = ttg.local_alloc : () -> !ttg.memdesc<128x4xf32, #shared, #smem, mutable>
    %b0 = ttg.local_alloc : () -> !ttg.memdesc<4x64xf32, #shared1, #smem, mutable>
    %a1 = ttg.local_alloc : () -> !ttg.memdesc<128x4xf32, #shared, #smem, mutable>
    %b1 = ttg.local_alloc : () -> !ttg.memdesc<4x64xf32, #shared1, #smem, mutable>
    %al0 = ttg.local_load %a0 : !ttg.memdesc<128x4xf32, #shared, #smem, mutable> -> tensor<128x4xf32, #blocked1>
    %bl0 = ttg.local_load %b0 : !ttg.memdesc<4x64xf32, #shared1, #smem, mutable> -> tensor<4x64xf32, #blocked>
    %ac0 = ttg.convert_layout %al0 : tensor<128x4xf32, #blocked1> -> tensor<128x4xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %bc0 = ttg.convert_layout %bl0 : tensor<4x64xf32, #blocked> -> tensor<4x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %d0 = tt.dot %ac0, %bc0, %acc : tensor<128x4xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<4x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x64xf32, #blocked2>
    %al1 = ttg.local_load %a1 : !ttg.memdesc<128x4xf32, #shared, #smem, mutable> -> tensor<128x4xf32, #blocked1>
    %bl1 = ttg.local_load %b1 : !ttg.memdesc<4x64xf32, #shared1, #smem, mutable> -> tensor<4x64xf32, #blocked>
    %ac1 = ttg.convert_layout %al1 : tensor<128x4xf32, #blocked1> -> tensor<128x4xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %bc1 = ttg.convert_layout %bl1 : tensor<4x64xf32, #blocked> -> tensor<4x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %d1 = tt.dot %ac1, %bc1, %d0 : tensor<128x4xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<4x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x64xf32, #blocked2>
    tt.return %d1 : tensor<128x64xf32, #blocked2>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// A 256x128 tile over 64 threads leaves 512 accumulators in each one, which is
// the whole target on its own, so no segment width gets the body under it. A
// body still holds `accs * dotK`, so narrowing to a single K per iteration is
// what divides this block the furthest, taking it from 2048 FMAs down to 512.

// CHECK-LABEL: tt.func @roll_to_accumulator_floor
// CHECK-DAG:     %[[A:.*]] = ttg.memdesc_reinterpret %{{.*}} : !ttg.memdesc<256x4xf32, {{.*}}> -> !ttg.memdesc<4x256x1xf32, {{.*}}>
// CHECK-DAG:     %[[B:.*]] = ttg.memdesc_reinterpret %{{.*}} : !ttg.memdesc<4x128xf32, {{.*}}> -> !ttg.memdesc<4x1x128xf32, {{.*}}>
// CHECK:         scf.for
// CHECK:           tt.dot {{.*}} tensor<256x1xf32, {{.*}}> * tensor<1x128xf32, {{.*}}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @roll_to_accumulator_floor(%acc: tensor<256x128xf32, #blocked2>) -> tensor<256x128xf32, #blocked2> {
    %a = ttg.local_alloc : () -> !ttg.memdesc<256x4xf32, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<4x128xf32, #shared1, #smem, mutable>
    %al = ttg.local_load %a : !ttg.memdesc<256x4xf32, #shared, #smem, mutable> -> tensor<256x4xf32, #blocked1>
    %bl = ttg.local_load %b : !ttg.memdesc<4x128xf32, #shared1, #smem, mutable> -> tensor<4x128xf32, #blocked>
    %ac = ttg.convert_layout %al : tensor<256x4xf32, #blocked1> -> tensor<256x4xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %bc = ttg.convert_layout %bl : tensor<4x128xf32, #blocked> -> tensor<4x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %d = tt.dot %ac, %bc, %acc : tensor<256x4xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<4x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<256x128xf32, #blocked2>
    tt.return %d : tensor<256x128xf32, #blocked2>
  }
}
