// Unit tests for the rocmlirTriton rock-set-reduction-layout pass.
//
// The pass redistributes the reduction-operand global-load #blocked layout so
// that every warp owns a reduction (K) strip. 
//
// These tests exercise the rewrite mechanics on kernels that do not carry the
// `rock.conv_kernel` attribute, so they force the rewrite on with
// `use-reduction-layout=1` (the default only rewrites convolution kernels).
// RUN: rocmlir-opt -rock-set-reduction-layout="use-reduction-layout=1" --mlir-print-local-scope --split-input-file %s | FileCheck %s
//
// The conflict path (a single load feeding dot operands with disagreeing
// reduction dims) emits a warning to stderr; check it separately.
// RUN: rocmlir-opt -rock-set-reduction-layout="use-reduction-layout=1" --split-input-file %s 2>&1 | FileCheck %s --check-prefix=CONFLICT

// The B / reduction operand (64x64) is loaded "gather"-style: its reduction dim
// (dim 0) is the slowest-varying dim (order = [1, 0]). Its warpsPerCTA = [2, 2]
// is collapsed onto K -> [4, 1], and the encoding is propagated to the masked
// load's pointer/mask/other operands.
// The A / free operand (128x64) keeps its layout: its reduction dim (dim 1) is
// the fastest-varying dim, so it is not a gather and is left untouched.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduction_redistributed
  // CHECK-DAG:     tt.load {{.*}}tensor<128x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>>
  // CHECK-DAG:     tt.load {{.*}}tensor<64x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     arith.constant dense<0> : tensor<64x64xi8, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  tt.func @reduction_redistributed(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>) -> tensor<128x64xi32, #blocked> {
    %cst = arith.constant dense<0> : tensor<64x64xi8, #blocked1>
    %cst_0 = arith.constant dense<true> : tensor<64x64xi1, #blocked1>
    %cst_1 = arith.constant dense<0> : tensor<128x64xi32, #blocked>
    %0 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blocked1>
    %2 = tt.load %0 : tensor<128x64x!tt.ptr<i8>, #blocked>
    %3 = tt.load %1, %cst_0, %cst : tensor<64x64x!tt.ptr<i8>, #blocked1>
    %4 = ttg.convert_layout %2 : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %5 = ttg.convert_layout %3 : tensor<64x64xi8, #blocked1> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %6 = tt.dot %4, %5, %cst_1 : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xi32, #blocked>
    tt.return %6 : tensor<128x64xi32, #blocked>
  }
}


// -----

// Gate check: when the reduction operand is already reduction-contiguous (its
// reduction dim, dim 0, is the fastest-varying dim, order = [0, 1]), it is not
// a gather load and the pass leaves every layout untouched.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduction_already_contiguous
  //   CHECK-NOT: warpsPerCTA = [4, 1]
  tt.func @reduction_already_contiguous(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>) -> tensor<128x64xi32, #blocked> {
    %cst = arith.constant dense<0> : tensor<128x64xi32, #blocked>
    %0 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blocked1>
    %2 = tt.load %0 : tensor<128x64x!tt.ptr<i8>, #blocked>
    %3 = tt.load %1 : tensor<64x64x!tt.ptr<i8>, #blocked1>
    %4 = ttg.convert_layout %2 : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %5 = ttg.convert_layout %3 : tensor<64x64xi8, #blocked1> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %6 = tt.dot %4, %5, %cst : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xi32, #blocked>
    tt.return %6 : tensor<128x64xi32, #blocked>
  }
}

// -----

// Contiguity is recomputed for every changed buffer load after all gather
// layouts in the module are final. The first kernel proves that a safe vector
// width can be recovered and that an unrelated load keeps its original hint.
// The second uses the same contiguous offsets, but its dynamic mask limits the
// safe width to one.

#blockedA = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedB = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @buffer_load_contiguity_recomputed
  // The unrelated A load is not retyped, so its explicit hint is preserved.
  // CHECK-DAG:     amdg.buffer_load {{.*}} {contiguity = 8 : i32} : tensor<128x64xi8, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>>
  // The B load's original hint is replaced with the width proven after relayout.
  // CHECK-DAG:     amdg.buffer_load {{.*}} {contiguity = 4 : i32} : tensor<64x64xi8, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
  tt.func @buffer_load_contiguity_recomputed(%argA: !tt.ptr<i8>, %argB: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<128x64xi32, #blockedA> {
    %offA = arith.constant dense<0> : tensor<128x64xi32, #blockedA>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blockedB}>>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blockedB}>> -> tensor<1x64xi32, #blockedB>
    %offB = tt.broadcast %expanded : tensor<1x64xi32, #blockedB> -> tensor<64x64xi32, #blockedB>
    %acc = arith.constant dense<0> : tensor<128x64xi32, #blockedA>
    %a = amdg.buffer_load %argA[%offA] {contiguity = 8 : i32} : tensor<128x64xi8, #blockedA>
    %b = amdg.buffer_load %argB[%offB] {contiguity = 2 : i32} : tensor<64x64xi8, #blockedB>
    %adot = ttg.convert_layout %a : tensor<128x64xi8, #blockedA> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>>
    %bdot = ttg.convert_layout %b : tensor<64x64xi8, #blockedB> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>>
    %out = tt.dot %adot, %bdot, %acc : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>> -> tensor<128x64xi32, #blockedA>
    tt.return %out : tensor<128x64xi32, #blockedA>
  }

  // CHECK-LABEL: tt.func @buffer_load_contiguity_limited_by_mask
  // CHECK:           amdg.buffer_load %{{.*}}[%{{.*}}], %{{.*}} : tensor<64x64xi8, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
  tt.func @buffer_load_contiguity_limited_by_mask(%argA: !tt.ptr<i8>, %argB: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %limit: i32) -> tensor<128x64xi32, #blockedA> {
    %offA = arith.constant dense<0> : tensor<128x64xi32, #blockedA>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blockedB}>>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blockedB}>> -> tensor<1x64xi32, #blockedB>
    %offB = tt.broadcast %expanded : tensor<1x64xi32, #blockedB> -> tensor<64x64xi32, #blockedB>
    %limitSplat = tt.splat %limit : i32 -> tensor<1x64xi32, #blockedB>
    %rowMask = arith.cmpi slt, %expanded, %limitSplat : tensor<1x64xi32, #blockedB>
    %mask = tt.broadcast %rowMask : tensor<1x64xi1, #blockedB> -> tensor<64x64xi1, #blockedB>
    %acc = arith.constant dense<0> : tensor<128x64xi32, #blockedA>
    %a = amdg.buffer_load %argA[%offA] : tensor<128x64xi8, #blockedA>
    %b = amdg.buffer_load %argB[%offB], %mask {contiguity = 8 : i32} : tensor<64x64xi8, #blockedB>
    %adot = ttg.convert_layout %a : tensor<128x64xi8, #blockedA> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>>
    %bdot = ttg.convert_layout %b : tensor<64x64xi8, #blockedB> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>>
    %out = tt.dot %adot, %bdot, %acc : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>> -> tensor<128x64xi32, #blockedA>
    tt.return %out : tensor<128x64xi32, #blockedA>
  }
}

// -----

// Real-pipeline case: by the time the pass runs (after ConvertToBufferOps /
// ConvertWarpPipeline) the global gather load is an amdgpu.buffer_load, not a
// tt.load, and it is consumed through amdgpu.in_thread_transpose. Layouts and
// the transpose's paired #linear are taken from a real conv kernel
// (i8, N1 C128 28x28 K128 3x3). The gather operand (64x64, #blocked, order
// [1, 0]) has warpsPerCTA = [2, 2] collapsed onto K -> [4, 1]; the change is
// propagated to the buffer_load's offsets/mask and to the in_thread_transpose
// #linear. The A / free operand (128x64, #blocked1) already owns K per warp and
// is left untouched.

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [16, 0], [32, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [8, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduction_redistributed_buffer_load
  // Gather buffer_load + its offsets/mask are moved onto K (warpsPerCTA = [4, 1]).
  // Its vector-width hint is recomputed for the new layout; these splat offsets
  // safely produce scalar loads, so the default contiguity = 1 is omitted.
  // CHECK-DAG:     amdg.buffer_load %{{.*}}[%{{.*}}], %{{.*}} : tensor<64x64xi8, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     arith.constant dense<true> : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // The in_thread_transpose #linear is remapped to the redistributed layout.
  // CHECK-DAG:     amdg.in_thread_transpose {{.*}} -> tensor<64x64xi8, #ttg.linear<{register = {{\[}}[1, 0], [2, 0], [4, 0], [0, 32], [32, 0]], lane = {{\[}}[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = {{\[}}[8, 0], [16, 0]], block = []}>>
  // The A / free operand keeps its layout (already one warp-strip per K).
  // CHECK-DAG:     amdg.buffer_load {{.*}} : tensor<128x64xi8, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
  tt.func @reduction_redistributed_buffer_load(%argA: !tt.ptr<i8>, %argB: !tt.ptr<i8>) -> tensor<128x64xi32, #mma> {
    %offA = arith.constant dense<0> : tensor<128x64xi32, #blocked1>
    %offB = arith.constant dense<0> : tensor<64x64xi32, #blocked>
    %maskB = arith.constant dense<true> : tensor<64x64xi1, #blocked>
    %cstOut = arith.constant dense<0> : tensor<128x64xi32, #mma>
    %a = amdg.buffer_load %argA[%offA] : tensor<128x64xi8, #blocked1>
    %b = amdg.buffer_load %argB[%offB], %maskB {contiguity = 8 : i32} : tensor<64x64xi8, #blocked>
    %bt = amdg.in_thread_transpose %b : tensor<64x64xi8, #blocked> -> tensor<64x64xi8, #linear>
    %adot = ttg.convert_layout %a : tensor<128x64xi8, #blocked1> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %bdot = ttg.convert_layout %bt : tensor<64x64xi8, #linear> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %out = tt.dot %adot, %bdot, %cstOut : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xi32, #mma>
    tt.return %out : tensor<128x64xi32, #mma>
  }
}

// -----

// Real-pipeline case with shared-memory staging: by the time the pass runs the
// dot operands are produced by ttg.local_load, and the global gather
// buffer_load only reaches the dot through in_thread_transpose ->
// ttg.local_alloc -> ttg.local_load (the free operand is staged buffer_load ->
// ttg.local_alloc -> ttg.local_load). The pass must walk the dot operand back
// through the local_alloc/local_load staging to reach the feeding load;
// otherwise it fails to find any feeding load, no-ops, and leaves the gather
// operand on the slow layout. Layouts match a real conv kernel (i8, N1 C256
// 200x200 K256 3x3). The gather operand (64x64, #blocked, order [1, 0]) has
// warpsPerCTA = [2, 2] collapsed onto K -> [4, 1]; the change is propagated to
// the buffer_load's offsets/mask and to the in_thread_transpose #linear. The
// A / free operand (128x64, #blocked1) already owns K per warp and is left
// untouched.

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [16, 0], [32, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [8, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.amd_rotating_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduction_redistributed_through_shared
  // CHECK-DAG:     amdg.buffer_load {{.*}} : tensor<64x64xi8, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     arith.constant dense<true> : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     amdg.in_thread_transpose {{.*}} -> tensor<64x64xi8, #ttg.linear<{register = {{\[}}[1, 0], [2, 0], [4, 0], [0, 32], [32, 0]], lane = {{\[}}[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = {{\[}}[8, 0], [16, 0]], block = []}>>
  // CHECK-DAG:     amdg.buffer_load {{.*}} : tensor<128x64xi8, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
  tt.func @reduction_redistributed_through_shared(%argA: !tt.ptr<i8>, %argB: !tt.ptr<i8>) -> tensor<128x64xi32, #mma> {
    %offA = arith.constant dense<0> : tensor<128x64xi32, #blocked1>
    %offB = arith.constant dense<0> : tensor<64x64xi32, #blocked>
    %maskB = arith.constant dense<true> : tensor<64x64xi1, #blocked>
    %cstOut = arith.constant dense<0> : tensor<128x64xi32, #mma>
    %a = amdg.buffer_load %argA[%offA] : tensor<128x64xi8, #blocked1>
    %b = amdg.buffer_load %argB[%offB], %maskB : tensor<64x64xi8, #blocked>
    %bt = amdg.in_thread_transpose %b : tensor<64x64xi8, #blocked> -> tensor<64x64xi8, #linear>
    %aAlloc = ttg.local_alloc %a : (tensor<128x64xi8, #blocked1>) -> !ttg.memdesc<128x64xi8, #shared, #smem>
    %aLoad = ttg.local_load %aAlloc : !ttg.memdesc<128x64xi8, #shared, #smem> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %bAlloc = ttg.local_alloc %bt : (tensor<64x64xi8, #linear>) -> !ttg.memdesc<64x64xi8, #shared1, #smem>
    %bLoad = ttg.local_load %bAlloc : !ttg.memdesc<64x64xi8, #shared1, #smem> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %out = tt.dot %aLoad, %bLoad, %cstOut : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xi32, #mma>
    tt.return %out : tensor<128x64xi32, #mma>
  }
}

// -----

// Gate check on the buffer_load path: when the reduction operand is already
// reduction-contiguous (dim 0 is the fastest-varying dim, order = [0, 1]) it is
// not a gather and every layout is left untouched, even though it is an
// amdgpu.buffer_load.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduction_already_contiguous_buffer_load
  // CHECK-NOT:     warpsPerCTA = [4, 1]
  // A skipped load keeps both its original layout and its contiguity hint.
  // CHECK:         amdg.buffer_load {{.*}} {contiguity = 8 : i32} : tensor<64x64xi8, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [0, 1]}>>
  // CHECK-NOT:     warpsPerCTA = [4, 1]
  tt.func @reduction_already_contiguous_buffer_load(%argA: !tt.ptr<i8>, %argB: !tt.ptr<i8>) -> tensor<128x64xi32, #blocked> {
    %offA = arith.constant dense<0> : tensor<128x64xi32, #blocked>
    %offB = arith.constant dense<0> : tensor<64x64xi32, #blocked1>
    %cstOut = arith.constant dense<0> : tensor<128x64xi32, #blocked>
    %a = amdg.buffer_load %argA[%offA] : tensor<128x64xi8, #blocked>
    %b = amdg.buffer_load %argB[%offB] {contiguity = 8 : i32} : tensor<64x64xi8, #blocked1>
    %adot = ttg.convert_layout %a : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bdot = ttg.convert_layout %b : tensor<64x64xi8, #blocked1> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %out = tt.dot %adot, %bdot, %cstOut : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xi32, #blocked>
    tt.return %out : tensor<128x64xi32, #blocked>
  }
}

// -----

// When the same blocked encoding is shared by multiple values, the rewrite is
// scoped to the dot operand, so a value that merely shares the blocked encoding is left alone.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedA = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @aliasing_unrelated_load  
  // CHECK-DAG:     tt.load {{.*}}tensor<64x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     tt.load {{.*}}tensor<256x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
  tt.func @aliasing_unrelated_load(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i8>, %arg3: !tt.ptr<i8>) -> tensor<128x64xi32, #blockedA> {
    %cst = arith.constant dense<0> : tensor<128x64xi32, #blockedA>
    %0 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blockedA>
    %1 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blocked>
    %2 = tt.load %0 : tensor<128x64x!tt.ptr<i8>, #blockedA>
    %3 = tt.load %1 : tensor<64x64x!tt.ptr<i8>, #blocked>
    %4 = ttg.convert_layout %2 : tensor<128x64xi8, #blockedA> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>>
    %5 = ttg.convert_layout %3 : tensor<64x64xi8, #blocked> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>>
    %6 = tt.dot %4, %5, %cst : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>> -> tensor<128x64xi32, #blockedA>
    // Unrelated global load/store that merely reuses #blocked; not a dot operand.
    %u0 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<256x64x!tt.ptr<i8>, #blocked>
    %u1 = tt.load %u0 : tensor<256x64x!tt.ptr<i8>, #blocked>
    %u2 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<256x64x!tt.ptr<i8>, #blocked>
    tt.store %u2, %u1 : tensor<256x64x!tt.ptr<i8>, #blocked>
    tt.return %6 : tensor<128x64xi32, #blockedA>
  }
}

// -----

// Two loads share the same #blocked encoding, but only the first (64x64) tiles
// under the redistributed [4, 1] layout. The pass bails on the second (2x64).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedA = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedA2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @shared_encoding_guard_defeated
  // CHECK-DAG:     tt.load {{.*}}tensor<64x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     tt.load {{.*}}tensor<2x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
  // The rewrite is weighed on a clone that then replaces the module, and the
  // clone's diagnostics reach the same engine as the module's. Pin that the
  // second gather's bail-out is therefore reported exactly once.
  // CONFLICT-COUNT-1: warps do not tile the reduction dim
  // CONFLICT-NOT:     warps do not tile the reduction dim
  tt.func @shared_encoding_guard_defeated(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i8>, %arg3: !tt.ptr<i8>) -> (tensor<128x64xi32, #blockedA>, tensor<128x64xi32, #blockedA2>) {
    %cst = arith.constant dense<0> : tensor<128x64xi32, #blockedA>
    %cst2 = arith.constant dense<0> : tensor<128x64xi32, #blockedA2>
    // dot 1: B1 = 64x64 gather (#blocked) -> tiles under [4, 1].
    %0 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blockedA>
    %1 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blocked>
    %2 = tt.load %0 : tensor<128x64x!tt.ptr<i8>, #blockedA>
    %3 = tt.load %1 : tensor<64x64x!tt.ptr<i8>, #blocked>
    %4 = ttg.convert_layout %2 : tensor<128x64xi8, #blockedA> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>>
    %5 = ttg.convert_layout %3 : tensor<64x64xi8, #blocked> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>>
    %6 = tt.dot %4, %5, %cst : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>> -> tensor<128x64xi32, #blockedA>
    // dot 2: B2 = 2x64 gather (same #blocked) -> does NOT tile under [4, 1].
    %10 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<128x2x!tt.ptr<i8>, #blockedA2>
    %11 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<2x64x!tt.ptr<i8>, #blocked>
    %12 = tt.load %10 : tensor<128x2x!tt.ptr<i8>, #blockedA2>
    %13 = tt.load %11 : tensor<2x64x!tt.ptr<i8>, #blocked>
    %14 = ttg.convert_layout %12 : tensor<128x2xi8, #blockedA2> -> tensor<128x2xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedA2}>>
    %15 = ttg.convert_layout %13 : tensor<2x64xi8, #blocked> -> tensor<2x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedA2}>>
    %16 = tt.dot %14, %15, %cst2 : tensor<128x2xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedA2}>> * tensor<2x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedA2}>> -> tensor<128x64xi32, #blockedA2>
    tt.return %6, %16 : tensor<128x64xi32, #blockedA>, tensor<128x64xi32, #blockedA2>
  }
}

// -----

// Gather load feeding from an scf.for loop-carried value: the gather
// buffer_load's pointer/mask are built inside a loop from an scf.for iter_arg
// (broadcast in the loop body). The backward slice of the load therefore
// crosses the loop boundary through block arguments, so the pass must retype
// the whole loop-carried chain.

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [16, 0], [32, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [8, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @gather_load_through_loop_iter_arg
  // CHECK-DAG:     amdg.buffer_load {{.*}} : tensor<64x64xi8, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     arith.divui {{.*}} : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     scf.for {{.*}} -> ({{.*}}, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>)
  // CHECK-DAG:     tt.broadcast {{.*}} : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     arith.addi {{.*}} : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // 
  // Nothing is left behind at the old [2, 2] layout.
  //   CHECK-NOT:   warpsPerCTA = [2, 2]
  tt.func @gather_load_through_loop_iter_arg(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>) -> tensor<128x64xi32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c36_i32 = arith.constant 36 : i32
    %cst = arith.constant dense<0> : tensor<128x64xi32, #blocked>
    %cst_0 = arith.constant dense<200> : tensor<64x64xi32, #blocked1>
    %cst_1 = arith.constant dense<1> : tensor<64x1xi32, #blocked1>
    %cst_2 = arith.constant dense<9> : tensor<64x1xi32, #blocked1>
    %cst_3 = arith.constant dense<0> : tensor<128x64xi32, #mma>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %2 = arith.divui %1, %cst_2 : tensor<64x1xi32, #blocked1>
    %3:2 = scf.for %arg2 = %c0_i32 to %c36_i32 step %c1_i32 iter_args(%arg3 = %cst_3, %arg4 = %2) -> (tensor<128x64xi32, #mma>, tensor<64x1xi32, #blocked1>)  : i32 {
      %4 = tt.broadcast %arg4 : tensor<64x1xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
      %5 = arith.cmpi ult, %4, %cst_0 : tensor<64x64xi32, #blocked1>
      %6 = amdg.buffer_load %arg0[%cst] : tensor<128x64xi8, #blocked>
      %7 = amdg.buffer_load %arg1[%4], %5 : tensor<64x64xi8, #blocked1>
      %8 = amdg.in_thread_transpose %7 : tensor<64x64xi8, #blocked1> -> tensor<64x64xi8, #linear>
      %9 = ttg.convert_layout %6 : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %10 = ttg.convert_layout %8 : tensor<64x64xi8, #linear> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %11 = tt.dot %9, %10, %arg3 : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xi32, #mma>
      %12 = arith.addi %arg4, %cst_1 : tensor<64x1xi32, #blocked1>
      scf.yield %11, %12 : tensor<128x64xi32, #mma>, tensor<64x1xi32, #blocked1>
    }
    tt.return %3#0 : tensor<128x64xi32, #mma>
  }
}

// -----

// If we have a loop carried value, but its not affected by the rewrite,
// we don't bail out and we actually run the pass on such IR.

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @gather_load_through_scalar_loop_iter_arg
  // CHECK-DAG:     tt.splat {{.*}} : i32 -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     amdg.buffer_load {{.*}} : tensor<64x64xi8, #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     scf.for {{.*}} -> ({{.*}}, i32)
  // CHECK-DAG:     tt.return %{{.*}}, %{{.*}}#1 : tensor<128x64xi32, {{.*}}>, i32
  tt.func @gather_load_through_scalar_loop_iter_arg(%arg0: !tt.ptr<i8>) -> (tensor<128x64xi32, #mma>, i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c36_i32 = arith.constant 36 : i32
    %cst = arith.constant dense<0> : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %cst_0 = arith.constant dense<0> : tensor<128x64xi32, #mma>
    %0:2 = scf.for %arg1 = %c0_i32 to %c36_i32 step %c1_i32 iter_args(%arg2 = %cst_0, %arg3 = %c0_i32) -> (tensor<128x64xi32, #mma>, i32)  : i32 {
      %1 = tt.splat %arg3 : i32 -> tensor<64x64xi32, #blocked>
      %2 = amdg.buffer_load %arg0[%1] : tensor<64x64xi8, #blocked>
      %3 = ttg.convert_layout %2 : tensor<64x64xi8, #blocked> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %4 = tt.dot %cst, %3, %arg2 : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xi32, #mma>
      %5 = arith.addi %arg3, %c1_i32 : i32
      scf.yield %4, %5 : tensor<128x64xi32, #mma>, i32
    }
    tt.return %0#0, %0#1 : tensor<128x64xi32, #mma>, i32
  }
}

// -----

// Same structure as above, but here the loop carried value has a layout
// (%cst_1, with the #blocked layout being rewritten) instead of a scalar.
// Retyping it would also retype the loop result, malforming the IR,
// so the pass bails.

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @loop_carried_offsets_read_after_loop
  //       CHECK:   warpsPerCTA = [2, 2]
  //   CHECK-NOT:   warpsPerCTA = [4, 1]
  //       CHECK:   tt.return
  tt.func @loop_carried_offsets_read_after_loop(%arg0: !tt.ptr<i8>) -> (tensor<128x64xi32, #mma>, tensor<64x1xi32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c36_i32 = arith.constant 36 : i32
    %cst = arith.constant dense<0> : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %cst_0 = arith.constant dense<0> : tensor<128x64xi32, #mma>
    %cst_1 = arith.constant dense<0> : tensor<64x1xi32, #blocked>
    %cst_2 = arith.constant dense<1> : tensor<64x1xi32, #blocked>
    %0:2 = scf.for %arg1 = %c0_i32 to %c36_i32 step %c1_i32 iter_args(%arg2 = %cst_0, %arg3 = %cst_1) -> (tensor<128x64xi32, #mma>, tensor<64x1xi32, #blocked>)  : i32 {
      %1 = tt.broadcast %arg3 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
      %2 = amdg.buffer_load %arg0[%1] : tensor<64x64xi8, #blocked>
      %3 = ttg.convert_layout %2 : tensor<64x64xi8, #blocked> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %4 = tt.dot %cst, %3, %arg2 : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xi32, #mma>
      %5 = arith.addi %arg3, %cst_2 : tensor<64x1xi32, #blocked>
      scf.yield %4, %5 : tensor<128x64xi32, #mma>, tensor<64x1xi32, #blocked>
    }
    tt.return %0#0, %0#1 : tensor<128x64xi32, #mma>, tensor<64x1xi32, #blocked>
  }
}

// -----

// Multiple dots (as non-power-of-two tile decomposition produces: one dot per
// sub-tile). Each load is redistributed according to the specific dot operand
// it feeds.

#blockedGather = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedFree = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @two_dots_shared_tile_shape
  // dot 1 gather B (64x64) is collapsed onto K -> warpsPerCTA = [4, 1].
  // CHECK-DAG:     tt.load {{.*}}tensor<64x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // dot 2 gather B (64x32) is likewise collapsed onto K -> warpsPerCTA = [4, 1].
  // CHECK-DAG:     tt.load {{.*}}tensor<64x32x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // The free A operands keep their layout, including dot 2's 64x64 A that shares
  // the gather B tile shape but is contiguous in K.
  // CHECK-DAG:     tt.load {{.*}}tensor<32x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>>
  // CHECK-DAG:     tt.load {{.*}}tensor<64x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>>
  tt.func @two_dots_shared_tile_shape(%aptr1: !tt.ptr<i8>, %bptr1: !tt.ptr<i8>, %aptr2: !tt.ptr<i8>, %bptr2: !tt.ptr<i8>) -> (tensor<32x64xi32, #blockedFree>, tensor<64x32xi32, #blockedFree>) {
    %c1 = arith.constant dense<0> : tensor<32x64xi32, #blockedFree>
    %c2 = arith.constant dense<0> : tensor<64x32xi32, #blockedFree>
    // dot 1: A1 (32x64) free, B1 (64x64) gather.
    %sa1 = tt.splat %aptr1 : !tt.ptr<i8> -> tensor<32x64x!tt.ptr<i8>, #blockedFree>
    %sb1 = tt.splat %bptr1 : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blockedGather>
    %la1 = tt.load %sa1 : tensor<32x64x!tt.ptr<i8>, #blockedFree>
    %lb1 = tt.load %sb1 : tensor<64x64x!tt.ptr<i8>, #blockedGather>
    %ca1 = ttg.convert_layout %la1 : tensor<32x64xi8, #blockedFree> -> tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedFree}>>
    %cb1 = ttg.convert_layout %lb1 : tensor<64x64xi8, #blockedGather> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedFree}>>
    %d1 = tt.dot %ca1, %cb1, %c1 : tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedFree}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedFree}>> -> tensor<32x64xi32, #blockedFree>
    // dot 2: A2 (64x64) free -- same tile shape as dot 1's gather B -- B2 (64x32) gather.
    %sa2 = tt.splat %aptr2 : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blockedFree>
    %sb2 = tt.splat %bptr2 : !tt.ptr<i8> -> tensor<64x32x!tt.ptr<i8>, #blockedGather>
    %la2 = tt.load %sa2 : tensor<64x64x!tt.ptr<i8>, #blockedFree>
    %lb2 = tt.load %sb2 : tensor<64x32x!tt.ptr<i8>, #blockedGather>
    %ca2 = ttg.convert_layout %la2 : tensor<64x64xi8, #blockedFree> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedFree}>>
    %cb2 = ttg.convert_layout %lb2 : tensor<64x32xi8, #blockedGather> -> tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedFree}>>
    %d2 = tt.dot %ca2, %cb2, %c2 : tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blockedFree}>> * tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #blockedFree}>> -> tensor<64x32xi32, #blockedFree>
    tt.return %d1, %d2 : tensor<32x64xi32, #blockedFree>, tensor<64x32xi32, #blockedFree>
  }
}

// -----

// Conflict: one load feeds both dot operands, so it is the A operand (reduction
// dim 1, contiguous) and the B operand (reduction dim 0, strided) of the same
// dot at once. The two roles want opposite reduction dims, so the pass cannot
// pick one and bails, emitting a diagnostic.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @conflicting_operand_roles
  // CHECK:         tt.load {{.*}}tensor<64x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
  // CHECK-NOT:     warpsPerCTA = [4, 1]
  // CONFLICT-COUNT-1: load feeds dot operands with conflicting reduction dims
  // CONFLICT-NOT:     load feeds dot operands with conflicting reduction dims
  tt.func @conflicting_operand_roles(%ptr: !tt.ptr<i8>) -> tensor<64x64xi32, #blocked> {
    %cst = arith.constant dense<0> : tensor<64x64xi32, #blocked>
    %s = tt.splat %ptr : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blocked>
    %l = tt.load %s : tensor<64x64x!tt.ptr<i8>, #blocked>
    %a = ttg.convert_layout %l : tensor<64x64xi8, #blocked> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %l : tensor<64x64xi8, #blocked> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %d = tt.dot %a, %b, %cst : tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xi32, #blocked>
    tt.return %d : tensor<64x64xi32, #blocked>
  }
}

// -----

// Shared producer escaping the scope. In a square (64x64) dot the free A load
// and the gather B load carry the same encoding, and CSE has merged their
// `dense<true>` masks into one constant feeding both. Redistributing B rewrites
// that constant in place, which would also retype the free A load's mask and
// leave A inconsistent. We should bail and leave B unchanged rather than corrupting A.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @shared_mask_escapes_scope
  // CHECK-DAG:     tt.load {{.*}}tensor<64x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
  // CHECK:         warpsPerCTA = [2, 2]
  tt.func @shared_mask_escapes_scope(%pa: !tt.ptr<i8>, %pb: !tt.ptr<i8>) -> tensor<64x64xi32, #blocked> {
    %cst = arith.constant dense<0> : tensor<64x64xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<64x64xi1, #blocked>
    %sa = tt.splat %pa : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blocked>
    %sb = tt.splat %pb : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blocked>
    %la = tt.load %sa, %mask : tensor<64x64x!tt.ptr<i8>, #blocked>
    %lb = tt.load %sb, %mask : tensor<64x64x!tt.ptr<i8>, #blocked>
    %ca = ttg.convert_layout %la : tensor<64x64xi8, #blocked> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %cb = ttg.convert_layout %lb : tensor<64x64xi8, #blocked> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %d = tt.dot %ca, %cb, %cst : tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xi32, #blocked>
    tt.return %d : tensor<64x64xi32, #blocked>
  }
}

// -----

// Fused kernel: the dot operand is not produced by a load but by a fusion
// prologue (here an arith.select standing in for a fused batchnorm/where) that
// consumes the gather. The walk back from the dot operand ends on the select,
// so the pass anchors on it rather than insisting on reaching a load. The new
// encoding still reaches the load and the prologue's constants, because they
// are all in the anchor's backward slice.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @gather_through_fusion_prologue
  // CHECK-DAG:     tt.load {{.*}}tensor<64x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     arith.select {{.*}} : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     arith.constant dense<true> : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     tt.load {{.*}}tensor<128x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>>
  tt.func @gather_through_fusion_prologue(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>) -> tensor<128x64xi32, #blocked> {
    %cst = arith.constant dense<0> : tensor<64x64xi8, #blocked1>
    %cstMask = arith.constant dense<true> : tensor<64x64xi1, #blocked1>
    %cstAcc = arith.constant dense<0> : tensor<128x64xi32, #blocked>
    %0 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blocked1>
    %2 = tt.load %0 : tensor<128x64x!tt.ptr<i8>, #blocked>
    %3 = tt.load %1 : tensor<64x64x!tt.ptr<i8>, #blocked1>
    %sel = arith.select %cstMask, %3, %cst : tensor<64x64xi1, #blocked1>, tensor<64x64xi8, #blocked1>
    %4 = ttg.convert_layout %2 : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %5 = ttg.convert_layout %sel : tensor<64x64xi8, #blocked1> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %6 = tt.dot %4, %5, %cstAcc : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xi32, #blocked>
    tt.return %6 : tensor<128x64xi32, #blocked>
  }
}

// -----

// A prologue ending in a math library call. tt.extern_elementwise is how an
// OCML symbol is carried at this level, and it is accepted as long as its marked as pure.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedA = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @gather_through_pure_ocml_call
  // CHECK-DAG:     tt.load {{.*}}tensor<32x64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK-DAG:     tt.extern_elementwise {{.*}}__ocml_exp_f32{{.*}} -> tensor<32x64xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
  tt.func @gather_through_pure_ocml_call(%pa: !tt.ptr<f32>, %pb: !tt.ptr<f32>) -> tensor<128x64xf32, #blockedA> {
    %acc = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blockedA>
    %sa = tt.splat %pa : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blockedA>
    %sb = tt.splat %pb : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %la = tt.load %sa : tensor<128x32x!tt.ptr<f32>, #blockedA>
    %lb = tt.load %sb : tensor<32x64x!tt.ptr<f32>, #blocked>
    %e = tt.extern_elementwise %lb {libname = "libdevice", libpath = "", symbol = "__ocml_exp_f32", pure = true} : (tensor<32x64xf32, #blocked>) -> tensor<32x64xf32, #blocked>
    %ca = ttg.convert_layout %la : tensor<128x32xf32, #blockedA> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>>
    %cb = ttg.convert_layout %e : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>>
    %d = tt.dot %ca, %cb, %acc : tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>> -> tensor<128x64xf32, #blockedA>
    tt.return %d : tensor<128x64xf32, #blockedA>
  }
}

// -----

// The same call marked impure is refused by the memory-effect guard

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedA = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @gather_through_impure_ocml_call
  //   CHECK-NOT:     warpsPerCTA = [4, 1]
  tt.func @gather_through_impure_ocml_call(%pa: !tt.ptr<f32>, %pb: !tt.ptr<f32>) -> tensor<128x64xf32, #blockedA> {
    %acc = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blockedA>
    %sa = tt.splat %pa : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blockedA>
    %sb = tt.splat %pb : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %la = tt.load %sa : tensor<128x32x!tt.ptr<f32>, #blockedA>
    %lb = tt.load %sb : tensor<32x64x!tt.ptr<f32>, #blocked>
    %e = tt.extern_elementwise %lb {libname = "libdevice", libpath = "", symbol = "__ocml_exp_f32", pure = false} : (tensor<32x64xf32, #blocked>) -> tensor<32x64xf32, #blocked>
    %ca = ttg.convert_layout %la : tensor<128x32xf32, #blockedA> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>>
    %cb = ttg.convert_layout %e : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>>
    %d = tt.dot %ca, %cb, %acc : tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>> -> tensor<128x64xf32, #blockedA>
    tt.return %d : tensor<128x64xf32, #blockedA>
  }
}

// -----

// Reject tranpose 
//
// The guard that rejects it today is the trait check: tt.trans carries
// SameOperandsAndResultElementType, and neither SameOperandsAndResultEncoding
// nor Elementwise, so it never reaches the encoding and shape comparison.

#blockedT = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedA = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @anchor_rejects_trans
  //   CHECK-NOT:     warpsPerCTA = [4, 1]
  tt.func @anchor_rejects_trans(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>) -> tensor<128x64xf32, #blockedA> {
    %acc = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blockedA>
    %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blockedA>
    %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blockedT>
    %la = tt.load %sa : tensor<128x32x!tt.ptr<f16>, #blockedA>
    %lb = tt.load %sb : tensor<64x32x!tt.ptr<f16>, #blockedT>
    %tb = tt.trans %lb {order = array<i32: 1, 0>} : tensor<64x32xf16, #blockedT> -> tensor<32x64xf16, #blocked>
    %ca = ttg.convert_layout %la : tensor<128x32xf16, #blockedA> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>>
    %cb = ttg.convert_layout %tb : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>>
    %d = tt.dot %ca, %cb, %acc : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>> -> tensor<128x64xf32, #blockedA>
    tt.return %d : tensor<128x64xf32, #blockedA>
  }
}

// -----

// Same hazard for tt.reshape, and the same guard turns it away: it carries
// SameOperandsAndResultElementType and is not Elementwise either. Were the
// trait check relaxed, the shape comparison would still catch a reshape that
// changes the shape, as this one does -- but not an identity reshape, nor a
// rank-1 tt.trans, whose encodings a transpose leaves alone.

#blockedR = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedA = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @anchor_rejects_reshape
  //   CHECK-NOT:     warpsPerCTA = [4, 1]
  tt.func @anchor_rejects_reshape(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>) -> tensor<128x64xf32, #blockedA> {
    %acc = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blockedA>
    %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blockedA>
    %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<16x128x!tt.ptr<f16>, #blockedR>
    %la = tt.load %sa : tensor<128x32x!tt.ptr<f16>, #blockedA>
    %lb = tt.load %sb : tensor<16x128x!tt.ptr<f16>, #blockedR>
    %rb = tt.reshape %lb allow_reorder : tensor<16x128xf16, #blockedR> -> tensor<32x64xf16, #blocked>
    %ca = ttg.convert_layout %la : tensor<128x32xf16, #blockedA> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>>
    %cb = ttg.convert_layout %rb : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>>
    %d = tt.dot %ca, %cb, %acc : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>> -> tensor<128x64xf32, #blockedA>
    tt.return %d : tensor<128x64xf32, #blockedA>
  }
}

// -----

// A dot whose blocked result feeds another dot's operand. Anchoring on the
// producer would redistribute its warps onto the consumer's reduction dim:
// valid IR, but not a trade this pass is in a position to make. The producer's
// own operands are deliberately reduction-contiguous, so it has no gather of
// its own and any [4, 1] here would have to come from anchoring on it.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedK = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [0, 1]}>
#blockedA = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @anchor_rejects_producer_dot
  //   CHECK-NOT:     warpsPerCTA = [4, 1]
  tt.func @anchor_rejects_producer_dot(%pa: !tt.ptr<f32>, %pb: !tt.ptr<f32>) -> tensor<128x64xf32, #blockedA> {
    %acc = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blockedA>
    %acc1 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #blocked>
    %s0 = tt.splat %pa : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>, #blocked>
    %s1 = tt.splat %pb : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #blockedK>
    %l0 = tt.load %s0 : tensor<32x16x!tt.ptr<f32>, #blocked>
    %l1 = tt.load %s1 : tensor<16x64x!tt.ptr<f32>, #blockedK>
    %c0 = ttg.convert_layout %l0 : tensor<32x16xf32, #blocked> -> tensor<32x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %c1 = ttg.convert_layout %l1 : tensor<16x64xf32, #blockedK> -> tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %d1 = tt.dot %c0, %c1, %acc1 : tensor<32x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf32, #blocked>
    %sa = tt.splat %pa : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blockedA>
    %la = tt.load %sa : tensor<128x32x!tt.ptr<f32>, #blockedA>
    %ca = ttg.convert_layout %la : tensor<128x32xf32, #blockedA> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>>
    %cb = ttg.convert_layout %d1 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>>
    %d2 = tt.dot %ca, %cb, %acc : tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>> -> tensor<128x64xf32, #blockedA>
    tt.return %d2 : tensor<128x64xf32, #blockedA>
  }
}

// -----

// A region-carrying op. The backward slice does not enter the region, so
// retyping the result would leave the yielded value on the old encoding and
// produce invalid IR.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blockedA = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @anchor_rejects_region_op
  //   CHECK-NOT:     warpsPerCTA = [4, 1]
  tt.func @anchor_rejects_region_op(%pa: !tt.ptr<f16>, %pb: !tt.ptr<f16>, %cond: i1) -> tensor<128x64xf32, #blockedA> {
    %acc = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blockedA>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #blocked>
    %sa = tt.splat %pa : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blockedA>
    %sb = tt.splat %pb : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %la = tt.load %sa : tensor<128x32x!tt.ptr<f16>, #blockedA>
    %lb = tt.load %sb : tensor<32x64x!tt.ptr<f16>, #blocked>
    %guarded = scf.if %cond -> tensor<32x64xf16, #blocked> {
      scf.yield %lb : tensor<32x64xf16, #blocked>
    } else {
      scf.yield %zero : tensor<32x64xf16, #blocked>
    }
    %ca = ttg.convert_layout %la : tensor<128x32xf16, #blockedA> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>>
    %cb = ttg.convert_layout %guarded : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>>
    %d = tt.dot %ca, %cb, %acc : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blockedA}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blockedA}>> -> tensor<128x64xf32, #blockedA>
    tt.return %d : tensor<128x64xf32, #blockedA>
  }
}

// -----

// The B operand is read from a mutable shared buffer that is allocated empty
// (no source tensor), as pipelining leaves behind. There is no distributed
// producer to anchor on, so the walk has to give up instead of dereferencing
// the missing source. The kernel must come through unchanged.

#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.amd_rotating_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @gather_through_sourceless_alloc
  // CHECK:         ttg.local_alloc : () -> !ttg.memdesc<64x64xi8, #ttg.amd_rotating_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [0, 1]}>, #ttg.shared_memory, mutable>
  // CHECK:         ttg.local_load
  tt.func @gather_through_sourceless_alloc(%argA: !tt.ptr<i8>) -> tensor<128x64xi32, #mma> {
    %offA = arith.constant dense<0> : tensor<128x64xi32, #blocked1>
    %cstOut = arith.constant dense<0> : tensor<128x64xi32, #mma>
    %a = amdg.buffer_load %argA[%offA] : tensor<128x64xi8, #blocked1>
    %aAlloc = ttg.local_alloc %a : (tensor<128x64xi8, #blocked1>) -> !ttg.memdesc<128x64xi8, #shared, #smem>
    %aLoad = ttg.local_load %aAlloc : !ttg.memdesc<128x64xi8, #shared, #smem> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %bAlloc = ttg.local_alloc : () -> !ttg.memdesc<64x64xi8, #shared1, #smem, mutable>
    %bLoad = ttg.local_load %bAlloc : !ttg.memdesc<64x64xi8, #shared1, #smem, mutable> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %out = tt.dot %aLoad, %bLoad, %cstOut : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x64xi32, #mma>
    tt.return %out : tensor<128x64xi32, #mma>
  }
}

// -----

// Shared-memory gate, permitting case. Same kernel as the first test, but the
// module now carries a target, so the pass can size shared memory and actually
// compares the footprints instead of falling back to applying the rewrite
// blind. The gather's layout is far enough from the dot operand's that the
// conversion between them already needs a 2 KB staging buffer, and it still
// needs exactly 2 KB after the warps move onto K, so the rewrite is kept.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @shared_memory_flat_keeps_rewrite
  // CHECK:         tt.load {{.*}}tensor<64x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
  tt.func @shared_memory_flat_keeps_rewrite(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>) -> tensor<128x64xi32, #blocked> {
    %cst = arith.constant dense<0> : tensor<64x64xi8, #blocked1>
    %cst_0 = arith.constant dense<true> : tensor<64x64xi1, #blocked1>
    %cst_1 = arith.constant dense<0> : tensor<128x64xi32, #blocked>
    %0 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blocked1>
    %2 = tt.load %0 : tensor<128x64x!tt.ptr<i8>, #blocked>
    %3 = tt.load %1, %cst_0, %cst : tensor<64x64x!tt.ptr<i8>, #blocked1>
    %4 = ttg.convert_layout %2 : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %5 = ttg.convert_layout %3 : tensor<64x64xi8, #blocked1> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %6 = tt.dot %4, %5, %cst_1 : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xi32, #blocked>
    tt.return %6 : tensor<128x64xi32, #blocked>
  }
}

// -----

// Shared-memory gate, declining case. Here the gather is loaded in exactly the
// layout the dot wants, so feeding the dot costs no shared memory at all.
// Moving the warps onto K would pull the gather away from that layout and force
// the conversion through a 2 KB staging buffer (0 -> 2048 bytes for this 64x64
// i8 tile, confirmed with -allocate-amdgpu-shared-memory). Trading shared
// memory -- and with it occupancy -- for the layout change is not a deal we
// want, so the pass throws the rewrite away and emits the original kernel.
//
// Keep this case in sync with @gather_matching_dot_layout_no_target below: the
// two are the same kernel and differ only in whether the module names a target,
// which is what makes the gate observable here.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @shared_memory_growth_declines_rewrite
  // CHECK:         tt.load {{.*}}tensor<64x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>>
  // CHECK-NOT:     warpsPerCTA = [4, 1]
  tt.func @shared_memory_growth_declines_rewrite(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>) -> tensor<128x64xi32, #blocked> {
    %cst = arith.constant dense<0> : tensor<64x64xi8, #blocked>
    %cst_0 = arith.constant dense<true> : tensor<64x64xi1, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x64xi32, #blocked>
    %0 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blocked>
    %2 = tt.load %0 : tensor<128x64x!tt.ptr<i8>, #blocked>
    %3 = tt.load %1, %cst_0, %cst : tensor<64x64x!tt.ptr<i8>, #blocked>
    %4 = ttg.convert_layout %2 : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %5 = ttg.convert_layout %3 : tensor<64x64xi8, #blocked> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %6 = tt.dot %4, %5, %cst_1 : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xi32, #blocked>
    tt.return %6 : tensor<128x64xi32, #blocked>
  }
}

// -----

// The kernel that the gate declines above, with the target dropped. Without a
// target there is no way to size shared memory, so the gate cannot weigh the
// rewrite and the pass applies it. This pins down that it really is the
// footprint comparison rejecting the case above, and not the gather walk or one
// of the other bail-outs: the only difference between the two kernels is the
// ttg.target attribute.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @gather_matching_dot_layout_no_target
  // CHECK:         tt.load {{.*}}tensor<64x64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
  tt.func @gather_matching_dot_layout_no_target(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>) -> tensor<128x64xi32, #blocked> {
    %cst = arith.constant dense<0> : tensor<64x64xi8, #blocked>
    %cst_0 = arith.constant dense<true> : tensor<64x64xi1, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x64xi32, #blocked>
    %0 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<64x64x!tt.ptr<i8>, #blocked>
    %2 = tt.load %0 : tensor<128x64x!tt.ptr<i8>, #blocked>
    %3 = tt.load %1, %cst_0, %cst : tensor<64x64x!tt.ptr<i8>, #blocked>
    %4 = ttg.convert_layout %2 : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %5 = ttg.convert_layout %3 : tensor<64x64xi8, #blocked> -> tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %6 = tt.dot %4, %5, %cst_1 : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xi32, #blocked>
    tt.return %6 : tensor<128x64xi32, #blocked>
  }
}
