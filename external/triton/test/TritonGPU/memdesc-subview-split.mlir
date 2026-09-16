// RUN: triton-opt %s | FileCheck %s


#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#padded = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [256, 128]}>
#crossing = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: memdesc_subslice_spliting
  tt.func public @memdesc_subslice_spliting() {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    %2 = ttg.memdesc_subslice %1 [0, 0]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %3 = ttg.memdesc_subslice %1 [0, 32]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %4 = ttg.memdesc_subslice %1 [0, 64]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %5 = ttg.memdesc_subslice %1 [0, 96]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %6 = ttg.memdesc_subslice %1 [128, 0]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %7 = ttg.memdesc_subslice %1 [128, 32]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %8 = ttg.memdesc_subslice %1 [128, 64]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %9 = ttg.memdesc_subslice %1 [128, 96]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>

    %padded = ttg.local_alloc : () -> !ttg.memdesc<1x256x128xf16, #padded, #smem, mutable>
    %padded_indexed_explicit_alloc_shape = ttg.memdesc_index %padded[%c0_i32] : !ttg.memdesc<1x256x128xf16, #padded, #smem, mutable> -> !ttg.memdesc<256x128xf16, #padded, #smem, mutable>
    %10 = ttg.memdesc_subslice %padded_indexed_explicit_alloc_shape [128, 96]  : !ttg.memdesc<256x128xf16, #padded, #smem, mutable> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable, 256x128>
    %padded_indexed_implicit_alloc_shape = ttg.memdesc_index %padded[%c0_i32] : !ttg.memdesc<1x256x128xf16, #padded, #smem, mutable> -> !ttg.memdesc<256x128xf16, #padded, #smem, mutable>
    %11 = ttg.memdesc_subslice %padded_indexed_implicit_alloc_shape [128, 96]  : !ttg.memdesc<256x128xf16, #padded, #smem, mutable> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable, 256x128>
    tt.return
  }

  // An index names the tile at run time. The swizzle repeats every 16 rows, so
  // these 8-row tiles each hold a different phase, which is a split the static
  // form cannot describe and the index can.
  // CHECK-LABEL: memdesc_subslice_indexed
  tt.func public @memdesc_subslice_indexed(%i: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    // CHECK: ttg.memdesc_subslice %{{.*}}[0, 0] index %{{.*}} : !ttg.memdesc<256x128xf16, {{.*}}> -> !ttg.memdesc<8x128xf16, {{.*}}, 256x128>
    %2 = ttg.memdesc_subslice %1 [0, 0] index %i : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<8x128xf16, #shared, #smem, mutable, 256x128>
    // CHECK: ttg.memdesc_subslice %{{.*}}[0, 0] index %{{.*}} : !ttg.memdesc<256x128xf16, {{.*}}> -> !ttg.memdesc<256x32xf16, {{.*}}, 256x128>
    %3 = ttg.memdesc_subslice %1 [0, 0] index %i : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x32xf16, #shared, #smem, mutable, 256x128>
    tt.return
  }

  // Splitting a 16-column tile into 4-column pieces cuts across this swizzle,
  // so the split lands two bits in the physical offset rather than one. An
  // unpadded shared encoding is a GF(2)-linear map, so the offset of the piece
  // and the offset within it still compose by XOR, which is what the lowering
  // computes, and the split is exact.
  // CHECK-LABEL: memdesc_subslice_across_swizzling_pattern
  tt.func public @memdesc_subslice_across_swizzling_pattern(%arg0: !ttg.memdesc<8x16xf32, #crossing, #smem>) {
    // CHECK: ttg.memdesc_subslice %{{.*}}[0, 4] : !ttg.memdesc<8x16xf32, {{.*}}> -> !ttg.memdesc<8x4xf32, {{.*}}>
    %a = ttg.memdesc_subslice %arg0 [0, 4] : !ttg.memdesc<8x16xf32, #crossing, #smem> -> !ttg.memdesc<8x4xf32, #crossing, #smem, 8x16>
    tt.return
  }
}
