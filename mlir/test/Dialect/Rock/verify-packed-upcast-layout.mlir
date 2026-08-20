// Unit tests for the rock-verify-packed-upcast-layout pass.
//
// The pass rejects `ttg.fp4_to_fp`, `amdg.scaled_upcast_fp4` and
// `amdg.scaled_upcast_fp8` whose source layout leaves a thread with a partial
// group, because the AMD lowerings consume a thread's packed values a whole
// group at a time. These are the cases with the narrow group of four, which
// applies to every op on a target without `v_cvt_pk_scale_pk8_*`; the wider
// group lives in verify-packed-upcast-layout-gfx1250.mlir.
//
// RUN: rocmlir-opt -rock-verify-packed-upcast-layout --split-input-file %s -verify-diagnostics
//
// A rejected configuration must also be marked `rock.not_applicable` so the
// tuning driver skips it instead of reporting a compiler crash.
// RUN: rocmlir-opt -rock-verify-packed-upcast-layout --split-input-file %s -verify-diagnostics \
// RUN:   --mlir-print-ir-after-failure 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NA --implicit-check-not=rock.not_applicable

// #blocked covers 8x32 per CTA (threadsPerWarp = [2, 32] over warpsPerCTA =
// [4, 1]), so a 16x64 source repeats it twice along each dim and leaves each
// thread with 4 packed values. That is a whole group, so the pass accepts it.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @whole_group_accepted(%arg0: tensor<16x64xi8, #blocked>) -> tensor<16x128xbf16, #blocked1> {
    %0 = ttg.fp4_to_fp %arg0 {axis = 1 : i32} : tensor<16x64xi8, #blocked> -> tensor<16x128xbf16, #blocked1>
    tt.return %0 : tensor<16x128xbf16, #blocked1>
  }
}

// -----

// The same layout with a 16x32 source repeats only along dim 0, leaving each
// thread with 2 packed values -- half a group. This is the shape the FP4 GEMM
// reproducer produced, and the lowering used to read past the end of the
// thread's values here.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// NA: module attributes {rock.not_applicable
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @partial_group_rejected(%arg0: tensor<16x32xi8, #blocked>) -> tensor<16x64xbf16, #blocked1> {
    // expected-error @+1 {{upcast consumes 4 packed values per thread, but this layout provides 2}}
    %0 = ttg.fp4_to_fp %arg0 {axis = 1 : i32} : tensor<16x32xi8, #blocked> -> tensor<16x64xbf16, #blocked1>
    tt.return %0 : tensor<16x64xbf16, #blocked1>
  }
}

// -----

// `amdg.scaled_upcast_fp4` groups its packed values the same way, so a
// partial group is rejected there too.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// NA: module attributes {rock.not_applicable
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_upcast_fp4_partial_group_rejected(%arg0: tensor<16x32xi8, #blocked>, %arg1: tensor<16x64xbf16, #blocked1>) -> tensor<16x64xbf16, #blocked1> {
    // expected-error @+1 {{upcast consumes 4 packed values per thread, but this layout provides 2}}
    %0 = amdg.scaled_upcast_fp4 %arg0 scale %arg1 {axis = 1 : i32} : tensor<16x32xi8, #blocked>, tensor<16x64xbf16, #blocked1> -> tensor<16x64xbf16, #blocked1>
    tt.return %0 : tensor<16x64xbf16, #blocked1>
  }
}

// -----

// `amdg.scaled_upcast_fp8` carries one value per byte rather than two and does
// not widen the shape, but its lowering still walks the values four at a time.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// NA: module attributes {rock.not_applicable
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_upcast_fp8_partial_group_rejected(%arg0: tensor<16x32xf8E4M3FN, #blocked>, %arg1: tensor<16x32xbf16, #blocked>) -> tensor<16x32xbf16, #blocked> {
    // expected-error @+1 {{upcast consumes 4 packed values per thread, but this layout provides 2}}
    %0 = amdg.scaled_upcast_fp8 %arg0 scale %arg1 : tensor<16x32xf8E4M3FN, #blocked>, tensor<16x32xbf16, #blocked> -> tensor<16x32xbf16, #blocked>
    tt.return %0 : tensor<16x32xbf16, #blocked>
  }
}

// -----

// A source without an encoding cannot be checked -- the per-thread count is not
// known until a distributed layout has been assigned -- so the pass leaves it
// alone rather than guessing.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @unencoded_source_ignored(%arg0: tensor<16x32xi8>) -> tensor<16x64xbf16> {
    %0 = ttg.fp4_to_fp %arg0 {axis = 1 : i32} : tensor<16x32xi8> -> tensor<16x64xbf16>
    tt.return %0 : tensor<16x64xbf16>
  }
}
