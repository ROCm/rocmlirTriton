// Unit tests for the arch-dependent group size of rock-verify-packed-upcast-layout.
//
// Where the target has `v_cvt_pk_scale_pk8_*` (gfx1250), Triton lowers a scaled
// FP8 upcast through it and consumes a thread's packed values eight at a time
// instead of four, so a count of four -- fine everywhere else -- is half a
// group here. The FP4 upcast keeps the narrow group even on such a target,
// because its packed operand is half as wide.
//
// RUN: rocmlir-opt -rock-verify-packed-upcast-layout=arch=gfx1250 --split-input-file %s -verify-diagnostics
//
// RUN: rocmlir-opt -rock-verify-packed-upcast-layout=arch=gfx1250 --split-input-file %s -verify-diagnostics \
// RUN:   --mlir-print-ir-after-failure 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NA --implicit-check-not=rock.not_applicable

// #blocked covers 4x32 per CTA, so a 16x32 source repeats it four times along
// dim 0 and leaves each thread with 4 values -- a whole group on every other
// target, but half of one here.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// NA: module attributes {rock.not_applicable
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @scaled_upcast_fp8_half_wide_group_rejected(%arg0: tensor<16x32xf8E4M3FN, #blocked>, %arg1: tensor<16x32xbf16, #blocked>) -> tensor<16x32xbf16, #blocked> {
    // expected-error @+1 {{upcast consumes 8 packed values per thread, but this layout provides 4}}
    %0 = amdg.scaled_upcast_fp8 %arg0 scale %arg1 : tensor<16x32xf8E4M3FN, #blocked>, tensor<16x32xbf16, #blocked> -> tensor<16x32xbf16, #blocked>
    tt.return %0 : tensor<16x32xbf16, #blocked>
  }
}

// -----

// Doubling the source along dim 0 fills the wide group, so the same op is
// accepted.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @scaled_upcast_fp8_wide_group_accepted(%arg0: tensor<32x32xf8E4M3FN, #blocked>, %arg1: tensor<32x32xbf16, #blocked>) -> tensor<32x32xbf16, #blocked> {
    %0 = amdg.scaled_upcast_fp8 %arg0 scale %arg1 : tensor<32x32xf8E4M3FN, #blocked>, tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #blocked>
    tt.return %0 : tensor<32x32xbf16, #blocked>
  }
}

// -----

// The FP4 upcast packs two values per byte, so four bytes already fill the
// conversion operand and the narrow group still applies here.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @scaled_upcast_fp4_narrow_group_accepted(%arg0: tensor<16x32xi8, #blocked>, %arg1: tensor<16x64xbf16, #blocked1>) -> tensor<16x64xbf16, #blocked1> {
    %0 = amdg.scaled_upcast_fp4 %arg0 scale %arg1 {axis = 1 : i32} : tensor<16x32xi8, #blocked>, tensor<16x64xbf16, #blocked1> -> tensor<16x64xbf16, #blocked1>
    tt.return %0 : tensor<16x64xbf16, #blocked1>
  }
}
