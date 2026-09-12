// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -cse | FileCheck %s

// A 128x1 load and a dot both reach the reduce, and before accelerate-matmul
// has run neither candidate layout is of the preferred kind. The load governs
// 512 bytes across 4 warps of 32 lanes, exactly the one dword per thread that
// is still counted as negligible, so its layout must not be the one the reduce
// adopts: laying the whole tile out one element per thread would put every
// value in the epilogue through shared memory.

// CHECK-DAG: #[[$DOT:.+]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$NARROW:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

#dot = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#narrow = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#generic = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "hip:gfx1100"} {
  // CHECK-LABEL: @reduce_keeps_dot_layout_over_narrow_load
  tt.func @reduce_keeps_dot_layout_over_narrow_load(
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dot}>>,
      %b: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dot}>>,
      %pbias: tensor<128x1x!tt.ptr<f32>, #narrow>)
      -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #generic}>> {
    // The dot result is left alone and the 128x1 load is converted instead.
    // CHECK:      %[[ACC:.*]] = tt.dot
    // CHECK:      %[[LOAD:.*]] = tt.load {{.*}} : tensor<128x1x!tt.ptr<f32>, #[[$NARROW]]>
    // CHECK:      %[[CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<128x1xf32, #[[$NARROW]]> -> tensor<128x1xf32, #[[$DOT]]>
    // CHECK:      %[[BCAST:.*]] = tt.broadcast %[[CVT]]
    // CHECK:      %[[SUM:.*]] = arith.addf %[[ACC]], %[[BCAST]]
    // CHECK:      "tt.reduce"(%[[SUM]])
    // CHECK:      }) : (tensor<128x64xf32, #[[$DOT]]>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[$DOT]]}>>
    %zero = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #dot>
    %acc = tt.dot %a, %b, %zero : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dot}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dot}>> -> tensor<128x64xf32, #dot>
    %bias = tt.load %pbias : tensor<128x1x!tt.ptr<f32>, #narrow>
    %biasg = ttg.convert_layout %bias : tensor<128x1xf32, #narrow> -> tensor<128x1xf32, #generic>
    %bcast = tt.broadcast %biasg : tensor<128x1xf32, #generic> -> tensor<128x64xf32, #generic>
    %accg = ttg.convert_layout %acc : tensor<128x64xf32, #dot> -> tensor<128x64xf32, #generic>
    %sum = arith.addf %accg, %bcast : tensor<128x64xf32, #generic>
    %red = "tt.reduce"(%sum) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %s = arith.addf %x, %y : f32
      tt.reduce.return %s : f32
    }) : (tensor<128x64xf32, #generic>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #generic}>>
    tt.return %red : tensor<128xf32, #ttg.slice<{dim = 1, parent = #generic}>>
  }
}

// -----

// The same epilogue with a load that is not negligible: it governs a full
// 128x64 tile, so the traffic rule does not fire and the layout it anchors is
// left to win the conflict, as it did before that rule existed.

// CHECK-DAG: #[[$WIDE:.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

#dot = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#wide = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#generic = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "hip:gfx1100"} {
  // CHECK-LABEL: @reduce_keeps_wide_load_layout
  tt.func @reduce_keeps_wide_load_layout(
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dot}>>,
      %b: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dot}>>,
      %pbias: tensor<128x64x!tt.ptr<f32>, #wide>)
      -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #generic}>> {
    // CHECK:      "tt.reduce"
    // CHECK:      }) : (tensor<128x64xf32, #[[$WIDE]]>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[$WIDE]]}>>
    %zero = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #dot>
    %acc = tt.dot %a, %b, %zero : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dot}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dot}>> -> tensor<128x64xf32, #dot>
    %bias = tt.load %pbias : tensor<128x64x!tt.ptr<f32>, #wide>
    %biasg = ttg.convert_layout %bias : tensor<128x64xf32, #wide> -> tensor<128x64xf32, #generic>
    %accg = ttg.convert_layout %acc : tensor<128x64xf32, #dot> -> tensor<128x64xf32, #generic>
    %sum = arith.addf %accg, %biasg : tensor<128x64xf32, #generic>
    %red = "tt.reduce"(%sum) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %s = arith.addf %x, %y : f32
      tt.reduce.return %s : f32
    }) : (tensor<128x64xf32, #generic>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #generic}>>
    tt.return %red : tensor<128xf32, #ttg.slice<{dim = 1, parent = #generic}>>
  }
}
