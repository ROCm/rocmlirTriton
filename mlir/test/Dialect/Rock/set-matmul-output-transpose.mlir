// Unit tests for the rocmlirTriton rock-set-matmul-output-transpose pass.
//
// The pass runs right after tritonamdgpu-accelerate-matmul. For each
// accelerator dot carrying the rock.o_transposed metadata it overrides the
// WMMA result layout's isTranspose flag so the consuming epilogue store can be
// coalesced/vectorized. The version-dependent mapping is:
//   row-major output (oTransposed = false): isTranspose = (version > 1)
//   col-major output (oTransposed = true):  isTranspose = (version == 1)

// RUN: rocmlir-opt -rock-set-matmul-output-transpose --mlir-print-local-scope --split-input-file %s | FileCheck %s

// Column-major output (oTransposed = true) on WMMA v2 wants isTranspose = false,
// so the v2 default (isTranspose = true) is flipped. The dot result is rebuilt
// with the new layout and convert_layout ops bridge the operands/result. The
// discardable metadata is left in place once consumed.

#wmma2 = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[0, 1], [0, 2]]}}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @wmma_flip
  //      CHECK:   %[[D:.*]] = tt.dot
  // CHECK-SAME:     rock.o_transposed = #rock.o_transposed<true>
  // CHECK-SAME:     -> tensor<128x256xf32, #ttg.amd_wmma<{version = 2, isTranspose = false,
  //      CHECK:   ttg.convert_layout %[[D]]
  // CHECK-SAME:     -> tensor<128x256xf32, #ttg.amd_wmma<{version = 2, isTranspose = true,
  tt.func public @wmma_flip(
      %a: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma2, kWidth = 16}>>,
      %b: tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma2, kWidth = 16}>>,
      %c: tensor<128x256xf32, #wmma2>) -> tensor<128x256xf32, #wmma2> {
    %d = tt.dot %a, %b, %c {rock.o_transposed = #rock.o_transposed<true>}
       : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma2, kWidth = 16}>>
       * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma2, kWidth = 16}>>
      -> tensor<128x256xf32, #wmma2>
    tt.return %d : tensor<128x256xf32, #wmma2>
  }
}

// -----

// Row-major output (oTransposed = false) on WMMA v2 wants isTranspose = true,
// which already matches the default: the dot is left untouched (no bridging
// convert_layout ops, no layout change).

#wmma2 = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[0, 1], [0, 2]]}}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @wmma_no_change
  //  CHECK-NOT:   ttg.convert_layout
  //      CHECK:   tt.dot
  // CHECK-SAME:     rock.o_transposed = #rock.o_transposed<false>
  // CHECK-SAME:     -> tensor<128x256xf32, #ttg.amd_wmma<{version = 2, isTranspose = true,
  tt.func public @wmma_no_change(
      %a: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma2, kWidth = 16}>>,
      %b: tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma2, kWidth = 16}>>,
      %c: tensor<128x256xf32, #wmma2>) -> tensor<128x256xf32, #wmma2> {
    %d = tt.dot %a, %b, %c {rock.o_transposed = #rock.o_transposed<false>}
       : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma2, kWidth = 16}>>
       * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma2, kWidth = 16}>>
      -> tensor<128x256xf32, #wmma2>
    tt.return %d : tensor<128x256xf32, #wmma2>
  }
}

// -----

// Column-major output on WMMA v1 wants isTranspose = true; the v1 input here
// defaults to false, so it is flipped.

#wmma1 = #ttg.amd_wmma<{version = 1, isTranspose = false, ctaLayout = {warp = [[0, 1], [0, 2]]}}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @wmma_v1_flip
  //      CHECK:   %[[D:.*]] = tt.dot
  // CHECK-SAME:     rock.o_transposed = #rock.o_transposed<true>
  // CHECK-SAME:     -> tensor<128x256xf32, #ttg.amd_wmma<{version = 1, isTranspose = true,
  //      CHECK:   ttg.convert_layout %[[D]]
  // CHECK-SAME:     -> tensor<128x256xf32, #ttg.amd_wmma<{version = 1, isTranspose = false,
  tt.func public @wmma_v1_flip(
      %a: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma1, kWidth = 16}>>,
      %b: tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma1, kWidth = 16}>>,
      %c: tensor<128x256xf32, #wmma1>) -> tensor<128x256xf32, #wmma1> {
    %d = tt.dot %a, %b, %c {rock.o_transposed = #rock.o_transposed<true>}
       : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma1, kWidth = 16}>>
       * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma1, kWidth = 16}>>
      -> tensor<128x256xf32, #wmma1>
    tt.return %d : tensor<128x256xf32, #wmma1>
  }
}

// -----

// Non-WMMA accelerator dots are not adjusted (only WMMA is handled for now); the
// dot, including its metadata, is left untouched.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @non_wmma
  //  CHECK-NOT:   ttg.convert_layout
  //      CHECK:   tt.dot
  // CHECK-SAME:     rock.o_transposed = #rock.o_transposed<true>
  // CHECK-SAME:     -> tensor<128x256xf32, #ttg.blocked
  tt.func public @non_wmma(
      %a: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %b: tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %c: tensor<128x256xf32, #blocked>) -> tensor<128x256xf32, #blocked> {
    %d = tt.dot %a, %b, %c {rock.o_transposed = #rock.o_transposed<true>}
       : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
       * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      -> tensor<128x256xf32, #blocked>
    tt.return %d : tensor<128x256xf32, #blocked>
  }
}

// -----

// A WMMA dot without rock.o_transposed metadata is left completely untouched.

#wmma2 = #ttg.amd_wmma<{version = 2, isTranspose = true, ctaLayout = {warp = [[0, 1], [0, 2]]}}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @no_metadata
  //  CHECK-NOT:   ttg.convert_layout
  //      CHECK:   tt.dot
  // CHECK-SAME:     -> tensor<128x256xf32, #ttg.amd_wmma<{version = 2, isTranspose = true,
  tt.func public @no_metadata(
      %a: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma2, kWidth = 16}>>,
      %b: tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma2, kWidth = 16}>>,
      %c: tensor<128x256xf32, #wmma2>) -> tensor<128x256xf32, #wmma2> {
    %d = tt.dot %a, %b, %c
       : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma2, kWidth = 16}>>
       * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma2, kWidth = 16}>>
      -> tensor<128x256xf32, #wmma2>
    tt.return %d : tensor<128x256xf32, #wmma2>
  }
}
