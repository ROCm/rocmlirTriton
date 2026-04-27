// Error tests for rock-lower-stores pass.

// RUN: rocmlir-opt -rock-lower-stores -verify-diagnostics -split-input-file -mlir-print-local-scope %s

// ============================================================
// Error: rock.store source is a block arg — no StoreMarkerOp
// found in the trace-back.
// ============================================================

module {
  func.func @test_no_store_marker(
      %src: tensor<1x16x16xf32>,
      %dest: tensor<1x16x16xf32>) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    // expected-error @+1 {{No StoreMarkerOp found for store}}
    %r = rock.store %src to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }
}

// -----

// ============================================================
// Error: fusion op has a raw block arg operand (not from
// UntileOp or StoreMarkerOp), so convertToTile fails.
// ============================================================

#tmap = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 16 + d3, d2 * 16 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 16} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{1, 16} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 1, 16, 16] -> [1, 16, 16]>

module {
  func.func @test_blockarg_in_fusion(
      %tile: tensor<16x16xf32>,
      %bias: tensor<1x16x16xf32>,
      %dest: tensor<1x16x16xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %sm, %bias : tensor<1x16x16xf32>
    // expected-error @+1 {{Failed to convert to tile}}
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }
}
