// Error tests for rock-insert-output-fusion-loads pass.

// RUN: rocmlir-opt -rock-insert-output-fusion-loads -verify-diagnostics --split-input-file %s

// ============================================================
// Error: rock.kernel function with no StoreMarkerOp.
// ============================================================

module {
  // expected-error @below {{No StoreMarkerOp found}}
  func.func @error_no_store_marker(%a: tensor<16x16xf32>, %dest: tensor<16x16xf32>) -> tensor<16x16xf32> attributes {rock.kernel} {
    %r = rock.store %a to %dest by set : tensor<16x16xf32> -> tensor<16x16xf32> to tensor<16x16xf32>
    return %r : tensor<16x16xf32>
  }
}
