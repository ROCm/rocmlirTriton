// Error tests for rock-regularize-output pass.
// Each split block triggers a different error case.

// RUN: rocmlir-opt -rock-regularize-output -verify-diagnostics --split-input-file %s

// ============================================================
// Error: No stores found for fusion root.
// The gemm result flows only to a return, not to rock.store.
// ============================================================

module {
  func.func @error_no_store(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>) -> tensor<1x4x4xf16> attributes {rock.kernel} {
    // expected-error @below {{No stores found for fusion root}}
    %gemm = rock.gemm %a * %b : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    return %gemm : tensor<1x4x4xf16>
  }
}

// -----

// ============================================================
// Error: Non-splat constant through transforms.
// The pass cannot recreate a non-splat dense constant in gemm space.
// ============================================================

#map_unmerge = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 2 + d3)>
#tf_unmerge = #rock.transform_map<#map_unmerge by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Unmerge{2, 2} ["n0", "n1"] at [2, 3] -> ["n"] at [2]>] bounds = [1, 2, 2, 2] -> [1, 2, 4]>

module {
  func.func @error_non_splat_const(%a: tensor<1x2x4xf16>, %b: tensor<1x4x4xf16>, %dest: tensor<1x2x2x2xf16>) -> tensor<1x2x2x2xf16> attributes {rock.kernel} {
    // expected-error @below {{cannot regularize: non-splat constant extra operand is not supported}}
    %gemm = rock.gemm %a * %b : tensor<1x2x4xf16> * tensor<1x4x4xf16> -> tensor<1x2x4xf16>
    %t = rock.transform %gemm by #tf_unmerge : tensor<1x2x4xf16> to tensor<1x2x2x2xf16>
    %cst = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]> : tensor<1x2x2x2xf16>
    %fused = arith.subf %t, %cst : tensor<1x2x2x2xf16>
    %r = rock.store %fused to %dest by set : tensor<1x2x2x2xf16> -> tensor<1x2x2x2xf16> to tensor<1x2x2x2xf16>
    return %r : tensor<1x2x2x2xf16>
  }
}

// -----

// ============================================================
// Error: Non-invertible transforms with external operand.
// AddDim{2} is not invertible, so the external tensor cannot be
// transformed to gemm space.
// ============================================================

#map_adddim2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#tf_adddim2 = #rock.transform_map<#map_adddim2 by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <PassThrough ["n"] at [2] -> ["n"] at [2]>, <AddDim{2} ["extra"] at [3] -> [] at []>] bounds = [1, 4, 4, 2] -> [1, 4, 4]>

module {
  func.func @error_non_invertible_ext(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %ext: tensor<1x4x4x2xf16>, %dest: tensor<1x4x4x2xf16>) -> tensor<1x4x4x2xf16> attributes {rock.kernel} {
    // expected-error @below {{cannot regularize: transforms are not invertible and extra operand is not a splat constant}}
    %gemm = rock.gemm %a * %b : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %t = rock.transform %gemm by #tf_adddim2 : tensor<1x4x4xf16> to tensor<1x4x4x2xf16>
    %fused = arith.addf %t, %ext : tensor<1x4x4x2xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4x2xf16> -> tensor<1x4x4x2xf16> to tensor<1x4x4x2xf16>
    return %r : tensor<1x4x4x2xf16>
  }
}

// -----

// ============================================================
// Error: Non-invertible transforms for store destination rewrite.
// The fusion op (exp) has no external operands, but the store dest
// still needs inverse transforms which are not invertible.
// ============================================================

#map_adddim2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#tf_adddim2 = #rock.transform_map<#map_adddim2 by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <PassThrough ["n"] at [2] -> ["n"] at [2]>, <AddDim{2} ["extra"] at [3] -> [] at []>] bounds = [1, 4, 4, 2] -> [1, 4, 4]>

module {
  func.func @error_non_invertible_store(%a: tensor<1x4x4xf16>, %b: tensor<1x4x4xf16>, %dest: tensor<1x4x4x2xf16>) -> tensor<1x4x4x2xf16> attributes {rock.kernel} {
    // expected-error @below {{cannot regularize: transforms are not invertible for store destination rewrite}}
    %gemm = rock.gemm %a * %b : tensor<1x4x4xf16> * tensor<1x4x4xf16> -> tensor<1x4x4xf16>
    %t = rock.transform %gemm by #tf_adddim2 : tensor<1x4x4xf16> to tensor<1x4x4x2xf16>
    %fused = math.exp %t : tensor<1x4x4x2xf16>
    %r = rock.store %fused to %dest by set : tensor<1x4x4x2xf16> -> tensor<1x4x4x2xf16> to tensor<1x4x4x2xf16>
    return %r : tensor<1x4x4x2xf16>
  }
}
