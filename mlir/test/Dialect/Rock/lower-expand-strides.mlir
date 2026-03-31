// RUN: rocmlir-opt -rock-expand-strides-lowering %s | FileCheck %s

// After InsertOutputStores, we have:
//   gemm -> expand_strides -> transform(Merge) -> store(src=flat, dest=flat)
// This pass rewrites it to:
//   store(source=gemm_result, dest=Unmerge(flat->expanded) + Slice(expanded->input))

// CHECK-LABEL: func.func @lower_expand_strides
// CHECK-SAME: (%arg0: tensor<4x24xf16>, %arg1: tensor<24x24xf16>, %arg2: tensor<192xf16>)
// CHECK: %[[GEMM:.*]] = rock.gemm %arg0 * %arg1
// CHECK-NOT: rock.expand_strides
// CHECK: %[[UNMERGE:.*]] = rock.transform %arg2
// CHECK: %[[SLICE:.*]] = rock.transform %[[UNMERGE]]
// CHECK: rock.store %[[GEMM]] to %[[SLICE]] by set
#map_merge = affine_map<(d0) -> (d0 floordiv 48, d0 mod 48)>
#merge_map = #rock.transform_map<#map_merge by [<Merge{4, 48} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [192] -> [4, 48]>
func.func @lower_expand_strides(%arg0: tensor<4x24xf16>, %arg1: tensor<24x24xf16>, %arg2: tensor<192xf16>) -> tensor<192xf16> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<4x24xf16> * tensor<24x24xf16> -> tensor<4x24xf16>
  %1 = rock.expand_strides %0 : tensor<4x24xf16> -> tensor<4x48xf16>
  %2 = rock.transform %1 by #merge_map : tensor<4x48xf16> to tensor<192xf16>
  %3 = rock.store %2 to %arg2 by set : tensor<192xf16> -> tensor<192xf16> to tensor<192xf16>
  return %3 : tensor<192xf16>
}

// -----

// Test case: expand_strides directly feeds store (no forward transforms).
// The dest already has the expanded shape, so only a Slice is needed.

// CHECK-LABEL: func.func @lower_expand_strides_no_merge
// CHECK-SAME: (%arg0: tensor<4x24xf16>, %arg1: tensor<24x24xf16>, %arg2: tensor<4x48xf16>)
// CHECK: %[[GEMM:.*]] = rock.gemm %arg0 * %arg1
// CHECK-NOT: rock.expand_strides
// CHECK: %[[SLICE:.*]] = rock.transform %arg2
// CHECK: rock.store %[[GEMM]] to %[[SLICE]] by set
func.func @lower_expand_strides_no_merge(%arg0: tensor<4x24xf16>, %arg1: tensor<24x24xf16>, %arg2: tensor<4x48xf16>) -> tensor<4x48xf16> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<4x24xf16> * tensor<24x24xf16> -> tensor<4x24xf16>
  %1 = rock.expand_strides %0 : tensor<4x24xf16> -> tensor<4x48xf16>
  %2 = rock.store %1 to %arg2 by set : tensor<4x48xf16> -> tensor<4x48xf16> to tensor<4x48xf16>
  return %2 : tensor<4x48xf16>
}

// -----

// Test case: non-multiple expansion (e.g., concat slice: 4x5 into a 4x12 buffer).
// The expanded dimension is not an integer multiple of the input dimension.

// CHECK-LABEL: func.func @lower_expand_strides_non_multiple
// CHECK-SAME: (%arg0: tensor<4x5xf16>, %arg1: tensor<5x5xf16>, %arg2: tensor<48xf16>)
// CHECK: %[[GEMM:.*]] = rock.gemm %arg0 * %arg1
// CHECK-NOT: rock.expand_strides
// CHECK: %[[UNMERGE:.*]] = rock.transform %arg2
// CHECK: %[[SLICE:.*]] = rock.transform %[[UNMERGE]]
// CHECK: rock.store %[[GEMM]] to %[[SLICE]] by set
#map_merge_nm = affine_map<(d0) -> (d0 floordiv 12, d0 mod 12)>
#merge_map_nm = #rock.transform_map<#map_merge_nm by [<Merge{4, 12} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [48] -> [4, 12]>
func.func @lower_expand_strides_non_multiple(%arg0: tensor<4x5xf16>, %arg1: tensor<5x5xf16>, %arg2: tensor<48xf16>) -> tensor<48xf16> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<4x5xf16> * tensor<5x5xf16> -> tensor<4x5xf16>
  %1 = rock.expand_strides %0 : tensor<4x5xf16> -> tensor<4x12xf16>
  %2 = rock.transform %1 by #merge_map_nm : tensor<4x12xf16> to tensor<48xf16>
  %3 = rock.store %2 to %arg2 by set : tensor<48xf16> -> tensor<48xf16> to tensor<48xf16>
  return %3 : tensor<48xf16>
}
