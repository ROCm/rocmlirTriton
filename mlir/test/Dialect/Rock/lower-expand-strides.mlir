// RUN: rocmlir-opt -rock-regularize-output %s | FileCheck %s

// After InsertOutputStores, we have:
//   gemm -> transform(Pad) -> transform(Merge) -> store(src=flat, dest=flat)
// RegularizeOutput rewrites it to:
//   store(source=gemm_result, dest=Unmerge(flat->expanded) + Slice(expanded->input))

// CHECK-LABEL: func.func @regularize_pad_with_merge
// CHECK-SAME: (%arg0: tensor<4x24xf16>, %arg1: tensor<24x24xf16>, %arg2: tensor<192xf16>)
// CHECK: %[[GEMM:.*]] = rock.gemm %arg0 * %arg1
// CHECK: %[[UNMERGE:.*]] = rock.transform %arg2
// CHECK: %[[SLICE:.*]] = rock.transform %[[UNMERGE]]
// CHECK: rock.store %[[GEMM]] to %[[SLICE]] by set
#pad_amap1 = affine_map<(d0, d1) -> (d0, d1)>
#pad_map1 = #rock.transform_map<#pad_amap1 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Pad{0, 24} ["exp1"] at [1] -> ["dim1"] at [1]>] bounds = [4, 48] -> [4, 24]>
#merge_amap1 = affine_map<(d0) -> (d0 floordiv 48, d0 mod 48)>
#merge_map1 = #rock.transform_map<#merge_amap1 by [<Merge{4, 48} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [192] -> [4, 48]>
func.func @regularize_pad_with_merge(%arg0: tensor<4x24xf16>, %arg1: tensor<24x24xf16>, %arg2: tensor<192xf16>) -> tensor<192xf16> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<4x24xf16> * tensor<24x24xf16> -> tensor<4x24xf16>
  %1 = rock.transform %0 by #pad_map1 : tensor<4x24xf16> to tensor<4x48xf16>
  %2 = rock.transform %1 by #merge_map1 : tensor<4x48xf16> to tensor<192xf16>
  %3 = rock.store %2 to %arg2 by set : tensor<192xf16> -> tensor<192xf16> to tensor<192xf16>
  return %3 : tensor<192xf16>
}

// -----

// Test case: Pad directly feeds store (no Merge transform).
// The dest already has the expanded shape, so only a Slice is needed.

// CHECK-LABEL: func.func @regularize_pad_no_merge
// CHECK-SAME: (%arg0: tensor<4x24xf16>, %arg1: tensor<24x24xf16>, %arg2: tensor<4x48xf16>)
// CHECK: %[[GEMM:.*]] = rock.gemm %arg0 * %arg1
// CHECK: %[[SLICE:.*]] = rock.transform %arg2
// CHECK: rock.store %[[GEMM]] to %[[SLICE]] by set
#pad_amap2 = affine_map<(d0, d1) -> (d0, d1)>
#pad_map2 = #rock.transform_map<#pad_amap2 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Pad{0, 24} ["exp1"] at [1] -> ["dim1"] at [1]>] bounds = [4, 48] -> [4, 24]>
func.func @regularize_pad_no_merge(%arg0: tensor<4x24xf16>, %arg1: tensor<24x24xf16>, %arg2: tensor<4x48xf16>) -> tensor<4x48xf16> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<4x24xf16> * tensor<24x24xf16> -> tensor<4x24xf16>
  %1 = rock.transform %0 by #pad_map2 : tensor<4x24xf16> to tensor<4x48xf16>
  %2 = rock.store %1 to %arg2 by set : tensor<4x48xf16> -> tensor<4x48xf16> to tensor<4x48xf16>
  return %2 : tensor<4x48xf16>
}

// -----

// Test case: non-multiple expansion (e.g., concat slice: 4x5 into a 4x12 buffer).
// The expanded dimension is not an integer multiple of the input dimension.

// CHECK-LABEL: func.func @regularize_pad_non_multiple
// CHECK-SAME: (%arg0: tensor<4x5xf16>, %arg1: tensor<5x5xf16>, %arg2: tensor<48xf16>)
// CHECK: %[[GEMM:.*]] = rock.gemm %arg0 * %arg1
// CHECK: %[[UNMERGE:.*]] = rock.transform %arg2
// CHECK: %[[SLICE:.*]] = rock.transform %[[UNMERGE]]
// CHECK: rock.store %[[GEMM]] to %[[SLICE]] by set
#pad_amap3 = affine_map<(d0, d1) -> (d0, d1)>
#pad_map3 = #rock.transform_map<#pad_amap3 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Pad{0, 7} ["exp1"] at [1] -> ["dim1"] at [1]>] bounds = [4, 12] -> [4, 5]>
#merge_amap3 = affine_map<(d0) -> (d0 floordiv 12, d0 mod 12)>
#merge_map3 = #rock.transform_map<#merge_amap3 by [<Merge{4, 12} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [48] -> [4, 12]>
func.func @regularize_pad_non_multiple(%arg0: tensor<4x5xf16>, %arg1: tensor<5x5xf16>, %arg2: tensor<48xf16>) -> tensor<48xf16> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<4x5xf16> * tensor<5x5xf16> -> tensor<4x5xf16>
  %1 = rock.transform %0 by #pad_map3 : tensor<4x5xf16> to tensor<4x12xf16>
  %2 = rock.transform %1 by #merge_map3 : tensor<4x12xf16> to tensor<48xf16>
  %3 = rock.store %2 to %arg2 by set : tensor<48xf16> -> tensor<48xf16> to tensor<48xf16>
  return %3 : tensor<48xf16>
}
