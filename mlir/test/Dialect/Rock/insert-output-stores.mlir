// RUN: rocmlir-opt -rock-insert-output-stores -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @simple_gemm
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm %arg0 * %arg1
// CHECK: %[[STORE:.*]] = rock.store %[[GEMM]] to %arg2 by set
// CHECK: return %[[STORE]]
func.func @simple_gemm(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func.func @gemm_arith_fusion
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<8x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm %arg0 * %arg1
// CHECK: %[[ADD:.*]] = arith.addf %[[GEMM]], %arg2
// CHECK: %[[STORE:.*]] = rock.store %[[ADD]] to %arg3 by set
// CHECK: return %[[STORE]]
func.func @gemm_arith_fusion(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = arith.addf %0, %arg2 : tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}

// Pass should skip: rock.store already exists
// CHECK-LABEL: func.func @gemm_existing_store
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>)
// CHECK-SAME: -> tensor<8x32xf32>
// CHECK: rock.gemm
// CHECK-COUNT-1: rock.store
// CHECK: return
func.func @gemm_existing_store(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = rock.store %0 to %arg2 by set : tensor<8x32xf32> -> tensor<8x32xf32> to tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}

// Pass should skip: no FusionRoot ops
// CHECK-LABEL: func.func @no_fusion_root
// CHECK-SAME: (%arg0: tensor<8x32xf32>)
// CHECK-SAME: -> tensor<8x32xf32>
// CHECK-NOT: rock.store
// CHECK: return %arg0
func.func @no_fusion_root(%arg0: tensor<8x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel} {
  return %arg0 : tensor<8x32xf32>
}

// Attention with two returns: both get stores, returns updated to store results
// CHECK-LABEL: func.func @attention_two_returns
// CHECK-SAME: (%arg0: tensor<1x32x32xf32>, %arg1: tensor<1x32x32xf32>, %arg2: tensor<1x32x32xf32>, %arg3: tensor<1x32x32xf32>, %arg4: tensor<1x32xf32>) -> (tensor<1x32x32xf32>, tensor<1x32xf32>) attributes {rock.kernel}
// CHECK: %[[RESULT:.*]], %[[LSE:.*]] = rock.attention
// CHECK: %[[STORE_R:.*]] = rock.store %[[RESULT]] to %arg3 by set
// CHECK: %[[STORE_L:.*]] = rock.store %[[LSE]] to %arg4 by set
// CHECK: return %[[STORE_R]], %[[STORE_L]]
func.func @attention_two_returns(%arg0: tensor<1x32x32xf32>, %arg1: tensor<1x32x32xf32>, %arg2: tensor<1x32x32xf32>) -> (tensor<1x32x32xf32>, tensor<1x32xf32>) attributes {rock.kernel} {
  %result, %lse = rock.attention{
   qk = %arg0 * %arg1 : tensor<1x32x32xf32>, tensor<1x32x32xf32>
   softmax(qk) * %arg2 : tensor<1x32x32xf32>
  } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32} -> tensor<1x32x32xf32>, tensor<1x32xf32>
  return %result, %lse : tensor<1x32x32xf32>, tensor<1x32xf32>
}

// Pass should skip: not a kernel function
// CHECK-LABEL: func.func @non_kernel
// CHECK-SAME: (%arg0: tensor<8x32xf32>)
// CHECK-SAME: -> tensor<8x32xf32>
// CHECK-NOT: rock.store
// CHECK: return %arg0
func.func @non_kernel(%arg0: tensor<8x32xf32>) -> tensor<8x32xf32> {
  return %arg0 : tensor<8x32xf32>
}

// Gemm -> transform -> return: new arg has transformed type, return updated
// CHECK-LABEL: func.func @gemm_with_transform
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm
// CHECK: %[[TR:.*]] = rock.transform %[[GEMM]]
// CHECK: %[[STORE:.*]] = rock.store %[[TR]] to %arg2 by set
// CHECK: return %[[STORE]]
#map_merge = affine_map<(d0) -> (d0 floordiv 32, d0 mod 32)>
#merge_map = #rock.transform_map<#map_merge by [<Merge{8, 32} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [256] -> [8, 32]>
func.func @gemm_with_transform(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<256xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = rock.transform %0 by #merge_map : tensor<8x32xf32> to tensor<256xf32>
  return %1 : tensor<256xf32>
}

// Gemm -> reduce -> return: reduce stops backward trace, new arg created
// CHECK-LABEL: func.func @gemm_with_reduce
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x1xf32>) -> tensor<8x1xf32> attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm
// CHECK: %[[RED:.*]] = rock.reduce sum %[[GEMM]]
// CHECK: %[[STORE:.*]] = rock.store %[[RED]] to %arg2 by set
// CHECK: return %[[STORE]]
func.func @gemm_with_reduce(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x1xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = rock.reduce sum %0 {axis = 1 : index} : tensor<8x32xf32> -> tensor<8x1xf32>
  return %1 : tensor<8x1xf32>
}


// Attention with transforms on both outputs: new args with transformed types
// CHECK-LABEL: func.func @attention_with_transforms
// CHECK-SAME: (%arg0: tensor<1x32x32xf32>, %arg1: tensor<1x32x32xf32>, %arg2: tensor<1x32x32xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<32xf32>) -> (tensor<1024xf32>, tensor<32xf32>) attributes {rock.kernel}
// CHECK: %[[RESULT:.*]], %[[LSE:.*]] = rock.attention
// CHECK: %[[TR_R:.*]] = rock.transform %[[RESULT]]
// CHECK: %[[TR_L:.*]] = rock.transform %[[LSE]]
// CHECK: %[[STORE_R:.*]] = rock.store %[[TR_R]] to %arg3 by set
// CHECK: %[[STORE_L:.*]] = rock.store %[[TR_L]] to %arg4 by set
// CHECK: return %[[STORE_R]], %[[STORE_L]]
#map_merge3 = affine_map<(d0) -> (d0 floordiv 1024, (d0 mod 1024) floordiv 32, d0 mod 32)>
#merge3_map = #rock.transform_map<#map_merge3 by [<Merge{1, 32, 32} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [1024] -> [1, 32, 32]>
#map_merge2 = affine_map<(d0) -> (d0 floordiv 32, d0 mod 32)>
#merge2_map = #rock.transform_map<#map_merge2 by [<Merge{1, 32} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [32] -> [1, 32]>
func.func @attention_with_transforms(%arg0: tensor<1x32x32xf32>, %arg1: tensor<1x32x32xf32>, %arg2: tensor<1x32x32xf32>) -> (tensor<1024xf32>, tensor<32xf32>) attributes {rock.kernel} {
  %result, %lse = rock.attention{
   qk = %arg0 * %arg1 : tensor<1x32x32xf32>, tensor<1x32x32xf32>
   softmax(qk) * %arg2 : tensor<1x32x32xf32>
  } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32} -> tensor<1x32x32xf32>, tensor<1x32xf32>
  %0 = rock.transform %result by #merge3_map : tensor<1x32x32xf32> to tensor<1024xf32>
  %1 = rock.transform %lse by #merge2_map : tensor<1x32xf32> to tensor<32xf32>
  return %0, %1 : tensor<1024xf32>, tensor<32xf32>
}

// Mixed returns: gemm result stored, passthrough also gets store
// CHECK-LABEL: func.func @mixed_returns
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<4xf32>, %arg3: tensor<8x32xf32>, %arg4: tensor<4xf32>)
// CHECK-SAME: -> (tensor<8x32xf32>, tensor<4xf32>)
// CHECK-SAME: attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm
// CHECK: %[[STORE1:.*]] = rock.store %[[GEMM]] to %arg3 by set
// CHECK: %[[STORE2:.*]] = rock.store %arg2 to %arg4 by set
// CHECK: return %[[STORE1]], %[[STORE2]]
func.func @mixed_returns(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<4xf32>) -> (tensor<8x32xf32>, tensor<4xf32>) attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  return %0, %arg2 : tensor<8x32xf32>, tensor<4xf32>
}

// Gemm -> arith fusion -> transform -> return: new arg has transformed type
// CHECK-LABEL: func.func @gemm_fusion_then_transform
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm
// CHECK: %[[ADD:.*]] = arith.addf %[[GEMM]], %arg2
// CHECK: %[[TR:.*]] = rock.transform %[[ADD]]
// CHECK: %[[STORE:.*]] = rock.store %[[TR]] to %arg3 by set
// CHECK: return %[[STORE]]
func.func @gemm_fusion_then_transform(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>) -> tensor<256xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = arith.addf %0, %arg2 : tensor<8x32xf32>
  %2 = rock.transform %1 by #merge_map : tensor<8x32xf32> to tensor<256xf32>
  return %2 : tensor<256xf32>
}

// Gemm -> transform -> arith fusion -> return: new arg has fused transformed type
// CHECK-LABEL: func.func @gemm_transform_then_fusion
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm
// CHECK: %[[TR:.*]] = rock.transform %[[GEMM]]
// CHECK: %[[ADD:.*]] = arith.addf %[[TR]], %arg2
// CHECK: %[[STORE:.*]] = rock.store %[[ADD]] to %arg3 by set
// CHECK: return %[[STORE]]
func.func @gemm_transform_then_fusion(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = rock.transform %0 by #merge_map : tensor<8x32xf32> to tensor<256xf32>
  %2 = arith.addf %1, %arg2 : tensor<256xf32>
  return %2 : tensor<256xf32>
}

// Gemm -> arith fusion -> reduce -> return: new arg has reduced type
// CHECK-LABEL: func.func @gemm_fusion_then_reduce
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<8x1xf32>) -> tensor<8x1xf32> attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm
// CHECK: %[[ADD:.*]] = arith.addf %[[GEMM]], %arg2
// CHECK: %[[RED:.*]] = rock.reduce sum %[[ADD]]
// CHECK: %[[STORE:.*]] = rock.store %[[RED]] to %arg3 by set
// CHECK: return %[[STORE]]
func.func @gemm_fusion_then_reduce(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>) -> tensor<8x1xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = arith.addf %0, %arg2 : tensor<8x32xf32>
  %2 = rock.reduce sum %1 {axis = 1 : index} : tensor<8x32xf32> -> tensor<8x1xf32>
  return %2 : tensor<8x1xf32>
}

// Gemm -> reduce -> transform -> return: new arg has final transformed type
// CHECK-LABEL: func.func @gemm_reduce_then_transform
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm
// CHECK: %[[RED:.*]] = rock.reduce sum %[[GEMM]]
// CHECK: %[[TR:.*]] = rock.transform %[[RED]]
// CHECK: %[[STORE:.*]] = rock.store %[[TR]] to %arg2 by set
// CHECK: return %[[STORE]]
#map_merge_8x1 = affine_map<(d0) -> (d0, 0)>
#merge_8x1_map = #rock.transform_map<#map_merge_8x1 by [<Merge{8, 1} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [8] -> [8, 1]>
func.func @gemm_reduce_then_transform(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = rock.reduce sum %0 {axis = 1 : index} : tensor<8x32xf32> -> tensor<8x1xf32>
  %2 = rock.transform %1 by #merge_8x1_map : tensor<8x1xf32> to tensor<8xf32>
  return %2 : tensor<8xf32>
}

// Gemm -> arith fusion -> reduce -> transform -> return: all three combined
// CHECK-LABEL: func.func @gemm_fusion_reduce_transform
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<8xf32>) -> tensor<8xf32> attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm
// CHECK: %[[MUL:.*]] = arith.mulf %[[GEMM]], %arg2
// CHECK: %[[RED:.*]] = rock.reduce sum %[[MUL]]
// CHECK: %[[TR:.*]] = rock.transform %[[RED]]
// CHECK: %[[STORE:.*]] = rock.store %[[TR]] to %arg3 by set
// CHECK: return %[[STORE]]
func.func @gemm_fusion_reduce_transform(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>) -> tensor<8xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = arith.mulf %0, %arg2 : tensor<8x32xf32>
  %2 = rock.reduce sum %1 {axis = 1 : index} : tensor<8x32xf32> -> tensor<8x1xf32>
  %3 = rock.transform %2 by #merge_8x1_map : tensor<8x1xf32> to tensor<8xf32>
  return %3 : tensor<8xf32>
}

// Gemm -> fusion -> fan-out to two transforms -> return both:
// chainSet from the single gemm root covers both return operands,
// so both get separate output args and stores.
// CHECK-LABEL: func.func @gemm_fusion_fanout_transforms
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>)
// CHECK-SAME: -> (tensor<256xf32>, tensor<256xf32>)
// CHECK-SAME: attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm
// CHECK: %[[ADD:.*]] = arith.addf %[[GEMM]], %arg2
// CHECK: %[[TR1:.*]] = rock.transform %[[ADD]]
// CHECK: %[[TR2:.*]] = rock.transform %[[ADD]]
// CHECK: %[[STORE1:.*]] = rock.store %[[TR2]] to %arg3 by set
// CHECK: %[[STORE2:.*]] = rock.store %[[TR1]] to %arg4 by set
// CHECK: return %[[STORE1]], %[[STORE2]]
func.func @gemm_fusion_fanout_transforms(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>) -> (tensor<256xf32>, tensor<256xf32>) attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = arith.addf %0, %arg2 : tensor<8x32xf32>
  %2 = rock.transform %1 by #merge_map : tensor<8x32xf32> to tensor<256xf32>
  %3 = rock.transform %1 by #merge_map : tensor<8x32xf32> to tensor<256xf32>
  return %3, %2 : tensor<256xf32>, tensor<256xf32>
}

// Gemm fans out to an existing rock.store and a fusion chain without a store.
// The existing store return is left alone; only the fusion path gets a new store.
// CHECK-LABEL: func.func @gemm_partial_store
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<8x32xf32>, %arg4: tensor<8x32xf32>)
// CHECK-SAME: -> (tensor<8x32xf32>, tensor<8x32xf32>)
// CHECK-SAME: attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm
// CHECK: %[[EXISTING:.*]] = rock.store %[[GEMM]] to %arg2 by set
// CHECK: %[[ADD:.*]] = arith.addf %[[GEMM]], %arg3
// CHECK: %[[NEW_STORE:.*]] = rock.store %[[ADD]] to %arg4 by set
// CHECK: return %[[EXISTING]], %[[NEW_STORE]]
func.func @gemm_partial_store(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<8x32xf32>) -> (tensor<8x32xf32>, tensor<8x32xf32>) attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = rock.store %0 to %arg2 by set : tensor<8x32xf32> -> tensor<8x32xf32> to tensor<8x32xf32>
  %2 = arith.addf %0, %arg3 : tensor<8x32xf32>
  return %1, %2 : tensor<8x32xf32>, tensor<8x32xf32>
}

// Reversed partial store: existing store at return index 1, new store needed
// at return index 0. The new output arg must be inserted before the existing
// store's dest arg so output args follow return-index order.
// CHECK-LABEL: func.func @gemm_partial_store_reversed
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<8x32xf32>, %arg4: tensor<8x32xf32>)
// CHECK-SAME: -> (tensor<8x32xf32>, tensor<8x32xf32>)
// CHECK-SAME: attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm
// CHECK: %[[EXISTING:.*]] = rock.store %[[GEMM]] to %arg3 by set
// CHECK: %[[ADD:.*]] = arith.addf %[[GEMM]], %arg4
// CHECK: %[[NEW_STORE:.*]] = rock.store %[[ADD]] to %arg2 by set
// CHECK: return %[[NEW_STORE]], %[[EXISTING]]
func.func @gemm_partial_store_reversed(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<8x32xf32>) -> (tensor<8x32xf32>, tensor<8x32xf32>) attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = rock.store %0 to %arg2 by set : tensor<8x32xf32> -> tensor<8x32xf32> to tensor<8x32xf32>
  %2 = arith.addf %0, %arg3 : tensor<8x32xf32>
  return %2, %1 : tensor<8x32xf32>, tensor<8x32xf32>
}

// Fusion op uses the gemm output twice (both operands): exercises the
// duplicate-user path in floodFillFromRoot.
// CHECK-LABEL: func.func @gemm_fusion_same_operand_twice
// CHECK-SAME: (%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<8x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel}
// CHECK: %[[GEMM:.*]] = rock.gemm
// CHECK: %[[ADD:.*]] = arith.addf %[[GEMM]], %[[GEMM]]
// CHECK: %[[STORE:.*]] = rock.store %[[ADD]] to %arg2 by set
// CHECK-NOT: rock.store
// CHECK: return
func.func @gemm_fusion_same_operand_twice(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %1 = arith.addf %0, %0 : tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}

// -----

// A return operand that is not a block argument and not reachable from any
// FusionRoot should be flagged as an error.
func.func @uncovered_return_operand(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> (tensor<8x32xf32>, tensor<8x32xf32>) attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  %cst = arith.constant dense<0.0> : tensor<8x32xf32>
  // expected-error @below {{return operand 1 is not a block argument and is not reachable from any FusionRoot}}
  return %0, %cst : tensor<8x32xf32>, tensor<8x32xf32>
}

// -----

// A FusionRoot result that is unused and has no store is an error.
func.func @root_not_reaching_return(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  // expected-error @below {{FusionRoot result has no rock.store and does not reach a function return}}
  %1 = rock.gemm %arg0 * %arg2 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// A chain value used by an op that is not a fusion op, transform, reduce,
// store, or return is an error.
func.func @unexpected_chain_use(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  // expected-error @below {{unexpected use of FusionRoot chain value by tosa.abs}}
  %1 = "tosa.abs"(%0) {} : (tensor<8x32xf32>) -> tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}

// -----

// A kernel with callers should error — the pass expects no call sites.
// expected-error @below {{kernel has callers; InsertOutputStores expects no call sites}}
func.func @kernel_with_caller(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> attributes {rock.kernel} {
  %0 = rock.gemm %arg0 * %arg1 : tensor<8x16xf32> * tensor<16x32xf32> -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}
func.func @caller(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
  %0 = func.call @kernel_with_caller(%arg0, %arg1) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

