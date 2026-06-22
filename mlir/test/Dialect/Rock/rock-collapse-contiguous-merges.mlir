// RUN: rocmlir-opt -rock-collapse-contiguous-merges -split-input-file %s | FileCheck %s

// End-to-end check of the RockCollapseContiguousMerges pass on the IR shape it
// sees just before RockTransformsToPointerArith: a rock.kernel function whose
// rock.transforms_to_ptr source is a pure rock.transform chain. The pass walks
// each transforms_to_ptr, isolates its source chain, and collapses contiguous
// merges so the composed pointer map stays stride-1 (which lets Triton's
// AxisInfoAnalysis vectorize the load/store).

// A contiguous {a, b} tile merge collapses to Merge{1, 24}; the trailing
// Unmerge widens to Unmerge{2, 1, 24}, and transforms_to_ptr is rewired onto
// the collapsed chain.
// CHECK: [[FLAT:#.+]] = #rock.transform_map<{{.*}}Unmerge{2, 1, 24} ["block", "a", "b"] at [0, 1, 2] -> ["raw"] at [0]{{.*}}bounds = [2, 1, 24] -> [48]>
// CHECK: [[MERGE:#.+]] = #rock.transform_map<{{.*}}Merge{1, 24} ["tile"] at [1] -> ["a", "b"] at [1, 2]{{.*}}bounds = [2, 24] -> [2, 1, 24]>
// CHECK: func @collapse_load
// CHECK-SAME: ([[ARG0:%.+]]: tensor<48xf16>)
// CHECK: [[U:%.+]] = rock.transform [[ARG0]] by [[FLAT]] : tensor<48xf16> to tensor<2x1x24xf16>
// CHECK: [[M:%.+]] = rock.transform [[U]] by [[MERGE]] : tensor<2x1x24xf16> to tensor<2x24xf16>
// CHECK: rock.transforms_to_ptr [[M]][%{{.+}}] : tensor<2x24xf16> -> tensor<24xi32>, tensor<24xi1>
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2) -> ((d0 * 3 + d1) * 8 + d2)>
  by [<Unmerge{2, 3, 8} ["block", "a", "b"] at [0, 1, 2] -> ["raw"] at [0]>]
  bounds = [2, 3, 8] -> [48]>
#merge = #rock.transform_map<
  affine_map<(d0, d1) -> (d0, d1 floordiv 8, d1 mod 8)>
  by [<PassThrough ["block"] at [0] -> ["block"] at [0]>,
    <Merge{3, 8} ["tile"] at [1] -> ["a", "b"] at [1, 2]>]
  bounds = [2, 24] -> [2, 3, 8]>

func.func @collapse_load(%arg0: tensor<48xf16>) -> tensor<24xf16>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  %c0 = arith.constant 0 : i32
  %0 = rock.transform %arg0 by #flatten : tensor<48xf16> to tensor<2x3x8xf16>
  %1 = rock.transform %0 by #merge : tensor<2x3x8xf16> to tensor<2x24xf16>
  %ptr, %mask = rock.transforms_to_ptr %1[%c0] : tensor<2x24xf16> -> tensor<24xi32>, tensor<24xi1>
  %2 = rock.blockwise_load_ptr %ptr[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<24xi32>, tensor<24xi1> -> tensor<24xf16>
  return %2 : tensor<24xf16>
}

// -----

// A function without rock.kernel is skipped: the chain is left untouched, so
// the original Merge{3, 8} survives and nothing is collapsed.
// CHECK: [[MERGE:#.+]] = #rock.transform_map<{{.*}}Merge{3, 8} ["tile"] at [1] -> ["a", "b"] at [1, 2]{{.*}}bounds = [2, 24] -> [2, 3, 8]>
// CHECK: func @not_a_kernel
// CHECK-SAME: ([[ARG0:%.+]]: tensor<48xf16>)
// CHECK: [[U:%.+]] = rock.transform [[ARG0]] by #{{.+}} : tensor<48xf16> to tensor<2x3x8xf16>
// CHECK: [[M:%.+]] = rock.transform [[U]] by [[MERGE]] : tensor<2x3x8xf16> to tensor<2x24xf16>
// CHECK: rock.transforms_to_ptr [[M]][%{{.+}}] : tensor<2x24xf16> -> tensor<24xi32>, tensor<24xi1>
// CHECK-NOT: Merge{1, 24}
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2) -> ((d0 * 3 + d1) * 8 + d2)>
  by [<Unmerge{2, 3, 8} ["block", "a", "b"] at [0, 1, 2] -> ["raw"] at [0]>]
  bounds = [2, 3, 8] -> [48]>
#merge = #rock.transform_map<
  affine_map<(d0, d1) -> (d0, d1 floordiv 8, d1 mod 8)>
  by [<PassThrough ["block"] at [0] -> ["block"] at [0]>,
    <Merge{3, 8} ["tile"] at [1] -> ["a", "b"] at [1, 2]>]
  bounds = [2, 24] -> [2, 3, 8]>

func.func @not_a_kernel(%arg0: tensor<48xf16>) -> tensor<24xf16>
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  %c0 = arith.constant 0 : i32
  %0 = rock.transform %arg0 by #flatten : tensor<48xf16> to tensor<2x3x8xf16>
  %1 = rock.transform %0 by #merge : tensor<2x3x8xf16> to tensor<2x24xf16>
  %ptr, %mask = rock.transforms_to_ptr %1[%c0] : tensor<2x24xf16> -> tensor<24xi32>, tensor<24xi1>
  %2 = rock.blockwise_load_ptr %ptr[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<24xi32>, tensor<24xi1> -> tensor<24xf16>
  return %2 : tensor<24xf16>
}

// -----

// The pass also collapses the chain behind a store: the transforms_to_ptr that
// feeds blockwise_store_ptr is rewired onto the collapsed Merge{1, 24}.
// CHECK: [[FLAT:#.+]] = #rock.transform_map<{{.*}}Unmerge{2, 1, 24} ["block", "a", "b"] at [0, 1, 2] -> ["raw"] at [0]{{.*}}bounds = [2, 1, 24] -> [48]>
// CHECK: [[MERGE:#.+]] = #rock.transform_map<{{.*}}Merge{1, 24} ["tile"] at [1] -> ["a", "b"] at [1, 2]{{.*}}bounds = [2, 24] -> [2, 1, 24]>
// CHECK: func @collapse_store
// CHECK-SAME: ([[VAL:%.+]]: tensor<24xf16>, [[DEST:%.+]]: tensor<48xf16>)
// CHECK: [[U:%.+]] = rock.transform [[DEST]] by [[FLAT]] : tensor<48xf16> to tensor<2x1x24xf16>
// CHECK: [[M:%.+]] = rock.transform [[U]] by [[MERGE]] : tensor<2x1x24xf16> to tensor<2x24xf16>
// CHECK: [[PTR:%.+]], [[MASK:%.+]] = rock.transforms_to_ptr [[M]][%{{.+}}] : tensor<2x24xf16> -> tensor<24xi32>, tensor<24xi1>
// CHECK: rock.blockwise_store_ptr [[VAL]] -> [[PTR]]([[MASK]]) by set
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2) -> ((d0 * 3 + d1) * 8 + d2)>
  by [<Unmerge{2, 3, 8} ["block", "a", "b"] at [0, 1, 2] -> ["raw"] at [0]>]
  bounds = [2, 3, 8] -> [48]>
#merge = #rock.transform_map<
  affine_map<(d0, d1) -> (d0, d1 floordiv 8, d1 mod 8)>
  by [<PassThrough ["block"] at [0] -> ["block"] at [0]>,
    <Merge{3, 8} ["tile"] at [1] -> ["a", "b"] at [1, 2]>]
  bounds = [2, 24] -> [2, 3, 8]>

func.func @collapse_store(%val: tensor<24xf16>, %dest: tensor<48xf16>)
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  %c0 = arith.constant 0 : i32
  %0 = rock.transform %dest by #flatten : tensor<48xf16> to tensor<2x3x8xf16>
  %1 = rock.transform %0 by #merge : tensor<2x3x8xf16> to tensor<2x24xf16>
  %ptr, %mask = rock.transforms_to_ptr %1[%c0] : tensor<2x24xf16> -> tensor<24xi32>, tensor<24xi1>
  rock.blockwise_store_ptr %val -> %ptr(%mask) by set : tensor<24xf16> -> tensor<24xi32>(tensor<24xi1>)
  return
}

// -----

// Non-contiguous merge: `a` and `b` are separated by `block` in memory, so the
// {a, b} merge is not contiguous and the chain is left unchanged.
// CHECK: [[MERGE:#.+]] = #rock.transform_map<{{.*}}Merge{3, 8} ["tile"] at [1] -> ["a", "b"] at [0, 2]{{.*}}bounds = [2, 24] -> [3, 2, 8]>
// CHECK: func @no_collapse_non_contiguous
// CHECK-SAME: ([[ARG0:%.+]]: tensor<48xf16>)
// CHECK: [[U:%.+]] = rock.transform [[ARG0]] by #{{.+}} : tensor<48xf16> to tensor<3x2x8xf16>
// CHECK: [[M:%.+]] = rock.transform [[U]] by [[MERGE]] : tensor<3x2x8xf16> to tensor<2x24xf16>
// CHECK: rock.transforms_to_ptr [[M]][%{{.+}}] : tensor<2x24xf16> -> tensor<24xi32>, tensor<24xi1>
// CHECK-NOT: Merge{1, 24}
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2) -> ((d0 * 2 + d1) * 8 + d2)>
  by [<Unmerge{3, 2, 8} ["a", "block", "b"] at [0, 1, 2] -> ["raw"] at [0]>]
  bounds = [3, 2, 8] -> [48]>
#merge = #rock.transform_map<
  affine_map<(d0, d1) -> (d1 floordiv 8, d0, d1 mod 8)>
  by [<PassThrough ["block"] at [0] -> ["block"] at [1]>,
    <Merge{3, 8} ["tile"] at [1] -> ["a", "b"] at [0, 2]>]
  bounds = [2, 24] -> [3, 2, 8]>

func.func @no_collapse_non_contiguous(%arg0: tensor<48xf16>) -> tensor<24xf16>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  %c0 = arith.constant 0 : i32
  %0 = rock.transform %arg0 by #flatten : tensor<48xf16> to tensor<3x2x8xf16>
  %1 = rock.transform %0 by #merge : tensor<3x2x8xf16> to tensor<2x24xf16>
  %ptr, %mask = rock.transforms_to_ptr %1[%c0] : tensor<2x24xf16> -> tensor<24xi32>, tensor<24xi1>
  %2 = rock.blockwise_load_ptr %ptr[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<24xi32>, tensor<24xi1> -> tensor<24xf16>
  return %2 : tensor<24xf16>
}

// -----

// A kernel with two transforms_to_ptr ops: the pass collapses both chains.
// CHECK: [[FLAT:#.+]] = #rock.transform_map<{{.*}}Unmerge{2, 1, 24} ["block", "a", "b"] at [0, 1, 2] -> ["raw"] at [0]{{.*}}bounds = [2, 1, 24] -> [48]>
// CHECK: [[MERGE:#.+]] = #rock.transform_map<{{.*}}Merge{1, 24} ["tile"] at [1] -> ["a", "b"] at [1, 2]{{.*}}bounds = [2, 24] -> [2, 1, 24]>
// CHECK: func @collapse_two_loads
// CHECK-SAME: ([[ARG0:%.+]]: tensor<48xf16>, [[ARG1:%.+]]: tensor<48xf16>)
// First load collapses.
// CHECK: [[U0:%.+]] = rock.transform [[ARG0]] by [[FLAT]] : tensor<48xf16> to tensor<2x1x24xf16>
// CHECK: [[M0:%.+]] = rock.transform [[U0]] by [[MERGE]] : tensor<2x1x24xf16> to tensor<2x24xf16>
// CHECK: rock.transforms_to_ptr [[M0]][%{{.+}}] : tensor<2x24xf16> -> tensor<24xi32>, tensor<24xi1>
// Second load collapses too.
// CHECK: [[U1:%.+]] = rock.transform [[ARG1]] by [[FLAT]] : tensor<48xf16> to tensor<2x1x24xf16>
// CHECK: [[M1:%.+]] = rock.transform [[U1]] by [[MERGE]] : tensor<2x1x24xf16> to tensor<2x24xf16>
// CHECK: rock.transforms_to_ptr [[M1]][%{{.+}}] : tensor<2x24xf16> -> tensor<24xi32>, tensor<24xi1>
#flatten = #rock.transform_map<
  affine_map<(d0, d1, d2) -> ((d0 * 3 + d1) * 8 + d2)>
  by [<Unmerge{2, 3, 8} ["block", "a", "b"] at [0, 1, 2] -> ["raw"] at [0]>]
  bounds = [2, 3, 8] -> [48]>
#merge = #rock.transform_map<
  affine_map<(d0, d1) -> (d0, d1 floordiv 8, d1 mod 8)>
  by [<PassThrough ["block"] at [0] -> ["block"] at [0]>,
    <Merge{3, 8} ["tile"] at [1] -> ["a", "b"] at [1, 2]>]
  bounds = [2, 24] -> [2, 3, 8]>

func.func @collapse_two_loads(%arg0: tensor<48xf16>, %arg1: tensor<48xf16>)
    -> (tensor<24xf16>, tensor<24xf16>)
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  %c0 = arith.constant 0 : i32
  %0 = rock.transform %arg0 by #flatten : tensor<48xf16> to tensor<2x3x8xf16>
  %1 = rock.transform %0 by #merge : tensor<2x3x8xf16> to tensor<2x24xf16>
  %ptr0, %mask0 = rock.transforms_to_ptr %1[%c0] : tensor<2x24xf16> -> tensor<24xi32>, tensor<24xi1>
  %2 = rock.blockwise_load_ptr %ptr0[%mask0] {cacheModifier = #rock<CacheModifier none>} : tensor<24xi32>, tensor<24xi1> -> tensor<24xf16>
  %3 = rock.transform %arg1 by #flatten : tensor<48xf16> to tensor<2x3x8xf16>
  %4 = rock.transform %3 by #merge : tensor<2x3x8xf16> to tensor<2x24xf16>
  %ptr1, %mask1 = rock.transforms_to_ptr %4[%c0] : tensor<2x24xf16> -> tensor<24xi32>, tensor<24xi1>
  %5 = rock.blockwise_load_ptr %ptr1[%mask1] {cacheModifier = #rock<CacheModifier none>} : tensor<24xi32>, tensor<24xi1> -> tensor<24xf16>
  return %2, %5 : tensor<24xf16>, tensor<24xf16>
}
