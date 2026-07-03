// RUN: rocmlir-opt -rock-split-cross-tile-fusion %s | FileCheck %s --check-prefix=TAGS
// RUN: rocmlir-opt -rock-split-cross-tile-fusion -rock-elementwise-to-gridwise -rock-link-split-kernels %s | FileCheck %s --check-prefix=ORCH
// RUN: rocmlir-driver -kernel-pipeline=gpu -arch gfx942 %s | FileCheck %s --check-prefix=HOST

// AIROCMLIR-709 Phase 3: host driver for a cross-tile fusion split.
//
// rock-split-cross-tile-fusion outlines @mlir_double_slice_add into a gemm
// kernel + an elementwise kernel and tags each half with split-link metadata.
// rock-link-split-kernels (run after rock-elementwise-to-gridwise has
// appended the elementwise output argument) reconstructs a host function named
// after the original kernel that allocates the intermediate device buffer and
// launches both kernels in order. The kernel func.call ops are later rewritten
// to gpu.launch_func by rock-emit-gpu-binary.

// --- The split records split-link metadata on both halves. ---
// TAGS: func.func @mlir_double_slice_add_gemm
// TAGS-SAME:  rock.split_arg_src = array<i64: 0, 1, -1>
// TAGS-SAME:  rock.split_group = "mlir_double_slice_add"
// TAGS-SAME:  rock.split_role = "gemm"
// TAGS: func.func @mlir_double_slice_add_elementwise
// TAGS-SAME:  rock.split_arg_src = array<i64: -1>
// TAGS-SAME:  rock.split_group = "mlir_double_slice_add"
// TAGS-SAME:  rock.split_out_src = array<i64: 2>
// TAGS-SAME:  rock.split_role = "elementwise"
// --- The split leaves a private, bodyless declaration with the original
//     kernel's name so the host's call stays resolvable until
//     rock-link-split-kernels fills in the body. ---
// TAGS: func.func private @mlir_double_slice_add(tensor<2048xf16>, tensor<1024xf16>, tensor<1024xf16>) -> tensor<1024xf16>{{$}}

// --- The host driver function has the original kernel's name and
//     signature, is NOT a rock.kernel, and chains the two launches through an
//     intermediate device buffer. The metadata is stripped from the kernels.
//     It is the private declaration left by the split (so the host call stays
//     resolvable), filled in here, so it prints after the two kernels. ---
// ORCH:       func.func @mlir_double_slice_add_gemm
// ORCH-NOT:     rock.split_
// ORCH:       func.func @mlir_double_slice_add_elementwise
// ORCH-NOT:     rock.split_
// ORCH:       func.func private @mlir_double_slice_add(%[[A0:.*]]: tensor<2048xf16>, %[[A1:.*]]: tensor<1024xf16>, %[[OUT:.*]]: tensor<1024xf16>) -> tensor<1024xf16> {
// ORCH-NEXT:    %[[DEV:.*]] = gpu.alloc () : memref<2048xf16>
// ORCH-NEXT:    %[[INTER:.*]] = bufferization.to_tensor %[[DEV]] restrict writable : memref<2048xf16> to tensor<2048xf16>
// ORCH-NEXT:    %[[G:.*]] = call @mlir_double_slice_add_gemm(%[[A0]], %[[A1]], %[[INTER]]) : (tensor<2048xf16>, tensor<1024xf16>, tensor<2048xf16>) -> tensor<2048xf16>
// ORCH-NEXT:    %[[O:.*]] = call @mlir_double_slice_add_elementwise(%[[G]], %[[OUT]]) : (tensor<2048xf16>, tensor<1024xf16>) -> tensor<1024xf16>
// ORCH-NEXT:    gpu.dealloc %[[DEV]] : memref<2048xf16>
// ORCH-NEXT:    return %[[O]] : tensor<1024xf16>

// --- Through the full gpu kernel pipeline the host driver function is
//     serialized as a host function (rock.host_functions) while both kernels
//     become tt.func, and each kernel gets its own grid size. ---
// HOST: rock.grid_size.mlir_double_slice_add_elementwise
// HOST: rock.grid_size.mlir_double_slice_add_gemm
// HOST: rock.host_functions =
// HOST-SAME: gpu.alloc
// HOST-SAME: @mlir_double_slice_add_gemm
// HOST-SAME: @mlir_double_slice_add_elementwise
// HOST-SAME: gpu.dealloc
// HOST: tt.func @mlir_double_slice_add_gemm
// HOST: tt.func @mlir_double_slice_add_elementwise

#map = affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 16 + d2)>
#map1 = affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 16 + d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1 + 16, d2)>
#map4 = affine_map<(d0) -> (d0 floordiv 256, (d0 mod 256) floordiv 16, d0 mod 16)>
#transform_map = #rock.transform_map<#map by [<Unmerge{4, 16, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 16, 16] -> [1024]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{4, 32, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 32, 16] -> [2048]>
#transform_map2 = #rock.transform_map<#map2 by [<Slice{0, 4, 0, 16, 0, 16} ["dim0_sliced", "dim1_sliced", "dim2_sliced"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [4, 16, 16] -> [4, 32, 16]>
#transform_map3 = #rock.transform_map<#map3 by [<Slice{0, 4, 16, 32, 0, 16} ["dim0_sliced", "dim1_sliced", "dim2_sliced"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [4, 16, 16] -> [4, 32, 16]>
#transform_map4 = #rock.transform_map<#map4 by [<Merge{4, 16, 16} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [1024] -> [4, 16, 16]>

module attributes {rock.arch = "gfx942"} {
  func.func @mlir_double_slice_add(%arg0: tensor<2048xf16>, %arg1: tensor<1024xf16>, %arg2: tensor<1024xf16>) -> tensor<1024xf16> attributes {rock.kernel = "mixr"} {
    %0 = rock.transform %arg1 by #transform_map : tensor<1024xf16> to tensor<4x16x16xf16>
    %1 = rock.transform %arg0 by #transform_map1 : tensor<2048xf16> to tensor<4x32x16xf16>
    %2 = rock.gemm %1 * %0 : tensor<4x32x16xf16> * tensor<4x16x16xf16> -> tensor<4x32x16xf16>
    %3 = rock.transform %2 by #transform_map2 : tensor<4x32x16xf16> to tensor<4x16x16xf16>
    %4 = rock.transform %2 by #transform_map3 : tensor<4x32x16xf16> to tensor<4x16x16xf16>
    %5 = arith.addf %3, %4 : tensor<4x16x16xf16>
    %6 = rock.transform %5 by #transform_map4 : tensor<4x16x16xf16> to tensor<1024xf16>
    %7 = rock.store %6 to %arg2 by set : tensor<1024xf16> -> tensor<1024xf16> to tensor<1024xf16>
    return %7 : tensor<1024xf16>
  }
}
