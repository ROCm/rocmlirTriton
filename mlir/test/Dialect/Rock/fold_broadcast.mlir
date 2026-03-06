// RUN: rocmlir-opt --rock-fold-broadcast %s | FileCheck %s

// CHECK-LABEL: func.func @fold_broadcast_b
// CHECK:         %[[MERGE_A:.*]] = rock.transform %arg0 by {{.*}} : tensor<4x8x16xf16> to tensor<32x16xf16>
// CHECK:         %[[UNBCAST_B:.*]] = rock.transform %{{.*}} by {{.*}} : tensor<4x16x32xf16> to tensor<16x32xf16>
// CHECK:         %[[GEMM:.*]] = rock.gemm %[[MERGE_A]] * %[[UNBCAST_B]] : tensor<32x16xf16> * tensor<16x32xf16> -> tensor<32x32xf16>
// CHECK:         %[[RESHAPE:.*]] = rock.transform %[[GEMM]] by {{.*}} : tensor<32x32xf16> to tensor<4x8x32xf16>
// CHECK:         return %[[RESHAPE]] : tensor<4x8x32xf16>
#map_bcast = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#bcast_b = #rock.transform_map<#map_bcast by [
  <Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>,
  <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>,
  <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>
] bounds = [4, 16, 32] -> [1, 16, 32]>
func.func @fold_broadcast_b(%arg0: tensor<4x8x16xf16>, %arg1: tensor<1x16x32xf16>) -> tensor<4x8x32xf16> attributes {rock.kernel} {
  %0 = rock.transform %arg1 by #bcast_b : tensor<1x16x32xf16> to tensor<4x16x32xf16>
  %1 = rock.gemm %arg0 * %0 : tensor<4x8x16xf16> * tensor<4x16x32xf16> -> tensor<4x8x32xf16>
  return %1 : tensor<4x8x32xf16>
}

// CHECK-LABEL: func.func @fold_broadcast_a
// CHECK:         %[[UNBCAST_A:.*]] = rock.transform %{{.*}} by {{.*}} : tensor<4x8x16xf16> to tensor<8x16xf16>
// CHECK:         %[[MERGE_B:.*]] = rock.transform %arg1 by {{.*}} : tensor<4x16x32xf16> to tensor<16x128xf16>
// CHECK:         %[[GEMM:.*]] = rock.gemm %[[UNBCAST_A]] * %[[MERGE_B]] : tensor<8x16xf16> * tensor<16x128xf16> -> tensor<8x128xf16>
// CHECK:         %[[RESHAPE:.*]] = rock.transform %[[GEMM]] by {{.*}} : tensor<8x128xf16> to tensor<4x8x32xf16>
// CHECK:         return %[[RESHAPE]] : tensor<4x8x32xf16>
#bcast_a = #rock.transform_map<#map_bcast by [
  <Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>,
  <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>,
  <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>
] bounds = [4, 8, 16] -> [1, 8, 16]>
func.func @fold_broadcast_a(%arg0: tensor<1x8x16xf16>, %arg1: tensor<4x16x32xf16>) -> tensor<4x8x32xf16> attributes {rock.kernel} {
  %0 = rock.transform %arg0 by #bcast_a : tensor<1x8x16xf16> to tensor<4x8x16xf16>
  %1 = rock.gemm %0 * %arg1 : tensor<4x8x16xf16> * tensor<4x16x32xf16> -> tensor<4x8x32xf16>
  return %1 : tensor<4x8x32xf16>
}

#map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0 * 16 + d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["dim2", "dim0", "dim1"] at [0, 1, 2] -> ["dim2", "dim0", "dim1"] at [2, 0, 1]>] bounds = [1, 8, 32] -> [8, 32, 1]>
#transform_map1 = #rock.transform_map<#map1 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [4, 8, 32] -> [1, 8, 32]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{1, 16} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 16, 32] -> [16, 32]>
#transform_map3 = #rock.transform_map<#map1 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [4, 16, 32] -> [1, 16, 32]>

func.func @mlir_dot_add_1(%arg0: tensor<8x32x1xf16>, %arg1: tensor<4x8x16xf16>, %arg2: tensor<16x32xf16>) -> tensor<4x8x32xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel} {
 %0 = rock.transform %arg0 by #transform_map : tensor<8x32x1xf16> to tensor<1x8x32xf16>
 %1 = rock.transform %0 by #transform_map1 : tensor<1x8x32xf16> to tensor<4x8x32xf16>
 %2 = rock.transform %arg2 by #transform_map2 : tensor<16x32xf16> to tensor<1x16x32xf16>
 %3 = rock.transform %2 by #transform_map3 : tensor<1x16x32xf16> to tensor<4x16x32xf16>
 // CHECK: %[[foldA:.*]] = rock.transform %arg1 by {{.*}} : tensor<4x8x16xf16> to tensor<32x16xf16>
 // CHECK: %[[unbroadcastB:.*]] = rock.transform {{.*}} by {{.*}} : tensor<4x16x32xf16> to tensor<16x32xf16>
 // CHECK: %[[gemmOut:.*]] = rock.gemm %[[foldA]] * %[[unbroadcastB]] : tensor<32x16xf16> * tensor<16x32xf16> -> tensor<32x32xf16>
 // CHECK: %[[reshape:.*]] = rock.transform %[[gemmOut]] by {{.*}} : tensor<32x32xf16> to tensor<4x8x32xf16>
 %5 = rock.gemm %arg1 * %3 : tensor<4x8x16xf16> * tensor<4x16x32xf16> -> tensor<4x8x32xf16>
 %6 = tensor.empty() : tensor<4x8x32xf16>
 // CHECK: linalg.generic {{.*}} ins(%[[reshape]], {{.*}}, {{.*}})
 %7 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %1 : tensor<4x8x32xf16>, tensor<4x8x32xf16>) outs(%6 : tensor<4x8x32xf16>) {
 ^bb0(%in: f16, %in_0: f16, %out: f16):
   %8 = arith.addf %in, %in_0 : f16
   linalg.yield %8 : f16
 } -> tensor<4x8x32xf16>
 return %7 : tensor<4x8x32xf16>
}

#map4 = affine_map<(d0, d1, d2) -> (d1 * 16 + d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1, 0, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#transform_map4 = #rock.transform_map<#map4 by [<Unmerge{1, 16} ["exp0", "exp1"] at [1, 0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [16, 1, 32] -> [16, 32]>
#transform_map5 = #rock.transform_map<#map5 by [<Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [16, 4, 32] -> [16, 1, 32]>
#transform_map6 = #rock.transform_map<#map6 by [<PassThrough ["dim0", "dim1", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [4, 16, 32] -> [16, 4, 32]>
// CHECK-LABEL: func.func @mlir_dot_add_2
// CHECK: %[[foldA2:.*]] = rock.transform %arg1 by {{.*}} : tensor<4x8x16xf16> to tensor<32x16xf16>
// CHECK: %[[unbroadcastB2:.*]] = rock.transform {{.*}} by {{.*}} : tensor<4x16x32xf16> to tensor<16x32xf16>
// CHECK: %[[gemmOut2:.*]] = rock.gemm %[[foldA2]] * %[[unbroadcastB2]] {{.*}} : tensor<32x16xf16> * tensor<16x32xf16> -> tensor<32x32xf16>
// CHECK: %[[reshape2:.*]] = rock.transform %[[gemmOut2]] by {{.*}} : tensor<32x32xf16> to tensor<4x8x32xf16>
// CHECK: linalg.generic {{.*}} ins(%[[reshape2]], {{.*}}, {{.*}})
func.func @mlir_dot_add_2(%arg0: tensor<8x32x1xf16>, %arg1: tensor<4x8x16xf16>, %arg2: tensor<16x32xf16>) -> tensor<4x8x32xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel} {
 %0 = rock.transform %arg0 by #transform_map : tensor<8x32x1xf16> to tensor<1x8x32xf16>
 %1 = rock.transform %0 by #transform_map1 : tensor<1x8x32xf16> to tensor<4x8x32xf16>

 %2 = rock.transform %arg2 by #transform_map4 : tensor<16x32xf16> to tensor<16x1x32xf16>
 %3 = rock.transform %2 by #transform_map5 : tensor<16x1x32xf16> to tensor<16x4x32xf16>
 %p = rock.transform %3 by #transform_map6 : tensor<16x4x32xf16> to tensor<4x16x32xf16>

 %5 = rock.gemm %arg1 * %p {perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : tensor<4x8x16xf16> * tensor<4x16x32xf16> -> tensor<4x8x32xf16>
 %6 = tensor.empty() : tensor<4x8x32xf16>
 %7 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %1 : tensor<4x8x32xf16>, tensor<4x8x32xf16>) outs(%6 : tensor<4x8x32xf16>) {
 ^bb0(%in: f16, %in_0: f16, %out: f16):
   %8 = arith.addf %in, %in_0 : f16
   linalg.yield %8 : f16
 } -> tensor<4x8x32xf16>
 return %7 : tensor<4x8x32xf16>
}

#map7 = affine_map<(d0, d1, d2, d3) -> (d0 * 16 + d1 * 16  + d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map9 = affine_map<(d0, d1, d2) -> (d0 floordiv 4, d0 mod 4, d1, d2)>
#transform_map7 = #rock.transform_map<#map7 by [<Unmerge{1, 1, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>, <PassThrough ["exp3"] at [3] -> ["dim1"] at [1]>] bounds = [1, 1, 16, 32] -> [16, 32]>
#transform_map8 = #rock.transform_map<#map8 by [<Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 4, 16, 32] -> [1, 1, 16, 32]>
#transform_map9 = #rock.transform_map<#map9 by [<Merge{1, 4} ["dim0"] at [0] -> ["dim0", "dim1"] at [0, 1]>, <PassThrough ["dim1", "dim2"] at [1, 2] -> ["dim2", "dim3"] at [2, 3]>] bounds = [4, 16, 32] -> [1, 4, 16, 32]>

// CHECK-LABEL: func.func @mlir_dot_add_3
// CHECK: %[[foldA3:.*]] = rock.transform %arg1 by {{.*}} : tensor<4x8x16xf16> to tensor<32x16xf16>
// CHECK: %[[unbroadcastB3:.*]] = rock.transform {{.*}} by {{.*}} : tensor<4x16x32xf16> to tensor<16x32xf16>
// CHECK: %[[gemmOut3:.*]] = rock.gemm %[[foldA3]] * %[[unbroadcastB3]] : tensor<32x16xf16> * tensor<16x32xf16> -> tensor<32x32xf16>
// CHECK: %[[reshape3:.*]] = rock.transform %[[gemmOut3]] by {{.*}} : tensor<32x32xf16> to tensor<4x8x32xf16>
// CHECK: linalg.generic {{.*}} ins(%[[reshape3]], {{.*}}, {{.*}})
func.func @mlir_dot_add_3(%arg0: tensor<8x32x1xf16>, %arg1: tensor<4x8x16xf16>, %arg2: tensor<16x32xf16>) -> tensor<4x8x32xf16> attributes {rock.arch = "gfx1100", rock.kernel} {
  %0 = rock.transform %arg0 by #transform_map : tensor<8x32x1xf16> to tensor<1x8x32xf16>
  %1 = rock.transform %0 by #transform_map1 : tensor<1x8x32xf16> to tensor<4x8x32xf16>

  %2 = rock.transform %arg2 by #transform_map7 : tensor<16x32xf16> to tensor<1x1x16x32xf16>
  %3 = rock.transform %2 by #transform_map8 : tensor<1x1x16x32xf16> to tensor<1x4x16x32xf16>
  %p = rock.transform %3 by #transform_map9 : tensor<1x4x16x32xf16> to tensor<4x16x32xf16>

  %5 = rock.gemm %arg1 * %p : tensor<4x8x16xf16> * tensor<4x16x32xf16> -> tensor<4x8x32xf16>
  %6 = tensor.empty() : tensor<4x8x32xf16>
  %7 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %1 : tensor<4x8x32xf16>, tensor<4x8x32xf16>) outs(%6 : tensor<4x8x32xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.addf %in, %in_0 : f16
    linalg.yield %8 : f16
  } -> tensor<4x8x32xf16>
  return %7 : tensor<4x8x32xf16>
}

#map10 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#transform_map10 = #rock.transform_map<#map10 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [3, 2, 3] -> [1, 2, 3]>

// CHECK-LABEL: func.func @mlir_dot_broadcastA
// CHECK: %[[unbroadcastA:.*]] = rock.transform {{.*}} by {{.*}} : tensor<3x2x3xf16> to tensor<2x3xf16>
// CHECK: %[[foldB:.*]] = rock.transform %arg1 by {{.*}} : tensor<3x3x4xf16> to tensor<3x12xf16>
// CHECK: %[[gemmOut4:.*]] = rock.gemm %[[unbroadcastA]] * %[[foldB]] : tensor<2x3xf16> * tensor<3x12xf16> -> tensor<2x12xf16>
// CHECK: %[[reshape4:.*]] = rock.transform %[[gemmOut4]] by {{.*}} : tensor<2x12xf16> to tensor<3x2x4xf16>
// CHECK: return %[[reshape4]] : tensor<3x2x4xf16>
func.func @mlir_dot_broadcastA(%arg0: tensor<1x2x3xf16>, %arg1: tensor<3x3x4xf16>) -> tensor<3x2x4xf16> attributes {rock.arch = "gfx1100", rock.kernel = "mixr", num_cu = 42 : i64} {
  %0 = rock.transform %arg0 by #transform_map10 : tensor<1x2x3xf16> to tensor<3x2x3xf16>
  %1 = rock.gemm %0 * %arg1 : tensor<3x2x3xf16> * tensor<3x3x4xf16> -> tensor<3x2x4xf16>
  return %1 : tensor<3x2x4xf16>
}

#transform_map11 = #rock.transform_map<#map10 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [3, 3, 4] -> [1, 3, 4]>

// Both A and B are broadcast on batch dim — fold-broadcast pass does nothing.
// CHECK-LABEL: func.func @mlir_dot_both_broadcast
// CHECK: rock.gemm {{.*}} : tensor<3x2x3xf16> * tensor<3x3x4xf16> -> tensor<3x2x4xf16>
func.func @mlir_dot_both_broadcast(%arg0: tensor<1x2x3xf16>, %arg1: tensor<1x3x4xf16>) -> tensor<3x2x4xf16> attributes {rock.arch = "gfx1100", rock.kernel = "mixr", num_cu = 42 : i64} {
  %0 = rock.transform %arg0 by #transform_map10 : tensor<1x2x3xf16> to tensor<3x2x3xf16>
  %1 = rock.transform %arg1 by #transform_map11 : tensor<1x3x4xf16> to tensor<3x3x4xf16>
  %2 = rock.gemm %0 * %1 : tensor<3x2x3xf16> * tensor<3x3x4xf16> -> tensor<3x2x4xf16>
  return %2 : tensor<3x2x4xf16>
}

#map11 = affine_map<(d0, d1, d2) -> (d1 * 3 + d2)>
#transform_map12 = #rock.transform_map<#map11 by [<Unmerge{2, 3} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 2, 3] -> [6]>

// CHECK-LABEL: func.func @mlir_dot_broadcastA_addDim
// CHECK: %[[unbroadcastA5:.*]] = rock.transform {{.*}} by {{.*}} : tensor<3x2x3xf16> to tensor<2x3xf16>
// CHECK: %[[foldB5:.*]] = rock.transform %arg1 by {{.*}} : tensor<3x3x4xf16> to tensor<3x12xf16>
// CHECK: %[[gemmOut5:.*]] = rock.gemm %[[unbroadcastA5]] * %[[foldB5]] : tensor<2x3xf16> * tensor<3x12xf16> -> tensor<2x12xf16>
// CHECK: %[[reshape5:.*]] = rock.transform %[[gemmOut5]] by {{.*}} : tensor<2x12xf16> to tensor<3x2x4xf16>
// CHECK: return %[[reshape5]] : tensor<3x2x4xf16>
func.func @mlir_dot_broadcastA_addDim(%arg0: tensor<6xf16>, %arg1: tensor<3x3x4xf16>) -> tensor<3x2x4xf16> attributes {rock.arch = "gfx1100", rock.kernel = "mixr", num_cu = 42 : i64} {
  %0 = rock.transform %arg0 by #transform_map12 : tensor<6xf16> to tensor<1x2x3xf16>
  %1 = rock.transform %0 by #transform_map10 : tensor<1x2x3xf16> to tensor<3x2x3xf16>
  %2 = rock.gemm %1 * %arg1 : tensor<3x2x3xf16> * tensor<3x3x4xf16> -> tensor<3x2x4xf16>
  return %2 : tensor<3x2x4xf16>
}

// Scaled GEMM test: broadcast on B, so fold A and its scale, unbroadcast B and its scale
// CHECK-LABEL: func.func @mlir_dot_scaled_broadcastB
// CHECK: %[[foldA6:.*]] = rock.transform %arg0 by {{.*}} : tensor<4x8x16xf4E2M1FN> to tensor<32x16xf4E2M1FN>
// CHECK: %[[unbroadcastB6:.*]] = rock.transform {{.*}} by {{.*}} : tensor<4x16x32xf4E2M1FN> to tensor<16x32xf4E2M1FN>
// CHECK: %[[foldScaleA6:.*]] = rock.transform %arg2 by {{.*}} : tensor<4x8x16xf8E8M0FNU> to tensor<32x16xf8E8M0FNU>
// CHECK: %[[unbroadcastScaleB6:.*]] = rock.transform {{.*}} by {{.*}} : tensor<4x16x32xf8E8M0FNU> to tensor<16x32xf8E8M0FNU>
// CHECK: %[[gemmOut6:.*]] = rock.gemm %[[foldA6]] scaled by %[[foldScaleA6]] * %[[unbroadcastB6]] scaled by %[[unbroadcastScaleB6]] : tensor<32x16xf4E2M1FN> scaled by tensor<32x16xf8E8M0FNU> * tensor<16x32xf4E2M1FN> scaled by tensor<16x32xf8E8M0FNU> -> tensor<32x32xf16>
// CHECK: %[[reshape6:.*]] = rock.transform %[[gemmOut6]] by {{.*}} : tensor<32x32xf16> to tensor<4x8x32xf16>
// CHECK: return %[[reshape6]] : tensor<4x8x32xf16>
func.func @mlir_dot_scaled_broadcastB(%arg0: tensor<4x8x16xf4E2M1FN>, %arg1: tensor<1x16x32xf4E2M1FN>,
                                       %scaleA: tensor<4x8x16xf8E8M0FNU>, %scaleB: tensor<1x16x32xf8E8M0FNU>)
                                       -> tensor<4x8x32xf16> attributes {rock.arch = "gfx1100", rock.kernel} {
  %0 = rock.transform %arg1 by #transform_map3 : tensor<1x16x32xf4E2M1FN> to tensor<4x16x32xf4E2M1FN>
  %1 = rock.transform %scaleB by #transform_map3 : tensor<1x16x32xf8E8M0FNU> to tensor<4x16x32xf8E8M0FNU>
  %2 = rock.gemm %arg0 scaled by %scaleA * %0 scaled by %1 :
       tensor<4x8x16xf4E2M1FN> scaled by tensor<4x8x16xf8E8M0FNU> * tensor<4x16x32xf4E2M1FN> scaled by tensor<4x16x32xf8E8M0FNU> -> tensor<4x8x32xf16>
  return %2 : tensor<4x8x32xf16>
}

// Scaled GEMM test: broadcast on A, so unbroadcast A and its scale, fold B and its scale
// CHECK-LABEL: func.func @mlir_dot_scaled_broadcastA
// CHECK: %[[unbroadcastA7:.*]] = rock.transform {{.*}} by {{.*}} : tensor<3x2x3xf4E2M1FN> to tensor<2x3xf4E2M1FN>
// CHECK: %[[foldB7:.*]] = rock.transform %arg1 by {{.*}} : tensor<3x3x4xf4E2M1FN> to tensor<3x12xf4E2M1FN>
// CHECK: %[[unbroadcastScaleA7:.*]] = rock.transform {{.*}} by {{.*}} : tensor<3x2x3xf8E8M0FNU> to tensor<2x3xf8E8M0FNU>
// CHECK: %[[foldScaleB7:.*]] = rock.transform %arg3 by {{.*}} : tensor<3x3x4xf8E8M0FNU> to tensor<3x12xf8E8M0FNU>
// CHECK: %[[gemmOut7:.*]] = rock.gemm %[[unbroadcastA7]] scaled by %[[unbroadcastScaleA7]] * %[[foldB7]] scaled by %[[foldScaleB7]] : tensor<2x3xf4E2M1FN> scaled by tensor<2x3xf8E8M0FNU> * tensor<3x12xf4E2M1FN> scaled by tensor<3x12xf8E8M0FNU> -> tensor<2x12xf16>
// CHECK: %[[reshape7:.*]] = rock.transform %[[gemmOut7]] by {{.*}} : tensor<2x12xf16> to tensor<3x2x4xf16>
// CHECK: return %[[reshape7]] : tensor<3x2x4xf16>
func.func @mlir_dot_scaled_broadcastA(%arg0: tensor<1x2x3xf4E2M1FN>, %arg1: tensor<3x3x4xf4E2M1FN>,
                                       %scaleA: tensor<1x2x3xf8E8M0FNU>, %scaleB: tensor<3x3x4xf8E8M0FNU>)
                                       -> tensor<3x2x4xf16> attributes {rock.arch = "gfx1100", rock.kernel} {
  %0 = rock.transform %arg0 by #transform_map10 : tensor<1x2x3xf4E2M1FN> to tensor<3x2x3xf4E2M1FN>
  %1 = rock.transform %scaleA by #transform_map10 : tensor<1x2x3xf8E8M0FNU> to tensor<3x2x3xf8E8M0FNU>
  %2 = rock.gemm %0 scaled by %1 * %arg1 scaled by %scaleB :
       tensor<3x2x3xf4E2M1FN> scaled by tensor<3x2x3xf8E8M0FNU> * tensor<3x3x4xf4E2M1FN> scaled by tensor<3x3x4xf8E8M0FNU> -> tensor<3x2x4xf16>
  return %2 : tensor<3x2x4xf16>
}
