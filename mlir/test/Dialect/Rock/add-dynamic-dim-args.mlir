// RUN: rocmlir-opt -rock-add-dynamic-dim-args -split-input-file %s | FileCheck %s
#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#a_map = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1 * 72 + d2)> by [<Unmerge{?, 72} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, ?, 72] -> [?]>
#b_map = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{72, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 72, 64] -> [4608]>
#c_map = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{1, ?, 64} ["col0", "col1", "col2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, ?, 64] -> [?]>

// All four gemm dimensions are appended, in G, M, N, K order, even though only
// M is dynamic here: the order is what identifies them, so it cannot depend on
// which ones happen to be known.

// CHECK-LABEL: func.func @dot_dynamic_m
// CHECK-SAME: (%{{.*}}: tensor<?xf32>, %{{.*}}: tensor<4608xf32>, %{{.*}}: tensor<?xf32>,
// CHECK-SAME: %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32)
// CHECK-NOT: rock.dyn_dim
func.func @dot_dynamic_m(%a: tensor<?xf32>, %b: tensor<4608xf32>, %c: tensor<?xf32>) -> tensor<?xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel = "mixr"} {
  %0 = rock.transform %a by #a_map : tensor<?xf32> to tensor<1x?x72xf32>
  %1 = rock.transform %b by #b_map : tensor<4608xf32> to tensor<1x72x64xf32>
  %2 = rock.gemm %0 * %1 {params = #gemm_params} : tensor<1x?x72xf32> * tensor<1x72x64xf32> -> tensor<1x?x64xf32>
  %3 = rock.transform %c by #c_map : tensor<?xf32> to tensor<1x?x64xf32>
  %4 = rock.store %2 to %3 by set : tensor<1x?x64xf32> -> tensor<?xf32> to tensor<1x?x64xf32>
  return %4 : tensor<?xf32>
}

// -----

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// A kernel whose shapes are fully known needs nothing at runtime.

// CHECK-LABEL: func.func @dot_static
// CHECK-SAME: (%{{.*}}: tensor<1x128x72xf32>, %{{.*}}: tensor<1x72x64xf32>)
// CHECK-NOT: i32
func.func @dot_static(%a: tensor<1x128x72xf32>, %b: tensor<1x72x64xf32>) -> tensor<1x128x64xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel = "mixr"} {
  %0 = rock.gemm %a * %b {params = #gemm_params} : tensor<1x128x72xf32> * tensor<1x72x64xf32> -> tensor<1x128x64xf32>
  return %0 : tensor<1x128x64xf32>
}

// -----

// A function that is not a kernel is left alone even if it has dynamic
// arguments.

// CHECK-LABEL: func.func @not_a_kernel
// CHECK-SAME: (%{{.*}}: tensor<?xf32>)
// CHECK-NOT: i32
func.func @not_a_kernel(%a: tensor<?xf32>) -> tensor<?xf32> {
  return %a : tensor<?xf32>
}

// -----

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#a_map = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1 * 72 + d2)> by [<Unmerge{?, 72} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, ?, 72] -> [?]>
#b_map = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{72, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 72, 64] -> [4608]>
#c_map = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{1, ?, 64} ["col0", "col1", "col2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, ?, 64] -> [?]>

// The caller derives M from the buffer it already passes. Undoing the
// `Unmerge{?, 72}` that the kernel views that buffer through is the compiler's
// job, because the coordinate transforms are the only place that relationship
// is written down.

// CHECK-LABEL: func.func @host
// CHECK-SAME: (%[[A:.*]]: tensor<?xf32>,
// CHECK: %[[G:.*]] = arith.constant 1 : i32
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[NUMEL:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?xf32>
// CHECK: %[[C72:.*]] = arith.constant 72 : index
// CHECK: %[[ROWS:.*]] = arith.divui %[[NUMEL]], %[[C72]] : index
// CHECK: %[[M:.*]] = arith.index_cast %[[ROWS]] : index to i32
// CHECK: %[[N:.*]] = arith.constant 64 : i32
// CHECK: %[[K:.*]] = arith.constant 72 : i32
// CHECK: call @kernel(%[[A]], %{{.*}}, %{{.*}}, %[[G]], %[[M]], %[[N]], %[[K]])
func.func @host(%a: tensor<?xf32>, %b: tensor<4608xf32>, %c: tensor<?xf32>) -> tensor<?xf32> {
  %0 = call @kernel(%a, %b, %c) : (tensor<?xf32>, tensor<4608xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func.func @kernel
// CHECK-SAME: %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32)
func.func @kernel(%a: tensor<?xf32>, %b: tensor<4608xf32>, %c: tensor<?xf32>) -> tensor<?xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel = "mixr"} {
  %0 = rock.transform %a by #a_map : tensor<?xf32> to tensor<1x?x72xf32>
  %1 = rock.transform %b by #b_map : tensor<4608xf32> to tensor<1x72x64xf32>
  %2 = rock.gemm %0 * %1 {params = #gemm_params} : tensor<1x?x72xf32> * tensor<1x72x64xf32> -> tensor<1x?x64xf32>
  %3 = rock.transform %c by #c_map : tensor<?xf32> to tensor<1x?x64xf32>
  %4 = rock.store %2 to %3 by set : tensor<1x?x64xf32> -> tensor<?xf32> to tensor<1x?x64xf32>
  return %4 : tensor<?xf32>
}
