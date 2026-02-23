// TODO(rocmlirTriton): Adapt this test
// RUN: rocmlir-opt %s | FileCheck %s

// TODO(rocmlirTriton): Add dummy check to make FileCheck happy
// CHECK: module

// XXX: rocmlir-opt %s -split-input-file -rock-gemm-to-gridwise \
// XXX:    -rock-gridwise-gemm-to-blockwise -rock-blockwise-load-tile-to-threadwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering \
// XXX: | FileCheck %s --check-prefixes=BLOCKWISE
// XXX: rocmlir-opt %s -split-input-file -rock-gemm-to-gridwise \
// XXX:    -rock-gridwise-gemm-to-blockwise -rock-blockwise-load-tile-to-threadwise -rock-blockwise-gemm-to-threadwise \
// XXX:    -canonicalize -rock-threadwise-gemm-lowering -rock-analyze-memory-use \
// XXX: | FileCheck %s --check-prefixes=ANALYZE
// XXX: rocmlir-opt %s -split-input-file -rock-kernel-pipeline | FileCheck %s --check-prefixes=GPU

// Arbitrary testcase: the tuning parameters are set to prevent needing to go
// through `-rock-affix-params` and can be replaced as needed.
// #general_gemm_params = #rock.general_gemm_params<blockSize = 64, kPerBlock = 16, mPerBlock = 64, nPerBlock = 32, kPerThread = 1, mPerThread = 4, nPerThread = 2, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
// module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
// DISABLED-BLOCKWISE-LABEL: @rock_gemm
// DISABLED-BLOCKWISE: needs64BitIdx
// DISABLED-ANALYZE-LABEL: @rock_gemm
// DISABLED-ANALYZE-SAME: rock.64bitindex
// DISABLED-GPU-LABEL: @rock_gemm_module
// DISABLED-GPU-SAME: dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>
//   func.func @rock_gemm(%arg0: memref<1x32768x32768xf32>, %arg1: memref<1x32768x1xf32>, %arg2: memref<1x32768x1xf32>) attributes {block_size = 64 : i32, grid_size = 512 : i32, kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100", wave_size = 32 : i32} {
//     rock.gemm %arg2 = %arg0 * %arg1 storeMethod =  set {gridSize = 512 : i32, params = #general_gemm_params} : memref<1x32768x1xf32> = memref<1x32768x32768xf32> * memref<1x32768x1xf32>
//     return
//   }
// }

// -----

// #general_gemm_params = #rock.general_gemm_params<blockSize = 64, kPerBlock = 16, mPerBlock = 64, nPerBlock = 32, kPerThread = 1, mPerThread = 4, nPerThread = 2, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
// module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
// DISABLED-BLOCKWISE-LABEL: @rock_gemm
// DISABLED-BLOCKWISE-NOT: rock.64bitindex
// DISABLED-ANALYZE-LABEL: @rock_gemm
// DISABLED-ANALYZE-NOT: rock.64bitindex
// DISABLED-GPU-LABEL: @rock_gemm_module
// DISABLED-GPU-SAME: dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>
//   func.func @rock_gemm(%arg0: memref<1x8192x8192xf32>, %arg1: memref<1x8192x1xf32>, %arg2: memref<1x8192x1xf32>) attributes {block_size = 64 : i32, grid_size = 128 : i32, kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100", wave_size = 32 : i32} {
//     rock.gemm %arg2 = %arg0 * %arg1 storeMethod =  set {gridSize = 128 : i32, params = #general_gemm_params} : memref<1x8192x1xf32> = memref<1x8192x8192xf32> * memref<1x8192x1xf32>
//     return
//   }
// }
