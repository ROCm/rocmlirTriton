// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Without a perf config, the tuning parameters fall back to the front of the
// quick-tuning list, which must compile. Triton's pipeliner gives every global
// load that feeds the dot its own LDS tile per stage, so `add(x, y) * z` keeps
// two A tiles per stage live. The first gfx950 f32 entry (64x64x128,
// numStages=2) needs (2 A + B) * 2 = 192 KiB of the 160 KiB on gfx950, and was
// picked while the estimate charged one A tile. This is the default MIGraphX
// compiles with benchmarking skipped (e.g. eager mode in test_add_dot).
//
// RUN: rocmlir-driver --kernel-pipeline=gpu,triton --arch gfx950 %s | FileCheck %s

// The triton stage ends in the LLVM dialect, after rock-check-lds-limit.
// CHECK: ttg.shared = {{[0-9]+}}
// CHECK: llvm.func @mlir_add_dot

#map = affine_map<(d0, d1) -> (d0 * 256 + d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1 * 256 + d2)>
#map4 = affine_map<(d0) -> (0, d0 floordiv 256, d0 mod 256)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 256} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>] bounds = [256, 256] -> [65536]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{256} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 256, 256] -> [256, 256]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{256, 256} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 256, 256] -> [65536]>
#transform_map5 = #rock.transform_map<#map4 by [<Merge{1, 256, 256} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [65536] -> [1, 256, 256]>
module {
  func.func @mlir_add_dot(%arg0: tensor<65536xf32>, %arg1: tensor<65536xf32>, %arg2: tensor<65536xf32>, %arg3: tensor<65536xf32>) -> tensor<65536xf32> attributes {rock.arch = "gfx950:sramecc+:xnack-", rock.input_fusions = ["addf"], rock.kernel = "mixr", rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
    %0 = rock.transform %arg1 by #transform_map : tensor<65536xf32> to tensor<256x256xf32>
    %1 = rock.transform %arg0 by #transform_map : tensor<65536xf32> to tensor<256x256xf32>
    %2 = arith.addf %1, %0 : tensor<256x256xf32>
    %3 = rock.transform %2 by #transform_map1 : tensor<256x256xf32> to tensor<1x256x256xf32>
    %4 = rock.transform %arg2 by #transform_map2 : tensor<65536xf32> to tensor<1x256x256xf32>
    %5 = rock.gemm %3 * %4 : tensor<1x256x256xf32> * tensor<1x256x256xf32> -> tensor<1x256x256xf32>
    %6 = rock.transform %5 by #transform_map5 : tensor<1x256x256xf32> to tensor<65536xf32>
    %7 = rock.store %6 to %arg3 by set : tensor<65536xf32> -> tensor<65536xf32> to tensor<65536xf32>
    return %7 : tensor<65536xf32>
  }
}
