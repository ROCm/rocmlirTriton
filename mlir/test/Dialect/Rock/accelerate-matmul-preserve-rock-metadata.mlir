// End-to-end (rock IR -> Triton) check for the rock.o_transposed metadata flow.
//
// The input is a transposed-output GEMM (rock.gemm with {oTransposed}) with a
// trailing arith.addf bias fusion. Driving it through the rocmlir Triton kernel
// pipeline exercises:
//   * rock-add-triton-metadata, which (seeing the column-major / M-fast store
//     reached through the addf fusion) tags the rock.blockwise_gemm with
//     rock.o_transposed = <true>;
//   * RockToTTIR, which carries that discardable attribute onto the lowered
//     tt.dot;
//   * the downstream Triton patch
//     (triton-patches/patch-wmma-preserve-discardable-attrs.patch), which keeps
//     the discardable attribute when tritonamdgpu-accelerate-matmul rewrites the
//     tt.dot into a WMMA (ttg.amd_wmma) dot;
//   * rock-set-matmul-output-transpose, which consumes the metadata and selects
//     the WMMA result layout's isTranspose flag (version-dependent).
//
// gfx1200 uses WMMA version 2, where a column-major output wants isTranspose =
// false, so the pass flips the accelerate-matmul default (true -> false).
// gfx1250 uses WMMA version 3, which behaves like version 2 (column-major wants
// isTranspose = false), so the pass also flips true -> false.
// gfx1100 uses WMMA version 1, where a column-major output already wants
// isTranspose = true, so the layout is left as-is and only the metadata is
// dropped.

// --- gfx1200 (WMMA v2): metadata survives accelerate-matmul ---
// RUN: sed s/##ARCH##/gfx1200/g %s | rocmlir-driver -c -o %t -arch gfx1200 --mlir-print-ir-after=tritonamdgpu-accelerate-matmul 2>&1 | FileCheck %s --check-prefix=ACCEL12
// --- gfx1200 (WMMA v2): set-matmul-output-transpose flips true -> false ---
// RUN: sed s/##ARCH##/gfx1200/g %s | rocmlir-driver -c -o %t -arch gfx1200 --mlir-print-ir-after=rock-set-matmul-output-transpose 2>&1 | FileCheck %s --check-prefix=SETT12

// --- gfx1250 (WMMA v3): metadata survives accelerate-matmul ---
// RUN: sed s/##ARCH##/gfx1250/g %s | rocmlir-driver -c -o %t -arch gfx1250 --mlir-print-ir-after=tritonamdgpu-accelerate-matmul 2>&1 | FileCheck %s --check-prefix=ACCEL1250
// --- gfx1250 (WMMA v3): set-matmul-output-transpose flips true -> false ---
// RUN: sed s/##ARCH##/gfx1250/g %s | rocmlir-driver -c -o %t -arch gfx1250 --mlir-print-ir-after=rock-set-matmul-output-transpose 2>&1 | FileCheck %s --check-prefix=SETT1250

// --- gfx1100 (WMMA v1): metadata survives accelerate-matmul ---
// RUN: sed s/##ARCH##/gfx1100/g %s | rocmlir-driver -c -o %t -arch gfx1100 --mlir-print-ir-after=tritonamdgpu-accelerate-matmul 2>&1 | FileCheck %s --check-prefix=ACCEL11
// --- gfx1100 (WMMA v1): set-matmul-output-transpose keeps isTranspose = true ---
// RUN: sed s/##ARCH##/gfx1100/g %s | rocmlir-driver -c -o %t -arch gfx1100 --mlir-print-ir-after=rock-set-matmul-output-transpose 2>&1 | FileCheck %s --check-prefix=SETT11

// ACCEL12: #[[$WMMA:.+]] = #ttg.amd_wmma<{version = 2, isTranspose = true,
// ACCEL12: tt.dot
// ACCEL12-SAME: rock.o_transposed = #rock.o_transposed<true>
// ACCEL12-SAME: -> tensor<64x64xf32, #[[$WMMA]]>

// The metadata is discardable and left in place once consumed.
// SETT12: #[[$WMMAT:.+]] = #ttg.amd_wmma<{version = 2, isTranspose = false,
// SETT12: tt.dot
// SETT12-SAME: rock.o_transposed = #rock.o_transposed<true>
// SETT12-SAME: -> tensor<64x64xf32, #[[$WMMAT]]>

// ACCEL1250: #[[$WMMA1250:.+]] = #ttg.amd_wmma<{version = 3, isTranspose = true,
// ACCEL1250: tt.dot
// ACCEL1250-SAME: rock.o_transposed = #rock.o_transposed<true>
// ACCEL1250-SAME: -> tensor<64x64xf32, #[[$WMMA1250]]>

// The metadata is discardable and left in place once consumed.
// SETT1250: #[[$WMMA1250T:.+]] = #ttg.amd_wmma<{version = 3, isTranspose = false,
// SETT1250: tt.dot
// SETT1250-SAME: rock.o_transposed = #rock.o_transposed<true>
// SETT1250-SAME: -> tensor<64x64xf32, #[[$WMMA1250T]]>

// ACCEL11: #[[$WMMA11:.+]] = #ttg.amd_wmma<{version = 1, isTranspose = true,
// ACCEL11: tt.dot
// ACCEL11-SAME: rock.o_transposed = #rock.o_transposed<true>
// ACCEL11-SAME: -> tensor<64x64xf32, #[[$WMMA11]]>

// For WMMA v1 a column-major output already wants isTranspose = true, so the dot
// is left untouched (no flip, no new layout).
// SETT11: #[[$WMMA11T:.+]] = #ttg.amd_wmma<{version = 1, isTranspose = true,
// SETT11: tt.dot
// SETT11-SAME: rock.o_transposed = #rock.o_transposed<true>
// SETT11-SAME: -> tensor<64x64xf32, #[[$WMMA11T]]>
// SETT11-NOT: isTranspose = false

#map = affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1 * 256 + d2)>
#map2 = affine_map<(d0) -> (0, d0 floordiv 128, d0 mod 128)>
#transform_map = #rock.transform_map<#map by [<Unmerge{128, 64} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 64] -> [8192]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{64, 256} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 256] -> [16384]>
#transform_map2 = #rock.transform_map<#map2 by [<Merge{256, 128} ["raw"] at [0] -> ["n", "m"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [32768] -> [1, 256, 128]>
#map_bias = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#transform_map_bias = #rock.transform_map<#map_bias by [<Unmerge{256, 128} ["n", "m"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:##ARCH##"} {
  func.func @rock_gemm(%arg0: tensor<8192xf16>, %arg1: tensor<16384xf16>, %bias: tensor<32768xf16>, %arg2: tensor<32768xf16>) -> tensor<32768xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:##ARCH##", rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 12 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<8192xf16> to tensor<1x128x64xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<16384xf16> to tensor<1x64x256xf16>
    // Pin the tuning params so the accelerator output tile is a fixed 64x64 for
    // every arch (otherwise the default perf_config -- and thus the dot tile
    // shape checked below -- can vary by arch / toolchain version).
    %2 = rock.gemm %0 * %1 {oTransposed, perf_config = "gemm:v1:64,64,64,1,1,4,0,1,2,0,0"} : tensor<1x128x64xf16> * tensor<1x64x256xf16> -> tensor<1x256x128xf16>
    %biasT = rock.transform %bias by #transform_map_bias : tensor<32768xf16> to tensor<1x256x128xf16>
    %add = arith.addf %2, %biasT : tensor<1x256x128xf16>
    %3 = rock.transform %add by #transform_map2 : tensor<1x256x128xf16> to tensor<32768xf16>
    %4 = rock.store %3 to %arg2 by set : tensor<32768xf16> -> tensor<32768xf16> to tensor<32768xf16>
    return %4 : tensor<32768xf16>
  }
}
