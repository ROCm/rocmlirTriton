// End-to-end (rock IR -> Triton) check for the rock.o_transposed metadata flow
// on a rock.attention kernel (the companion gemm case lives in
// accelerate-matmul-preserve-rock-metadata.mlir).
//
// The input is a transposed-output attention (rock.attention with {oTransposed},
// i.e. --transO) with a trailing arith.addf bias fusion. Both of its matmuls
// (Q*K^T and softmax(qk)*V) lower to rock.blockwise_gemm and reach the
// column-major / M-fast output store (through the addf fusion), so
// rock-add-triton-metadata tags both with rock.o_transposed = <true>. Driving it
// through the rocmlir Triton kernel pipeline exercises:
//   * RockToTTIR, which carries the discardable attribute onto both lowered
//     tt.dot ops;
//   * the downstream Triton patch
//     (triton-patches/patch-wmma-preserve-discardable-attrs.patch), which keeps
//     the attribute when tritonamdgpu-accelerate-matmul rewrites each tt.dot into
//     a WMMA (ttg.amd_wmma) dot;
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

// --- gfx1200 (WMMA v2): metadata survives accelerate-matmul on both dots ---
// RUN: sed s/##ARCH##/gfx1200/g %s | rocmlir-driver -c -o %t -arch gfx1200 --mlir-print-ir-after=tritonamdgpu-accelerate-matmul 2>&1 | FileCheck %s --check-prefix=ACCEL12
// --- gfx1200 (WMMA v2): set-matmul-output-transpose flips true -> false ---
// RUN: sed s/##ARCH##/gfx1200/g %s | rocmlir-driver -c -o %t -arch gfx1200 --mlir-print-ir-after=rock-set-matmul-output-transpose 2>&1 | FileCheck %s --check-prefix=SETT12

// --- gfx1250 (WMMA v3): metadata survives accelerate-matmul on both dots ---
// RUN: sed s/##ARCH##/gfx1250/g %s | rocmlir-driver -c -o %t -arch gfx1250 --mlir-print-ir-after=tritonamdgpu-accelerate-matmul 2>&1 | FileCheck %s --check-prefix=ACCEL1250
// --- gfx1250 (WMMA v3): set-matmul-output-transpose flips true -> false ---
// RUN: sed s/##ARCH##/gfx1250/g %s | rocmlir-driver -c -o %t -arch gfx1250 --mlir-print-ir-after=rock-set-matmul-output-transpose 2>&1 | FileCheck %s --check-prefix=SETT1250

// --- gfx1100 (WMMA v1): metadata survives accelerate-matmul on both dots ---
// RUN: sed s/##ARCH##/gfx1100/g %s | rocmlir-driver -c -o %t -arch gfx1100 --mlir-print-ir-after=tritonamdgpu-accelerate-matmul 2>&1 | FileCheck %s --check-prefix=ACCEL11
// --- gfx1100 (WMMA v1): set-matmul-output-transpose keeps isTranspose = true ---
// RUN: sed s/##ARCH##/gfx1100/g %s | rocmlir-driver -c -o %t -arch gfx1100 --mlir-print-ir-after=rock-set-matmul-output-transpose 2>&1 | FileCheck %s --check-prefix=SETT11

// ACCEL12: #[[$WMMA:.+]] = #ttg.amd_wmma<{version = 2, isTranspose = true,
// ACCEL12: tt.dot
// ACCEL12-SAME: rock.o_transposed = #rock.o_transposed<true>
// ACCEL12-SAME: -> tensor<64x64xf32, #[[$WMMA]]>
// ACCEL12: tt.dot
// ACCEL12-SAME: rock.o_transposed = #rock.o_transposed<true>
// ACCEL12-SAME: -> tensor<64x64xf32, #[[$WMMA]]>

// The metadata is discardable and left in place once consumed.
// SETT12: #[[$WMMAT:.+]] = #ttg.amd_wmma<{version = 2, isTranspose = false,
// SETT12: tt.dot
// SETT12-SAME: rock.o_transposed = #rock.o_transposed<true>
// SETT12-SAME: -> tensor<64x64xf32, #[[$WMMAT]]>
// SETT12: tt.dot
// SETT12-SAME: rock.o_transposed = #rock.o_transposed<true>
// SETT12-SAME: -> tensor<64x64xf32, #[[$WMMAT]]>

// ACCEL1250: #[[$WMMA1250:.+]] = #ttg.amd_wmma<{version = 3, isTranspose = true,
// ACCEL1250: tt.dot
// ACCEL1250-SAME: rock.o_transposed = #rock.o_transposed<true>
// ACCEL1250-SAME: -> tensor<64x64xf32, #[[$WMMA1250]]>
// ACCEL1250: tt.dot
// ACCEL1250-SAME: rock.o_transposed = #rock.o_transposed<true>
// ACCEL1250-SAME: -> tensor<64x64xf32, #[[$WMMA1250]]>

// The metadata is discardable and left in place once consumed.
// SETT1250: #[[$WMMA1250T:.+]] = #ttg.amd_wmma<{version = 3, isTranspose = false,
// SETT1250: tt.dot
// SETT1250-SAME: rock.o_transposed = #rock.o_transposed<true>
// SETT1250-SAME: -> tensor<64x64xf32, #[[$WMMA1250T]]>
// SETT1250: tt.dot
// SETT1250-SAME: rock.o_transposed = #rock.o_transposed<true>
// SETT1250-SAME: -> tensor<64x64xf32, #[[$WMMA1250T]]>

// ACCEL11: #[[$WMMA11:.+]] = #ttg.amd_wmma<{version = 1, isTranspose = true,
// ACCEL11: tt.dot
// ACCEL11-SAME: rock.o_transposed = #rock.o_transposed<true>
// ACCEL11-SAME: -> tensor<64x64xf32, #[[$WMMA11]]>
// ACCEL11: tt.dot
// ACCEL11-SAME: rock.o_transposed = #rock.o_transposed<true>
// ACCEL11-SAME: -> tensor<64x64xf32, #[[$WMMA11]]>

// For WMMA v1 a column-major output already wants isTranspose = true, so both
// dots are left untouched (no flip).
// SETT11: #[[$WMMA11T:.+]] = #ttg.amd_wmma<{version = 1, isTranspose = true,
// SETT11: tt.dot
// SETT11-SAME: rock.o_transposed = #rock.o_transposed<true>
// SETT11-SAME: -> tensor<64x64xf32, #[[$WMMA11T]]>
// SETT11: tt.dot
// SETT11-SAME: rock.o_transposed = #rock.o_transposed<true>
// SETT11-SAME: -> tensor<64x64xf32, #[[$WMMA11T]]>
// SETT11-NOT: isTranspose = false

#map = affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1 * 256 + d2)>
#map2 = affine_map<(d0) -> (0, d0 floordiv 256, d0 mod 256)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 64} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 64] -> [16384]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{64, 256} ["head_qk", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 256] -> [16384]>
#transform_map2 = #rock.transform_map<#map by [<Unmerge{256, 64} ["seq_k", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 64] -> [16384]>
#transform_map3 = #rock.transform_map<#map2 by [<Merge{64, 256} ["raw"] at [0] -> ["head_v", "seq_q"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [16384] -> [1, 64, 256]>
#map_bias = affine_map<(d0, d1, d2) -> (d1 * 256 + d2)>
#transform_map_bias = #rock.transform_map<#map_bias by [<Unmerge{64, 256} ["head_v", "seq_q"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 256] -> [16384]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:##ARCH##"} {
  func.func @rock_attention(%arg0: tensor<16384xf16>, %arg1: tensor<16384xf16>, %arg2: tensor<16384xf16>, %bias: tensor<16384xf16>, %arg3: tensor<16384xf16>) -> tensor<16384xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:##ARCH##", rock.kernel} {
    %0 = rock.transform %arg0 by #transform_map : tensor<16384xf16> to tensor<1x256x64xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<16384xf16> to tensor<1x64x256xf16>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<16384xf16> to tensor<1x256x64xf16>
    %result = rock.attention{
     qk = %0 * %1 : tensor<1x256x64xf16>, tensor<1x64x256xf16>
     qk = elementwise {
    ^bb0(%arg4: tensor<1x256x256xf16>):
      rock.yield %arg4 : tensor<1x256x256xf16>
    }
     softmax(qk) * %2 : tensor<1x256x64xf16>
    // Pin the tuning params so both accelerator dot tiles are a fixed 64x64 for
    // every arch (otherwise the default perf_config -- and thus the dot tile
    // shapes checked below -- can vary by arch / toolchain version).
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, oTransposed, perf_config = "attn:v1:64,64,64,1,1,4,16,1,1,0,0", softmaxType = f32, splitKV = 1 : i32} -> tensor<1x64x256xf16>
    %biasT = rock.transform %bias by #transform_map_bias : tensor<16384xf16> to tensor<1x64x256xf16>
    %add = arith.addf %result, %biasT : tensor<1x64x256xf16>
    %3 = rock.transform %add by #transform_map3 : tensor<1x64x256xf16> to tensor<16384xf16>
    %4 = rock.store %3 to %arg3 by set : tensor<16384xf16> -> tensor<16384xf16> to tensor<16384xf16>
    return %4 : tensor<16384xf16>
  }
}
