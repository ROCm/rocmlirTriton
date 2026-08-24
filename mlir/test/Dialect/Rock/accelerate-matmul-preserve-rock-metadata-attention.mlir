// End-to-end (rock IR -> Triton) check for the rock.o_transposed metadata flow
// on a rock.attention kernel (the companion gemm case lives in
// accelerate-matmul-preserve-rock-metadata.mlir).
//
// The input is a transposed-output attention (rock.attention with {oTransposed},
// i.e. --transO) with a trailing arith.addf bias fusion. It lowers to two
// rock.blockwise_gemm ops: the scores gemm (Q*K^T) and the value gemm
// (softmax(qk)*V). Only the value gemm is stored: its result reaches the
// column-major / M-fast output store (through the addf fusion), so
// rock-add-triton-metadata tags it with rock.o_transposed = <true>. The scores
// gemm feeds the value gemm as an operand (a chained-dot head), so it has no
// stored output layout and is intentionally left untagged -- even though it also
// reaches the store indirectly through the softmax statistics. Leaving it
// untagged keeps its accelerator-native layout, which on WMMA v1 (gfx1100) is
// what maps the softmax row reduction into registers instead of cross-lane DPP.
//
// Driving it through the rocmlir Triton kernel pipeline exercises:
//   * RockToTTIR, which carries the discardable attribute onto the tagged
//     (value) tt.dot only;
//   * the downstream Triton patch
//     (triton-patches/patch-wmma-preserve-discardable-attrs.patch), which keeps
//     the attribute when tritonamdgpu-accelerate-matmul rewrites the tt.dot into
//     a WMMA (ttg.amd_wmma) dot;
//   * rock-set-matmul-output-transpose, which consumes the metadata on the value
//     dot and selects its WMMA result layout's isTranspose flag (version-
//     dependent), while leaving the untagged scores dot at the default layout.
//
// gfx1200 uses WMMA version 2, where a column-major output wants isTranspose =
// false, so the pass flips the value dot from the accelerate-matmul default
// (true -> false) and leaves the scores dot at true.
// gfx1250 uses WMMA version 3, which behaves like version 2 (column-major wants
// isTranspose = false), so the pass also flips the value dot true -> false.
// gfx1100 uses WMMA version 1, where a column-major output already wants
// isTranspose = true, so both dots keep isTranspose = true and only the metadata
// on the value dot is dropped from the layout decision.

// --- gfx1200 (WMMA v2): metadata survives accelerate-matmul on the value dot ---
// RUN: sed s/##ARCH##/gfx1200/g %s | rocmlir-driver -c -arch gfx1200 --mlir-print-ir-after=tritonamdgpu-accelerate-matmul 2>&1 1>/dev/null | FileCheck %s --check-prefix=ACCEL12
// --- gfx1200 (WMMA v2): set-matmul-output-transpose flips true -> false ---
// RUN: sed s/##ARCH##/gfx1200/g %s | rocmlir-driver -c -arch gfx1200 --mlir-print-ir-after=rock-set-matmul-output-transpose 2>&1 1>/dev/null | FileCheck %s --check-prefix=SETT12

// --- gfx1250 (WMMA v3): metadata survives accelerate-matmul on the value dot ---
// RUN: sed s/##ARCH##/gfx1250/g %s | rocmlir-driver -c -arch gfx1250 --mlir-print-ir-after=tritonamdgpu-accelerate-matmul 2>&1 1>/dev/null | FileCheck %s --check-prefix=ACCEL1250
// --- gfx1250 (WMMA v3): set-matmul-output-transpose flips true -> false ---
// RUN: sed s/##ARCH##/gfx1250/g %s | rocmlir-driver -c -arch gfx1250 --mlir-print-ir-after=rock-set-matmul-output-transpose 2>&1 1>/dev/null | FileCheck %s --check-prefix=SETT1250

// --- gfx1100 (WMMA v1): metadata survives accelerate-matmul on the value dot ---
// RUN: sed s/##ARCH##/gfx1100/g %s | rocmlir-driver -c -arch gfx1100 --mlir-print-ir-after=tritonamdgpu-accelerate-matmul 2>&1 1>/dev/null | FileCheck %s --check-prefix=ACCEL11
// --- gfx1100 (WMMA v1): set-matmul-output-transpose keeps isTranspose = true ---
// RUN: sed s/##ARCH##/gfx1100/g %s | rocmlir-driver -c -arch gfx1100 --mlir-print-ir-after=rock-set-matmul-output-transpose 2>&1 1>/dev/null | FileCheck %s --check-prefix=SETT11

// Both dots share the accelerate-matmul default WMMA (isTranspose = true). The
// scores dot is untagged; only the value dot carries the metadata.
// ACCEL12: #[[$WMMA:.+]] = #ttg.amd_wmma<{version = 2, isTranspose = true,
// ACCEL12:      tt.dot
// ACCEL12-SAME:   -> tensor<64x64xf32, #[[$WMMA]]>
// ACCEL12-NOT:  rock.o_transposed
// ACCEL12:      tt.dot {{.*}}rock.o_transposed = #rock.o_transposed<true>
// ACCEL12-SAME:   -> tensor<64x64xf32, #[[$WMMA]]>

// The value dot is flipped to isTranspose = false; the scores dot stays at the
// default isTranspose = true. The metadata is discardable and left in place.
// SETT12-DAG: #[[$WMMA:.+]] = #ttg.amd_wmma<{version = 2, isTranspose = true,
// SETT12-DAG: #[[$WMMAT:.+]] = #ttg.amd_wmma<{version = 2, isTranspose = false,
// SETT12:      tt.dot
// SETT12-SAME:   -> tensor<64x64xf32, #[[$WMMA]]>
// SETT12-NOT:  rock.o_transposed
// SETT12:      tt.dot {{.*}}rock.o_transposed = #rock.o_transposed<true>
// SETT12-SAME:   -> tensor<64x64xf32, #[[$WMMAT]]>

// ACCEL1250: #[[$WMMA1250:.+]] = #ttg.amd_wmma<{version = 3, isTranspose = true,
// ACCEL1250:      tt.dot
// ACCEL1250-SAME:   -> tensor<64x64xf32, #[[$WMMA1250]]>
// ACCEL1250-NOT:  rock.o_transposed
// ACCEL1250:      tt.dot {{.*}}rock.o_transposed = #rock.o_transposed<true>
// ACCEL1250-SAME:   -> tensor<64x64xf32, #[[$WMMA1250]]>

// The value dot is flipped to isTranspose = false; the scores dot stays at the
// default isTranspose = true. The metadata is discardable and left in place.
// SETT1250-DAG: #[[$WMMA1250:.+]] = #ttg.amd_wmma<{version = 3, isTranspose = true,
// SETT1250-DAG: #[[$WMMA1250T:.+]] = #ttg.amd_wmma<{version = 3, isTranspose = false,
// SETT1250:      tt.dot
// SETT1250-SAME:   -> tensor<64x64xf32, #[[$WMMA1250]]>
// SETT1250-NOT:  rock.o_transposed
// SETT1250:      tt.dot {{.*}}rock.o_transposed = #rock.o_transposed<true>
// SETT1250-SAME:   -> tensor<64x64xf32, #[[$WMMA1250T]]>

// ACCEL11: #[[$WMMA11:.+]] = #ttg.amd_wmma<{version = 1, isTranspose = true,
// ACCEL11:      tt.dot
// ACCEL11-SAME:   -> tensor<64x64xf32, #[[$WMMA11]]>
// ACCEL11-NOT:  rock.o_transposed
// ACCEL11:      tt.dot {{.*}}rock.o_transposed = #rock.o_transposed<true>
// ACCEL11-SAME:   -> tensor<64x64xf32, #[[$WMMA11]]>

// For WMMA v1 a column-major output already wants isTranspose = true, so the
// value dot is left untouched (no flip); the untagged scores dot also stays at
// isTranspose = true. Both keep the same layout.
// SETT11: #[[$WMMA11T:.+]] = #ttg.amd_wmma<{version = 1, isTranspose = true,
// SETT11:      tt.dot
// SETT11-SAME:   -> tensor<64x64xf32, #[[$WMMA11T]]>
// SETT11-NOT:  rock.o_transposed
// SETT11:      tt.dot {{.*}}rock.o_transposed = #rock.o_transposed<true>
// SETT11-SAME:   -> tensor<64x64xf32, #[[$WMMA11T]]>
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
