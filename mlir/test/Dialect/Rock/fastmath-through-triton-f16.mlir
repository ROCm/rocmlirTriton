// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The f16 arithmetic kernel of the fastmath-through-triton family: what happens
// to `rock-allow-fast-math-flags`' output once it crosses into Triton. See
// fastmath-through-triton.mlir for what the family checks and why each file
// holds exactly one kernel.
//
// f16 is the width the fused elementwise kernels actually run at, and it is
// where `arcp` is worth the most in the emitted code, so this file is the one
// that carries the assembly-level expectations.

// RUN: rocmlir-driver -arch=gfx942 -kernel-pipeline=gpu,triton %s | FileCheck %s

// gfx1250 swaps in the packed conversions for f32 and bf16, but not for f16: the
// f16 arithmetic stays scalar there and only the loads/truncations vectorize.
// Compiling it keeps that pinned, so a future change that starts packing f16
// arithmetic has to come here and say so.
// RUN: rocmlir-driver -arch=gfx1250 -kernel-pipeline=gpu,triton %s \
// RUN: | FileCheck %s --check-prefix=PACKED

// The expectations above stop at the LLVM dialect, which can only show that a
// flag survived, not that it bought anything. These two runs go all the way to
// assembly so the codegen difference is pinned too. The implicit-check-nots
// carry the negative half: with the flags on, no f16 divide falls back to the
// fixup sequence, and with them off, none reaches a bare reciprocal.
// RUN: AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx942 -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=ASM --implicit-check-not=v_div_fixup_f16
// RUN: AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx942 --disable-fast-math -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=NOFM-ASM --implicit-check-not=v_rcp_f16

#flat_to_gemm = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
  by [<Unmerge{64, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>,
      <AddDim{1} ["unit0"] at [0] -> [] at []>]
  bounds = [1, 64, 64] -> [4096]>
#gemm_to_flat = #rock.transform_map<affine_map<(d0) -> (0, d0 floordiv 64, d0 mod 64)>
  by [<Merge{1, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>]
  bounds = [4096] -> [1, 64, 64]>

module attributes {rock.arch = "gfx942"} {
  // On this arch f16 takes the same four conversions as f32, which is why there
  // is no pass-off direction for it here -- the f32 kernel already covers that
  // code.
  // CHECK-LABEL: llvm.func @arith_f16
  // CHECK-DAG: llvm.fdiv %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, arcp, afn>} : f16
  // CHECK-DAG: llvm.fadd %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f16
  // CHECK-DAG: llvm.fsub %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f16
  // CHECK-DAG: llvm.fmul %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f16
  // CHECK-DAG: llvm.fneg %{{.*}} {fastmathFlags = #llvm.fastmath<nsz>} : f16
  // CHECK-DAG: llvm.frem %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz>} : f16
  // CHECK-DAG: llvm.intr.maxnum(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nnan, nsz>} : (f16, f16) -> f16
  // CHECK-DAG: llvm.intr.maximum(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz>} : (f16, f16) -> f16
  // CHECK-DAG: llvm.intr.minimum(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz>} : (f16, f16) -> f16
  // CHECK-DAG: llvm.intr.fma(%{{.*}}, %{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz, contract>} : (f16, f16, f16) -> f16
  //
  // Still scalar on gfx1250, and the flags ride along unchanged.
  // PACKED-LABEL: llvm.func @arith_f16
  // PACKED-DAG: llvm.fadd %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f16
  // PACKED-DAG: llvm.fsub %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f16
  // PACKED-DAG: llvm.fmul %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f16
  // PACKED-DAG: llvm.fdiv %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, arcp, afn>} : f16
  //
  // This is where `arcp` is worth the most in the emitted code: the divide
  // collapses to the bare reciprocal, where without the flag it has to run the
  // IEEE fixup sequence to get the last bit right.
  // ASM-LABEL: arith_f16:
  // ASM: v_rcp_f16
  // NOFM-ASM-LABEL: arith_f16:
  // NOFM-ASM: v_div_fixup_f16
  func.func @arith_f16(%a: tensor<4096xf16>, %b: tensor<4096xf16>,
                       %c: tensor<4096xf16>, %out: tensor<4096xf16>)
      -> tensor<4096xf16> attributes {rock.kernel} {
    %ta = rock.transform %a by #flat_to_gemm : tensor<4096xf16> to tensor<1x64x64xf16>
    %tb = rock.transform %b by #flat_to_gemm : tensor<4096xf16> to tensor<1x64x64xf16>
    %tc = rock.transform %c by #flat_to_gemm : tensor<4096xf16> to tensor<1x64x64xf16>
    %g = rock.gemm %ta * %tb : tensor<1x64x64xf16> * tensor<1x64x64xf16> -> tensor<1x64x64xf16>
    %div = arith.divf %g, %tc : tensor<1x64x64xf16>
    %add = arith.addf %div, %tc : tensor<1x64x64xf16>
    %sub = arith.subf %add, %tc : tensor<1x64x64xf16>
    %mul = arith.mulf %sub, %tc : tensor<1x64x64xf16>
    %neg = arith.negf %mul : tensor<1x64x64xf16>
    %rem = arith.remf %neg, %tc : tensor<1x64x64xf16>
    %maxnum = arith.maxnumf %rem, %tc : tensor<1x64x64xf16>
    %maximum = arith.maximumf %maxnum, %tc : tensor<1x64x64xf16>
    %minimum = arith.minimumf %maximum, %tc : tensor<1x64x64xf16>
    %fma = math.fma %minimum, %tc, %tc : tensor<1x64x64xf16>
    %flat = rock.transform %fma by #gemm_to_flat : tensor<1x64x64xf16> to tensor<4096xf16>
    %s = rock.store %flat to %out by set : tensor<4096xf16> -> tensor<4096xf16> to tensor<4096xf16>
    return %s : tensor<4096xf16>
  }
}
