// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The bf16 arithmetic kernel of the fastmath-through-triton family: what happens
// to `rock-allow-fast-math-flags`' output once it crosses into Triton. See
// fastmath-through-triton.mlir for what the family checks and why each file
// holds exactly one kernel.
//
// bf16 is here because add/sub/mul are emulated by widening to f32 on gfx942, so
// the flags have to land on the f32 op the emulation builds rather than on a
// bf16 one -- a separate path from the one f16 and f32 take.

// RUN: rocmlir-driver -arch=gfx942 -kernel-pipeline=gpu,triton %s | FileCheck %s

// The same compile with the pass gated off. The emulation receives the flags as
// an explicit argument to the op builder rather than by copying an attribute
// across, so it needs its own pass-off direction: this is the shape where
// accidentally materializing an empty attribute instead of none at all would go
// unnoticed.
// RUN: rocmlir-driver -arch=gfx942 --disable-fast-math -kernel-pipeline=gpu,triton %s \
// RUN: | FileCheck %s --check-prefix=NOFM

// bf16 is native on gfx1250, so there the same three ops take the packed
// conversion instead of the widening emulation.
// RUN: rocmlir-driver -arch=gfx1250 -kernel-pipeline=gpu,triton %s \
// RUN: | FileCheck %s --check-prefix=PACKED

// The expectations above stop at the LLVM dialect, which can only show that a
// flag survived, not that it bought anything. These two runs go all the way to
// assembly so the codegen difference is pinned too.
// RUN: AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx942 -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=ASM
// RUN: AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx942 --disable-fast-math -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=NOFM-ASM

#flat_to_gemm = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
  by [<Unmerge{64, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>,
      <AddDim{1} ["unit0"] at [0] -> [] at []>]
  bounds = [1, 64, 64] -> [4096]>
#gemm_to_flat = #rock.transform_map<affine_map<(d0) -> (0, d0 floordiv 64, d0 mod 64)>
  by [<Merge{1, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>]
  bounds = [4096] -> [1, 64, 64]>

module attributes {rock.arch = "gfx942"} {
  // `divf` needs no emulation and stays at width, which is what makes it the
  // control here.
  // CHECK-LABEL: llvm.func @arith_bf16
  // CHECK-DAG: llvm.fadd %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
  // CHECK-DAG: llvm.fsub %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
  // CHECK-DAG: llvm.fmul %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
  // CHECK-DAG: llvm.fdiv %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, arcp, afn>} : bf16
  //
  // NOFM-LABEL: llvm.func @arith_bf16
  // NOFM-DAG: llvm.fadd %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fsub %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fmul %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fdiv %{{.*}}, %{{.*}} : bf16
  //
  // On gfx1250 the same three ops keep their flags on a vector rather than on a
  // widened f32.
  // PACKED-LABEL: llvm.func @arith_bf16
  // PACKED-DAG: llvm.fadd %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : vector<2xbf16>
  // PACKED-DAG: llvm.fsub %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : vector<2xbf16>
  // PACKED-DAG: llvm.fmul %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : vector<2xbf16>
  // PACKED-DAG: llvm.fdiv %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, arcp, afn>} : bf16
  //
  // The bf16 divide widens to f32 either way, so what `arcp` removes here is
  // the whole three-instruction scale/fmas/fixup sequence around it rather than
  // a single fixup.
  // ASM-LABEL: arith_bf16:
  // ASM-NOT: v_div_scale_f32
  // ASM-NOT: v_div_fmas_f32
  // ASM-NOT: v_div_fixup_f32
  // NOFM-ASM-LABEL: arith_bf16:
  // NOFM-ASM-DAG: v_div_scale_f32
  // NOFM-ASM-DAG: v_div_fmas_f32
  // NOFM-ASM-DAG: v_div_fixup_f32
  func.func @arith_bf16(%a: tensor<4096xbf16>, %b: tensor<4096xbf16>,
                        %c: tensor<4096xbf16>, %out: tensor<4096xbf16>)
      -> tensor<4096xbf16> attributes {rock.kernel} {
    %ta = rock.transform %a by #flat_to_gemm : tensor<4096xbf16> to tensor<1x64x64xbf16>
    %tb = rock.transform %b by #flat_to_gemm : tensor<4096xbf16> to tensor<1x64x64xbf16>
    %tc = rock.transform %c by #flat_to_gemm : tensor<4096xbf16> to tensor<1x64x64xbf16>
    %g = rock.gemm %ta * %tb : tensor<1x64x64xbf16> * tensor<1x64x64xbf16> -> tensor<1x64x64xbf16>
    %div = arith.divf %g, %tc : tensor<1x64x64xbf16>
    %add = arith.addf %div, %tc : tensor<1x64x64xbf16>
    %sub = arith.subf %add, %tc : tensor<1x64x64xbf16>
    %mul = arith.mulf %sub, %tc : tensor<1x64x64xbf16>
    %flat = rock.transform %mul by #gemm_to_flat : tensor<1x64x64xbf16> to tensor<4096xbf16>
    %s = rock.store %flat to %out by set : tensor<4096xbf16> -> tensor<4096xbf16> to tensor<4096xbf16>
    return %s : tensor<4096xbf16>
  }
}
