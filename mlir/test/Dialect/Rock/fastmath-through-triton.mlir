// Pins what happens to `rock-allow-fast-math-flags`' output once it crosses into
// Triton: for every op that pass can tag and that Triton will legalize, this
// records whether the flags are still there by the time the op is in the LLVM
// dialect.
//
// The input is Rock IR as it stands at the start of `rocmlir-driver -c`, and the
// run below is that same compile minus the `binary` stage, so it stops with the
// kernel in the LLVM dialect. Nothing here writes a fastmath attribute by hand:
// the flags in the expectations can only come from the pass running inside the
// pipeline.

// RUN: rocmlir-driver -arch=gfx942 -kernel-pipeline=gpu,triton %s | FileCheck %s

// The same compile with the pass gated off, which keeps the expectations above
// honest in both directions. It confirms the flags can only have come from the
// pass, and that with no flags to translate the LLVM ops come out bare instead
// of carrying an explicit `fastmath<none>` -- the property that keeps the
// Triton-side translation inert for upstream Triton, whose frontend never sets
// fast-math flags at all.
// RUN: rocmlir-driver -arch=gfx942 --disable-fast-math -kernel-pipeline=gpu,triton %s \
// RUN: | FileCheck %s --check-prefix=NOFM

// gfx1250 is compiled as well because it swaps in a different set of conversions
// for the same source ops: add/sub/mul go through the packed conversion and come
// out as `vector<2x...>`, and bf16 is native there so it takes that packed path
// instead of the widening emulation the gfx942 run above exercises. Without this
// the packed conversion has no coverage at all.
// RUN: rocmlir-driver -arch=gfx1250 -kernel-pipeline=gpu,triton %s \
// RUN: | FileCheck %s --check-prefix=PACKED

// The expectations above stop at the LLVM dialect, which can only show that a
// flag survived, not that it bought anything. These two runs go all the way to
// assembly so the codegen difference is pinned too. The implicit-check-nots
// carry the negative half: with the flags on, no f16 divide anywhere in the
// module falls back to the fixup sequence, and with them off, none reaches a
// bare reciprocal.
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
  // Every float `arith` op the pass tags keeps its flags. `divf` is the one that
  // changes codegen the most: without `arcp` the backend cannot use a bare
  // reciprocal and has to emit the fixup sequence.
  // CHECK-LABEL: llvm.func @arith_f32
  // CHECK-DAG: llvm.fdiv %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, arcp, afn>} : f32
  // CHECK-DAG: llvm.fadd %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
  // CHECK-DAG: llvm.fsub %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
  // CHECK-DAG: llvm.fmul %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
  // CHECK-DAG: llvm.fneg %{{.*}} {fastmathFlags = #llvm.fastmath<nsz>} : f32
  // CHECK-DAG: llvm.frem %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz>} : f32
  // CHECK-DAG: llvm.intr.maxnum(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz>}
  // CHECK-DAG: llvm.intr.maximum(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz>}
  // CHECK-DAG: llvm.intr.minimum(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz>}
  // CHECK-DAG: llvm.intr.fma(%{{.*}}, %{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz, contract>}
  //
  // NOFM-LABEL: llvm.func @arith_f32
  // NOFM-DAG: llvm.fdiv %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fadd %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fsub %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fmul %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fneg %{{.*}} : f32
  // NOFM-DAG: llvm.frem %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.intr.maxnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // NOFM-DAG: llvm.intr.maximum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // NOFM-DAG: llvm.intr.minimum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // NOFM-DAG: llvm.intr.fma(%{{.*}}, %{{.*}}, %{{.*}}) : (f32, f32, f32) -> f32
  //
  // The packed conversion applies at f32 too, so it is checked here rather than
  // only on bf16.
  // PACKED-LABEL: llvm.func @arith_f32
  // PACKED-DAG: llvm.fadd %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : vector<2xf32>
  // PACKED-DAG: llvm.fsub %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : vector<2xf32>
  // PACKED-DAG: llvm.fmul %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : vector<2xf32>
  // PACKED-DAG: llvm.fdiv %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, arcp, afn>} : f32
  func.func @arith_f32(%a: tensor<4096xf32>, %b: tensor<4096xf32>,
                       %c: tensor<4096xf32>, %out: tensor<4096xf32>)
      -> tensor<4096xf32> attributes {rock.kernel} {
    %ta = rock.transform %a by #flat_to_gemm : tensor<4096xf32> to tensor<1x64x64xf32>
    %tb = rock.transform %b by #flat_to_gemm : tensor<4096xf32> to tensor<1x64x64xf32>
    %tc = rock.transform %c by #flat_to_gemm : tensor<4096xf32> to tensor<1x64x64xf32>
    %g = rock.gemm %ta * %tb : tensor<1x64x64xf32> * tensor<1x64x64xf32> -> tensor<1x64x64xf32>
    %div = arith.divf %g, %tc : tensor<1x64x64xf32>
    %add = arith.addf %div, %tc : tensor<1x64x64xf32>
    %sub = arith.subf %add, %tc : tensor<1x64x64xf32>
    %mul = arith.mulf %sub, %tc : tensor<1x64x64xf32>
    %neg = arith.negf %mul : tensor<1x64x64xf32>
    %rem = arith.remf %neg, %tc : tensor<1x64x64xf32>
    %maxnum = arith.maxnumf %rem, %tc : tensor<1x64x64xf32>
    %maximum = arith.maximumf %maxnum, %tc : tensor<1x64x64xf32>
    %minimum = arith.minimumf %maximum, %tc : tensor<1x64x64xf32>
    %fma = math.fma %minimum, %tc, %tc : tensor<1x64x64xf32>
    %flat = rock.transform %fma by #gemm_to_flat : tensor<1x64x64xf32> to tensor<4096xf32>
    %s = rock.store %flat to %out by set : tensor<4096xf32> -> tensor<4096xf32> to tensor<4096xf32>
    return %s : tensor<4096xf32>
  }

  // f16 is the width the fused elementwise kernels actually run at. On this arch
  // it takes the same four conversions as f32, which is why there is no pass-off
  // direction for it below -- the f32 kernel already covers that code. The
  // packed f16 conversion belongs to a different arch and is covered by the
  // gfx1250 run.
  // CHECK-LABEL: llvm.func @arith_f16
  // CHECK-DAG: llvm.fdiv %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, arcp, afn>} : f16
  // CHECK-DAG: llvm.fadd %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f16
  // CHECK-DAG: llvm.fsub %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f16
  // CHECK-DAG: llvm.fmul %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f16
  // CHECK-DAG: llvm.fneg %{{.*}} {fastmathFlags = #llvm.fastmath<nsz>} : f16
  // CHECK-DAG: llvm.frem %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz>} : f16
  // CHECK-DAG: llvm.intr.maxnum(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz>} : (f16, f16) -> f16
  // CHECK-DAG: llvm.intr.maximum(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz>} : (f16, f16) -> f16
  // CHECK-DAG: llvm.intr.minimum(%{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz>} : (f16, f16) -> f16
  // CHECK-DAG: llvm.intr.fma(%{{.*}}, %{{.*}}, %{{.*}}) {fastmathFlags = #llvm.fastmath<nsz, contract>} : (f16, f16, f16) -> f16
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

  // bf16 add/sub/mul are emulated by widening to f32, so the flags have to land
  // on the f32 op the emulation builds rather than on a bf16 one -- a separate
  // path from the one f16 and f32 take. `divf` needs no emulation and stays at
  // width, which is what makes it the control here.
  // CHECK-LABEL: llvm.func @arith_bf16
  // CHECK-DAG: llvm.fadd %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
  // CHECK-DAG: llvm.fsub %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
  // CHECK-DAG: llvm.fmul %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
  // CHECK-DAG: llvm.fdiv %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, arcp, afn>} : bf16
  //
  // The emulation receives the flags as an explicit argument to the op builder
  // rather than by copying an attribute across, so it needs its own pass-off
  // direction: this is the shape where accidentally materializing an empty
  // attribute instead of none at all would go unnoticed.
  // NOFM-LABEL: llvm.func @arith_bf16
  // NOFM-DAG: llvm.fadd %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fsub %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fmul %{{.*}}, %{{.*}} : f32
  // NOFM-DAG: llvm.fdiv %{{.*}}, %{{.*}} : bf16
  //
  // bf16 is native on gfx1250, so there the same three ops take the packed
  // conversion and keep their flags on a vector rather than on a widened f32.
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

  // Whether a transcendental keeps its flags depends on what the lowering
  // committed it to. An op that stays in the LLVM dialect -- `llvm.intr.*` or a
  // call to an `llvm.*` intrinsic -- has somewhere to put the attribute and
  // keeps it. A ROCDL op is a raw hardware intrinsic and an OCML call is a
  // function call, and neither does, so those come out bare. Matching through
  // to the trailing type is what pins each case, since a `fastmathFlags`
  // dictionary prints just before it.
  //
  // What that costs is a separate question from the flag. Measured per element
  // on gfx942 against a gemm-only baseline: exp, exp2, sqrt and rsqrt cost one
  // instruction each. exp2, sqrt and rsqrt get there by being `rocdl.*`, which
  // is the hardware instruction outright. exp gets there through
  // `llvm.exp2.f32`, which the backend leaves as a bare `v_exp_f32` because the
  // kernel's denormal mode flushes f32 denorms -- not because of `afn`, which
  // moves it not at all.
  //
  // log2, sin, cos and erf stay expensive because their OCML calls inline into
  // full software argument reduction, and an attribute cannot redirect a call.
  // `rocdl.log` exists and is unused, so closing that one is an op-selection
  // change in Triton rather than anything more flags would fix. `nsz` on
  // `llvm.intr.fabs` would be inert regardless.
  //
  // The `llvm.fmul` is the log2(e) scaling inside the exp expansion.
  // CHECK-LABEL: llvm.func @transcendentals_f32
  // CHECK-DAG: llvm.fmul %{{.*}}, %{{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract, afn>} : f32
  // CHECK-DAG: llvm.call @llvm.exp2.f32(%{{.*}}) {fastmathFlags = #llvm.fastmath<nsz, contract, afn>} : (f32) -> f32
  // CHECK-DAG: llvm.intr.log(%{{.*}}) {fastmathFlags = #llvm.fastmath<nsz, contract, afn>} : (f32) -> f32
  // CHECK-DAG: rocdl.exp2 %{{.*}} f32 -> f32
  // CHECK-DAG: llvm.call @__ocml_log2_f32(%{{.*}}) : (f32) -> f32
  // CHECK-DAG: llvm.call @__ocml_sin_f32(%{{.*}}) : (f32) -> f32
  // CHECK-DAG: llvm.call @__ocml_cos_f32(%{{.*}}) : (f32) -> f32
  // CHECK-DAG: llvm.call @__ocml_erf_f32(%{{.*}}) : (f32) -> f32
  // CHECK-DAG: rocdl.sqrt %{{.*}} f32 -> f32
  // CHECK-DAG: rocdl.rsq %{{.*}} f32 -> f32
  // CHECK-DAG: llvm.intr.fabs(%{{.*}}) : (f32) -> f32

  // ASM-LABEL: transcendentals_f32:
  // ASM-DAG: v_exp_f32
  // ASM-DAG: v_log_f32
  // ASM-DAG: v_sqrt_f32
  // ASM-DAG: v_rsq_f32
  // These four lower to rocdl hardware ops either way, so fast math changes the
  // LLVM attributes above but not the emitted instructions.
  // NOFM-ASM-LABEL: transcendentals_f32:
  // NOFM-ASM-DAG: v_exp_f32
  // NOFM-ASM-DAG: v_log_f32
  // NOFM-ASM-DAG: v_sqrt_f32
  // NOFM-ASM-DAG: v_rsq_f32
  func.func @transcendentals_f32(%a: tensor<4096xf32>, %b: tensor<4096xf32>,
                                 %out: tensor<4096xf32>)
      -> tensor<4096xf32> attributes {rock.kernel} {
    %ta = rock.transform %a by #flat_to_gemm : tensor<4096xf32> to tensor<1x64x64xf32>
    %tb = rock.transform %b by #flat_to_gemm : tensor<4096xf32> to tensor<1x64x64xf32>
    %g = rock.gemm %ta * %tb : tensor<1x64x64xf32> * tensor<1x64x64xf32> -> tensor<1x64x64xf32>
    %exp = math.exp %g : tensor<1x64x64xf32>
    %exp2 = math.exp2 %exp : tensor<1x64x64xf32>
    %log = math.log %exp2 : tensor<1x64x64xf32>
    %log2 = math.log2 %log : tensor<1x64x64xf32>
    %sin = math.sin %log2 : tensor<1x64x64xf32>
    %cos = math.cos %sin : tensor<1x64x64xf32>
    %erf = math.erf %cos : tensor<1x64x64xf32>
    %sqrt = math.sqrt %erf : tensor<1x64x64xf32>
    %rsqrt = math.rsqrt %sqrt : tensor<1x64x64xf32>
    %absf = math.absf %rsqrt : tensor<1x64x64xf32>
    %flat = rock.transform %absf by #gemm_to_flat : tensor<1x64x64xf32> to tensor<4096xf32>
    %s = rock.store %flat to %out by set : tensor<4096xf32> -> tensor<4096xf32> to tensor<4096xf32>
    return %s : tensor<4096xf32>
  }
}
