// The transcendentals kernel of the fastmath-through-triton family: what happens
// to `rock-allow-fast-math-flags`' output once it crosses into Triton. See
// fastmath-through-triton.mlir for what the family checks and why each file
// holds exactly one kernel.

// RUN: rocmlir-driver -arch=gfx942 -kernel-pipeline=gpu,triton %s | FileCheck %s

// The expectations above stop at the LLVM dialect, which can only show that a
// flag survived, not that it bought anything. This run goes all the way to
// assembly so the instruction each op actually reaches is pinned too.
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
