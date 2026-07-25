// Exercise rock-allow-fast-math-flags: each op is tagged with the fast-math
// flag(s) the AMDGPU backend can exploit for that specific kind of op.
//   arith.divf                              -> nsz + arcp + afn         (hw reciprocal + approx)
//   arith.{add,sub,mul}f, math.fma          -> nsz + contract           (nsz peepholes + fma fusion)
//   arith.{neg,rem,maximum,maxnum,minimum}f -> nsz                      (sign-bit XOR / ±0 peepholes)
//   math.{absf,copysign,clampf}             -> nsz                      (±0 peepholes only; not approximated)
//   math.* transcendental (incl. sincos)    -> nsz + contract + afn     (hw approximate impl)
//
// The per-op tests live inside a nested `module @perop_tests`. The pass is
// gated on `rock.kernel`, so the per-op funcs carry that attribute and the
// pass-pipeline below explicitly descends into the nested module to run the
// pass on them. The migraphx pipeline used by the TOSA/ROCK/FAST runs only
// walks top-level funcs (via `module.getOps<func::FuncOp>()`), so the nested
// module is opaque to it and only the single MIGraphX kernel at the bottom
// of this file is lowered through that pipeline.
// RUN: rocmlir-opt --pass-pipeline='builtin.module(func.func(rock-allow-fast-math-flags),builtin.module(func.func(rock-allow-fast-math-flags)))' -mlir-print-local-scope %s | FileCheck %s

// End-to-end check
// migraphx -> tosa
// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | FileCheck %s --check-prefix=TOSA

// tosa -> rock
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | FileCheck %s --check-prefix=ROCK

// rock-allow-fast-math-flags
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-opt -rock-allow-fast-math-flags -mlir-print-local-scope | FileCheck %s --check-prefix=FAST

// CHECK-LABEL: module @perop_tests
module @perop_tests {

  // Individual op check
  // CHECK-LABEL: func.func @divf_scalar_adds_arcp_nsz_afn
  // CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<nsz,arcp,afn> : f32
  func.func @divf_scalar_adds_arcp_nsz_afn(%a: f32, %b: f32) -> f32
      attributes {rock.kernel} {
    %0 = arith.divf %a, %b : f32
    return %0 : f32
  }

  // CHECK-LABEL: func.func @divf_tensor_adds_arcp_nsz_afn
  // CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<nsz,arcp,afn> : tensor<2x3xf32>
  func.func @divf_tensor_adds_arcp_nsz_afn(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xf32>
      attributes {rock.kernel} {
    %0 = arith.divf %x, %y : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // Prior fast-math bits are kept; new flags are merged in.
  // CHECK-LABEL: func.func @divf_preserves_other_fastmath
  // CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<nnan,nsz,arcp,afn> : f32
  func.func @divf_preserves_other_fastmath(%a: f32, %b: f32) -> f32
      attributes {rock.kernel} {
    %0 = arith.divf %a, %b fastmath<nnan> : f32
    return %0 : f32
  }

  // Binary float arith ops get `contract` (so LLVM can fuse mul+add into
  // fma) and `nsz` (so LLVM can apply ±0 peepholes around them).
  // CHECK-LABEL: func.func @binary_arith_adds_contract_nsz
  // CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<nsz,contract> : f32
  // CHECK: arith.subf %{{.*}}, %{{.*}} fastmath<nsz,contract> : f32
  // CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<nsz,contract> : f32
  func.func @binary_arith_adds_contract_nsz(%a: f32, %b: f32) -> f32
      attributes {rock.kernel} {
    %0 = arith.addf %a, %b : f32
    %1 = arith.subf %0, %b : f32
    %2 = arith.mulf %1, %a : f32
    return %2 : f32
  }

  // arith ops in the `nszOnly` bucket: `negf` (sign-bit XOR for `0 - x`),
  // `remf` (±0 peepholes around the IEEE remainder), and
  // `maximumf`/`maxnumf`/`minimumf`. `nsz` does not imply `nnan`, so maximumf
  // still propagates NaNs.
  // These must NOT receive `contract`/`arcp`/`afn` -- a wrong flag set on the
  // registration would show up as extra bits here.
  // CHECK-LABEL: func.func @nsz_only_arith_ops_add_nsz
  // CHECK: arith.negf %{{.*}} fastmath<nsz> : f32
  // CHECK: arith.remf %{{.*}}, %{{.*}} fastmath<nsz> : f32
  // CHECK: arith.maximumf %{{.*}}, %{{.*}} fastmath<nsz> : f32
  // CHECK: arith.maxnumf %{{.*}}, %{{.*}} fastmath<nsz> : f32
  // CHECK: arith.minimumf %{{.*}}, %{{.*}} fastmath<nsz> : f32
  func.func @nsz_only_arith_ops_add_nsz(%a: f32, %b: f32) -> f32
      attributes {rock.kernel} {
    %0 = arith.negf %a : f32
    %1 = arith.remf %0, %b : f32
    %2 = arith.maximumf %1, %b : f32
    %3 = arith.maxnumf %2, %b : f32
    %4 = arith.minimumf %3, %a : f32
    return %4 : f32
  }

  // Non-transcendental math ops (`absf`, `copysign`, `clampf`) are exact
  // operations -- they get `nsz` only so the backend may apply ±0 peepholes
  // (e.g. `absf(-0.0) -> 0.0`) without enabling `afn`-style approximations.
  // CHECK-LABEL: func.func @non_transcendental_math_ops_add_nsz
  // CHECK: math.absf %{{.*}} fastmath<nsz> : f32
  // CHECK: math.copysign %{{.*}}, %{{.*}} fastmath<nsz> : f32
  // CHECK: math.clampf %{{.*}} to [%{{.*}}, %{{.*}}] fastmath<nsz> : f32
  func.func @non_transcendental_math_ops_add_nsz(%a: f32, %lo: f32, %hi: f32) -> f32
      attributes {rock.kernel} {
    %0 = math.absf %a : f32
    %1 = math.copysign %0, %a : f32
    %2 = math.clampf %1 to [%lo, %hi] : f32
    return %2 : f32
  }

  // math.* transcendentals get `afn` so the backend may use approximate
  // hardware implementations (v_exp_f32, v_log_f32, v_sqrt_f32, ...).
  // CHECK-LABEL: func.func @math_transcendentals_add_afn
  // CHECK: math.exp %{{.*}} fastmath<nsz,contract,afn> : f32
  // CHECK: math.log %{{.*}} fastmath<nsz,contract,afn> : f32
  // CHECK: math.sqrt %{{.*}} fastmath<nsz,contract,afn> : f32
  // CHECK: math.rsqrt %{{.*}} fastmath<nsz,contract,afn> : f32
  // CHECK: math.sin %{{.*}} fastmath<nsz,contract,afn> : f32
  // CHECK: math.tanh %{{.*}} fastmath<nsz,contract,afn> : f32
  func.func @math_transcendentals_add_afn(%x: f32) -> f32
      attributes {rock.kernel} {
    %0 = math.exp %x : f32
    %1 = math.log %0 : f32
    %2 = math.sqrt %1 : f32
    %3 = math.rsqrt %2 : f32
    %4 = math.sin %3 : f32
    %5 = math.tanh %4 : f32
    return %5 : f32
  }

  // `math.sincos` is the only multi-result op in the transcendental bucket; if
  // the registration mistakenly walked only single-result ops or used the wrong
  // flag set, this check would fail.
  // CHECK-LABEL: func.func @sincos_multi_result_adds_transcendental_flags
  // CHECK: %{{.*}}, %{{.*}} = math.sincos %{{.*}} fastmath<nsz,contract,afn> : f32
  func.func @sincos_multi_result_adds_transcendental_flags(%x: f32) -> (f32, f32)
      attributes {rock.kernel} {
    %s, %c = math.sincos %x : f32
    return %s, %c : f32, f32
  }

  // `math.fma` is the only `math.*` op in the FMA-fusion bucket -- it gets
  // `nsz + contract` rather than transcendental's `afn`.
  // CHECK-LABEL: func.func @math_fma_adds_contract_nsz
  // CHECK: math.fma %{{.*}}, %{{.*}}, %{{.*}} fastmath<nsz,contract> : f32
  func.func @math_fma_adds_contract_nsz(%a: f32, %b: f32, %c: f32) -> f32
      attributes {rock.kernel} {
    %0 = math.fma %a, %b, %c : f32
    return %0 : f32
  }

  // Sanity check that the pass leaves non-kernel funcs alone (the `rock.kernel`
  // gate is what makes the per-op tests above meaningful).
  // CHECK-LABEL: func.func @non_kernel_is_skipped
  // CHECK: arith.divf %{{.*}}, %{{.*}} : f32
  // CHECK-NOT: fastmath
  func.func @non_kernel_is_skipped(%a: f32, %b: f32) -> f32 {
    %0 = arith.divf %a, %b : f32
    return %0 : f32
  }
}

// End-to-end test (top-level so the migraphx pipeline picks it up).
// TOSA-LABEL: func.func @migraphx_pipeline_adds_per_op_flags
// TOSA-DAG: tosa.sub
// TOSA-DAG: tosa.exp
// TOSA-DAG: tosa.add
// TOSA-DAG: tosa.reciprocal
// TOSA-DAG: tosa.mul
// TOSA-DAG: tosa.negate

// ROCK-LABEL: func.func @migraphx_pipeline_adds_per_op_flags
// ROCK:      arith.subf %{{.*}}, %{{.*}} : tensor<2x3xf32>
// ROCK-NEXT: math.exp %{{.*}} : tensor<2x3xf32>
// ROCK-NEXT: arith.addf %{{.*}}, %{{.*}} : tensor<2x3xf32>
// ROCK-NEXT: arith.divf %{{.*}}, %{{.*}} : tensor<2x3xf32>
// ROCK-NEXT: arith.mulf %{{.*}}, %{{.*}} : tensor<2x3xf32>
// ROCK-NEXT: arith.mulf %{{.*}}, %{{.*}} : tensor<2x3xf32>

// FAST-LABEL: func.func @migraphx_pipeline_adds_per_op_flags
// FAST:      arith.subf %{{.*}}, %{{.*}} fastmath<nsz,contract> : tensor<2x3xf32>
// FAST-NEXT: math.exp %{{.*}} fastmath<nsz,contract,afn> : tensor<2x3xf32>
// FAST-NEXT: arith.addf %{{.*}}, %{{.*}} fastmath<nsz,contract> : tensor<2x3xf32>
// FAST-NEXT: arith.divf %{{.*}}, %{{.*}} fastmath<nsz,arcp,afn> : tensor<2x3xf32>
// FAST-NEXT: arith.mulf %{{.*}}, %{{.*}} fastmath<nsz,contract> : tensor<2x3xf32>
// FAST-NEXT: arith.mulf %{{.*}}, %{{.*}} fastmath<nsz,contract> : tensor<2x3xf32>
func.func @migraphx_pipeline_adds_per_op_flags(
    %a: !migraphx.shaped<2x3xf32, 3x1>,
    %b: !migraphx.shaped<2x3xf32, 3x1>,
    %c: !migraphx.shaped<2x3xf32, 3x1>) -> !migraphx.shaped<2x3xf32, 3x1>
    attributes {kernel, arch = "gfx90a", rock.kernel} {
  %sub = migraphx.sub %a, %b : <2x3xf32, 3x1>, <2x3xf32, 3x1> -> <2x3xf32, 3x1>
  %exp = migraphx.exp %sub : <2x3xf32, 3x1> -> <2x3xf32, 3x1>
  %add = migraphx.add %exp, %c : <2x3xf32, 3x1>, <2x3xf32, 3x1> -> <2x3xf32, 3x1>
  %div = migraphx.div %exp, %add : <2x3xf32, 3x1>, <2x3xf32, 3x1> -> <2x3xf32, 3x1>
  %neg = migraphx.neg %div : <2x3xf32, 3x1> -> <2x3xf32, 3x1>
  return %neg : !migraphx.shaped<2x3xf32, 3x1>
}

// Max requests NaN propagation through TOSA and lowers to `maximumf`. The
// fast-math pass adds only `nsz`, which permits signed-zero optimizations but
// does not imply `nnan`.
// TOSA-LABEL: func.func @migraphx_max_propagates_nan_with_nsz
// TOSA: tosa.maximum

// ROCK-LABEL: func.func @migraphx_max_propagates_nan_with_nsz
// ROCK: arith.maximumf %{{[^,]+}}, %{{[^ ]+}} : tensor<2x3xf32>

// FAST-LABEL: func.func @migraphx_max_propagates_nan_with_nsz
// FAST: arith.maximumf %{{[^,]+}}, %{{[^ ]+}} fastmath<nsz> : tensor<2x3xf32>
func.func @migraphx_max_propagates_nan_with_nsz(
    %a: !migraphx.shaped<2x3xf32, 3x1>,
    %b: !migraphx.shaped<2x3xf32, 3x1>)
    -> !migraphx.shaped<2x3xf32, 3x1>
    attributes {kernel, arch = "gfx90a", rock.kernel} {
  %max = migraphx.max %a, %b
      : <2x3xf32, 3x1>, <2x3xf32, 3x1> -> <2x3xf32, 3x1>
  return %max : !migraphx.shaped<2x3xf32, 3x1>
}
