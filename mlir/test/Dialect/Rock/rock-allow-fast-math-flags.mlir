// Exercise rock-allow-fast-math-flags: each op is tagged with the fast-math
// flag(s) the AMDGPU backend can exploit for that specific kind of op.
//   arith.divf                              -> nsz + arcp + afn         (hw reciprocal + approx)
//   arith.{add,sub,mul}f, math.fma          -> nsz + contract           (nsz peepholes + fma fusion)
//   arith.{neg,rem,maximum,maxnum,minimum}f -> nsz                      (sign-bit XOR / ±0 peepholes)
//   math.{absf,copysign,clampf}             -> nsz                      (±0 peepholes only; not approximated)
//   math.* transcendental (incl. sincos)    -> nsz + contract + afn     (hw approximate impl)
//
// The pass is gated on `rock.kernel` and silently skips any other func, which
// is what keeps the host/CPU path free of these flags: `buildKernelPipeline`
// schedules it while the host funcs are still in the module, so the gate rather
// than the schedule is what excludes them.
//
// The per-op tests live inside a nested `module @perop_tests` so that the
// migraphx pipeline used by the TOSA/ROCK/FAST runs cannot see them: that
// pipeline only walks top-level funcs (via `module.getOps<func::FuncOp>()`), so
// only the single MIGraphX kernel at the bottom of this file is lowered through
// it. The pass-pipeline below explicitly descends into the nested module.
// RUN: rocmlir-opt --pass-pipeline='builtin.module(func.func(rock-allow-fast-math-flags),builtin.module(func.func(rock-allow-fast-math-flags)))' -mlir-print-local-scope %s | FileCheck %s

// End-to-end check
// migraphx -> tosa
// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | FileCheck %s --check-prefix=TOSA

// tosa -> rock
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | FileCheck %s --check-prefix=ROCK

// rock-allow-fast-math-flags
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-opt -rock-allow-fast-math-flags -mlir-print-local-scope | FileCheck %s --check-prefix=FAST

// The same two stages with `-disable-fast-math`, which the driver has to thread
// all the way back to migraphx-to-tosa in phase 1 for the NaN mode to change.
// RUN: rocmlir-driver -disable-fast-math -kernel-pipeline=migraphx,highlevel %s | FileCheck %s --check-prefix=ROCK_IEEE
// RUN: rocmlir-driver -disable-fast-math -kernel-pipeline=migraphx,highlevel %s | rocmlir-opt -rock-allow-fast-math-flags -mlir-print-local-scope | FileCheck %s --check-prefix=FAST_IEEE

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
  // `remf` (±0 peepholes around the IEEE remainder), and the whole min/max
  // family. These must NOT receive `contract`/`arcp`/`afn` -- a wrong flag set
  // on the registration would show up as extra bits here.
  //
  // The already-non-propagating maxnumf/minnumf additionally get `nnan`, which
  // is what lets a clamp pair fold into one v_med3 later. The propagating
  // maximumf/minimumf are deliberately left alone: `nsz` does not imply `nnan`,
  // and flagging them would quietly undo a request to propagate. Reaching this
  // pass at all already means fast math is on, since `-disable-fast-math` makes
  // the kernel pipeline skip it entirely.
  // CHECK-LABEL: func.func @nsz_only_arith_ops_add_nsz
  // CHECK: arith.negf %{{.*}} fastmath<nsz> : f32
  // CHECK: arith.remf %{{.*}}, %{{.*}} fastmath<nsz> : f32
  // CHECK: arith.maximumf %{{.*}}, %{{.*}} fastmath<nsz> : f32
  // CHECK: arith.maxnumf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f32
  // CHECK: arith.minimumf %{{.*}}, %{{.*}} fastmath<nsz> : f32
  // CHECK: arith.minnumf %{{.*}}, %{{.*}} fastmath<nnan,nsz> : f32
  func.func @nsz_only_arith_ops_add_nsz(%a: f32, %b: f32) -> f32
      attributes {rock.kernel} {
    %0 = arith.negf %a : f32
    %1 = arith.remf %0, %b : f32
    %2 = arith.maximumf %1, %b : f32
    %3 = arith.maxnumf %2, %b : f32
    %4 = arith.minimumf %3, %a : f32
    %5 = arith.minnumf %4, %a : f32
    return %5 : f32
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

  // Combiners nested in a `tt.reduce` region are added by RockToTTIR. Make sure
  // that they can get the proper flags added.
  // CHECK-LABEL: func.func @reduce_combiner_is_reached
  // CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<nsz,contract> : f32
  func.func @reduce_combiner_is_reached(%arg0: tensor<64x64xf32>) -> tensor<64xf32>
      attributes {rock.kernel} {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.addf %lhs, %rhs : f32
      "tt.reduce.return"(%1) : (f32) -> ()
    }) : (tensor<64x64xf32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }

  // A func without `rock.kernel` is left exactly as it came in, even for the two
  // ops the kernels above get the most aggressive flags on. This is what the CPU
  // reference relies on: it is still in the module when the pass runs, and it
  // has to keep its IEEE division and its NaN-clamping maxnum  f to be worth
  // comparing the kernel against.
  // CHECK-LABEL: func.func @no_kernel_attr_is_untouched
  // CHECK: arith.divf %{{[^ ]+}}, %{{[^ ]+}} : f32
  // CHECK: arith.maxnumf %{{[^ ]+}}, %{{[^ ]+}} : f32
  func.func @no_kernel_attr_is_untouched(%a: f32, %b: f32) -> f32 {
    %0 = arith.divf %a, %b : f32
    %1 = arith.maxnumf %0, %b : f32
    return %1 : f32
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
// ROCK-NEXT: arith.negf %{{.*}} : tensor<2x3xf32>

// FAST-LABEL: func.func @migraphx_pipeline_adds_per_op_flags
// FAST:      arith.subf %{{.*}}, %{{.*}} fastmath<nsz,contract> : tensor<2x3xf32>
// FAST-NEXT: math.exp %{{.*}} fastmath<nsz,contract,afn> : tensor<2x3xf32>
// FAST-NEXT: arith.addf %{{.*}}, %{{.*}} fastmath<nsz,contract> : tensor<2x3xf32>
// FAST-NEXT: arith.divf %{{.*}}, %{{.*}} fastmath<nsz,arcp,afn> : tensor<2x3xf32>
// FAST-NEXT: arith.mulf %{{.*}}, %{{.*}} fastmath<nsz,contract> : tensor<2x3xf32>
// FAST-NEXT: arith.negf %{{.*}} fastmath<nsz> : tensor<2x3xf32>
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

// Which `arith` op a `migraphx.max` becomes is decided back in phase 1, by the
// NaN mode migraphx-to-tosa puts on the `tosa.maximum`. This is the end-to-end
// check that `-disable-fast-math` actually reaches that far: the driver has to
// carry the flag into the migraphx phase, not just into the kernel/backend
// phases, or the two runs below would be identical.
//
// By default the kernel assumes no NaN occurs, so the max is the
// non-propagating `maxnumf` and the fast-math pass gives it `nnan` on top of
// `nsz`. Under `-disable-fast-math` it is the propagating `maximumf`, which
// gets `nsz` only -- `nsz` permits signed-zero optimizations but does not imply
// `nnan`.
// TOSA-LABEL: func.func @migraphx_max_nan_mode_follows_fast_math
// TOSA: tosa.maximum

// ROCK-LABEL: func.func @migraphx_max_nan_mode_follows_fast_math
// ROCK: arith.maxnumf %{{[^,]+}}, %{{[^ ]+}} : tensor<2x3xf32>

// FAST-LABEL: func.func @migraphx_max_nan_mode_follows_fast_math
// FAST: arith.maxnumf %{{[^,]+}}, %{{[^ ]+}} fastmath<nnan,nsz> : tensor<2x3xf32>

// ROCK_IEEE-LABEL: func.func @migraphx_max_nan_mode_follows_fast_math
// ROCK_IEEE: arith.maximumf %{{[^,]+}}, %{{[^ ]+}} : tensor<2x3xf32>

// FAST_IEEE-LABEL: func.func @migraphx_max_nan_mode_follows_fast_math
// FAST_IEEE: arith.maximumf %{{[^,]+}}, %{{[^ ]+}} fastmath<nsz> : tensor<2x3xf32>
func.func @migraphx_max_nan_mode_follows_fast_math(
    %a: !migraphx.shaped<2x3xf32, 3x1>,
    %b: !migraphx.shaped<2x3xf32, 3x1>)
    -> !migraphx.shaped<2x3xf32, 3x1>
    attributes {kernel, arch = "gfx90a", rock.kernel} {
  %max = migraphx.max %a, %b
      : <2x3xf32, 3x1>, <2x3xf32, 3x1> -> <2x3xf32, 3x1>
  return %max : !migraphx.shaped<2x3xf32, 3x1>
}
