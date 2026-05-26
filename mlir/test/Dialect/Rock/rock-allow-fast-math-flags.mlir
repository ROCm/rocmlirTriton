// Exercise rock-allow-fast-math-flags: each op is tagged with the fast-math
// flag(s) the AMDGPU backend can exploit for that specific kind of op.
//   arith.divf                -> nsz + arcp + afn         (hw reciprocal + approx)
//   arith.{add,sub,mul}f      -> nsz + contract           (nsz peepholes + fma fusion)
//   arith.negf                -> nsz                      (sign-bit XOR peephole)
//   math.* transcendental     -> nsz + contract + afn     (hw approximate impl)
// RUN: rocmlir-opt -rock-allow-fast-math-flags -mlir-print-local-scope %s | FileCheck %s

// End-to-end check
// migraphx -> tosa
// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | FileCheck %s --check-prefix=TOSA

// tosa -> rock
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | FileCheck %s --check-prefix=ROCK

// rock-allow-fast-math-flags
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-opt -rock-allow-fast-math-flags -mlir-print-local-scope | FileCheck %s --check-prefix=FAST


module {

  // Individual op check
  // CHECK-LABEL: func.func @divf_scalar_adds_arcp_nsz_afn
  // CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<nsz,arcp,afn> : f32
  func.func @divf_scalar_adds_arcp_nsz_afn(%a: f32, %b: f32) -> f32 {
    %0 = arith.divf %a, %b : f32
    return %0 : f32
  }

  // CHECK-LABEL: func.func @divf_tensor_adds_arcp_nsz_afn
  // CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<nsz,arcp,afn> : tensor<2x3xf32>
  func.func @divf_tensor_adds_arcp_nsz_afn(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = arith.divf %x, %y : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // Prior fast-math bits are kept; new flags are merged in.
  // CHECK-LABEL: func.func @divf_preserves_other_fastmath
  // CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<nnan,nsz,arcp,afn> : f32
  func.func @divf_preserves_other_fastmath(%a: f32, %b: f32) -> f32 {
    %0 = arith.divf %a, %b fastmath<nnan> : f32
    return %0 : f32
  }

  // Binary float arith ops get `contract` (so LLVM can fuse mul+add into
  // fma) and `nsz` (so LLVM can apply ±0 peepholes around them).
  // CHECK-LABEL: func.func @binary_arith_adds_contract_nsz
  // CHECK: arith.addf %{{.*}}, %{{.*}} fastmath<nsz,contract> : f32
  // CHECK: arith.subf %{{.*}}, %{{.*}} fastmath<nsz,contract> : f32
  // CHECK: arith.mulf %{{.*}}, %{{.*}} fastmath<nsz,contract> : f32
  func.func @binary_arith_adds_contract_nsz(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    %1 = arith.subf %0, %b : f32
    %2 = arith.mulf %1, %a : f32
    return %2 : f32
  }

  // arith.negf gets `nsz` so `0 - x` style negations can lower to a sign-bit
  // XOR even when the source value might be -0.0.
  // CHECK-LABEL: func.func @negf_adds_nsz
  // CHECK: arith.negf %{{.*}} fastmath<nsz> : f32
  func.func @negf_adds_nsz(%a: f32) -> f32 {
    %0 = arith.negf %a : f32
    return %0 : f32
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
  func.func @math_transcendentals_add_afn(%x: f32) -> f32 {
    %0 = math.exp %x : f32
    %1 = math.log %0 : f32
    %2 = math.sqrt %1 : f32
    %3 = math.rsqrt %2 : f32
    %4 = math.sin %3 : f32
    %5 = math.tanh %4 : f32
    return %5 : f32
  }


  // End-to-end test
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
}