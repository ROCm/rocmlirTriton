// Exercise rock-allow-fast-math-flags: each arith.divf is replaced with the
// same operands and type, with fastmath extended by `arcp`.
// RUN: rocmlir-opt -rock-allow-fast-math-flags -mlir-print-local-scope %s | FileCheck %s

// End-to-end check
// migraphx -> tosa
// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | FileCheck %s --check-prefix=TOSA

// tosa -> rock
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | FileCheck %s --check-prefix=ROCK

// rock-allow-fast-math-flags
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-opt -rock-allow-fast-math-flags -mlir-print-local-scope | FileCheck %s --check-prefix=ARCP


module {

  // CHECK-LABEL: func.func @divf_scalar_adds_arcp
  // CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<arcp> : f32
  func.func @divf_scalar_adds_arcp(%a: f32, %b: f32) -> f32 {
    %0 = arith.divf %a, %b : f32
    return %0 : f32
  }

  // CHECK-LABEL: func.func @divf_tensor_adds_arcp
  // CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<arcp> : tensor<2x3xf32>
  func.func @divf_tensor_adds_arcp(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = arith.divf %x, %y : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // Prior fast-math bits are kept; `arcp` is merged in.
  // CHECK-LABEL: func.func @divf_preserves_other_fastmath
  // CHECK: arith.divf %{{.*}}, %{{.*}} fastmath<{{(nnan,arcp|arcp,nnan)}}> : f32
  func.func @divf_preserves_other_fastmath(%a: f32, %b: f32) -> f32 {
    %0 = arith.divf %a, %b fastmath<nnan> : f32
    return %0 : f32
  }

  // TOSA-LABEL: func.func @migraphx_div_adds_arcp
  // TOSA:      tosa.reciprocal %{{.*}} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // TOSA-NEXT: tosa.mul %{{.*}}, %{{.*}}, %{{.*}} : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>

  // ROCK-LABEL: func.func @migraphx_div_adds_arcp
  // ROCK:      %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<2x3xf32>
  // ROCK:      %[[RECIP:.*]] = arith.divf %[[ONE]], %{{.*}} : tensor<2x3xf32>
  // ROCK-NEXT: arith.mulf %{{.*}}, %[[RECIP]] : tensor<2x3xf32>

  // ARCP-LABEL: func.func @migraphx_div_adds_arcp
  // ARCP:      %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<2x3xf32>
  // ARCP:      %[[RECIP:.*]] = arith.divf %[[ONE]], %{{.*}} fastmath<arcp> : tensor<2x3xf32>
  // ARCP-NEXT: arith.mulf %{{.*}}, %[[RECIP]] : tensor<2x3xf32>
  func.func @migraphx_div_adds_arcp(%a: !migraphx.shaped<2x3xf32, 3x1>, %b: !migraphx.shaped<2x3xf32, 3x1>) -> !migraphx.shaped<2x3xf32, 3x1> attributes {kernel, arch = "gfx90a", rock.kernel} {
    %0 = migraphx.div %a, %b : <2x3xf32, 3x1>, <2x3xf32, 3x1> -> <2x3xf32, 3x1>
    return %0 : !migraphx.shaped<2x3xf32, 3x1>
  }
}