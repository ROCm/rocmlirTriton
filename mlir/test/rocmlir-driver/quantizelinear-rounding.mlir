// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | FileCheck %s --check-prefix=TOSA
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | FileCheck %s --check-prefix=HIGHLEVEL

// QuantizeLinear rounds to nearest, ties to even. The values include numbers
// just below integers and exact .5 ties so truncation produces different
// results. The zero point is deliberately kept separate from this check.
//
// TOSA-LABEL: func.func @quantizelinear_rne
// TOSA-NOT: fp_to_int_cast
// TOSA: tosa.cast {{.*}} : (tensor<8xf32>) -> tensor<8xi32>
// TOSA-NOT: fp_to_int_cast
//
// HIGHLEVEL-LABEL: func.func @quantizelinear_rne
// HIGHLEVEL: arith.constant dense<[-123, -42, 76, -21, 956, -769, 160, -312]> : tensor<8xi32>
func.func @quantizelinear_rne() -> !migraphx.shaped<8xsi32, 1>
    attributes {rock.kernel} {
  %x = migraphx.literal(dense<[-1.22999985e+02, -4.19999962e+01,
                               7.59999924e+01, -2.09999981e+01,
                               9.565000e+02, -7.686000e+02,
                               1.595000e+02, -3.115000e+02]>
      : tensor<8xf32>) : <8xf32, 1>
  %scale = migraphx.literal(dense<1.000000e+00>
      : tensor<8xf32>) : <8xf32, 1>
  %zeroPoint = migraphx.literal(dense<[123, 123, 123, 123, 0, 0, 0, 0]>
      : tensor<8xsi32>) : <8xsi32, 1>
  %result = migraphx.quantizelinear %x, %scale, %zeroPoint
      : <8xf32, 1>, <8xf32, 1>, !migraphx.shaped<8xsi32, 1>
        -> <8xsi32, 1>
  return %result : !migraphx.shaped<8xsi32, 1>
}

// migraphx.convert retains saturating round-toward-zero semantics.
//
// TOSA-LABEL: func.func @convert_rtz
// TOSA-NOT: tosa.cast
// TOSA: operator_name = "fp_to_int_cast"
//
// HIGHLEVEL-LABEL: func.func @convert_rtz
// HIGHLEVEL-NOT: math.roundeven
// HIGHLEVEL: arith.fptosi
func.func @convert_rtz(%input: !migraphx.shaped<8xf32, 1>)
    -> !migraphx.shaped<8xsi32, 1> attributes {rock.kernel} {
  %result = migraphx.convert %input : <8xf32, 1> to <8xsi32, 1>
  return %result : !migraphx.shaped<8xsi32, 1>
}
