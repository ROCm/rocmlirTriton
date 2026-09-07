// RUN: rocmlir-opt --migraphx-to-tosa -verify-diagnostics -split-input-file %s

func.func @dot_dynamic_m(%arg0: !migraphx.shaped<?x72xf32, 72x1>,
                         %arg1: !migraphx.shaped<72x64xf32, 64x1>)
    -> !migraphx.shaped<?x64xf32, 64x1> {
  // expected-error @+2 {{dynamic shapes are not supported}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.dot' that was explicitly marked illegal}}
  %0 = migraphx.dot %arg0, %arg1 : <?x72xf32, 72x1>, <72x64xf32, 64x1> -> <?x64xf32, 64x1>
  return %0 : !migraphx.shaped<?x64xf32, 64x1>
}

// -----

func.func @add_dynamic_stride(%arg0: !migraphx.shaped<32x64xf32, ?x1>)
    -> !migraphx.shaped<32x64xf32, ?x1> {
  %0 = migraphx.add %arg0, %arg0 : <32x64xf32, ?x1>, <32x64xf32, ?x1> -> <32x64xf32, ?x1>
  return %0 : !migraphx.shaped<32x64xf32, ?x1>
}

// -----

func.func @dot_dynamic_result(%arg0: !migraphx.shaped<32x72xf32, 72x1>,
                              %arg1: !migraphx.shaped<72x64xf32, 64x1>)
    -> !migraphx.shaped<32x64xf32, 64x1> {
  // expected-error @+2 {{dynamic shapes are not supported}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.dot' that was explicitly marked illegal}}
  %0 = migraphx.dot %arg0, %arg1 : <32x72xf32, 72x1>, <72x64xf32, 64x1> -> <?x64xf32, 64x1>
  %1 = migraphx.relu %0 : <?x64xf32, 64x1> -> <32x64xf32, 64x1>
  return %1 : !migraphx.shaped<32x64xf32, 64x1>
}
