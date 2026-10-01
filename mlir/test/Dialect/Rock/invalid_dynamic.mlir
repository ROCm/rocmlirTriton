// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// Verifier errors for symbolic transform maps, #rock.arg_expr, and
// rock.transform on tensors with dynamic dimensions.

func.func @map_symbols_unbound(%arg0: tensor<?xf32>) {
  // expected-error @+1 {{Affine map must not have symbol inputs, but has 1}}
  %0 = rock.transform %arg0 by <affine_map<(d0)[s0] -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : tensor<?xf32> to tensor<?xf32>
  return
}

// -----

func.func @map_symbol_count_mismatch(%arg0: tensor<?x?xf32>) {
  // expected-error @+1 {{Affine map has 1 symbols but 2 are bound}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1)[s0] -> (d0, d1)> by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "b"] at [0, 1]>] symbols = [arg(0, 0), arg(0, 1)] bounds = [s0, s1] -> [s0, s1]> : tensor<?x?xf32> to tensor<?x?xf32>
  return
}

// -----

func.func @duplicate_binding(%arg0: tensor<?x?xf32>) {
  // expected-error @+1 {{Duplicate symbol binding arg(0, 0)}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1)[s0, s1] -> (d0, d1)> by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "b"] at [0, 1]>] symbols = [arg(0, 0), arg(0, 0)] bounds = [s0, s1] -> [s0, s1]> : tensor<?x?xf32> to tensor<?x?xf32>
  return
}

// -----

func.func @bounds_unbound_symbol(%arg0: tensor<?xf32>) {
  // expected-error @+1 {{Upper bounds use unbound symbols}}
  %0 = rock.transform %arg0 by <affine_map<(d0)[s0] -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] symbols = [arg(0, 0)] bounds = [s1] -> [s0]> : tensor<?xf32> to tensor<?xf32>
  return
}

// -----

func.func @transform_unbound_symbol(%arg0: tensor<?xf32>) {
  // expected-error @+1 {{Transform #rock.transform<Pad{0, s1} ["a"] at [0] -> ["a"] at [0]> uses unbound symbols}}
  %0 = rock.transform %arg0 by <affine_map<(d0)[s0] -> (d0)> by [<Pad{0, s1} ["a"] at [0] -> ["a"] at [0]>] symbols = [arg(0, 0)] bounds = [s0] -> [s0]> : tensor<?xf32> to tensor<?xf32>
  return
}

// -----

func.func @symbolic_params_static_map(%arg0: tensor<64xf32>) {
  // expected-error @+1 {{has symbolic parameters but the map binds no symbols}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<Pad{0, s0} ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : tensor<64xf32> to tensor<64xf32>
  return
}

// -----

func.func @passthrough_symbolic_vs_constant(%arg0: tensor<?xf32>) {
  // expected-error @+1 {{PassThrough: upper bound 64 does not match lower bound s0}}
  %0 = rock.transform %arg0 by <affine_map<(d0)[s0] -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] symbols = [arg(0, 0)] bounds = [64] -> [s0]> : tensor<?xf32> to tensor<64xf32>
  return
}

// -----

func.func @pad_bound_mismatch(%arg0: tensor<?xf32>) {
  // expected-error @+1 {{Pad: upper bound s0 + 2 does not match lower bound s0 + leftPad(0) + rightPad(1) = s0 + 1}}
  %0 = rock.transform %arg0 by <affine_map<(d0)[s0] -> (d0)> by [<Pad{0, 1} ["a"] at [0] -> ["a"] at [0]>] symbols = [arg(0, 0)] bounds = [s0 + 2] -> [s0]> : tensor<?xf32> to tensor<?xf32>
  return
}

// -----

func.func @merge_product_mismatch(%arg0: tensor<?x?xf32>) {
  // expected-error @+1 {{Merge: product of parameters (s0 * s1) does not match upper bound (s0 * 2)}}
  %0 = rock.transform %arg0 by <affine_map<(d0)[s0, s1] -> (d0 floordiv s1, d0 mod s1)> by [<Merge{s0, s1} ["mn"] at [0] -> ["m", "n"] at [0, 1]>] symbols = [arg(0, 0), arg(0, 1)] bounds = [s0 * 2] -> [s0, s1]> : tensor<?x?xf32> to tensor<?xf32>
  return
}

// -----

func.func @unmerge_param_mismatch(%arg0: tensor<?xf32>) {
  // expected-error @+1 {{Unmerge: upper bound s0 at dimension 0 does not match parameter s1}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + d1)> by [<Unmerge{s1, s0} ["m", "n"] at [0, 1] -> ["mn"] at [0]>] symbols = [arg(0, 0), arg(1, 0)] bounds = [s0, s1] -> [s0 * s1]> : tensor<?xf32> to tensor<?x?xf32>
  return
}

// -----

func.func @slice_bound_mismatch(%arg0: tensor<?xf32>) {
  // expected-error @+1 {{Slice: upper bound s0 does not match end(s0) - begin(1) = s0 - 1}}
  %0 = rock.transform %arg0 by <affine_map<(d0)[s0] -> (d0 + 1)> by [<Slice{1, s0} ["a"] at [0] -> ["a"] at [0]>] symbols = [arg(0, 0)] bounds = [s0] -> [s0]> : tensor<?xf32> to tensor<?xf32>
  return
}

// -----

func.func @symbol_names_static_dim(%arg0: tensor<64x?xf32>) {
  // expected-error @+1 {{'rock.transform' op symbol arg(0, 0) does not name a dynamic dimension of a function argument}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1)[s0] -> (d0, d1)> by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "b"] at [0, 1]>] symbols = [arg(0, 0)] bounds = [64, s0] -> [64, s0]> : tensor<64x?xf32> to tensor<64x?xf32>
  return
}

// -----

func.func @symbol_names_missing_arg(%arg0: tensor<?xf32>) {
  // expected-error @+1 {{'rock.transform' op symbol arg(3, 0) does not name a dynamic dimension of a function argument}}
  %0 = rock.transform %arg0 by <affine_map<(d0)[s0] -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] symbols = [arg(3, 0)] bounds = [s0] -> [s0]> : tensor<?xf32> to tensor<?xf32>
  return
}

// -----

func.func @symbol_names_out_of_rank(%arg0: tensor<?xf32>) {
  // expected-error @+1 {{'rock.transform' op symbol arg(0, 1) does not name a dynamic dimension of a function argument}}
  %0 = rock.transform %arg0 by <affine_map<(d0)[s0] -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] symbols = [arg(0, 1)] bounds = [s0] -> [s0]> : tensor<?xf32> to tensor<?xf32>
  return
}

// -----

func.func @dynamic_output_static_bound(%arg0: tensor<?xf32>) {
  // expected-error @+1 {{'rock.transform' op output shape must match transform upper bounds}}
  %0 = rock.transform %arg0 by <affine_map<(d0)[s0] -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] symbols = [arg(0, 0)] bounds = [s0] -> [s0]> : tensor<?xf32> to tensor<64xf32>
  return
}

// -----

// expected-error @+1 {{expression s0 * s1 uses more than the 1 bound symbols}}
module attributes {rock.grid_size.k = #rock.arg_expr<s0 * s1, [arg(0, 0)]>} {
}
