// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// rocmlirTriton, unlike the rocMLIR it was forked from, works on tensors
// throughout. Every Rock op must reject a memref operand or result, so
// that nothing can build a buffer-level Rock program by accident.
// 
// The custom assembly formats reject the type while parsing. The generic
// form (and any op built from C++) skips the parser, so it is exercised
// too: there the operand/result constraint of the op is what fires.

// -----

// rock.transform used to take any shaped type, memrefs included.
func.func @transform_memref_input(%arg0: memref<64xf32>) {
  // expected-error @+1 {{invalid kind of type specified: expected builtin.tensor, but found 'memref<64xf32>'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : memref<64xf32> to tensor<64xf32>
  return
}

// -----

func.func @transform_memref_result(%arg0: tensor<64xf32>) {
  // expected-error @+1 {{invalid kind of type specified: expected builtin.tensor, but found 'memref<64xf32>'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : tensor<64xf32> to memref<64xf32>
  return
}

// -----

func.func @transform_memref_input_and_result(%arg0: memref<4x16xf32>) {
  // expected-error @+1 {{invalid kind of type specified: expected builtin.tensor, but found 'memref<4x16xf32>'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "b"] at [0, 1]>] bounds = [4, 16] -> [4, 16]> : memref<4x16xf32> to memref<4x16xf32>
  return
}

// -----

// An unranked tensor is not a ranked one: the verifier and the type
// inference both need the shape.
func.func @transform_unranked_input(%arg0: tensor<*xf32>) {
  // expected-error @+1 {{invalid kind of type specified: expected builtin.tensor, but found 'tensor<*xf32>'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : tensor<*xf32> to tensor<64xf32>
  return
}

// -----

// Generic form: no custom parser in the way, so the ODS constraints speak.
func.func @transform_generic_memref_input(%arg0: memref<64xf32>) {
  // expected-error @+1 {{'rock.transform' op operand #0 must be ranked tensor of any non-token type values, but got 'memref<64xf32>'}}
  %0 = "rock.transform"(%arg0) <{transform = #rock.transform_map<affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]>}> : (memref<64xf32>) -> tensor<64xf32>
  return
}

// -----

func.func @transform_generic_memref_result(%arg0: tensor<64xf32>) {
  // expected-error @+1 {{'rock.transform' op result #0 must be ranked tensor of any non-token type values, but got 'memref<64xf32>'}}
  %0 = "rock.transform"(%arg0) <{transform = #rock.transform_map<affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]>}> : (tensor<64xf32>) -> memref<64xf32>
  return
}

// -----

func.func @gemm_memref_operand(%a: memref<1x32x64xf32>, %b: tensor<1x32x128xf32>) -> tensor<1x64x128xf32> {
  // expected-error @+1 {{invalid kind of type specified}}
  %0 = rock.gemm tr %a * %b : memref<1x32x64xf32> * tensor<1x32x128xf32> -> tensor<1x64x128xf32>
  return %0 : tensor<1x64x128xf32>
}

// -----

#gemm_params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_memref_operand(%a: memref<1x16x16xf32>, %b: tensor<1x16x16xf32>) -> tensor<1x16x16xf32> attributes {rock.arch = "gfx906"} {
  // expected-error @+1 {{invalid kind of type specified: expected builtin.tensor, but found 'memref<1x16x16xf32>'}}
  %0 = rock.gridwise_gemm(%a, %b) {params = #gemm_params} : memref<1x16x16xf32>, tensor<1x16x16xf32> -> tensor<1x16x16xf32>
  return %0 : tensor<1x16x16xf32>
}

// -----

func.func @store_memref_dest(%src: tensor<4x4xf32>, %dest: memref<4x4xf32>) {
  // expected-error @+1 {{invalid kind of type specified: expected builtin.tensor, but found 'memref<4x4xf32>'}}
  %0 = rock.store %src to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to memref<4x4xf32>
  return
}

// -----

func.func @reduce_memref_input(%in: memref<8x32xf32>) {
  // expected-error @+1 {{invalid kind of type specified: expected builtin.tensor, but found 'memref<8x32xf32>'}}
  %0 = rock.reduce sum %in {axis = 1 : index} : memref<8x32xf32> -> tensor<8x1xf32>
  return
}

// -----

func.func @untile_memref_source(%src: memref<4x4xf32>) {
  // expected-error @+1 {{invalid kind of type specified: expected builtin.tensor, but found 'memref<4x4xf32>'}}
  %0 = rock.untile %src : memref<4x4xf32> -> tensor<16x16xf32>
  return
}
