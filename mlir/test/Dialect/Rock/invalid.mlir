// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// -----

#gemm_params2 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_m_too_big(%a: tensor<1x2147483648x1xf32>,
                        %b: tensor<1x1x1xf32>,
                        %c: tensor<1x2147483648x1xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op M dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  %result = rock.gridwise_gemm(%a, %b) {
    params = #gemm_params2}
  : tensor<1x2147483648x1xf32>, tensor<1x1x1xf32> -> tensor<1x2147483648x1xf32>
  func.return
}

// -----

#gemm_params3 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_k_too_big(%a: tensor<1x1x2147483648xf32>,
                        %b: tensor<1x2147483648x1xf32>,
                        %c: tensor<1x1x1xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op K dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  %result = rock.gridwise_gemm(%a, %b) {
    params = #gemm_params3}
  : tensor<1x1x2147483648xf32>, tensor<1x2147483648x1xf32> -> tensor<1x1x1xf32>
  func.return
}
// -----

#gemm_params4 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_n_too_big(%a: tensor<1x1x1xf32>,
                        %b: tensor<1x1x2147483648xf32>,
                        %c: tensor<1x1x2147483648xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op N dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  %result = rock.gridwise_gemm(%a, %b) {
    params = #gemm_params4}
  : tensor<1x1x1xf32>, tensor<1x1x2147483648xf32> -> tensor<1x1x2147483648xf32>
  func.return
}

// -----

func.func @store_atomic_add_fp8(%source: tensor<4xf8E4M3FNUZ>,
                                %dest: tensor<4xf8E4M3FNUZ>) {
  // expected-error@+1 {{'rock.store' op source element type 'f8E4M3FNUZ' does not support atomic_add}}
  %0 = rock.store %source to %dest by atomic_add : tensor<4xf8E4M3FNUZ> -> tensor<4xf8E4M3FNUZ> to tensor<4xf8E4M3FNUZ>
  func.return
}

// -----

func.func @blockwise_store_atomic_add_fp8(
    %source: tensor<4xf8E4M3FNUZ>, %dest: tensor<4xf8E4M3FNUZ>) {
  // expected-error@+1 {{'rock.blockwise_store' op source element type 'f8E4M3FNUZ' does not support atomic_add}}
  %0 = rock.blockwise_store %source -> %dest by atomic_add : tensor<4xf8E4M3FNUZ> -> tensor<4xf8E4M3FNUZ> -> tensor<4xf8E4M3FNUZ>
  func.return
}

// -----

func.func @blockwise_store_ptr_atomic_add_fp8(
    %source: tensor<4xf8E4M3FNUZ>, %pointers: tensor<4xi32>,
    %mask: tensor<4xi1>) {
  // expected-error@+1 {{'rock.blockwise_store_ptr' op source element type 'f8E4M3FNUZ' does not support atomic_add}}
  rock.blockwise_store_ptr %source -> %pointers(%mask) by atomic_add : tensor<4xf8E4M3FNUZ> -> tensor<4xi32>(tensor<4xi1>)
  func.return
}

// -----

func.func @store_atomic_max_fp8(%source: tensor<4xf8E4M3FNUZ>,
                                %dest: tensor<4xf8E4M3FNUZ>) {
  // expected-error@+1 {{'rock.store' op source element type 'f8E4M3FNUZ' does not support atomic_max}}
  %0 = rock.store %source to %dest by atomic_max : tensor<4xf8E4M3FNUZ> -> tensor<4xf8E4M3FNUZ> to tensor<4xf8E4M3FNUZ>
  func.return
}

func.func @store_atomic_max_supported(
    %f16Source: tensor<4xf16>, %f16Dest: tensor<4xf16>,
    %bf16Source: tensor<4xbf16>, %bf16Dest: tensor<4xbf16>,
    %f32Source: tensor<4xf32>, %f32Dest: tensor<4xf32>,
    %f64Source: tensor<4xf64>, %f64Dest: tensor<4xf64>,
    %i4Source: tensor<4xi4>, %i4Dest: tensor<4xi4>,
    %i8Source: tensor<4xi8>, %i8Dest: tensor<4xi8>,
    %i16Source: tensor<4xi16>, %i16Dest: tensor<4xi16>,
    %i32Source: tensor<4xi32>, %i32Dest: tensor<4xi32>,
    %i64Source: tensor<4xi64>, %i64Dest: tensor<4xi64>) {
  %0 = rock.store %f16Source to %f16Dest by atomic_max : tensor<4xf16> -> tensor<4xf16> to tensor<4xf16>
  %1 = rock.store %bf16Source to %bf16Dest by atomic_max : tensor<4xbf16> -> tensor<4xbf16> to tensor<4xbf16>
  %2 = rock.store %f32Source to %f32Dest by atomic_max : tensor<4xf32> -> tensor<4xf32> to tensor<4xf32>
  %3 = rock.store %f64Source to %f64Dest by atomic_max : tensor<4xf64> -> tensor<4xf64> to tensor<4xf64>
  %4 = rock.store %i4Source to %i4Dest by atomic_max : tensor<4xi4> -> tensor<4xi4> to tensor<4xi4>
  %5 = rock.store %i8Source to %i8Dest by atomic_max : tensor<4xi8> -> tensor<4xi8> to tensor<4xi8>
  %6 = rock.store %i16Source to %i16Dest by atomic_max : tensor<4xi16> -> tensor<4xi16> to tensor<4xi16>
  %7 = rock.store %i32Source to %i32Dest by atomic_max : tensor<4xi32> -> tensor<4xi32> to tensor<4xi32>
  %8 = rock.store %i64Source to %i64Dest by atomic_max : tensor<4xi64> -> tensor<4xi64> to tensor<4xi64>
  func.return
}

// -----

// COM: Negative coverage for rock::TransformAttr::verify and
// COM: rock::TransformMapAttr::verify in
// COM: mlir/lib/Dialect/Rock/IR/RockDialect.cpp. Each section trips exactly one
// COM: verifier branch; the malformed #rock.transform_map attribute is rejected
// COM: at parse time via getChecked. Two diagnostics are emitted per case: the
// COM: verifier message and the generic "failed to parse ... parameter 'ops'"
// COM: wrapper from the attribute parser.

// COM: upperNames.size() != upperDims.size()
func.func @transform_upper_names_dims_mismatch(%arg0: tensor<64xf32>) {
  // expected-error @+2 {{Have 2 names for 1 dimensions}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a", "b"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : tensor<64xf32> to tensor<64xf32>
  return
}

// -----

// COM: lowerNames.size() != lowerDims.size()
func.func @transform_lower_names_dims_mismatch(%arg0: tensor<64xf32>) {
  // expected-error @+2 {{Have 2 names for 1 dimensions}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a", "b"] at [0]>] bounds = [64] -> [64]> : tensor<64xf32> to tensor<64xf32>
  return
}

// -----

// COM: non-AddDim transform with no outputs
func.func @transform_no_outputs(%arg0: tensor<64xf32>) {
  // expected-error @+2 {{The transformation must define outputs}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> [] at []>] bounds = [64] -> [64]> : tensor<64xf32> to tensor<64xf32>
  return
}

// -----

// COM: non-ConstDim transform with no inputs
func.func @transform_no_inputs(%arg0: tensor<64xf32>) {
  // expected-error @+2 {{The transformation must have at least one input}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough [] at [] -> ["a"] at [0]>] bounds = [64] -> [64]> : tensor<64xf32> to tensor<64xf32>
  return
}

// -----

// COM: PassThrough must have matching input/output rank
func.func @transform_passthrough_rank(%arg0: tensor<64x64xf32>) {
  // expected-error @+2 {{PassThrough must have the same number of inputs and outputs}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0)> by [<PassThrough ["a", "b"] at [0, 1] -> ["a"] at [0]>] bounds = [64, 64] -> [64]> : tensor<64x64xf32> to tensor<64xf32>
  return
}

// -----

// COM: PassThrough takes no parameters
func.func @transform_passthrough_params(%arg0: tensor<64xf32>) {
  // expected-error @+2 {{PassThrough has no parameters}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough{1} ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : tensor<64xf32> to tensor<64xf32>
  return
}

// -----

// COM: Embed can only have one output argument
func.func @transform_embed_one_output(%arg0: tensor<64x64xf32>) {
  // expected-error @+2 {{Embed can only have one output argument}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0, d1)> by [<Embed{1, 1} ["a", "b"] at [0, 1] -> ["x", "y"] at [0, 1]>] bounds = [64, 64] -> [64, 64]> : tensor<64x64xf32> to tensor<64x64xf32>
  return
}

// -----

// COM: Embed must specify one coefficient per input dimension
func.func @transform_embed_coeffs(%arg0: tensor<64x3xf32>) {
  // expected-error @+2 {{Embed must specify one coefficient per input dimension}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0)> by [<Embed{1} ["a", "b"] at [0, 1] -> ["x"] at [0]>] bounds = [64, 3] -> [64]> : tensor<64x3xf32> to tensor<64xf32>
  return
}

// -----

// COM: Unmerge can only have one output argument
func.func @transform_unmerge_one_output(%arg0: tensor<16x4xf32>) {
  // expected-error @+2 {{Unmerge can only have one output argument}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0, d1)> by [<Unmerge{16, 4} ["a", "b"] at [0, 1] -> ["x", "y"] at [0, 1]>] bounds = [16, 4] -> [16, 4]> : tensor<16x4xf32> to tensor<16x4xf32>
  return
}

// -----

// COM: Unmerge must specify one length per input dimension
func.func @transform_unmerge_lengths(%arg0: tensor<16x4xf32>) {
  // expected-error @+2 {{Unmerge must specify one length per input dimension}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0)> by [<Unmerge{16} ["a", "b"] at [0, 1] -> ["x"] at [0]>] bounds = [16, 4] -> [64]> : tensor<16x4xf32> to tensor<64xf32>
  return
}

// -----

// COM: Merge can only have one input dimension
func.func @transform_merge_one_input(%arg0: tensor<16x4xf32>) {
  // expected-error @+2 {{Merge can only have one input dimension}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0, d1)> by [<Merge{16, 4} ["a", "b"] at [0, 1] -> ["x", "y"] at [0, 1]>] bounds = [16, 4] -> [16, 4]> : tensor<16x4xf32> to tensor<16x4xf32>
  return
}

// -----

// COM: Merge has one parameter per output dimension
func.func @transform_merge_params(%arg0: tensor<64xf32>) {
  // expected-error @+2 {{Merge must have one parameter per output dimension (its size)}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0, d0)> by [<Merge{16} ["a"] at [0] -> ["x", "y"] at [0, 1]>] bounds = [64] -> [16, 4]> : tensor<64xf32> to tensor<16x4xf32>
  return
}

// -----

// COM: AddDim can only add one dimension at a time
func.func @transform_adddim_one(%arg0: tensor<16x4xf32>) {
  // expected-error @+2 {{Can only add one dimension at a time}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> ()> by [<AddDim{16, 4} ["a", "b"] at [0, 1] -> [] at []>] bounds = [16, 4] -> []> : tensor<16x4xf32> to tensor<f32>
  return
}

// -----

// COM: AddDim must supply a size parameter for each dimension
func.func @transform_adddim_size(%arg0: tensor<16xf32>) {
  // expected-error @+2 {{Must supply a size parameter for each dimension}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> ()> by [<AddDim ["a"] at [0] -> [] at []>] bounds = [16] -> []> : tensor<16xf32> to tensor<f32>
  return
}

// -----

// COM: AddDim output cannot be mapped anywhere
func.func @transform_adddim_mapped(%arg0: tensor<16xf32>) {
  // expected-error @+2 {{The added dimension cannot be mapped anywhere}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<AddDim{16} ["a"] at [0] -> ["x"] at [0]>] bounds = [16] -> [16]> : tensor<16xf32> to tensor<16xf32>
  return
}

// -----

// COM: Broadcast must have same rank
func.func @transform_broadcast_rank(%arg0: tensor<64x64xf32>) {
  // expected-error @+2 {{Broadcast must have same rank}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0)> by [<Broadcast{1} ["a", "b"] at [0, 1] -> ["x"] at [0]>] bounds = [64, 64] -> [64]> : tensor<64x64xf32> to tensor<64xf32>
  return
}

// -----

// COM: Broadcast must specify the output length for each dimension
func.func @transform_broadcast_lengths(%arg0: tensor<64xf32>) {
  // expected-error @+2 {{Broadcast must specify the output length for each dimension}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<Broadcast{1, 2} ["a"] at [0] -> ["x"] at [0]>] bounds = [64] -> [64]> : tensor<64xf32> to tensor<64xf32>
  return
}

// -----

// COM: ConstDim must not take any inputs
func.func @transform_constdim_inputs(%arg0: tensor<64xf32>) {
  // expected-error @+2 {{ConstDim must not take any inputs}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<ConstDim{0, 8} ["a"] at [0] -> ["x"] at [0]>] bounds = [64] -> [8]> : tensor<64xf32> to tensor<8xf32>
  return
}

// -----

// COM: ConstDim is parameterized by [value, length] pairs
func.func @transform_constdim_pairs(%arg0: tensor<f32>) {
  // expected-error @+2 {{ConstDim is parameterized by [value, length] pairs}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<() -> (0)> by [<ConstDim{1} [] at [] -> ["x"] at [0]>] bounds = [] -> [8]> : tensor<f32> to tensor<8xf32>
  return
}

// -----

// COM: ConstDim value must be less than dimension length
func.func @transform_constdim_value(%arg0: tensor<f32>) {
  // expected-error @+2 {{constant value 8 must be less than dimension length 8}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<() -> (0)> by [<ConstDim{8, 8} [] at [] -> ["x"] at [0]>] bounds = [] -> [8]> : tensor<f32> to tensor<8xf32>
  return
}

// -----

// COM: TransformMapAttr: affine map input count must match upper bounds
func.func @transform_map_input_count(%arg0: tensor<64xf32>) {
  // expected-error @+1 {{Affine map has 2 inputs but there are 1 input dimensions}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : tensor<64xf32> to tensor<64xf32>
  return
}

// -----

// COM: TransformMapAttr: affine map output count must match lower bounds
func.func @transform_map_output_count(%arg0: tensor<64xf32>) {
  // expected-error @+1 {{Affine map has 2 outputs but there are 1 output dimensions}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0, d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : tensor<64xf32> to tensor<64xf32>
  return
}

// -----

// COM: TransformMapAttr: non-positive upper bound rejected
func.func @transform_map_negative_upper(%arg0: tensor<64xf32>) {
  // expected-error @+1 {{Upper bound/shape component must be positive, got -1}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [-1] -> [64]> : tensor<64xf32> to tensor<64xf32>
  return
}

// -----

// COM: TransformMapAttr: non-positive lower bound rejected
func.func @transform_map_negative_lower(%arg0: tensor<64xf32>) {
  // expected-error @+1 {{Lower bound/shape component must be positive, got -1}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [-1]> : tensor<64xf32> to tensor<64xf32>
  return
}

// -----

// COM: TransformAttr::parse rejects an unknown transform name
func.func @transform_unknown_name(%arg0: tensor<64xf32>) {
  // expected-error @+3 {{expected a name of a known transform}}
  // expected-note @+2 {{The transforms are PassThrough, Pad, Slice, Embed, Unmerge, Merge, AddDim, Broadcast, ConstDim}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<NotARealTransform ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : tensor<64xf32> to tensor<64xf32>
  return
}
