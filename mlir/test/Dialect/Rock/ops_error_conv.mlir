// RUN: rocmlir-opt -split-input-file -verify-diagnostics %s

// =============================================================================
// rock.conv tests
// =============================================================================

// Rank mismatch: filter is rank 4, input and output are rank 5
func.func @conv_rank_mismatch(
    %filter: tensor<128x8x3x3xf32>,
    %input: tensor<128x1x8x32x32xf32>) {
  // expected-error @+1 {{filter, input, and output must have the same rank}}
  %result = rock.conv(%filter, %input) {
    filter_layout = ["k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index],
    dilations = [1 : index, 1 : index]
  } : tensor<128x8x3x3xf32>, tensor<128x1x8x32x32xf32> -> tensor<128x1x128x30x30xf32>
  func.return
}

// -----

// Filter float, input integer type mismatch (ConvOp uses GemmInputTypes)
func.func @conv_filter_input_type_mismatch(
    %filter: tensor<1x128x8x3x3xf32>,
    %input: tensor<128x1x8x32x32xi8>) {
  // expected-error @+1 {{filter and input must both be float or both be integer types}}
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index],
    dilations = [1 : index, 1 : index]
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xi8> -> tensor<128x1x128x30x30xf32>
  func.return
}

// -----

// Float inputs with integer output
func.func @conv_float_in_int_out(
    %filter: tensor<1x128x8x3x3xf32>,
    %input: tensor<128x1x8x32x32xf32>) {
  // expected-error @+1 {{float-valued inputs must have a floating-point output type}}
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index],
    dilations = [1 : index, 1 : index]
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32> -> tensor<128x1x128x30x30xi32>
  func.return
}

// -----

// Integer inputs with float output
func.func @conv_int_in_float_out(
    %filter: tensor<1x128x8x3x3xi8>,
    %input: tensor<128x1x8x32x32xi8>) {
  // expected-error @+1 {{integer-valued inputs must have an integer output type}}
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index],
    dilations = [1 : index, 1 : index]
  } : tensor<1x128x8x3x3xi8>, tensor<128x1x8x32x32xi8> -> tensor<128x1x128x30x30xf32>
  func.return
}

// -----

// Strides and dilations size mismatch
func.func @conv_strides_dilations_mismatch(
    %filter: tensor<1x128x8x3x3xf32>,
    %input: tensor<128x1x8x32x32xf32>) {
  // expected-error @+1 {{strides and dilations must have the same size}}
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index],
    dilations = [1 : index, 1 : index, 1 : index]
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32> -> tensor<128x1x128x30x30xf32>
  func.return
}

// -----

// Padding vs strides size mismatch (padding=6, strides=2, expected padding=4)
func.func @conv_padding_strides_mismatch(
    %filter: tensor<1x128x8x3x3xf32>,
    %input: tensor<128x1x8x32x32xf32>) {
  // expected-error @+1 {{padding must have twice as many elements as strides}}
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index],
    dilations = [1 : index, 1 : index]
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32> -> tensor<128x1x128x30x30xf32>
  func.return
}

// -----

// Strides count doesn't match spatial dims (3 strides for 2D conv)
func.func @conv_strides_spatial_mismatch(
    %filter: tensor<1x128x8x3x3xf32>,
    %input: tensor<128x1x8x32x32xf32>) {
  // expected-error @+1 {{number of strides must match number of spatial dimensions}}
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index, 1 : index],
    dilations = [1 : index, 1 : index, 1 : index]
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32> -> tensor<128x1x128x30x30xf32>
  func.return
}

// -----

// =============================================================================
// rock.conv_bwd_data tests
// =============================================================================

// Rank mismatch on conv_bwd_data: filter is rank 4, gradient is rank 5
func.func @conv_bwd_data_rank_mismatch(
    %filter: tensor<128x8x3x3xf32>,
    %gradient: tensor<128x1x128x30x30xf32>) {
  // expected-error @+1 {{filter, input, and output must have the same rank}}
  %result = rock.conv_bwd_data(%filter, %gradient) {
    filter_layout = ["k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index],
    dilations = [1 : index, 1 : index]
  } : tensor<128x8x3x3xf32>, tensor<128x1x128x30x30xf32> -> tensor<128x1x8x32x32xf32>
  func.return
}

// -----

// =============================================================================
// rock.conv_bwd_weight tests
// =============================================================================

// Rank mismatch on conv_bwd_weight: input is rank 5, gradient is rank 5, result (filter) is rank 4
func.func @conv_bwd_weight_rank_mismatch(
    %input: tensor<128x1x8x32x32xf32>,
    %gradient: tensor<128x1x128x30x30xf32>) {
  // expected-error @+1 {{filter, input, and output must have the same rank}}
  %result = rock.conv_bwd_weight(%input, %gradient) {
    filter_layout = ["k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index],
    dilations = [1 : index, 1 : index]
  } : tensor<128x1x8x32x32xf32>, tensor<128x1x128x30x30xf32> -> tensor<128x8x3x3xf32>
  func.return
}

// -----

// kBlocks must be positive
func.func @conv_bwd_weight_nonpositive_kblocks(
    %input: tensor<128x1x8x32x32xf32>,
    %gradient: tensor<128x1x128x30x30xf32>) {
  // expected-error @+1 {{kBlocks (0) must be positive and evenly divide batch size N (128)}}
  %result = rock.conv_bwd_weight(%input, %gradient) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index],
    dilations = [1 : index, 1 : index],
    kBlocks = 0 : index
  } : tensor<128x1x8x32x32xf32>, tensor<128x1x128x30x30xf32> -> tensor<1x128x8x3x3xf32>
  func.return
}

// -----

// kBlocks must evenly divide the batch dimension N
func.func @conv_bwd_weight_kblocks_does_not_divide_n(
    %input: tensor<128x1x8x32x32xf32>,
    %gradient: tensor<128x1x128x30x30xf32>) {
  // expected-error @+1 {{kBlocks (5) must be positive and evenly divide batch size N (128)}}
  %result = rock.conv_bwd_weight(%input, %gradient) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index],
    dilations = [1 : index, 1 : index],
    kBlocks = 5 : index
  } : tensor<128x1x8x32x32xf32>, tensor<128x1x128x30x30xf32> -> tensor<1x128x8x3x3xf32>
  func.return
}
