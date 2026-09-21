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
