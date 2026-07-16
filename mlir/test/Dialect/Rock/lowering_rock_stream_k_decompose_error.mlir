// Error tests for the rock-stream-k-decompose pass.
//
// The pass runs at the gridwise layer; each case is a rock.gridwise_gemm that
// is selected for stream-K decomposition (streamKMultiple >= 1, grid overflows
// the persistent wave, clean rectangular partition) but hits one of the pass's
// hard-error diagnostics on the output side.

// RUN: rocmlir-opt -rock-stream-k-decompose -verify-diagnostics -split-input-file -mlir-print-local-scope %s

#params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, streamKMultiple = 1>

// ============================================================
// Error: the output is stored by atomic_max. The split-K remainder can only
// reconstruct the result by summing partial products (atomic_add into a
// zero-prefill), which is incompatible with a max reduction.
// ============================================================

func.func @stream_k_atomic_max(%a: tensor<1x1280x256xf16>, %b: tensor<1x256x1280xf16>, %c: tensor<1x1280x1280xf32>) -> tensor<1x1280x1280xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 80 : i32, rock.grid_size = 100 : i32} {
  %r = rock.gridwise_gemm(%a, %b) {params = #params, rock.streamk_part_dim = #rock<rock.streamk_part_dim n>} : tensor<1x1280x256xf16>, tensor<1x256x1280xf16> -> tensor<1x1280x1280xf32>
  // expected-error @+1 {{atomic_max output is incompatible with the split-K remainder}}
  %out = rock.store %r to %c by atomic_max : tensor<1x1280x1280xf32> -> tensor<1x1280x1280xf32> to tensor<1x1280x1280xf32>
  return %out : tensor<1x1280x1280xf32>
}

// -----

#splitk_params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 2, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, streamKMultiple = 1>

// ============================================================
// Error: stream-K is explicitly requested (streamKMultiple = 1) on a gemm that
// is already split-K (splitKFactor = 2). The remainder wave re-splits K, which
// cannot compose with an existing split-K, so the pass fails.
// ============================================================

func.func @stream_k_already_split_k(%a: tensor<1x1280x256xf16>, %b: tensor<1x256x1280xf16>, %c: tensor<1x1280x1280xf32>) -> tensor<1x1280x1280xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 80 : i32, rock.grid_size = 100 : i32} {
  // expected-error @+1 {{streamKMultiple is incompatible with splitKFactor > 1}}
  %r = rock.gridwise_gemm(%a, %b) {params = #splitk_params} : tensor<1x1280x256xf16>, tensor<1x256x1280xf16> -> tensor<1x1280x1280xf32>
  %out = rock.store %r to %c by set : tensor<1x1280x1280xf32> -> tensor<1x1280x1280xf32> to tensor<1x1280x1280xf32>
  return %out : tensor<1x1280x1280xf32>
}
