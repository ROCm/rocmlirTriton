// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-transforms-to-pointer-arith --split-input-file --verify-diagnostics

// Transform chain root is the result of arith.addf, which is neither
// a block argument nor an arith.constant.
func.func @test_error_bad_buffer_root(%arg0: tensor<4096xf16>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %added = arith.addf %arg0, %arg0 : tensor<4096xf16>

  %0 = rock.transform %added by <affine_map<(d0, d1) -> (d0 * 64 + d1)> by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]> : tensor<4096xf16> to tensor<64x64xf16>

  // expected-error @+2 {{'rock.transforms_to_ptr' op expected transform chain root to be a block argument or arith.constant, but got:}}
  // expected-error @+1 {{failed to legalize operation 'rock.transforms_to_ptr' that was explicitly marked illegal}}
  %pointers, %mask = rock.transforms_to_ptr %0 : tensor<64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  %1 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

  return %1 : tensor<64x64xf16>
}

// -----

// Transform chain bottoms out at a 2D tensor with PassThrough transforms, so
// the composed affine map produces 2 results instead of 1 linearized index and
// the shared expansion engine reports the chain as not well formed.
func.func @test_error_multiple_results(%arg0: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["m"] at [0] -> ["m"] at [0]>, <PassThrough ["n"] at [1] -> ["n"] at [1]>] bounds = [64, 64] -> [64, 64]> : tensor<64x64xf16> to tensor<64x64xf16>

  // expected-error @+2 {{'rock.transforms_to_ptr' op Transforms are not well formed}}
  // expected-error @+1 {{failed to legalize operation 'rock.transforms_to_ptr' that was explicitly marked illegal}}
  %pointers, %mask = rock.transforms_to_ptr %0 : tensor<64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
  %1 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

  return %1 : tensor<64x64xf16>
}

// -----

// Only dense tensor constants have a defined compiler-owned storage layout.
// Other ElementsAttr implementations must be rejected rather than silently
// using the zero pointer reserved for splats.
func.func @test_error_non_dense_constant() attributes {rock.arch = "##TOKEN_ARCH##"} {
  %values = arith.constant sparse<[[0], [3]], [1.0, 4.0]> : tensor<4xf32>
  // expected-error @+2 {{'rock.transforms_to_ptr' op constant transform chain root must contain dense elements}}
  // expected-error @+1 {{failed to legalize operation 'rock.transforms_to_ptr' that was explicitly marked illegal}}
  %pointers, %mask = rock.transforms_to_ptr %values : tensor<4xf32> -> tensor<4xi32>, tensor<4xi1>
  return
}
