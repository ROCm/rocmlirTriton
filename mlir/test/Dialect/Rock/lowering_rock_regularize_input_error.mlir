// Error tests for rock-regularize-input pass.

// RUN: rocmlir-opt -rock-regularize-input -verify-diagnostics --split-input-file %s

// ============================================================
// Error: non-splat constant in fusion chain of load_marker source.
// ============================================================

#tmap_small = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1 * 2 + d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Unmerge{1, 2} ["m_block", "m_iter"] at [1, 2] -> ["m"] at [1]>] bounds = [1, 1, 2] -> [1, 2]>

module {
  func.func @error_non_splat_constant(%tile: tensor<2xf32>, %dest: tensor<1x2xf32>, %g: i32, %m: i32) -> tensor<1x2xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap_small] [%g, %m] : tensor<2xf32> -> tensor<1x2xf32>
    // expected-error @below {{'arith.constant' op non-splat constant in fusion chain not supported}}
    %cst = arith.constant dense<[[1.0, 2.0]]> : tensor<1x2xf32>
    %sum = arith.addf %cst, %cst : tensor<1x2xf32>
    // expected-error @below {{Failed to distribute load_marker past fusions}}
    %lm = rock.load_marker %sum views [#tmap_small] [%g, %m] {cacheModifier = #rock<CacheModifier none>} : tensor<1x2xf32> -> tensor<2xf32>
    %ut = rock.untile %lm : tensor<2xf32> -> tensor<1x2xf32>
    %fused = arith.addf %sm, %ut : tensor<1x2xf32>
    %r = rock.store %fused to %dest by set : tensor<1x2xf32> -> tensor<1x2xf32> to tensor<1x2xf32>
    return %r : tensor<1x2xf32>
  }
}
