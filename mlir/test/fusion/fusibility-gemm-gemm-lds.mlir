// gfx950: Requires too much LDS -> not fusible.
// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:32,32,32,1,1,4,0,1,1,0,0 - < %s | FileCheck %s --check-prefix=GFX950
// GFX950: fusible:0

// Same kernel retargeted to gfx942: Requires too much LDS -> not fusible.
// RUN: sed -e 's/gfx950:sramecc+:xnack-/gfx942:sramecc+:xnack-/' %s | rocmlir-gen -emit-module-fusibility-for=attn:v1:32,32,32,1,1,4,0,1,1,0,0 - | FileCheck %s --check-prefix=GFX942
// GFX942: fusible:0

// Non-gemm-gemm perfConfig (e.g. a regular `gemm:v1:...` solution string for
// a sibling rock.gemm kernel) also falls back through `obtainTuningParameters`
// to the same default tile, so it must also reject.
// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefix=NON-GEG
// NON-GEG: fusible:0

module {
  func.func @mlir_dot_add_mul_erf_add_mul_dot(%arg0: tensor<1920000xf32>, %arg1: tensor<6553600xf32>, %arg2: tensor<7680000xf32>, %arg3: tensor<6553600xf32>, %arg4: tensor<1920000xf32>) -> tensor<1920000xf32> attributes {rock.arch = "gfx950:sramecc+:xnack-", rock.enable_splitk_for_tuning, rock.kernel = "mixr", rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
    %cst = arith.constant dense<1.000000e+00> : tensor<1x1500x5120xf32>
    %cst_0 = arith.constant dense<0.707106769> : tensor<1x1500x5120xf32>
    %a = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 1280 + d2)> by [<Unmerge{1500, 1280} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1500, 1280] -> [1920000]> : tensor<1920000xf32> to tensor<1x1500x1280xf32>
    %b = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 5120 + d2)> by [<Unmerge{1280, 5120} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1280, 5120] -> [6553600]> : tensor<6553600xf32> to tensor<1x1280x5120xf32>
    %c = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 1280 + d2)> by [<Unmerge{5120, 1280} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 5120, 1280] -> [6553600]> : tensor<6553600xf32> to tensor<1x5120x1280xf32>
    %geg = rock.gemm_elementwise_gemm{
     ab = %a * %b : tensor<1x1500x1280xf32>, tensor<1x1280x5120xf32>
     ab = elementwise otherIns(%arg2 : tensor<7680000xf32>) {
    ^bb0(%abv: tensor<1x1500x5120xf32>, %arg2v: tensor<7680000xf32>):
      %t = rock.transform %arg2v by <affine_map<(d0, d1, d2) -> (d1 * 5120 + d2)> by [<Unmerge{1500, 5120} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1500, 5120] -> [7680000]> : tensor<7680000xf32> to tensor<1x1500x5120xf32>
      %add0 = arith.addf %abv, %t : tensor<1x1500x5120xf32>
      %scale = arith.mulf %add0, %cst_0 : tensor<1x1500x5120xf32>
      %erf = math.erf %scale : tensor<1x1500x5120xf32>
      %add1 = arith.addf %erf, %cst : tensor<1x1500x5120xf32>
      %mul = arith.mulf %add0, %add1 : tensor<1x1500x5120xf32>
      rock.yield %mul : tensor<1x1500x5120xf32>
    }
     out = ab * %c : tensor<1x5120x1280xf32>
    } -> tensor<1x1500x1280xf32>
    %flat = rock.transform %geg by <affine_map<(d0) -> (0, d0 floordiv 1280, d0 mod 1280)> by [<Merge{1, 1500, 1280} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [1920000] -> [1, 1500, 1280]> : tensor<1x1500x1280xf32> to tensor<1920000xf32>
    %out = rock.store %flat to %arg4 by set : tensor<1920000xf32> -> tensor<1920000xf32> to tensor<1920000xf32>
    return %out : tensor<1920000xf32>
  }
}
