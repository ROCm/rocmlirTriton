// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A redundant f32->f16->f32 cast must disappear before rock-to-ttir, including
// when the truncation has an explicit rounding mode. Downward and upward have
// no Triton lowering, and toward_zero is rewritten to tt.fp_to_fp, which the
// round-trip fold cannot match. Running the fast-math pass only after
// rock-to-ttir therefore rejects this kernel.

// RUN: rocmlir-driver -arch=gfx942 -kernel-pipeline=gpu %s | FileCheck %s

#flat_to_gemm = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
  by [<Unmerge{64, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>,
      <AddDim{1} ["unit0"] at [0] -> [] at []>]
  bounds = [1, 64, 64] -> [4096]>
#gemm_to_flat = #rock.transform_map<affine_map<(d0) -> (0, d0 floordiv 64, d0 mod 64)>
  by [<Merge{1, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>]
  bounds = [4096] -> [1, 64, 64]>

module attributes {rock.arch = "gfx942"} {
  // CHECK-LABEL: tt.func @rounded_cast_roundtrip
  // CHECK-NOT: arith.truncf
  // CHECK-NOT: arith.extf
  // CHECK-NOT: tt.fp_to_fp
  // CHECK: tt.return
  func.func @rounded_cast_roundtrip(%a: tensor<4096xf32>,
                                    %b: tensor<4096xf32>,
                                    %out: tensor<4096xf32>)
      -> tensor<4096xf32> attributes {rock.kernel} {
    %ta = rock.transform %a by #flat_to_gemm
        : tensor<4096xf32> to tensor<1x64x64xf32>
    %tb = rock.transform %b by #flat_to_gemm
        : tensor<4096xf32> to tensor<1x64x64xf32>
    %g = rock.gemm %ta * %tb
        : tensor<1x64x64xf32> * tensor<1x64x64xf32> -> tensor<1x64x64xf32>
    %down = arith.truncf %g downward
        : tensor<1x64x64xf32> to tensor<1x64x64xf16>
    %down_wide = arith.extf %down
        : tensor<1x64x64xf16> to tensor<1x64x64xf32>
    %up = arith.truncf %down_wide upward
        : tensor<1x64x64xf32> to tensor<1x64x64xf16>
    %up_wide = arith.extf %up
        : tensor<1x64x64xf16> to tensor<1x64x64xf32>
    %rtz = arith.truncf %up_wide toward_zero
        : tensor<1x64x64xf32> to tensor<1x64x64xf16>
    %rtz_wide = arith.extf %rtz
        : tensor<1x64x64xf16> to tensor<1x64x64xf32>
    %flat = rock.transform %rtz_wide by #gemm_to_flat
        : tensor<1x64x64xf32> to tensor<4096xf32>
    %stored = rock.store %flat to %out by set
        : tensor<4096xf32> -> tensor<4096xf32> to tensor<4096xf32>
    return %stored : tensor<4096xf32>
  }
}
