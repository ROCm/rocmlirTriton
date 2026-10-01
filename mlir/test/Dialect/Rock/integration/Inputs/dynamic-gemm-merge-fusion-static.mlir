// Static reference for ../dynamic-gemm-merge-fusion.e2e.mlir at M = 200,
// N = 100. Static kernel arguments are flat, so the logical views are rebuilt
// with Unmerge; the argument order and sizes match the dynamic harness.

func.func @static_merge(%a: tensor<12800xf16>, %b: tensor<6400xf16>, %arg2: tensor<20000xf16>, %arg3: tensor<20000xf16>) -> tensor<20000xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %arg0 = rock.transform %a by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{200, 64} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 200, 64] -> [12800]> : tensor<12800xf16> to tensor<1x200x64xf16>
  %arg1 = rock.transform %b by <affine_map<(d0, d1, d2) -> (d1 * 100 + d2)> by [<Unmerge{64, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 100] -> [6400]> : tensor<6400xf16> to tensor<1x64x100xf16>
  %g = rock.gemm %arg0 * %arg1 : tensor<1x200x64xf16> * tensor<1x64x100xf16> -> tensor<1x200x100xf16>
  %flat = rock.transform %g by <affine_map<(d0) -> (0, d0 floordiv 100, d0 mod 100)> by [<Merge{200, 100} ["raw"] at [0] -> ["m", "n"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [20000] -> [1, 200, 100]> : tensor<1x200x100xf16> to tensor<20000xf16>
  %fused = arith.addf %flat, %arg2 : tensor<20000xf16>
  %r = rock.store %fused to %arg3 by set : tensor<20000xf16> -> tensor<20000xf16> to tensor<20000xf16>
  return %r : tensor<20000xf16>
}
