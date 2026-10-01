// Static reference for ../dynamic-gemm-pad-fusion.e2e.mlir at M = 200,
// N = 100, padded N = 128. Static kernel arguments are flat, so the logical
// views are rebuilt with Unmerge and Merge; the argument order and sizes match
// the dynamic harness.

func.func @static_pad(%a: tensor<12800xf16>, %b: tensor<6400xf16>, %c: tensor<25600xf16>, %d: tensor<25600xf16>) -> tensor<25600xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %arg0 = rock.transform %a by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{200, 64} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 200, 64] -> [12800]> : tensor<12800xf16> to tensor<1x200x64xf16>
  %arg1 = rock.transform %b by <affine_map<(d0, d1, d2) -> (d1 * 100 + d2)> by [<Unmerge{64, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 100] -> [6400]> : tensor<6400xf16> to tensor<1x64x100xf16>
  %arg2 = rock.transform %c by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{200, 128} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 200, 128] -> [25600]> : tensor<25600xf16> to tensor<1x200x128xf16>
  %g = rock.gemm %arg0 * %arg1 : tensor<1x200x64xf16> * tensor<1x64x100xf16> -> tensor<1x200x100xf16>
  %padded = rock.transform %g by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Pad{0, 28} ["nPad"] at [2] -> ["n"] at [2]>] bounds = [1, 200, 128] -> [1, 200, 100]> : tensor<1x200x100xf16> to tensor<1x200x128xf16>
  %fused = arith.addf %padded, %arg2 : tensor<1x200x128xf16>
  %flat = rock.transform %fused by <affine_map<(d0) -> (0, d0 floordiv 128, d0 mod 128)> by [<Merge{200, 128} ["raw"] at [0] -> ["m", "n"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [25600] -> [1, 200, 128]> : tensor<1x200x128xf16> to tensor<25600xf16>
  %r = rock.store %flat to %d by set : tensor<25600xf16> -> tensor<25600xf16> to tensor<25600xf16>
  return %r : tensor<25600xf16>
}
