// RUN: rocmlir-opt -rock-regularize-output -mlir-print-local-scope %s | FileCheck %s

// Output fusions behind dynamic transforms are moved into GEMM space by
// applying the inverse transforms, whose sizes are expressions over the same
// argument dimensions.

// The GEMM result is flattened with a dynamic Merge, whose inverse is an
// Unmerge with the same sizes.
// CHECK-LABEL: func.func @dynamic_merge
// CHECK-SAME: (%{{.*}}: tensor<1x?x64xf16>, %{{.*}}: tensor<1x64x?xf16>, %[[BIAS:arg[0-9]+]]: tensor<1x?xf16>, %[[DEST:arg[0-9]+]]: tensor<1x?xf16>)
//      CHECK:   %[[G:.*]] = rock.gemm
//      CHECK:   %[[BIAS3D:.*]] = rock.transform %[[BIAS]] by <affine_map<(d0, d1, d2)[s0] -> (d0, d1 * s0 + d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Unmerge{s0, s1} ["m", "n"] at [1, 2] -> ["mn"] at [1]>] symbols = [arg(1, 2), arg(0, 1)] bounds = [1, s1, s0] -> [1, s1 * s0]>
//      CHECK:   %[[F:.*]] = arith.addf %[[G]], %[[BIAS3D]] : tensor<1x?x?xf16>
//      CHECK:   %[[DEST3D:.*]] = rock.transform %[[DEST]] by
//      CHECK:   rock.store %[[F]] to %[[DEST3D]] {{.*}}: tensor<1x?x?xf16>
func.func @dynamic_merge(%arg0: tensor<1x?x64xf16>, %arg1: tensor<1x64x?xf16>, %arg2: tensor<1x?xf16>, %arg3: tensor<1x?xf16>) -> tensor<1x?xf16> attributes {rock.kernel} {
  %g = rock.gemm %arg0 * %arg1 : tensor<1x?x64xf16> * tensor<1x64x?xf16> -> tensor<1x?x?xf16>
  %flat = rock.transform %g by <affine_map<(d0, d1)[s0, s1] -> (d0, d1 floordiv s1, d1 mod s1)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Merge{s0, s1} ["mn"] at [1] -> ["m", "n"] at [1, 2]>] symbols = [arg(0, 1), arg(1, 2)] bounds = [1, s0 * s1] -> [1, s0, s1]> : tensor<1x?x?xf16> to tensor<1x?xf16>
  %fused = arith.addf %flat, %arg2 : tensor<1x?xf16>
  %r = rock.store %fused to %arg3 by set : tensor<1x?xf16> -> tensor<1x?xf16> to tensor<1x?xf16>
  return %r : tensor<1x?xf16>
}

// -----

// The GEMM result is padded to a multiple of 64 along N; the inverse is a
// Slice of the unpadded range.
// CHECK-LABEL: func.func @dynamic_pad
//      CHECK:   %[[G:.*]] = rock.gemm
//      CHECK:   rock.transform %{{.*}} by <affine_map<(d0, d1, d2)[s0] -> (d0, d1, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Slice{0, s0} ["n"] at [2] -> ["nPad"] at [2]>] symbols = [arg(1, 2), arg(0, 1)] bounds = [1, s1, s0] -> [1, s1, (s0 ceildiv 64) * 64]>
//      CHECK:   arith.addf %[[G]], %{{.*}} : tensor<1x?x?xf16>
func.func @dynamic_pad(%arg0: tensor<1x?x64xf16>, %arg1: tensor<1x64x?xf16>, %arg2: tensor<1x?x?xf16>, %arg3: tensor<1x?x?xf16>) -> tensor<1x?x?xf16> attributes {rock.kernel} {
  %g = rock.gemm %arg0 * %arg1 : tensor<1x?x64xf16> * tensor<1x64x?xf16> -> tensor<1x?x?xf16>
  %padded = rock.transform %g by <affine_map<(d0, d1, d2)[s0, s1] -> (d0, d1, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Pad{0, -s1 + (s1 ceildiv 64) * 64} ["nPad"] at [2] -> ["n"] at [2]>] symbols = [arg(0, 1), arg(1, 2)] bounds = [1, s0, (s1 ceildiv 64) * 64] -> [1, s0, s1]> : tensor<1x?x?xf16> to tensor<1x?x?xf16>
  %fused = arith.addf %padded, %arg2 : tensor<1x?x?xf16>
  %r = rock.store %fused to %arg3 by set : tensor<1x?x?xf16> -> tensor<1x?x?xf16> to tensor<1x?x?xf16>
  return %r : tensor<1x?x?xf16>
}
