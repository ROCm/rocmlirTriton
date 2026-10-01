// A GEMM with dynamic M and N whose output is flattened by a symbolic Merge
// before a fused bias add. rock-regularize-output turns the Merge into an
// Unmerge on the bias and the destination, so the kernel must index them as
// m * N + n with N read from a dimension argument. Running the kernel for a
// 200x100 output must reproduce, bit for bit, the static twin in Inputs/.
//
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen -ph -print-results -rand 1 -rand_type float --dynamic-arg-shapes=1x200x64,1x64x100,1x20000,1x20000 - | rocmlir-driver -c -arch %arch | rocm-run | sed 's/base@ = 0x[0-9a-f]*//' > %t.dynamic
// RUN: sed s/##TOKEN_ARCH##/%arch/g %S/Inputs/dynamic-gemm-merge-fusion-static.mlir | rocmlir-gen -ph -print-results -rand 1 -rand_type float - | rocmlir-driver -c -arch %arch | rocm-run | sed 's/base@ = 0x[0-9a-f]*//' > %t.static
// RUN: FileCheck %s < %t.dynamic
// RUN: diff %t.dynamic %t.static

// CHECK: Unranked Memref rank = 1 offset = 0 sizes = [20000] strides = [1] data =

func.func @dynamic_merge(%arg0: tensor<1x?x64xf16>, %arg1: tensor<1x64x?xf16>, %arg2: tensor<1x?xf16>, %arg3: tensor<1x?xf16>) -> tensor<1x?xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %g = rock.gemm %arg0 * %arg1 : tensor<1x?x64xf16> * tensor<1x64x?xf16> -> tensor<1x?x?xf16>
  %flat = rock.transform %g by <affine_map<(d0, d1)[s0, s1] -> (d0, d1 floordiv s1, d1 mod s1)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Merge{s0, s1} ["mn"] at [1] -> ["m", "n"] at [1, 2]>] symbols = [arg(0, 1), arg(1, 2)] bounds = [1, s0 * s1] -> [1, s0, s1]> : tensor<1x?x?xf16> to tensor<1x?xf16>
  %fused = arith.addf %flat, %arg2 : tensor<1x?xf16>
  %r = rock.store %fused to %arg3 by set : tensor<1x?xf16> -> tensor<1x?xf16> to tensor<1x?xf16>
  return %r : tensor<1x?xf16>
}
