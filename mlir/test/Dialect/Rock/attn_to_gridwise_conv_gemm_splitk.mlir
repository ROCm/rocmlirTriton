// Split-k applied to a conv+gemm whose two GEMMs have an elementwise fusion
// between them. Driven through the real pipelines rather than a hand-written
// pass list, so the ordering the compiler actually uses is what gets checked:
// the body's view of the elementwise input is externalized, the convolution
// becomes a GEMM, and only then is the split applied.

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-driver -kernel-pipeline=gpu -mlir-print-ir-after=rock-attn-to-gridwise -mlir-print-local-scope -o /dev/null 2>&1 | FileCheck %s

// The convolution's output space is (G=1, M=128 image positions, N=16 output
// channels), and N is what the second GEMM reduces over, so the split folds a
// 4-way slice of the channels into G.
// CHECK-LABEL: @conv_gemm_splitk_intergemm
// CHECK: rock.transform %{{.*}} by <{{.*}}<Unmerge{4, 4} ["gemmNSplit", "gemmN"] at [1, 2] -> ["gemmN"] at [1]>{{.*}}> : tensor<1x16x32xf32> to tensor<1x4x4x32xf32>

// The inter-gemm elementwise input lives in that same output space, so it is
// split the same way and ends up indexed by the split-carrying G.
// CHECK: %[[EW_SPLIT:.*]] = rock.transform %{{.*}} by <{{.*}}<Unmerge{4, 4} ["gemmNSplit", "gemmN"] at [1, 2] -> ["gemmN"] at [2]>{{.*}}> : tensor<1x128x16xf32> to tensor<1x4x4x128xf32>
// CHECK: %[[EW:.*]] = rock.transform %[[EW_SPLIT]] by <{{.*}}<Merge{1, 4} ["gemmG"] at [0] -> ["gemmG", "gemmNSplit"] at [0, 1]>{{.*}}> : tensor<1x4x4x128xf32> to tensor<4x128x4xf32>

// The body is retyped to the per-split shape and passed the split input.
// CHECK: rock.gridwise_attention({{.*}}%[[EW]]) preSoftmaxOps = {
// CHECK-NEXT: ^bb0(%[[AB:.*]]: tensor<4x128x4xf32>, %[[OTHER:.*]]: tensor<4x128x4xf32>):
// CHECK-NEXT: arith.mulf %[[AB]], %[[OTHER]] : tensor<4x128x4xf32>

// splitKFactor rides on the second GEMM's params, the pre-pad extent of the
// split dimension is recorded for the masking, and the partials are summed.
// CHECK: params1 = #rock.gemm_params<{{.*}}splitKFactor = 4
// CHECK-SAME: prePadG0N = 4 : index
// CHECK: rock.store %{{.*}} by atomic_add
module {
  func.func @conv_gemm_splitk_intergemm(%arg0: !migraphx.shaped<2x8x8x16xf32, 1024x8x1x64>, %arg1: !migraphx.shaped<16x16x3x3xf32, 144x1x48x16>, %arg2: !migraphx.shaped<1x16x32xf32, 0x1x0>, %arg3: !migraphx.shaped<1x128x16xf32, 2048x16x1>) -> !migraphx.shaped<1x128x32xf32, 4096x32x1> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
    %transposed = migraphx.transpose %arg0 {permutation = [0, 3, 1, 2]} : <2x8x8x16xf32, 1024x8x1x64> -> <2x16x8x8xf32, 1024x64x8x1>
    %1 = migraphx.convolution %transposed, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <2x16x8x8xf32, 1024x64x8x1>, <16x16x3x3xf32, 144x1x48x16> -> <2x16x8x8xf32, 2048x1x128x16>
    %2 = migraphx.transpose %1 {permutation = [0, 2, 3, 1]} : <2x16x8x8xf32, 2048x1x128x16> -> <2x8x8x16xf32, 2048x128x16x1>
    %3 = migraphx.reshape %2 {dims = [1, 128, 16]} : <2x8x8x16xf32, 2048x128x16x1> -> <1x128x16xf32, 2048x16x1>
    %scaled = migraphx.mul %3, %arg3 : <1x128x16xf32, 2048x16x1>, <1x128x16xf32, 2048x16x1> -> <1x128x16xf32, 2048x16x1>
    %4 = migraphx.dot %scaled, %arg2 {perf_config="attn:mPerBlockG0=128,nPerBlockG0=64,kPerBlock=32,numWaves=4,matrixInstrNonkdim=0,splitKFactor=4"} : <1x128x16xf32, 2048x16x1>, <1x16x32xf32, 0x1x0> -> <1x128x32xf32, 4096x32x1>
    return %4 : !migraphx.shaped<1x128x32xf32, 4096x32x1>
  }
}
