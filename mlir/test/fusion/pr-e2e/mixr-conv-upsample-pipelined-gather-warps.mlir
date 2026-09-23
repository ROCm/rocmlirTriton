// RUN: rocmlir-gen -fut mlir_upsample_conv --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_upsample_conv --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// A 2x nearest upsample feeding a padded 3x3 convolution, compiled with a
// software-pipelined (numStages = 2) perf_config. The input gather's gemmK
// index is decomposed into (c, y, x) with divisions by 9 and 3 inside the main
// loop, so rock-set-gather-warps moves every warp of that load onto K, for
// both the prologue and the in-loop loads staged into the same shared-memory
// buffer.

module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_upsample_conv(%in: !migraphx.shaped<1x64x16x16xf16, 16384x256x16x1>,
                                %fil: !migraphx.shaped<256x64x3x3xf16, 576x9x3x1>)
      -> !migraphx.shaped<1x256x32x32xf16, 262144x1024x32x1> attributes {rock.kernel} {
    %0 = migraphx.reshape %in {dims = [1, 64, 16, 1, 16, 1]} : <1x64x16x16xf16, 16384x256x16x1> -> <1x64x16x1x16x1xf16, 16384x256x16x16x1x1>
    %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 64, 16, 2, 16, 2]} : <1x64x16x1x16x1xf16, 16384x256x16x16x1x1> -> <1x64x16x2x16x2xf16, 16384x256x16x0x1x0>
    %2 = migraphx.reshape %1 {dims = [1, 64, 32, 32]} : <1x64x16x2x16x2xf16, 16384x256x16x0x1x0> -> <1x64x32x32xf16, 65536x1024x32x1>
    %3 = migraphx.convolution %2, %fil {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1], perf_config = "gemm:mPerBlock=256,nPerBlock=128,kPerBlock=32,kpack=1,numCTAs=1,numWaves=8,matrixInstrNonkdim=0,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1"} : <1x64x32x32xf16, 65536x1024x32x1>, <256x64x3x3xf16, 576x9x3x1> -> <1x256x32x32xf16, 262144x1024x32x1>
    return %3 : !migraphx.shaped<1x256x32x32xf16, 262144x1024x32x1>
  }
}
