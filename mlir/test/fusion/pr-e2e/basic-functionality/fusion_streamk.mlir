// Fused GEMM test (migraphx IR): C_f32 = extf((A_f16 + input_f16) * (B_f16 + input_f16) + ofusion_f16)
//
// Same fused GEMM as fusion_splitk.mlir but pinning a stream-K perf_config on
// the migraphx.dot. The perf_config attribute flows through MIGraphXToTosa ->
// TosaToRock onto the rock.gemm (and the kernel result gets rock.prefill = 0,
// since the stream-K remainder atomic_adds into a zero-prefilled output). Where
// the grid leaves a ragged tail for the target's num_cu, rock-stream-k-decompose
// splits into data-parallel waves plus a split-K remainder; otherwise it falls
// back to plain data-parallel. Either way the clone verifier checks the fused
// f32 epilogue against the CPU reference.

// RUN: rocmlir-gen -fut rock_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rock_gemm --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
module {
  func.func @rock_gemm(%arg0: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %arg1: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %inputfusion: !migraphx.shaped<1x100x100xf16, 10000x100x1>,
                       %ofusion: !migraphx.shaped<1x100x100xf16, 10000x100x1>)
      -> !migraphx.shaped<1x100x100xf32, 10000x100x1>
      attributes {rock.kernel} {
    %a = migraphx.add %arg0, %inputfusion : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %b = migraphx.add %arg1, %inputfusion : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %gemm = migraphx.dot %a, %b {perf_config = "gemm:v5:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1,-1,-1"} : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %fused = migraphx.add %gemm, %ofusion : <1x100x100xf16, 10000x100x1>, <1x100x100xf16, 10000x100x1> -> <1x100x100xf16, 10000x100x1>
    %result = migraphx.convert %fused : <1x100x100xf16, 10000x100x1> to <1x100x100xf32, 10000x100x1>
    return %result : !migraphx.shaped<1x100x100xf32, 10000x100x1>
  }
}
