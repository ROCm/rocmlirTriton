// Test that perf_config set at MIGraphX dialect level propagates through
// the full pipeline to Rock dialect with correctly parsed attributes.

// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel,gpu -arch gfx908 --mlir-print-ir-after=rock-affix-params --mlir-print-local-scope -o /dev/null %s 2>&1 | FileCheck %s

// The pass processes functions in non-deterministic order, so use CHECK-DAG.

// CHECK-DAG: func.func @elem_default({{.*}}) -> {{.*}} attributes {perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>,{{.*}}rock.block_size = 256 : i32
// CHECK-DAG: func.func @elem_custom_config({{.*}}) -> {{.*}} attributes {perf_config = #rock.elementwise_params<tileSize = 512, numCTAs = 1, numWaves = 2, numStages = 1, wavesPerEU = 0>,{{.*}}rock.block_size = 128 : i32
// CHECK-DAG: func.func @elem_custom_waves_per_eu({{.*}}) -> {{.*}} attributes {perf_config = #rock.elementwise_params<tileSize = 128, numCTAs = 1, numWaves = 8, numStages = 1, wavesPerEU = 2>,{{.*}}rock.block_size = 512 : i32
// CHECK-DAG: func.func @elem_f16_default({{.*}}) -> {{.*}} attributes {perf_config = #rock.elementwise_params<tileSize = 256, numCTAs = 1, numWaves = 4, numStages = 1, wavesPerEU = 0>,{{.*}}rock.block_size = 256 : i32

module {
  func.func @elem_default(%arg0: !migraphx.shaped<16x64xf32, 64x1>, %arg1: !migraphx.shaped<16x64xf32, 64x1>) -> !migraphx.shaped<16x64xf32, 64x1> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
    %0 = migraphx.add %arg0, %arg1 {} : <16x64xf32, 64x1>, <16x64xf32, 64x1> -> <16x64xf32, 64x1>
    return %0 : !migraphx.shaped<16x64xf32, 64x1>
  }

  func.func @elem_custom_config(%arg0: !migraphx.shaped<32x64xf32, 64x1>, %arg1: !migraphx.shaped<32x64xf32, 64x1>) -> !migraphx.shaped<32x64xf32, 64x1> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", perf_config = "elem:v1:512,1,2,1,0"} {
    %0 = migraphx.add %arg0, %arg1 {} : <32x64xf32, 64x1>, <32x64xf32, 64x1> -> <32x64xf32, 64x1>
    return %0 : !migraphx.shaped<32x64xf32, 64x1>
  }

  func.func @elem_custom_waves_per_eu(%arg0: !migraphx.shaped<64x64xf32, 64x1>, %arg1: !migraphx.shaped<64x64xf32, 64x1>) -> !migraphx.shaped<64x64xf32, 64x1> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", perf_config = "elem:v1:128,1,8,1,2"} {
    %0 = migraphx.add %arg0, %arg1 {} : <64x64xf32, 64x1>, <64x64xf32, 64x1> -> <64x64xf32, 64x1>
    return %0 : !migraphx.shaped<64x64xf32, 64x1>
  }

  func.func @elem_f16_default(%arg0: !migraphx.shaped<32x64xf16, 64x1>, %arg1: !migraphx.shaped<32x64xf16, 64x1>) -> !migraphx.shaped<32x64xf16, 64x1> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
    %0 = migraphx.add %arg0, %arg1 {} : <32x64xf16, 64x1>, <32x64xf16, 64x1> -> <32x64xf16, 64x1>
    return %0 : !migraphx.shaped<32x64xf16, 64x1>
  }
}
