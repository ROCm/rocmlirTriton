// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 -n 128 -k 8 -m 256 --perf_config "gemm:v1:128,64,64,1,1,4,16,1,2,0,0" | FileCheck %s --check-prefix=GEN
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 -n 128 -k 8 -m 256 --perf_config "gemm:v1:128,64,64,1,1,4,16,1,2,0,0" | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=AFFIX
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 -n 128 -k 8 -m 256 --perf_config "gemm:v1:128,64,64,1,1,4,16,1,2,0,0" | rocmlir-driver -c --mlir-print-ir-after=rock-gemm-to-gridwise -o /dev/null 2>&1 | FileCheck %s --check-prefix=GRIDWISE

// GEN: rock.gemm
// CHECK-SAME: arch = "amdgcn-amd-amdhsa:gfx1030"
// CHECK-SAME: perf_config = "gemm:v1:128,64,64,1,1,4,16,1,2,0,0"
// AFFIX: #rock.gemm_params<mPerBlock = 128, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
// GRIDWISE: rock.gridwise_gemm

// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 -n 128 -k 8 -m 256 --perf_config "gemm:v1:128,64,64,1,1,4,16,2,2,0,0" | FileCheck %s --check-prefix=GEN_V2
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 -n 128 -k 8 -m 256 --perf_config "gemm:v1:128,64,64,1,1,4,16,2,2,0,0" | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=AFFIX_V2
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 -n 128 -k 8 -m 256 --perf_config "gemm:v1:128,64,64,1,1,4,16,2,2,0,0" | rocmlir-driver -c --mlir-print-ir-after=rock-gemm-to-gridwise -o /dev/null 2>&1 | FileCheck %s --check-prefix=GRIDWISE_V2

// GEN_V2: rock.gemm
// CHECK-SAME: arch = "amdgcn-amd-amdhsa:gfx1030"
// CHECK-SAME: perf_config = "gemm:v1:128,64,64,1,1,4,16,2,2,0,0"
// AFFIX_V2: #rock.gemm_params<mPerBlock = 128, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 2, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
// GRIDWISE_V2: rock.gridwise_gemm
