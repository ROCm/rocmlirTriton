// Architectures whose QuickTuning list has only a single (degenerate) config
// must fall back to the closest richer relative's list instead of using their
// lone entry. See ParamLookupTable::lookup / getRelatives.
//
// gfx1201 (single-config) -> gfx1200's list
// gfx1150 (single-config) -> gfx1100's list
//
// We assert this end-to-end: without an explicit perf_config, the affix pass
// materializes the first conservatively-applicable config from the fallback
// list. The resulting params for the degenerate arch must match those of its
// richer relative (and differ from the lone "64,64,64" entry).

// RUN: rocmlir-gen --operation gemm -t f16 --arch gfx1201 -g 1 -m 256 -k 128 -n 256 -transA=False -transB=False -p | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=GFX1201
// RUN: rocmlir-gen --operation gemm -t f16 --arch gfx1200 -g 1 -m 256 -k 128 -n 256 -transA=False -transB=False -p | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=GFX1200

// gfx1201 falls back to gfx1200, so both pick the same first applicable config.
// GFX1201: #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
// GFX1200: #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// RUN: rocmlir-gen --operation gemm -t f16 --arch gfx1150 -g 1 -m 256 -k 128 -n 256 -transA=False -transB=False -p | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=GFX1150
// RUN: rocmlir-gen --operation gemm -t f16 --arch gfx1100 -g 1 -m 256 -k 128 -n 256 -transA=False -transB=False -p | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=GFX1100

// gfx1150 falls back to gfx1100, so both pick the same first applicable config.
// GFX1150: #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 256, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
// GFX1100: #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 256, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// RUN: rocmlir-gen --operation gemm -t f16 --arch gfx908 -g 1 -m 256 -k 128 -n 256 -transA=False -transB=False -p | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=GFX908F16
// RUN: rocmlir-gen --operation gemm -t f16 --arch gfx90a -g 1 -m 256 -k 128 -n 256 -transA=False -transB=False -p | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=GFX90AF16

// gfx908 GEMM falls back to its closest gfx9 relative, gfx90a.
// GFX908F16: #rock.gemm_params<mPerBlock = 32, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 3, wavesPerEU = 0, gridGroupSize = 0>
// GFX90AF16: #rock.gemm_params<mPerBlock = 32, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 3, wavesPerEU = 0, gridGroupSize = 0>
