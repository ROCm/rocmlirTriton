// Check the naming of tuning parameters for xdlops and matrix c vectorization values

// RUN: rocmlir-gen --arch gfx908 -p | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s --check-prefix=STEP1
// RUN: rocmlir-gen --arch gfx908 -p | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gridwise-gemm-to-blockwise | FileCheck %s --check-prefix=STEP2

// STEP1: numWaves
// STEP1-NOT: mPerWave
// STEP1-NOT: nPerWave

// STEP2: numWaves
// STEP2-NOT: mPerWave
// STEP2-NOT: nPerWave
