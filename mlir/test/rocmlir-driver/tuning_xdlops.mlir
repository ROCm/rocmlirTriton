// Check the naming of tuning parameters for xdlops and matrix c vectorization values

// RUN: rocmlir-gen --arch gfx908 -p | rocmlir-driver -c -mlir-print-ir-after=rock-conv-to-gemm 2>&1 >/dev/null | FileCheck %s --check-prefix=STEP1
// RUN: rocmlir-gen --arch gfx908 -p | rocmlir-driver -c -mlir-print-ir-after=rock-gemm-to-gridwise 2>&1 >/dev/null | FileCheck %s --check-prefix=STEP2

// STEP1: numWaves
// STEP1-NOT: mPerWave
// STEP1-NOT: nPerWave

// STEP2: numWaves
// STEP2-NOT: mPerWave
// STEP2-NOT: nPerWave
