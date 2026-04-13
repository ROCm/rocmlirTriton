// Verify that rocmlir-driver accepts standard MLIR pass manager and timing
// command-line options. These are registered via registerMLIRCLOptions() and
// applied in every PassManager creation path; this test guards against
// silently losing support for them.

// RUN: rocmlir-driver --mlir-timing -dump-pipelines -kernel-pipeline=gpu -arch=gfx90a /dev/null -o /dev/null 2>&1 | FileCheck %s
// RUN: rocmlir-driver --mlir-pass-statistics -dump-pipelines -kernel-pipeline=gpu -arch=gfx90a /dev/null -o /dev/null 2>&1 | FileCheck %s
// RUN: rocmlir-driver --mlir-print-ir-before-all -dump-pipelines -kernel-pipeline=gpu -arch=gfx90a /dev/null -o /dev/null 2>&1 | FileCheck %s

// CHECK: Kernel pipeline:
