// RUN: rocmlir-opt -rock-shuffle-gemm-for-reductions -mlir-print-local-scope %s | FileCheck %s
// TODO(rocmlirTriton): Implement this test again if needed
