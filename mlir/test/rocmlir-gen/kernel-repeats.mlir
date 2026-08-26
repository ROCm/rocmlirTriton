// RUN: rocmlir-gen --arch gfx908 --operation gemm -p -ph --kernel-repeats=5 | FileCheck %s --check-prefix=GEMM

// GEMM-LABEL: @rock_gemm_gpu
// GEMM-DAG: %[[zero:.*]] = arith.constant 0 : index
// GEMM-DAG: %[[one:.*]] = arith.constant 1 : index
// GEMM-DAG: %[[five:.*]] = arith.constant 5 : index
// GEMM: scf.for %{{.*}} = %[[zero]] to %[[five]] step %[[one]] {
// GEMM: func.call @rock_gemm
// GEMM: }
