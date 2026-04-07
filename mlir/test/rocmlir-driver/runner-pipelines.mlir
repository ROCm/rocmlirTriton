// RUN: rocmlir-driver -dump-pipelines -host-pipeline=backend -arch=gfx90a /dev/null -o /dev/null 2>&1 | FileCheck %s --check-prefix=RUNNER

// RUNNER: Host Backend pipeline:
// RUNNER-NEXT: {{^}}builtin.module
// RUNNER-SAME: (cpu-lower-verifier{dump-schedules-path= phase=1},
// RUNNER-SAME: one-shot-bufferize{allow-return-allocs-from-loops=false allow-unknown-ops=false analysis-fuzzer-seed=0 analysis-heuristic=bottom-up buffer-alignment=64 bufferize-function-boundaries=true check-parallel-regions=true copy-before-write=false{{ ?}}dump-alias-sets=false function-boundary-type-conversion=identity-layout-map must-infer-memory-space=false{{ ?}}print-conflicts=false test-analysis-only=false unknown-type-conversion=fully-dynamic-layout-map use-encoding-for-memory-space=false},
// RUNNER-SAME: cpu-lower-verifier{dump-schedules-path= phase=2},
// RUNNER-SAME: convert-linalg-to-loops,
// RUNNER-SAME: arith-expand{include-bf16=false include-f4e2m1=true include-f8e8m0=true},
// RUNNER-SAME: func.func(rock-convert-narrow-type-signatures),
// RUNNER-SAME: func.func(rock-emulate-narrow-types),
// RUNNER-SAME: expand-strided-metadata,
// RUNNER-SAME: lower-affine,
// RUNNER-SAME: convert-scf-to-cf{allow-pattern-rollback=true},
// RUNNER-SAME: func.func(gpu-async-region),
// RUNNER-SAME: convert-cf-to-llvm{index-bitwidth=0},
// RUNNER-SAME: convert-math-to-llvm{approximate-log1p=true},
// RUNNER-SAME: convert-math-to-libm,
// RUNNER-SAME: convert-arith-to-llvm{index-bitwidth=0},
// RUNNER-SAME: finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false},
// RUNNER-SAME: gpu-to-llvm{intersperse-sizes-for-kernels=false use-bare-pointers-for-host=false use-bare-pointers-for-kernels=true},
// RUNNER-SAME: convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},
// RUNNER-SAME: canonicalize{{\{ *}}max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,reconcile-unrealized-casts){{ ?$}}
