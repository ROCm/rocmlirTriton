// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=migraphx -arch=gfx90a /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=MIGRAPHX --match-full-lines --strict-whitespace
// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=gpu -arch=gfx90a /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=GPU --match-full-lines --strict-whitespace
// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=binary -arch=gfx90a /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=BINARY --strict-whitespace
// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=binary -arch=gfx942 /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=BINARY --strict-whitespace
// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=binary -arch=gfx950 /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=BINARY --strict-whitespace
// RUN: rocmlir-driver -dump-pipelines -kernel-pipeline=highlevel -arch=gfx90a /dev/null -o /dev/null 2>&1 | sed -e 's/,/,\n/g' | FileCheck %s --check-prefix=HIGHLEVEL --match-full-lines --strict-whitespace

// COM: Do not put a leading space between the colon and the pass you're looking for
// MIGRAPHX:Kernel MIGraphX pipeline:
// MIGRAPHX-NEXT:builtin.module(func.func(migraphx-realize-int4,
// MIGRAPHX-NEXT:migraphx-transform,
// MIGRAPHX-NEXT:canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
// MIGRAPHX-NEXT:migraphx-to-tosa,
// MIGRAPHX-NEXT:cse,
// MIGRAPHX-NEXT:migraphx-tosa-simplify))

// GPU:Kernel pipeline:
// GPU-NEXT:builtin.module(func.func(rock-affix-params),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-lower-reduce),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-regularize-output),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-regularize-inter-gemm-fusion),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-conv-to-gemm),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-fusion-splitk-regularization),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-gemm-to-gridwise),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-attn-to-gridwise),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-gridwise-attn-to-blockwise),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-allow-fast-math-flags),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-gridwise-gemm-to-blockwise),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-insert-output-fusion-loads),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-regularize-input),
// GPU-NEXT:cse,
// GPU-NEXT:func.func(rock-lower-loads),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-lower-stores),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:func.func(rock-legalize-float-types),
// GPU-NEXT:remove-dead-values{canonicalize=true},
// GPU-NEXT:rock-serialize-host-funcs,
// GPU-NEXT:arith-emulate-unsupported-floats{source-types={f4E2M1FN,
// GPU-NEXT:f8E4M3FN,
// GPU-NEXT:f8E4M3FNUZ,
// GPU-NEXT:f8E5M2FNUZ,
// GPU-NEXT:f8E5M2,
// GPU-NEXT:f8E8M0FNU} target-type=f32},
// GPU-NEXT:arith-expand{include-bf16=false include-f4e2m1=true include-f8e8m0=true},
// GPU-NEXT:func.func(rock-lower-blockwise-to-ptr,
// GPU-NEXT:rock-preserve-masked-load-semantics,
// GPU-NEXT:rock-transforms-to-pointer-arith,
// GPU-NEXT:canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
// GPU-NEXT:rock-to-ttir),
// GPU-NEXT:rock-func-to-triton-func,
// GPU-NEXT:tt.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
// GPU-NEXT:cse))

// `--kernel-pipeline=binary` is now strictly the GPU-only compile: it must
// produce `gpu.binary` (via TritonToHsaco + RockEmitGpuBinary) but must NOT
// chain in any host-side lowering.  Host lowering is opt-in via
// `--host-pipeline=backend` (covered by runner-pipelines.mlir).
// BINARY:Kernel pipeline:
// BINARY-NEXT:builtin.module(resolve-kernel-launch-params,
// BINARY-NEXT:triton-to-hsaco{allow-flush-denorm=false arch={{gfx90a|gfx942|gfx950}} enable-fp-fusion=true features= num-ctas=1 num-warps=4 opt-level=3 scalarize-packed-fops=false schedule-hint=none triple=amdgcn-amd-amdhsa waves-per-eu=0},
// BINARY-NEXT:rock-emit-gpu-binary{arch={{gfx90a|gfx942|gfx950}} features= opt-level=3 triple=amdgcn-amd-amdhsa})

// HIGHLEVEL:Kernel Highlevel pipeline:
// HIGHLEVEL-NEXT:builtin.module(rock-flatten-tosa-func-args,
// HIGHLEVEL-NEXT:func.func(tosa-to-tensor,
// HIGHLEVEL-NEXT:tosa-to-rock,
// HIGHLEVEL-NEXT:rock-view-to-transform,
// HIGHLEVEL-NEXT:rock-detect-flash-decoding,
// HIGHLEVEL-NEXT:rocmlir-custom-tosa-decompose,
// HIGHLEVEL-NEXT:rocmlir-promote-softmax-precision,
// HIGHLEVEL-NEXT:rock-tosa-to-elementwise),
// HIGHLEVEL-NEXT:func.func(tosa-optional-decompositions),
// HIGHLEVEL-NEXT:func.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}),
// HIGHLEVEL-NEXT:func.func(tosa-infer-shapes{convert-function-boundaries=false fold-shape-expressions=false}),
// HIGHLEVEL-NEXT:func.func(tosa-make-broadcastable),
// HIGHLEVEL-NEXT:func.func(tosa-to-linalg-named{prefer-conv2d-kernel-layout-hwcf=false}),
// HIGHLEVEL-NEXT:func.func(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}),
// HIGHLEVEL-NEXT:func.func(tosa-layerwise-constant-fold{aggressive-reduce-constant=false}),
// HIGHLEVEL-NEXT:func.func(tosa-make-broadcastable),
// HIGHLEVEL-NEXT:func.func(tosa-to-linalg{aggressive-reduce-constant=false disable-tosa-decompositions=false}),
// HIGHLEVEL-NEXT:func.func(tosa-to-tensor,
// HIGHLEVEL-NEXT:tosa-to-scf,
// HIGHLEVEL-NEXT:tosa-to-arith{include-apply-rescale=false use-32-bit=false},
// HIGHLEVEL-NEXT:linalg-fuse-elementwise-ops,
// HIGHLEVEL-NEXT:linalg-fold-unit-extent-dims{use-rank-reducing-slices=false},
// HIGHLEVEL-NEXT:rock-view-to-transform,
// HIGHLEVEL-NEXT:rock-fold-broadcast,
// HIGHLEVEL-NEXT:canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}),
// HIGHLEVEL-NEXT:convert-tensor-to-linalg,
// HIGHLEVEL-NEXT:func.func(rock-sort-dimensions-memory-layout),
// HIGHLEVEL-NEXT:rock-insert-output-stores)
