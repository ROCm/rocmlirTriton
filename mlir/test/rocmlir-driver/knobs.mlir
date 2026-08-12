// Verify that Triton knobs encoded in a perfConfig string flow through
// `fillCompilationConfigs` and gate the relevant pass options in the
// Triton pipeline. The original five knobs
// (`useAsyncCopy`, `useBlockPingpong`, `useInThreadTranspose`,
//  `useBufferOps`, `useBufferAtomics`) are encoded as the trailing 5
// fields of the `gemm:v3:` perfConfig string (see
// `mlir/Dialect/Rock/IR/RockAttrDefs.td`). Knob values:
//   -1 -> per-arch / heuristic default
//    0 -> force off
//    1 -> force on (for the tri-state knobs)
//
// The tunable prefix `64,64,64,1,1,4,16,1,2,0,0` is a representative
// shape that the pipeline accepts on every arch we test below; only the
// trailing knob block changes between RUNs.

//===----------------------------------------------------------------------===//
// useAsyncCopy
//===----------------------------------------------------------------------===//

// gfx950 default: async-copy on by per-arch default.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ASYNC_GFX950_DEFAULT

// gfx950 with useAsyncCopy=0 (force off).
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,0,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ASYNC_GFX950_OFF

// gfx942 default: async-copy off by per-arch default.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ASYNC_GFX942_DEFAULT

// gfx942 with useAsyncCopy=1 (force on).
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ASYNC_GFX942_ON

// ASYNC_GFX950_DEFAULT: tritonamdgpu-pipeline{use_async_copy=true
// ASYNC_GFX950_DEFAULT: tritonamdgpu-coalesce-async-copy{gfx-arch=gfx950}

// ASYNC_GFX950_OFF: tritonamdgpu-pipeline{use_async_copy=false
// ASYNC_GFX950_OFF-NOT: tritonamdgpu-coalesce-async-copy

// ASYNC_GFX942_DEFAULT: tritonamdgpu-pipeline{use_async_copy=false
// ASYNC_GFX942_DEFAULT-NOT: tritonamdgpu-coalesce-async-copy

// ASYNC_GFX942_ON: tritonamdgpu-pipeline{use_async_copy=true
// ASYNC_GFX942_ON: tritonamdgpu-coalesce-async-copy{gfx-arch=gfx942}

//===----------------------------------------------------------------------===//
// useBlockPingpong
//===----------------------------------------------------------------------===//

// gfx942 default: pingpong on by per-arch default (numStages=2 -> pass is
// scheduled).
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=PP_GFX942_DEFAULT

// gfx942 with useBlockPingpong=0: pipeline pass sees use_pingpong=false and
// the block-pingpong pass is absent.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,0,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=PP_GFX942_OFF

// gfx1250 default: async-copy on, but pingpong is off by per-arch default
// (only gfx942 and gfx950+async-copy enable it).
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=PP_GFX1250_DEFAULT

// gfx1250 with useBlockPingpong=1: force on.
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=PP_GFX1250_ON

// PP_GFX942_DEFAULT: tritonamdgpu-pipeline{{.*}}use_pingpong=true
// PP_GFX942_DEFAULT: tritonamdgpu-block-pingpong

// PP_GFX942_OFF: tritonamdgpu-pipeline{{.*}}use_pingpong=false
// PP_GFX942_OFF-NOT: tritonamdgpu-block-pingpong

// PP_GFX1250_DEFAULT: tritonamdgpu-pipeline{{.*}}use_pingpong=false
// PP_GFX1250_DEFAULT-NOT: tritonamdgpu-block-pingpong

// PP_GFX1250_ON: tritonamdgpu-pipeline{{.*}}use_pingpong=true
// PP_GFX1250_ON: tritonamdgpu-block-pingpong

//===----------------------------------------------------------------------===//
// useInThreadTranspose
//===----------------------------------------------------------------------===//

// gfx942 default: in-thread-transpose pass is scheduled.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX942_DEFAULT

// gfx942 with useInThreadTranspose=0: pass absent.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,0,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX942_OFF

// gfx950 default: in-thread-transpose pass is absent.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX950_DEFAULT

// gfx950 with useInThreadTranspose=1: force on.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX950_ON

// gfx1170 default: in-thread-transpose pass is scheduled. gfx11.7 is in the
// per-arch default set (gfx942 / gfx110 / gfx115 / gfx117 / gfx120), matching
// compiler.py's is_in_thread_transpose_enabled.
// RUN: rocmlir-gen --arch gfx1170 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX1170_DEFAULT

// gfx1170 with useInThreadTranspose=0: override beats the per-arch default, so
// the pass is absent.
// RUN: rocmlir-gen --arch gfx1170 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,0,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX1170_OFF

// ITT_GFX942_DEFAULT: tritonamdgpu-in-thread-transpose
// ITT_GFX942_OFF-NOT: tritonamdgpu-in-thread-transpose
// ITT_GFX950_DEFAULT-NOT: tritonamdgpu-in-thread-transpose
// ITT_GFX950_ON: tritonamdgpu-in-thread-transpose
// ITT_GFX1170_DEFAULT: tritonamdgpu-in-thread-transpose
// ITT_GFX1170_OFF-NOT: tritonamdgpu-in-thread-transpose

//===----------------------------------------------------------------------===//
// useBufferOps / useBufferAtomics
//===----------------------------------------------------------------------===//
//
// `bufferOpsAnalyzeSmallTensorRange` is a debug-only `TritonOptions`
// field (not in perfConfig), so it's not exercised here. It's reached
// via `rocmlir-opt --pass-pipeline=...` and `rocmlir-driver` has no
// override path for it.

// Default: all three buffer-ops passes are scheduled, atomics on, small-tensor
// range analysis off (matches the historical hardcoded values).
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=BUF_DEFAULT

// useBufferOps=0: all three buffer-ops passes are skipped.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,0,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=BUF_OFF

// useBufferAtomics=0: convert-to-buffer-ops sees allow-buffer-atomics=false.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,0 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=BUF_NOATOMICS

// BUF_DEFAULT: tritonamdgpu-canonicalize-pointers
// BUF_DEFAULT: tritonamdgpu-convert-buffer-ops{allow-buffer-atomics=true analyze-small-tensor-ofst=false
// BUF_DEFAULT: tritonamdgpu-optimize-buffer-op-ptr

// BUF_OFF-NOT: tritonamdgpu-canonicalize-pointers
// BUF_OFF-NOT: tritonamdgpu-convert-buffer-ops
// BUF_OFF-NOT: tritonamdgpu-optimize-buffer-op-ptr

// BUF_NOATOMICS: tritonamdgpu-convert-buffer-ops{allow-buffer-atomics=false analyze-small-tensor-ofst=false

//===----------------------------------------------------------------------===//
// v2 perfConfig back-compat
//===----------------------------------------------------------------------===//
//
// A legacy `gemm:v2:` perfConfig carried a trailing `scheduleHint` field. It
// is still accepted read-only: the five bool knobs are honored and the
// trailing token is discarded. Here useAsyncCopy=0 (field 12) must still force
// async-copy off on gfx950 even though a stray scheduleHint=2 trails.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,0,-1,-1,-1,-1,2 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=V2_BACKCOMPAT

// V2_BACKCOMPAT: tritonamdgpu-pipeline{use_async_copy=false

//===----------------------------------------------------------------------===//
// useReductionLayout
//===----------------------------------------------------------------------===//
//
// `useReductionLayout` was introduced in the v4 perfConfig and is a tri-state
// like the other knobs: -1 (the default / heuristic), 0 (off), or 1 (on). The
// `rock-set-reduction-layout` pass is always scheduled and the knob is threaded
// to it as `use-reduction-layout`, which controls what the pass rewrites:
// -1 rewrites only convolution kernels (`rock.conv_kernel`), 0 disables the
// rewrite entirely, and 1 forces it on every kernel. So the knob controls the
// pass option, not whether the pass is present.

// Default (v3 string, knob absent -> defaults to -1): rewrite conv kernels only.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=RL_DEFAULT

// v5 with useReductionLayout=-1 (heuristic default): rewrite conv kernels only.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v5:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=RL_HEURISTIC

// v5 with useReductionLayout=0 (explicit off): disable the rewrite entirely.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v5:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,0,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=RL_OFF

// v5 with useReductionLayout=1 (force on): force option on.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v5:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=RL_ON

// RL_DEFAULT: rock-set-reduction-layout{use-reduction-layout=-1}
// RL_HEURISTIC: rock-set-reduction-layout{use-reduction-layout=-1}
// RL_OFF: rock-set-reduction-layout{use-reduction-layout=0}
// RL_ON: rock-set-reduction-layout{use-reduction-layout=1}

//===----------------------------------------------------------------------===//
// useOptimizeEpilogue
//===----------------------------------------------------------------------===//
//
// `useOptimizeEpilogue` gates Triton's `tritonamdgpu-optimize-epilogue` pass.
// The automatic policy (-1) and explicit on (1) schedule the pass; explicit off
// (0) omits it from makeTTGIR.

// v5 with useOptimizeEpilogue=-1 (automatic policy): pass is scheduled.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v5:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=OE_DEFAULT

// v5 with useOptimizeEpilogue=0 (explicit off): pass is absent.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v5:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,0 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=OE_OFF

// v5 with useOptimizeEpilogue=1 (explicit on): pass is scheduled.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v5:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=OE_ON

// OE_DEFAULT: rock-set-matmul-output-transpose
// OE_DEFAULT: tritonamdgpu-optimize-epilogue
// OE_DEFAULT: tritonamdgpu-optimize-dot-operands

// OE_OFF: rock-set-matmul-output-transpose
// OE_OFF-NOT: tritonamdgpu-optimize-epilogue
// OE_OFF: tritonamdgpu-optimize-dot-operands

// OE_ON: rock-set-matmul-output-transpose
// OE_ON: tritonamdgpu-optimize-epilogue
// OE_ON: tritonamdgpu-optimize-dot-operands

//===----------------------------------------------------------------------===//
// useBf16x3ForF32
//===----------------------------------------------------------------------===//
//
// `useBf16x3ForF32` picks the `tt.dot` input precision for f32 operands, so unlike
// the other knobs it shows up in the lowered IR rather than the pass pipeline.
// -1 follows the arch default (on for gfx950, off elsewhere), 0 forces the
// IEEE dot, and 1 forces the 3xBF16 decomposition.

// useBf16x3ForF32=-1 on gfx950: the arch default decomposes.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f32 -p --perf_config=gemm:mPerBlock=64,nPerBlock=64,kPerBlock=64,matrixInstrNonkdim=16,numStages=2,useBf16x3ForF32=-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu \
// RUN:   | FileCheck %s --check-prefix=BF16X3_DEFAULT_ON

// useBf16x3ForF32=0 on gfx950: explicit off overrides the arch default.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f32 -p --perf_config=gemm:mPerBlock=64,nPerBlock=64,kPerBlock=64,matrixInstrNonkdim=16,numStages=2,useBf16x3ForF32=0 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu \
// RUN:   | FileCheck %s --check-prefix=BF16X3_OFF

// useBf16x3ForF32=-1 on gfx942: the arch default keeps the IEEE dot.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -p --perf_config=gemm:mPerBlock=64,nPerBlock=64,kPerBlock=64,matrixInstrNonkdim=16,numStages=2,useBf16x3ForF32=-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu \
// RUN:   | FileCheck %s --check-prefix=BF16X3_DEFAULT_OFF

// useBf16x3ForF32=1 on gfx942: explicit on overrides the arch default.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -p --perf_config=gemm:mPerBlock=64,nPerBlock=64,kPerBlock=64,matrixInstrNonkdim=16,numStages=2,useBf16x3ForF32=1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu \
// RUN:   | FileCheck %s --check-prefix=BF16X3_ON

// The tri-state is consumed by rock-to-ttir, so the bridge attribute that
// carries it from rock-affix-params must not reach Triton IR. It would sit on
// the kernel function, ahead of the dot, hence the leading CHECK-NOT.
// BF16X3_DEFAULT_ON-NOT: rock.use_bf16x3_for_f32
// BF16X3_DEFAULT_ON: tt.dot {{.*}} inputPrecision = bf16x3

// BF16X3_OFF-NOT: inputPrecision = bf16x3
// BF16X3_OFF: tt.dot

// BF16X3_DEFAULT_OFF-NOT: inputPrecision = bf16x3
// BF16X3_DEFAULT_OFF: tt.dot

// BF16X3_ON: tt.dot {{.*}} inputPrecision = bf16x3

//===----------------------------------------------------------------------===//
// `--pass-pipeline=...` validation
//===----------------------------------------------------------------------===//

// Boolean knob = 2 (out of {-1, 0, 1}) is rejected.
// RUN: echo 'module {}' \
// RUN:   | not rocmlir-opt --pass-pipeline='builtin.module(rock-triton-pipeline{arch=gfx942 useBlockPingpong=2})' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD_USEBLOCKPINGPONG

// Boolean knob = -2 is rejected (only -1 is a sentinel).
// RUN: echo 'module {}' \
// RUN:   | not rocmlir-opt --pass-pipeline='builtin.module(rock-triton-pipeline{arch=gfx942 useAsyncCopy=-2})' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD_USEASYNCCOPY

// useReductionLayout is a tri-state gate like the boolean knobs above, so -1
// (the knob default) is accepted; out-of-range values are rejected.
// RUN: echo 'module {}' \
// RUN:   | not rocmlir-opt --pass-pipeline='builtin.module(rock-triton-pipeline{arch=gfx942 useReductionLayout=2})' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD_USEREDUCTIONLAYOUT_TWO

// BAD_USEBLOCKPINGPONG: LLVM ERROR: invalid `--pass-pipeline=triton{useBlockPingpong=2}`
// BAD_USEBLOCKPINGPONG-SAME: expected -1 (arch default), 0 (off), or 1 (on)

// BAD_USEASYNCCOPY: LLVM ERROR: invalid `--pass-pipeline=triton{useAsyncCopy=-2}`
// BAD_USEASYNCCOPY-SAME: expected -1 (arch default), 0 (off), or 1 (on)

// BAD_USEREDUCTIONLAYOUT_TWO: LLVM ERROR: invalid `--pass-pipeline=triton{useReductionLayout=2}`
// BAD_USEREDUCTIONLAYOUT_TWO-SAME: expected -1 (arch default), 0 (off), or 1 (on)
