// Verify that Triton knobs encoded in a perfConfig string flow through
// `fillCompilationConfigs` and gate the relevant pass options in the
// Triton pipeline. The five knobs
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

// ITT_GFX942_DEFAULT: tritonamdgpu-in-thread-transpose
// ITT_GFX942_OFF-NOT: tritonamdgpu-in-thread-transpose
// ITT_GFX950_DEFAULT-NOT: tritonamdgpu-in-thread-transpose
// ITT_GFX950_ON: tritonamdgpu-in-thread-transpose

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
// `useReductionLayout` is the v4 perfConfig knob and a tri-state gate like the
// other knobs: -1 (the knob default / heuristic, currently off), 0 (off), or
// 1 (on). Both the default (-1, elided in the backward-compatible v3
// serialization) and an explicit 0 keep the `rock-set-reduction-layout` pass
// out of the pipeline. Only an explicit `gemm:v4:...,1` schedules it.

// Default (v3 string, knob absent -> defaults to -1): pass is not scheduled.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v3:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=RL_DEFAULT

// v4 with useReductionLayout=-1 (heuristic default, currently off): absent.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v4:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=RL_HEURISTIC

// v4 with useReductionLayout=0 (explicit off): still absent.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v4:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,0 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=RL_OFF

// v4 with useReductionLayout=1 (force on): the pass is scheduled.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v4:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=RL_ON

// RL_DEFAULT-NOT: rock-set-reduction-layout
// RL_HEURISTIC-NOT: rock-set-reduction-layout
// RL_OFF-NOT: rock-set-reduction-layout
// RL_ON: rock-set-reduction-layout

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
