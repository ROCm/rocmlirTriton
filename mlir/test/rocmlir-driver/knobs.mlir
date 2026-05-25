// Verify that Triton knobs encoded in a perfConfig string flow through
// `fillCompilationConfigs` and gate the relevant pass options in the
// Triton pipeline. The seven knobs
// (`useAsyncCopy`, `useBlockPingpong`, `useInThreadTranspose`,
//  `useBufferOps`, `useBufferAtomics`, `bufferOpsAnalyzeSmallTensorRange`,
//  `scheduleHint`) are encoded as the trailing 7 fields of the
// `gemm:v2:` perfConfig string (see
// `mlir/Dialect/Rock/IR/RockAttrDefs.td`). Knob values:
//   -1 -> per-arch / heuristic default
//    0 -> force off
//    1 -> force on (for the tri-state knobs)
// `scheduleHint` is a bitfield (see KnobUtils.h):
//    bit 0 (=1) = attention, bit 1 (=2) = memory-bound-attention.
// Combinations are expressible -- e.g. 3 means both bits set, matching
// upstream Triton's `schedule_hint="attention,memory-bound-attention"`.
//
// The tunable prefix `64,64,64,1,1,4,16,1,2,0,0` is a representative
// shape that the pipeline accepts on every arch we test below; only the
// trailing knob block changes between RUNs.

//===----------------------------------------------------------------------===//
// useAsyncCopy
//===----------------------------------------------------------------------===//

// gfx950 default: async-copy on by per-arch default.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ASYNC_GFX950_DEFAULT

// gfx950 with useAsyncCopy=0 (force off).
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,0,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ASYNC_GFX950_OFF

// gfx942 default: async-copy off by per-arch default.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ASYNC_GFX942_DEFAULT

// gfx942 with useAsyncCopy=1 (force on).
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ASYNC_GFX942_ON

// ASYNC_GFX950_DEFAULT: tritonamdgpu-pipeline{use_async_copy=true
// ASYNC_GFX950_DEFAULT: tritonamdgpu-coalesce-async-copy{arch-generation-name=gfx950}

// ASYNC_GFX950_OFF: tritonamdgpu-pipeline{use_async_copy=false
// ASYNC_GFX950_OFF-NOT: tritonamdgpu-coalesce-async-copy

// ASYNC_GFX942_DEFAULT: tritonamdgpu-pipeline{use_async_copy=false
// ASYNC_GFX942_DEFAULT-NOT: tritonamdgpu-coalesce-async-copy

// ASYNC_GFX942_ON: tritonamdgpu-pipeline{use_async_copy=true
// ASYNC_GFX942_ON: tritonamdgpu-coalesce-async-copy{arch-generation-name=gfx942}

//===----------------------------------------------------------------------===//
// useBlockPingpong
//===----------------------------------------------------------------------===//

// gfx942 default: pingpong on by per-arch default (numStages=2 -> pass is
// scheduled).
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=PP_GFX942_DEFAULT

// gfx942 with useBlockPingpong=0: pipeline pass sees use_pingpong=false and
// the block-pingpong pass is absent.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,0,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=PP_GFX942_OFF

// gfx1250 default: async-copy on, but pingpong is off by per-arch default
// (only gfx942 and gfx950+async-copy enable it).
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=PP_GFX1250_DEFAULT

// gfx1250 with useBlockPingpong=1: force on.
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,1,-1,-1,-1,-1,-1 \
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
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX942_DEFAULT

// gfx942 with useInThreadTranspose=0: pass absent.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,0,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX942_OFF

// gfx950 default: in-thread-transpose pass is absent.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX950_DEFAULT

// gfx950 with useInThreadTranspose=1: force on.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX950_ON

// ITT_GFX942_DEFAULT: tritonamdgpu-in-thread-transpose
// ITT_GFX942_OFF-NOT: tritonamdgpu-in-thread-transpose
// ITT_GFX950_DEFAULT-NOT: tritonamdgpu-in-thread-transpose
// ITT_GFX950_ON: tritonamdgpu-in-thread-transpose

//===----------------------------------------------------------------------===//
// useBufferOps / useBufferAtomics / bufferOpsAnalyzeSmallTensorRange
//===----------------------------------------------------------------------===//

// Default: all three buffer-ops passes are scheduled, atomics on, small-tensor
// range analysis off (matches the historical hardcoded values).
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=BUF_DEFAULT

// useBufferOps=0: all three buffer-ops passes are skipped.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,0,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=BUF_OFF

// useBufferAtomics=0: convert-to-buffer-ops sees allow-buffer-atomics=false.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,0,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=BUF_NOATOMICS

// bufferOpsAnalyzeSmallTensorRange=1: analyze-small-tensor-ofst=true.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=BUF_SMALL

// BUF_DEFAULT: tritonamdgpu-canonicalize-pointers
// BUF_DEFAULT: tritonamdgpu-convert-buffer-ops{allow-buffer-atomics=true analyze-small-tensor-ofst=false
// BUF_DEFAULT: tritonamdgpu-optimize-buffer-op-ptr

// BUF_OFF-NOT: tritonamdgpu-canonicalize-pointers
// BUF_OFF-NOT: tritonamdgpu-convert-buffer-ops
// BUF_OFF-NOT: tritonamdgpu-optimize-buffer-op-ptr

// BUF_NOATOMICS: tritonamdgpu-convert-buffer-ops{allow-buffer-atomics=false analyze-small-tensor-ofst=false

// BUF_SMALL: tritonamdgpu-convert-buffer-ops{allow-buffer-atomics=true analyze-small-tensor-ofst=true

//===----------------------------------------------------------------------===//
// scheduleHint
//===----------------------------------------------------------------------===//
//
// Mirrors upstream `HIPOptions.schedule_hint` as a multi-select bitfield
// (see `mlir/Dialect/Rock/utility/KnobUtils.h`):
//   -1   = arch default (equivalent to "none" today)
//    0   = none (explicit)
//    0x1 = bit for "attention"
//    0x2 = bit for "memory-bound-attention"
//    0x3 = both bits set (matches upstream
//          `schedule_hint="attention,memory-bound-attention"`)
//
// Per-variant pass scheduling exactly mirrors compiler.py:
//   make_ttgir():  if schedule_hint != "none":
//                    for hint in schedule_hint.split(","):
//                      insert_instruction_sched_hints(pm, hint)
//   make_llir():   if schedule_hint != "none":
//                    lower_instruction_sched_hints(pm, ...)
// We expand the bitfield in stable order, so bit 0 ("attention") emits
// the TTGIR insert pass; bit 1 ("memory-bound-attention") emits an
// insert pass too (it's harmless before lowering) and is also consumed
// by `setKernelAttributes` in TritonToHsaco to set the
// `amdgpu-sched-strategy=iterative-ilp` kernel attribute. Any non-zero
// bit enables the single LLIR lower pass.

// Default (knob = -1): no schedule-hint passes are scheduled.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=SH_DEFAULT

// scheduleHint=0 (explicit none): identical to default.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,0 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=SH_NONE

// scheduleHint=1 (attention): TTGIR insert pass with variant=attention is
// scheduled, and the LLIR lower pass is enabled.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=SH_ATTN

// scheduleHint=2 (memory-bound-attention): upstream's loop emits a TTGIR
// insert pass for this token too. The LLIR lower pass runs so the
// kernel-attribute step (setKernelAttributes in TritonToHsaco) can see
// the request.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,2 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=SH_MBA

// scheduleHint=3 (attention | memory-bound-attention): combined
// bitfield matching upstream `schedule_hint="attention,memory-bound-attention"`.
// The TTGIR insert pass is scheduled once per set bit, in stable order
// (attention first, then memory-bound-attention).
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p --perf_config=gemm:v2:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,3 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=SH_COMBO

// SH_DEFAULT-NOT: triton-amdgpu-insert-instruction-sched-hints
// SH_DEFAULT-NOT: triton-amdgpu-lower-insert-instruction-sched-hints

// SH_NONE-NOT: triton-amdgpu-insert-instruction-sched-hints
// SH_NONE-NOT: triton-amdgpu-lower-insert-instruction-sched-hints

// SH_ATTN: triton-amdgpu-insert-instruction-sched-hints{variant=attention}
// SH_ATTN: triton-amdgpu-lower-insert-instruction-sched-hints

// SH_MBA: triton-amdgpu-insert-instruction-sched-hints{variant=memory-bound-attention}
// SH_MBA: triton-amdgpu-lower-insert-instruction-sched-hints

// SH_COMBO: triton-amdgpu-insert-instruction-sched-hints{variant=attention}
// SH_COMBO: triton-amdgpu-insert-instruction-sched-hints{variant=memory-bound-attention}
// SH_COMBO: triton-amdgpu-lower-insert-instruction-sched-hints

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

// scheduleHint with an unknown high bit (4 = bit 2, no variant assigned)
// is rejected by the TTGIR pipeline.
// RUN: echo 'module {}' \
// RUN:   | not rocmlir-opt --pass-pipeline='builtin.module(rock-triton-pipeline{arch=gfx942 scheduleHint=4})' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD_SCHEDHINT_TRITON

// scheduleHint with an unknown high bit is also rejected by the backend
// pipeline (it consumes the same knob via TritonToHsaco).
// RUN: echo 'module {}' \
// RUN:   | not rocmlir-opt --pass-pipeline='builtin.module(rock-backend-pipeline{chip=gfx942 triple=amdgcn-amd-amdhsa scheduleHint=4})' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD_SCHEDHINT_BACKEND

// BAD_USEBLOCKPINGPONG: LLVM ERROR: invalid `--pass-pipeline=triton{useBlockPingpong=2}`
// BAD_USEBLOCKPINGPONG-SAME: expected -1 (arch default), 0 (off), or 1 (on)

// BAD_USEASYNCCOPY: LLVM ERROR: invalid `--pass-pipeline=triton{useAsyncCopy=-2}`
// BAD_USEASYNCCOPY-SAME: expected -1 (arch default), 0 (off), or 1 (on)

// BAD_SCHEDHINT_TRITON: LLVM ERROR: invalid `--pass-pipeline=triton{scheduleHint=4}`
// BAD_SCHEDHINT_TRITON-SAME: expected -1 (arch default) or a subset of bitmask 3
// BAD_SCHEDHINT_TRITON-SAME: bit 0 = attention, bit 1 = memory-bound-attention

// BAD_SCHEDHINT_BACKEND: LLVM ERROR: invalid `--pass-pipeline=backend{scheduleHint=4}`
// BAD_SCHEDHINT_BACKEND-SAME: expected -1 (arch default) or a subset of bitmask 3
