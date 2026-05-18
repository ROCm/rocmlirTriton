// Verify the Triton knobs on rocmlir-driver gate the relevant
// pass options in the Triton pipeline, mirroring Triton's `knobs.amd.*`
// semantics.
//
// Triton knobs (`--use-async-copy`, `--use-block-pingpong`,
// `--use-in-thread-transpose`, `--use-buffer-ops`, `--use-buffer-atomics`,
// `--buffer-ops-analyze-small-tensor-range`) accept:
//   auto (default) -> per-arch default
//   false          -> force off
//   true           -> force on
//
// `--schedule-hint` is a string variant selector ("none", "attention",
// "memory-bound-attention", or a comma-separated combination).

//===----------------------------------------------------------------------===//
// --use-async-copy
//===----------------------------------------------------------------------===//

// gfx950 default: async-copy is on by per-arch default.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ASYNC_GFX950_DEFAULT

// gfx950 with explicit --use-async-copy=false.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --use-async-copy=false --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ASYNC_GFX950_OFF

// gfx942 default: async-copy off by per-arch default.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ASYNC_GFX942_DEFAULT

// gfx942 with --use-async-copy=true.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --use-async-copy=true --dump-pipelines 2>&1 >/dev/null \
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
// --use-block-pingpong
//===----------------------------------------------------------------------===//

// gfx942 default: pingpong on by per-arch default (numStages=2 -> pass is
// scheduled).
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=PP_GFX942_DEFAULT

// gfx942 with --use-block-pingpong=false: pipeline pass sees use_pingpong=
// false and the block-pingpong pass is absent.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --use-block-pingpong=false --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=PP_GFX942_OFF

// gfx1250 default: async-copy on, but pingpong is off by per-arch default
// (only gfx942 and gfx950+async-copy enable it).
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=PP_GFX1250_DEFAULT

// gfx1250 with --use-block-pingpong=true: force on.
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --use-block-pingpong=true --dump-pipelines 2>&1 >/dev/null \
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
// --use-in-thread-transpose
//===----------------------------------------------------------------------===//

// gfx942 default: in-thread-transpose pass is scheduled.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX942_DEFAULT

// gfx942 with --use-in-thread-transpose=false: pass absent.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --use-in-thread-transpose=false --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX942_OFF

// gfx950 default: in-thread-transpose pass is absent.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX950_DEFAULT

// gfx950 with --use-in-thread-transpose=true: force on.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --use-in-thread-transpose=true --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=ITT_GFX950_ON

// ITT_GFX942_DEFAULT: tritonamdgpu-in-thread-transpose
// ITT_GFX942_OFF-NOT: tritonamdgpu-in-thread-transpose
// ITT_GFX950_DEFAULT-NOT: tritonamdgpu-in-thread-transpose
// ITT_GFX950_ON: tritonamdgpu-in-thread-transpose

//===----------------------------------------------------------------------===//
// --use-buffer-ops / --use-buffer-atomics /
// --buffer-ops-analyze-small-tensor-range
//===----------------------------------------------------------------------===//

// Default: all three buffer-ops passes are scheduled, atomics on, small-tensor
// range analysis off (matches the historical hardcoded values).
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=BUF_DEFAULT

// --use-buffer-ops=false: all three buffer-ops passes are skipped.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --use-buffer-ops=false --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=BUF_OFF

// --use-buffer-atomics=false: convert-to-buffer-ops sees allow-buffer-atomics=
// false.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --use-buffer-atomics=false --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=BUF_NOATOMICS

// --buffer-ops-analyze-small-tensor-range=true: analyze-small-tensor-ofst=
// true.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --buffer-ops-analyze-small-tensor-range=true --dump-pipelines 2>&1 >/dev/null \
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
// --triton-schedule-hint
//===----------------------------------------------------------------------===//
//
// Mirrors upstream `HIPOptions.schedule_hint`. The string value selects
// one or more scheduling-hint variants (comma-separated). The default
// ("none") leaves both the TTGIR insert pass and the LLIR lower pass
// out of the pipeline. Only `attention` triggers the TTGIR insert pass;
// `memory-bound-attention` is an LLIR-only variant (consumed by
// setKernelAttributes in TritonToHsaco) so it does not schedule the
// TTGIR insert pass, but it does keep the LLIR lower pass.

// Default: no schedule-hint passes are scheduled.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=SH_DEFAULT

// --triton-schedule-hint=attention: insert-sched-hints with
// variant=attention is scheduled, and the LLIR lower pass is enabled.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --triton-schedule-hint=attention --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=SH_ATTN

// --triton-schedule-hint=memory-bound-attention: TTGIR insert pass is
// _not_ scheduled (this variant has no TTGIR hint), but the LLIR lower
// pass still runs so the kernel-attribute step can see the request.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --triton-schedule-hint=memory-bound-attention --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=SH_MBA

// --triton-schedule-hint=attention,memory-bound-attention: the insert
// pass is scheduled once (for attention only) and the LLIR lower pass
// runs once.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --triton-schedule-hint=attention,memory-bound-attention --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=SH_BOTH

// --triton-schedule-hint=NONE is case-insensitive and behaves like the
// default.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --triton-schedule-hint=NONE --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=SH_NONE

// An unknown variant must be rejected at the driver boundary with a
// clear error and non-zero exit (no pass-pipeline dump).
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | not rocmlir-driver --kernel-pipeline=gpu,triton --triton-schedule-hint=bogus --dump-pipelines 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SH_BAD

// SH_DEFAULT-NOT: triton-amdgpu-insert-instruction-sched-hints
// SH_DEFAULT-NOT: triton-amdgpu-lower-insert-instruction-sched-hints

// SH_ATTN: triton-amdgpu-insert-instruction-sched-hints{variant=attention}
// SH_ATTN: triton-amdgpu-lower-insert-instruction-sched-hints

// SH_MBA-NOT: triton-amdgpu-insert-instruction-sched-hints
// SH_MBA: triton-amdgpu-lower-insert-instruction-sched-hints

// SH_BOTH: triton-amdgpu-insert-instruction-sched-hints{variant=attention}
// SH_BOTH-NOT: triton-amdgpu-insert-instruction-sched-hints{variant=memory-bound-attention}
// SH_BOTH: triton-amdgpu-lower-insert-instruction-sched-hints

// SH_NONE-NOT: triton-amdgpu-insert-instruction-sched-hints
// SH_NONE-NOT: triton-amdgpu-lower-insert-instruction-sched-hints

// SH_BAD: unknown scheduleHint variant 'bogus'
