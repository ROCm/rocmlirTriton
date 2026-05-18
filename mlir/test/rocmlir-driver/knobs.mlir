// Verify the Triton knobs on rocmlir-driver gate the relevant
// pass options in the Triton pipeline, mirroring Triton's `knobs.amd.*`
// semantics.
//
// Tri-state knobs (`--use-async-copy`, `--use-block-pingpong`,
// `--use-in-thread-transpose`) accept:
//   auto (default) -> per-arch default
//   false          -> force off
//   true           -> force on
//
// Bool knobs (`--use-buffer-ops`, `--use-buffer-atomics`,
// `--buffer-ops-analyze-small-tensor-range`) are plain on/off.

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
