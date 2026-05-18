// Verify the Triton knobs gate the async-copy schedule in the Triton
// pipeline, mirroring Triton's `knobs.amd.use_async_copy` three-state
// semantics:
//   auto (default) -> per-arch default
//   false          -> force off
//   true           -> force on

// 1. gfx950 default: async-copy is on by per-arch default.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=GFX950_DEFAULT

// 2. gfx950 with explicit --use-async-copy=false: pipeline pass must see
//    use_async_copy=false.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --use-async-copy=false --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=GFX950_OFF

// 3. gfx942 default: async-copy is off by per-arch default.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=GFX942_DEFAULT

// 4. gfx942 with --use-async-copy=true: force on.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton --use-async-copy=true --dump-pipelines 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=GFX942_ON

// gfx950 default: async-copy on -> coalesce pass present, pipeline pass sees true.
// GFX950_DEFAULT: tritonamdgpu-pipeline{use_async_copy=true
// GFX950_DEFAULT: tritonamdgpu-coalesce-async-copy{arch-generation-name=gfx950}

// gfx950 forced off: pipeline pass sees false, coalesce pass absent.
// GFX950_OFF: tritonamdgpu-pipeline{use_async_copy=false
// GFX950_OFF-NOT: tritonamdgpu-coalesce-async-copy

// gfx942 default: async-copy off -> pipeline pass sees false, no coalesce.
// GFX942_DEFAULT: tritonamdgpu-pipeline{use_async_copy=false
// GFX942_DEFAULT-NOT: tritonamdgpu-coalesce-async-copy

// gfx942 forced on: pipeline pass sees true, coalesce pass appears for gfx942.
// GFX942_ON: tritonamdgpu-pipeline{use_async_copy=true
// GFX942_ON: tritonamdgpu-coalesce-async-copy{arch-generation-name=gfx942}
