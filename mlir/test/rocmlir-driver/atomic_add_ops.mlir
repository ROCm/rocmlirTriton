// RUN: rocmlir-gen --arch gfx90a --store-method atomic_add --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=ATOMICRMW
// RUN: rocmlir-gen --arch gfx90a --store-method atomic_add --operation gemm -t bf16 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=ATOMICRMW
// RUN: rocmlir-gen --arch gfx90a --store-method atomic_add --operation gemm -t f32 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=ATOMICRMW

// RUN: rocmlir-gen --arch gfx942 --store-method atomic_add --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD
// RUN: rocmlir-gen --arch gfx942 --store-method atomic_add --operation gemm -t bf16 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=ATOMICRMW
// RUN: rocmlir-gen --arch gfx942 --store-method atomic_add --operation gemm -t f32 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD

// RUN: rocmlir-gen --arch gfx950 --store-method atomic_add --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD
// RUN: rocmlir-gen --arch gfx950 --store-method atomic_add --operation gemm -t bf16 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD
// RUN: rocmlir-gen --arch gfx950 --store-method atomic_add --operation gemm -t f32 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD

// TODO(rocmlirTriton): Enable the XXX whenever we have a valid perfConfig for f32 in gfx1201
// RUN: rocmlir-gen --arch gfx1100 --store-method atomic_add --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=ATOMICRMW
// RUN: rocmlir-gen --arch gfx1100 --store-method atomic_add --operation gemm -t bf16 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=ATOMICRMW
// XXX: rocmlir-gen --arch gfx1100 --store-method atomic_add --operation gemm -t f32 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=XXX

// TODO(rocmlirTriton): Enable the XXX whenever we have a valid perfConfig for f32 in gfx1201
// RUN: rocmlir-gen --arch gfx1201 --store-method atomic_add --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD
// RUN: rocmlir-gen --arch gfx1201 --store-method atomic_add --operation gemm -t bf16 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD
// XXX: rocmlir-gen --arch gfx1201 --store-method atomic_add --operation gemm -t f32 -p | rocmlir-driver -c --debug-only=convert-triton-amdgpu-to-llvm | FileCheck %s --check-prefix=XXX

// BUFFER_ATOMIC_ADD: llvm.amdgcn.raw.ptr.buffer.atomic.fadd
// ATOMICRMW: llvm.atomicrmw
