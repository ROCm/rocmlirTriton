// A split-K gemm accumulates its partial tiles with `atomicrmw fadd`, so it is
// the shortest end-to-end path to the float atomics the `useFastAtomics`
// perfConfig knob gates. With the knob on, codegen keeps the native
// `global_atomic_add_f32`; with it off, `triton-to-hsaco` drops the subtarget
// features that provide it and AtomicExpandPass rewrites the accumulation into
// a compare-and-swap loop. The features never show up in the LLVM IR, so
// inspect the assembly (AMDGCN_ENABLE_DUMP).
//
// Both runs pass `useBufferAtomics=0`, so that they differ only in the knob
// under test: a buffer atomic bypasses AtomicExpandPass, which is why
// `rock-affix-params` rejects `useFastAtomics=0` with buffer atomics left on
// (see lowering_affix_params_negative.mlir).

// gfx942 (CDNA3), knob on: native instruction, no CAS loop.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -p \
// RUN:   --perf_config=gemm:mPerBlock=64,nPerBlock=64,kPerBlock=64,splitKFactor=4,useBufferAtomics=0,useFastAtomics=1 \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --arch gfx942 --kernel-pipeline=gpu,triton,binary 2>&1 >/dev/null \
// RUN: | FileCheck %s --check-prefix=CDNA-ON --implicit-check-not=global_atomic_cmpswap

// gfx942 (CDNA3), knob off: CAS loop, no native instruction.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -p \
// RUN:   --perf_config=gemm:mPerBlock=64,nPerBlock=64,kPerBlock=64,splitKFactor=4,useBufferAtomics=0,useFastAtomics=0 \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --arch gfx942 --kernel-pipeline=gpu,triton,binary 2>&1 >/dev/null \
// RUN: | FileCheck %s --check-prefix=CDNA-OFF --implicit-check-not=global_atomic_add_f32

// The knob works the same way on RDNA, where only the float add features may be
// dropped (see `appendFastAtomicDisables`): dropping a feature the gfx10+
// generation owns instead makes the AMDGPU backend assert on any kernel, so
// these two runs also cover that the disables stay compilable there.

// gfx1201 (RDNA4), knob on.
// RUN: rocmlir-gen --arch gfx1201 --operation gemm -t f32 -p \
// RUN:   --perf_config=gemm:mPerBlock=64,nPerBlock=64,kPerBlock=64,splitKFactor=4,useBufferAtomics=0,useFastAtomics=1 \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --arch gfx1201 --kernel-pipeline=gpu,triton,binary 2>&1 >/dev/null \
// RUN: | FileCheck %s --check-prefix=RDNA-ON --implicit-check-not=global_atomic_cmpswap

// gfx1201 (RDNA4), knob off.
// RUN: rocmlir-gen --arch gfx1201 --operation gemm -t f32 -p \
// RUN:   --perf_config=gemm:mPerBlock=64,nPerBlock=64,kPerBlock=64,splitKFactor=4,useBufferAtomics=0,useFastAtomics=0 \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --arch gfx1201 --kernel-pipeline=gpu,triton,binary 2>&1 >/dev/null \
// RUN: | FileCheck %s --check-prefix=RDNA-OFF --implicit-check-not=global_atomic_add_f32

// CDNA-ON: global_atomic_add_f32
// CDNA-OFF: global_atomic_cmpswap

// RDNA-ON: global_atomic_add_f32
// RDNA-OFF: global_atomic_cmpswap_b32
