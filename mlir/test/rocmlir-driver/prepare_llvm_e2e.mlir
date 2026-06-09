// End-to-end test: verify AnalyzeMemoryUse + RockPrepareLLVM annotate
// generated gemms correctly through the full pipeline.

// --- Regular gemm (gfx942, no atomics) ---
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f32 -p \
// RUN:   | rocmlir-driver -c --mlir-print-ir-after=rock-prepare-llvm 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GEMM

// AnalyzeMemoryUse: input args are readonly, output is writeonly.
// All pointer args get the full set of LLVM + Triton attributes.
// GEMM: llvm.func @rock_gemm(

// First arg (A matrix): readonly input with full attribute set
// GEMM-SAME: llvm.align = 16
// GEMM-SAME: llvm.dereferenceable
// GEMM-SAME: llvm.noalias
// GEMM-SAME: llvm.nocapture
// GEMM-SAME: llvm.nofree
// GEMM-SAME: llvm.nonnull
// GEMM-SAME: llvm.noundef
// GEMM-SAME: llvm.readonly
// GEMM-SAME: tt.divisibility = 16
// GEMM-SAME: tt.pointer_range = 32

// Second arg (B matrix): also readonly
// GEMM-SAME: llvm.readonly
// GEMM-SAME: tt.divisibility = 16
// GEMM-SAME: tt.pointer_range = 32

// Third arg (C matrix): writeonly output
// GEMM-SAME: llvm.writeonly
// GEMM-SAME: tt.divisibility = 16
// GEMM-SAME: tt.pointer_range = 32

// RockPrepareLLVM: GEPs are inbounds
// GEMM: llvm.getelementptr inbounds

// RockPrepareLLVM: buffer loads have alias scopes
// GEMM: rocdl.raw.ptr.buffer.load
// GEMM-SAME: alias_scopes
// GEMM-SAME: noalias_scopes

// RockPrepareLLVM: buffer stores have alias scopes
// GEMM: rocdl.raw.ptr.buffer.store
// GEMM-SAME: alias_scopes
// GEMM-SAME: noalias_scopes

// --- Atomic gemm (gfx1100, atomic_add f32) ---
// RUN: rocmlir-gen --arch gfx1100 --store-method atomic_add --operation gemm -t f32 -p \
// RUN:   | rocmlir-driver -c --mlir-print-ir-after=rock-prepare-llvm 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ATOMIC

// AnalyzeMemoryUse: first arg (A matrix) is readonly with full attribute set
// ATOMIC: llvm.func @rock_gemm(
// ATOMIC-SAME: llvm.align = 16
// ATOMIC-SAME: llvm.dereferenceable
// ATOMIC-SAME: llvm.noalias
// ATOMIC-SAME: llvm.nocapture
// ATOMIC-SAME: llvm.nofree
// ATOMIC-SAME: llvm.nonnull
// ATOMIC-SAME: llvm.noundef
// ATOMIC-SAME: llvm.readonly
// ATOMIC-SAME: tt.divisibility = 16
// ATOMIC-SAME: tt.pointer_range = 32

// Second arg (B matrix): also readonly
// ATOMIC-SAME: llvm.readonly
// ATOMIC-SAME: tt.divisibility = 16
// ATOMIC-SAME: tt.pointer_range = 32

// Third arg (C matrix): atomic output (no readonly, no writeonly)
// ATOMIC-SAME: llvm.noalias
// ATOMIC-SAME: tt.divisibility = 16
// ATOMIC-SAME: tt.pointer_range = 32

// RockPrepareLLVM: atomics get ROCDL metadata and alias scopes
// ATOMIC: llvm.atomicrmw fadd
// ATOMIC-SAME: alias_scopes
// ATOMIC-SAME: noalias_scopes
// ATOMIC-SAME: rocdl.ignore_denormal_mode
// ATOMIC-SAME: rocdl.no_fine_grained_memory
// ATOMIC-SAME: rocdl.no_remote_memory
