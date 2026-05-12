// RUN: rocmlir-opt -resolve-kernel-launch-params="waves-per-eu=4" --split-input-file %s -verify-diagnostics
// RUN: rocmlir-opt -resolve-kernel-launch-params="waves-per-eu=4" --split-input-file %s -verify-diagnostics --mlir-print-ir-after-failure 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NA --implicit-check-not=rock.not_applicable

// Exercises the LDS-bound waves-per-eu gate added to
// ResolveKernelLaunchParamsPass: a perfConfig that requests more occupancy
// than the kernel's static LDS allocation can support must be rejected and
// the module marked `rock.not_applicable`, so the tuner skips the candidate
// instead of measuring a kernel whose occupancy gets capped downstream.
// See plans/slow-attention-regalloc/TICKET.md for the underlying compile-time
// pathology this gate prevents. Behavior for `waves-per-eu = 0` (the default
// when no perfConfig occupancy is requested) is already covered by the other
// `resolve-kernel-launch-params*.mlir` tests in this directory.

// Achievable request -- passes through unchanged.
//
// gfx90a: LDS_per_CU=65536, waveSize=64, maxWavesPerEU=8, EUsPerLdsUnit=4.
// With ttg.shared=4096 and 4 warps (block_size=256), the LDS-bound ceiling
// is floor(65536/4096) * (256/64) / 4 = 16 * 4 / 4 = 16, clamped to
// maxWavesPerEU=8. waves-per-eu=4 is therefore achievable.
module attributes {
    "ttg.shared" = 4096 : i32,
    "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @achievable_wpu(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
    llvm.return
  }
}

// -----

// Over-request -- gate rejects this perfConfig.
//
// With ttg.shared=16384 and 1 warp (block_size=64) on gfx90a, the LDS-bound
// ceiling is floor(65536/16384) * (64/64) / 4 = 4 * 1 / 4 = 1. Requesting
// waves-per-eu=4 therefore exceeds the achievable maximum, the pass emits
// an error, and marks the module `rock.not_applicable`.
// expected-error @+2 {{perfConfig waves-per-eu (4) exceeds LDS-achievable maximum (1) for amdgcn-amd-amdhsa:gfx90a (kernel uses 16384 B LDS, block_size=64)}}
// NA: module attributes {rock.not_applicable, {{.*}}ttg.shared = 16384
module attributes {
    "ttg.shared" = 16384 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @oversubscribed_wpu(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a", rock.kernel} {
    llvm.return
  }
}

// -----

// gfx950 attention cfg1 reproducer: ttg.shared=19456, 1 warp (block_size=64).
// LDS_per_CU=163840, EUsPerLdsUnit=4, maxWavesPerEU=8. LDS-bound ceiling is
// floor(163840/19456) * (64/64) / 4 = 8 * 1 / 4 = 2. The perfConfig in the
// ticket requested waves-per-eu=8; our RUN line uses waves-per-eu=4, which is
// still > 2 and exercises the same gate without needing a second pass
// invocation.
// expected-error @+2 {{perfConfig waves-per-eu (4) exceeds LDS-achievable maximum (2) for amdgcn-amd-amdhsa:gfx950 (kernel uses 19456 B LDS, block_size=64)}}
// NA: module attributes {rock.not_applicable, {{.*}}ttg.shared = 19456
module attributes {
    "ttg.shared" = 19456 : i32,
    "ttg.num-warps" = 1 : i32,
    "ttg.threads-per-warp" = 64 : i32
} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  llvm.func @attention_cfg1_shape(%arg0: !llvm.ptr, %gs: !llvm.ptr<1>, %ps: !llvm.ptr<1>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
    llvm.return
  }
}
