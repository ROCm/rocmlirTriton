// Verifies that the AMDGPU backend gets an LDS-achievable
// `amdgpu-waves-per-eu` attribute stamped on the kernel for an LDS-heavy
// shape, so that `RegAllocGreedy` doesn't chase an unachievable target
// (cf. the slow-attention regression that motivated the cap in
// `setKernelAttributes` and the gate in `ResolveKernelLaunchParamsPass`).
//
// This test only locks the integration through the standard pipeline on
// the motivating arch (gfx950). The arithmetic of
// `rock::computeLdsBoundWavesPerEU` / `rock::resolveWavesPerEU` --
// including per-arch coverage (CDNA1-4, RDNA1-4, gfx1250), boundary
// cases, and the over-request cap path that isn't reachable from this
// driver entry point (the pass-level gate fires first) -- is covered
// exhaustively in `MLIRRockUnitTests`
// (`mlir/unittests/Dialect/Rock/AmdArchDbTests.cpp`).
//
// If `rocmlir-gen --operation gemm -p` or rock-affix-params changes the
// populated perfConfig (and hence the LDS allocation / block size), the
// expected sizes below will go stale. To refresh them:
//
//   build/bin/rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
//     | AMDGCN_ENABLE_LLVM_DUMP=1 build/bin/rocmlir-driver -c 2>&1 \
//     | grep -E '@global_smem = .* \[[0-9]+ x i8\]|amdgpu-flat-work-group-size|amdgpu-waves-per-eu'
//
// then update the CHECK lines below and re-run the math:
//
//   ldsBound = floor(LDS_per_unit / kernel_LDS_bytes) * (block / wave) / 4
//
// gfx950: LDS_per_unit=163840 B, wave_size=64, maxWavesPerEU=8.

// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
// RUN:   | AMDGCN_ENABLE_LLVM_DUMP=1 rocmlir-driver -c 2>&1 \
// RUN:   | FileCheck %s

// For gfx950 the populated GEMM allocates a 33728-byte @global_smem and
// runs with a 256-thread workgroup (4 wavefronts of 64 threads):
//   ldsBound = floor(163840 / 33728) * (256 / 64) / 4 = 4 * 4 / 4 = 4.
// 4 < maxWavesPerEU=8, so either the perfConfig requested `wavesPerEU=4`
// or `setKernelAttributes` filled in the LDS bound; both
// `rock::resolveWavesPerEU` branches land at 4 and stamp "4, 4".

// CHECK: @global_smem = {{.*}} addrspace(3) {{.*}} [33728 x i8]
// CHECK: "amdgpu-waves-per-eu"="4, 4"
