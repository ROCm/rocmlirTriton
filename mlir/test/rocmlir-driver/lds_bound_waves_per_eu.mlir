// Verifies that the AMDGPU backend gets an LDS-achievable
// `amdgpu-waves-per-eu` attribute stamped on the kernel for LDS-heavy
// shapes, so that `RegAllocGreedy` doesn't chase an unachievable target
// (cf. the slow-attention regression that motivated the cap in
// `setKernelAttributes` and the gate in `ResolveKernelLaunchParamsPass`).
//
// This test exercises the integration through the standard pipeline.
// The arithmetic of `rock::computeLdsBoundWavesPerEU` /
// `rock::resolveWavesPerEU` -- including the over-request cap path that
// isn't reachable from this driver entry point (the pass-level gate fires
// first) -- is covered exhaustively in `MLIRRockUnitTests`
// (`mlir/unittests/Dialect/Rock/AmdArchDbTests.cpp`).

// RUN: rocmlir-gen --arch gfx90a --operation gemm -t f16 -p \
// RUN:   | AMDGCN_ENABLE_LLVM_DUMP=1 rocmlir-driver -c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GFX90A
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p \
// RUN:   | AMDGCN_ENABLE_LLVM_DUMP=1 rocmlir-driver -c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GFX942
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p \
// RUN:   | AMDGCN_ENABLE_LLVM_DUMP=1 rocmlir-driver -c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GFX950
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t f16 -p \
// RUN:   | AMDGCN_ENABLE_LLVM_DUMP=1 rocmlir-driver -c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GFX1100

// For the CDNA arches the populated GEMM produces a 16384-byte @global_smem
// (gfx90a/gfx942) or 33728-byte @global_smem (gfx950) with a 256-thread
// workgroup (4 wavefronts of 64 threads). Per the formula in
// `computeLdsBoundWavesPerEU`:
//
//   floor(LDS_per_unit / kernel_LDS_bytes) * (block / wave) / 4
//
// gfx90a/gfx942 (LDS_per_unit=65536): floor(65536/16384) * (256/64) / 4
//                                    = 4 * 4 / 4 = 4.
// gfx950        (LDS_per_unit=163840): floor(163840/33728) * (256/64) / 4
//                                    = 4 * 4 / 4 = 4.
//
// Either the populated perfConfig requests `wavesPerEU = 4` or it leaves
// `wavesPerEU = 0` and `setKernelAttributes` fills in the LDS bound; both
// `rock::resolveWavesPerEU` branches land at 4, which is below the
// per-CDNA-arch maximum of 8, so the attribute must be present and equal
// to "4, 4".

// GFX90A:   @global_smem = {{.*}} addrspace(3) {{.*}} [16384 x i8]
// GFX90A:   "amdgpu-waves-per-eu"="4, 4"
// GFX942:   @global_smem = {{.*}} addrspace(3) {{.*}} [16384 x i8]
// GFX942:   "amdgpu-waves-per-eu"="4, 4"
// GFX950:   @global_smem = {{.*}} addrspace(3) {{.*}} [33728 x i8]
// GFX950:   "amdgpu-waves-per-eu"="4, 4"

// gfx1100 (RDNA3, wave_size=32, LDS_per_unit=65536, max=16) produces a
// 16384-byte @global_smem and a 128-thread workgroup (4 wavefronts of 32):
// floor(65536/16384) * (128/32) / 4 = 4 * 4 / 4 = 4. Below the per-arch
// maximum of 16, so the attribute must still be stamped at "4, 4".
// GFX1100:  @global_smem = {{.*}} addrspace(3) {{.*}} [16384 x i8]
// GFX1100:  "amdgpu-waves-per-eu"="4, 4"
