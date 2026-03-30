// Verify that the Triton pipeline correctly propagates numCTAs from
// the perf_config into the ttg.num-ctas module attribute, and that
// kernel launch parameters (grid_size, block_size) are set correctly.

// TODO: Enable when upstream supports multi-CTA:
//   - mgpuLaunchClusterKernel for HIP (RocmRuntimeWrappers.cpp)
//   - CGA layout fix for dot operands (Dialect.cpp)
// DISABLED: rocmlir-gen --arch gfx1250 --operation gemm -t f16 -p --perf_config "gemm:v1:64,64,64,1,2,4,16,1,2,0,0" | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=GFX1250_CTA2

// gfx1250 with default numCTAs=1 for comparison
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f16 -p --perf_config "gemm:v1:64,64,64,1,1,4,16,1,2,0,0" | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=GFX1250_CTA1

// gfx942 with default numCTAs=1
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=GFX942

// Check module-level attributes
// GFX1250_CTA2: rock.grid_size.rock_gemm = 128 : i32
// GFX1250_CTA2-SAME: "ttg.num-ctas" = 2 : i32

// GFX1250_CTA1: rock.grid_size.rock_gemm = 128 : i32
// GFX1250_CTA1-SAME: "ttg.num-ctas" = 1 : i32

// GFX942: rock.grid_size.rock_gemm = 128 : i32
// GFX942-SAME: "ttg.num-ctas" = 1 : i32

// Check kernel function attributes (grid_size and block_size)
// gfx1250: 4 waves * 32 wavesize = 128 block_size
// GFX1250_CTA2: rock.block_size = 128 : i32
// GFX1250_CTA2-SAME: rock.grid_size = 128 : i32

// GFX1250_CTA1: rock.block_size = 128 : i32
// GFX1250_CTA1-SAME: rock.grid_size = 128 : i32

// gfx942: 4 waves * 64 wavesize = 256 block_size
// GFX942: rock.block_size = 256 : i32
// GFX942-SAME: rock.grid_size = 128 : i32

// Verify that numCTAs>1 generates cluster intrinsics in the kernel,
// and numCTAs=1 uses regular workgroup id.
// GFX1250_CTA2: llvm.call_intrinsic "llvm.amdgcn.cluster.workgroup.id.x"
// GFX1250_CTA1-NOT: cluster.workgroup.id
// GFX942-NOT: cluster.workgroup.id
