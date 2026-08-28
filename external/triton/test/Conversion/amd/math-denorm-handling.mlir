// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=True" | FileCheck %s --check-prefixes=COMMON,LLVM_FTZ
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=False" | FileCheck %s --check-prefixes=COMMON,LLVM_NO_FTZ


#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_exp2(%arg0: tensor<64xf32, #blocked>) {
    // LLVM_FTZ: rocdl.exp2
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = math.exp2 %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_exp(%arg0: tensor<64xf32, #blocked>) {
    // LLVM_FTZ: llvm.exp2.f32
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = math.exp %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_rsqrt(%arg0: tensor<64xf32, #blocked>) {
    // LLVM_FTZ: rocdl.rsq {{.*}} f32 -> f32
    // LLVM_NO_FTZ: _ocml_rsqrt_f32
    %0 = math.rsqrt %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_sqrt_f32(%arg0: tensor<64xf32, #blocked>) {
    // LLVM_FTZ-LABEL: test_sqrt_f32
    // LLVM_FTZ-NOT: llvm.fcmp "ogt"
    // LLVM_FTZ: rocdl.sqrt
    // LLVM_FTZ-NOT: llvm.fmul
    // LLVM_FTZ-NOT: llvm.select
    //
    // LLVM_NO_FTZ-LABEL: test_sqrt_f32
    // LLVM_NO_FTZ: llvm.fcmp "ogt"
    // LLVM_NO_FTZ: llvm.fmul
    // LLVM_NO_FTZ-NEXT: llvm.select
    // LLVM_NO_FTZ-NEXT: rocdl.sqrt
    // LLVM_NO_FTZ: llvm.fmul
    // LLVM_NO_FTZ-NEXT: llvm.select
    %0 = math.sqrt %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_sqrt_rn_f32(%arg0: tensor<64xf32, #blocked>) {
    // COMMON-LABEL: test_sqrt_rn_f32
    // COMMON: llvm.intr.sqrt
    %0 = tt.precise_sqrt %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_sqrt_rn_f64(%arg0: tensor<64xf64, #blocked>) {
    // COMMON-LABEL: test_sqrt_rn_f64
    // COMMON: llvm.intr.sqrt
    %0 = tt.precise_sqrt %arg0 : tensor<64xf64, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_divf_rn_f32(%arg0: tensor<64xf32, #blocked>, %arg1: tensor<64xf32, #blocked>) {
    // COMMON-LABEL: test_divf_rn_f32
    // COMMON: llvm.fdiv
    %0 = tt.precise_divf %arg0, %arg1 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

// FP16 has a native exp2, so the expansion stays in FP16 instead of falling
// through to __ocml_exp_f16, which is defined in terms of FP32. The flags have
// to reach both operations or the backend will not select v_exp_f16. Neither
// choice depends on ftz, hence COMMON.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_exp_f16(%arg0: tensor<64xf16, #blocked>) {
    // COMMON-LABEL: test_exp_f16
    // COMMON: llvm.fmul
    // COMMON-SAME: {fastmathFlags = #llvm.fastmath<afn>}
    // COMMON: llvm.call @llvm.exp2.f16
    // COMMON-SAME: {fastmathFlags = #llvm.fastmath<afn>}
    // COMMON-NOT: __ocml_exp_f16
    %0 = math.exp %arg0 fastmath<afn> : tensor<64xf16, #blocked>
    tt.return
  }
}

// -----

// FP16 log goes to llvm.intr.log rather than MathToROCDL's __ocml_log_f16,
// which is defined as (half)(log2f((float)x) * ln2) and so widens by
// construction. Keeping the flags is what lets the backend pick v_log_f16.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_log_f16(%arg0: tensor<64xf16, #blocked>) {
    // COMMON-LABEL: test_log_f16
    // COMMON: llvm.intr.log
    // COMMON-SAME: {fastmathFlags = #llvm.fastmath<afn>}
    // COMMON-NOT: __ocml_log_f16
    %0 = math.log %arg0 fastmath<afn> : tensor<64xf16, #blocked>
    tt.return
  }
}
