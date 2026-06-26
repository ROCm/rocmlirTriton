// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=True" --convert-builtin-func-to-llvm="ftz=True" | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950 ftz=True" --convert-builtin-func-to-llvm="ftz=True" | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=False" --convert-builtin-func-to-llvm="ftz=False" | FileCheck %s --check-prefix=LLVM_NO_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950 ftz=False" --convert-builtin-func-to-llvm="ftz=False" | FileCheck %s --check-prefix=LLVM_NO_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1250 ftz=True" --convert-builtin-func-to-llvm="gfx-arch=gfx1250 ftz=True" | FileCheck %s --check-prefix=GFX1250

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fast_expf(%arg0: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // CHECK-LABEL: test_fast_expf
    // LLVM_FTZ: rocdl.exp2
    // LLVM_NO_FTZ: llvm.intr.exp2
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__triton_hip_fast_expf"} : (tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %0 : tensor<64xf32, #blocked>
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fast_tanhf(%arg0: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // CHECK-LABEL: test_fast_tanhf
    // LLVM_FTZ: rocdl.exp2
    // LLVM_NO_FTZ: llvm.intr.exp2
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__triton_hip_fast_tanhf"} : (tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %0 : tensor<64xf32, #blocked>
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_ocml_tanh_f32(%arg0: tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked> {
    // GFX1250-LABEL: test_ocml_tanh_f32
    // GFX1250: llvm.call_intrinsic "llvm.amdgcn.tanh.f32"
    // LLVM_FTZ-LABEL: test_ocml_tanh_f32
    // LLVM_FTZ: llvm.call @__ocml_tanh_f32
    // LLVM_NO_FTZ-LABEL: test_ocml_tanh_f32
    // LLVM_NO_FTZ: llvm.call @__ocml_tanh_f32
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__ocml_tanh_f32"} : (tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return %0 : tensor<64xf32, #blocked>
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_ocml_tanh_f16(%arg0: tensor<64xf16, #blocked>) -> tensor<64xf16, #blocked> {
    // GFX1250-LABEL: test_ocml_tanh_f16
    // GFX1250: llvm.call_intrinsic "llvm.amdgcn.tanh.f16"
    // LLVM_FTZ-LABEL: test_ocml_tanh_f16
    // LLVM_FTZ: llvm.call @__ocml_tanh_f16
    // LLVM_NO_FTZ-LABEL: test_ocml_tanh_f16
    // LLVM_NO_FTZ: llvm.call @__ocml_tanh_f16
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__ocml_tanh_f16"} : (tensor<64xf16, #blocked>) -> tensor<64xf16, #blocked>
    tt.return %0 : tensor<64xf16, #blocked>
  }
}

