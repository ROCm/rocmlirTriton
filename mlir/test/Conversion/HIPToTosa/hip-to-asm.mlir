// Compile HIP IR all the way down to an AMDGCN binary, checking both the
// emitted assembly and the module the backend pipeline hands back.
//
// RUN: rocmlir-driver -kernel-pipeline hipep,highlevel -arch gfx942 %s | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c -arch gfx942 2>&1 | FileCheck %s --check-prefix=ASM
// RUN: rocmlir-driver -kernel-pipeline hipep,highlevel -arch gfx942 %s | rocmlir-driver -c -arch gfx942 | FileCheck %s --check-prefix=MODULE

// The GEMM must reach the matrix cores rather than falling back to plain FMAs,
// and the kernel must be a complete, well-formed HSA kernel.
// ASM-LABEL: .globl main_graph
// ASM: v_mfma_f32_16x16x4_f32
// ASM: s_endpgm
// ASM: .amdhsa_kernel main_graph
// ASM: .end_amdhsa_kernel

// The backend pipeline emits a gpu.binary holding the HSACO, and the kernel
// signature is the three tensor arguments as pointers -- the !hip.context
// argument is gone.
// MODULE: module attributes
// MODULE-SAME: gpu.container_module
// MODULE-SAME: triton.hsaco = "{{.*}}ELF
// MODULE: gpu.binary @rock_kernels
// MODULE-SAME: chip = "gfx942"
// MODULE-SAME: kernel_metadata<"main_graph", (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// MODULE: llvm.func @main_graph
// MODULE-NOT: hip.
// MODULE-NOT: onnx.

// The signature carries the ONNX provenance metadata a tf2onnx frontend leaves
// behind, which Rock's kernel attribute allowlist rejects if it survives.
func.func @main_graph(%arg0: !hip.context, %arg1: tensor<4x16x256x4xf32> {onnx.name = "transpose:0"}, %arg2: tensor<4x256x64xf32> {onnx.name = "split:0"}) -> tensor<4x256x256x4xf32> attributes {onnx.graph.name = "Extracted from {tf2onnx}"} {
  %0 = tensor.empty() : tensor<4x4x16x256xf32>
  %1 = hip.transpose(%arg0) ins(%arg1 : tensor<4x16x256x4xf32>) outs(%0 : tensor<4x4x16x256xf32>) {perm = [0, 3, 1, 2]} : tensor<4x4x16x256xf32>
  %expanded = tensor.expand_shape %arg2 [[0], [1], [2, 3]] output_shape [4, 256, 4, 16] : tensor<4x256x64xf32> into tensor<4x256x4x16xf32>
  %2 = tensor.empty() : tensor<4x4x256x16xf32>
  %3 = hip.transpose(%arg0) ins(%expanded : tensor<4x256x4x16xf32>) outs(%2 : tensor<4x4x256x16xf32>) {perm = [0, 2, 1, 3]} : tensor<4x4x256x16xf32>
  %4 = tensor.empty() : tensor<4x4x256x256xf32>
  %5 = hip.matmul(%arg0) ins(%3, %1 : tensor<4x4x256x16xf32>, tensor<4x4x16x256xf32>) outs(%4 : tensor<4x4x256x256xf32>) : tensor<4x4x256x256xf32>
  %6 = tensor.empty() : tensor<4x256x256x4xf32>
  %7 = hip.transpose(%arg0) ins(%5 : tensor<4x4x256x256xf32>) outs(%6 : tensor<4x256x256x4xf32>) {perm = [0, 2, 3, 1]} : tensor<4x256x256x4xf32>
  return %7 : tensor<4x256x256x4xf32>
}
