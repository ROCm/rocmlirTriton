// Regression test for an AMDGPU register-allocator crash that surfaces in
// conv_bwd_data kernels. Without the fix, the Greedy Register
// Allocator hits a MachineVerifier failure ("Bad machine code: Using an
// undefined physical register") while spilling/hoisting SI_SPILL_WWM_AV32_SAVE,
// because LLVM's InlineSpiller::hoistSpillInsideBB does not extend the live
// range of the spilled value when its def is a PHI

// RUN: rocmlir-gen -pv --operation conv_bwd_data -t f16 --arch %arch --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 1 --in_channels 512 --in_h 12 --in_w 12 --out_channels 512 --fil_h 4 --fil_w 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --groupsize 1 --perf_config="gemm:v1:64,64,256,1,1,1,16,3,1,0,0" | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
