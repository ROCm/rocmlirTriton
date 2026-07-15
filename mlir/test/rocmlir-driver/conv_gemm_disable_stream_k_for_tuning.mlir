// RUN: rocmlir-gen --arch gfx942 --operation conv_gemm -t f32 -p | FileCheck %s

// CHECK: func.func @rock_conv_gemm
// CHECK-SAME: rock.enable_streamk_for_tuning

// RUN: rocmlir-gen --arch gfx942 --operation conv_gemm -t f32 -p -disable-stream-k-for-tuning | FileCheck %s --check-prefix=CHECK-NOSTREAMK

// CHECK-NOSTREAMK: func.func @rock_conv_gemm
// CHECK-NOSTREAMK-NOT: rock.enable_streamk_for_tuning
