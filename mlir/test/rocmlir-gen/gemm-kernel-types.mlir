// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t f16 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,F16
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t i8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,I8,INT
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t i8 -c_dtype i8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,I8-I8,INT
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t i8 -c_dtype f32 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,I8-F32,INT
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -ta fp8 -tb bf8 -tc f32 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,FP8-BF8
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t fp8_bf8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,FP8-BF8
// RUN: rocmlir-gen --arch gfx950:sramecc+:xnack- --operation gemm -ta fp8 -tb bf8 -tc f32 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,FP8-BF8-OCP
// RUN: rocmlir-gen --arch gfx950:sramecc+:xnack- --operation gemm -t fp8_bf8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,FP8-BF8-OCP

// CHECK-LABEL: func @rock_gemm
// F16-SAME: (%{{.*}}: tensor<2362368xf16>, %{{.*}}: tensor<1181184xf16>, %{{.*}}: tensor<1572864xf16>)
// I8-SAME: (%{{.*}}: tensor<2362368xi8>, %{{.*}}: tensor<1181184xi8>, %{{.*}}: tensor<1572864xi32>)
// I8-I8-SAME: (%{{.*}}: tensor<2362368xi8>, %{{.*}}: tensor<1181184xi8>, %{{.*}}: tensor<1572864xi8>)
// I8-F32-SAME: (%{{.*}}: tensor<2362368xi8>, %{{.*}}: tensor<1181184xi8>, %{{.*}}: tensor<1572864xf32>)
// FP8-BF8-SAME: (%{{.*}}: tensor<2362368xf8E4M3FNUZ>, %{{.*}}: tensor<1181184xf8E5M2FNUZ>, %{{.*}}: tensor<1572864xf32>)
// FP8-BF8-OCP-SAME: (%{{.*}}: tensor<2362368xf8E4M3FN>, %{{.*}}: tensor<1181184xf8E5M2>, %{{.*}}: tensor<1572864xf32>)

// CHECK-LABEL: func @host_naive_gemm
// F16-SAME: (%{{.*}}: tensor<2362368xf16>, %{{.*}}: tensor<1181184xf16>, %{{.*}}: tensor<1572864xf16>)
// I8-SAME: (%{{.*}}: tensor<2362368xi8>, %{{.*}}: tensor<1181184xi8>, %{{.*}}: tensor<1572864xi32>)
// I8-I8-SAME: (%{{.*}}: tensor<2362368xi8>, %{{.*}}: tensor<1181184xi8>, %{{.*}}: tensor<1572864xi32>)
// I8-F32-SAME: (%{{.*}}: tensor<2362368xi8>, %{{.*}}: tensor<1181184xi8>, %{{.*}}: tensor<1572864xf32>)
// FP8-BF8-SAME: (%{{.*}}: tensor<2362368xf8E4M3FNUZ>, %{{.*}}: tensor<1181184xf8E5M2FNUZ>, %{{.*}}: tensor<1572864xf32>)
// FP8-BF8-OCP-SAME: (%{{.*}}: tensor<2362368xf8E4M3FN>, %{{.*}}: tensor<1181184xf8E5M2>, %{{.*}}: tensor<1572864xf32>)

// Integer GEMM mirrors the GPU's i32 accumulator: the body sign-extends the i8
// operands and uses integer multiply/add.
// INT: arith.muli
// INT: arith.addi

// Floating-point GEMM accumulates in f32.
// F16: arith.mulf
// F16: arith.addf
// FP8-BF8: arith.mulf
// FP8-BF8: arith.addf
// FP8-BF8-OCP: arith.mulf
// FP8-BF8-OCP: arith.addf

// The accumulator is rounded to the kernel's output type at store time: sitofp
// for an f32 output, truncation for an i8 output, truncf for an f16 output.
// I8-F32: arith.sitofp
// I8-I8: arith.trunci
// F16: arith.truncf
