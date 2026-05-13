// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t f16 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,F16
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t i8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,I8,INT
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t i8 -c_dtype i8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,I8-I8,INT
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -ta fp8 -tb bf8 -tc f32 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,FP8-BF8
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t fp8_bf8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,FP8-BF8
// RUN: rocmlir-gen --arch gfx950:sramecc+:xnack- --operation gemm -ta fp8 -tb bf8 -tc f32 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,FP8-BF8-OCP
// RUN: rocmlir-gen --arch gfx950:sramecc+:xnack- --operation gemm -t fp8_bf8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,FP8-BF8-OCP

// CHECK-LABEL: func @rock_gemm
// F16-SAME: (%{{.*}}: tensor<2362368xf16>, %{{.*}}: tensor<1181184xf16>, %{{.*}}: tensor<1572864xf16>)
// I8-SAME: (%{{.*}}: tensor<2362368xi8>, %{{.*}}: tensor<1181184xi8>, %{{.*}}: tensor<1572864xi32>)
// I8-I8-SAME: (%{{.*}}: tensor<2362368xi8>, %{{.*}}: tensor<1181184xi8>, %{{.*}}: tensor<1572864xi8>)
// FP8-BF8-SAME: (%{{.*}}: tensor<2362368xf8E4M3FNUZ>, %{{.*}}: tensor<1181184xf8E5M2FNUZ>, %{{.*}}: tensor<1572864xf32>)
// FP8-BF8-OCP-SAME: (%{{.*}}: tensor<2362368xf8E4M3FN>, %{{.*}}: tensor<1181184xf8E5M2>, %{{.*}}: tensor<1572864xf32>)

// CHECK-LABEL: func @host_naive_gemm
// F16-SAME: (%{{.*}}: tensor<2362368xf16>, %{{.*}}: tensor<1181184xf16>, %{{.*}}: tensor<1572864xf16>)
// I8-SAME: (%{{.*}}: tensor<2362368xi8>, %{{.*}}: tensor<1181184xi8>, %{{.*}}: tensor<1572864xi32>)
// I8-I8-SAME: (%{{.*}}: tensor<2362368xi8>, %{{.*}}: tensor<1181184xi8>, %{{.*}}: tensor<1572864xi32>)
// FP8-BF8-SAME: (%{{.*}}: tensor<2362368xf8E4M3FNUZ>, %{{.*}}: tensor<1181184xf8E5M2FNUZ>, %{{.*}}: tensor<1572864xf32>)
// FP8-BF8-OCP-SAME: (%{{.*}}: tensor<2362368xf8E4M3FN>, %{{.*}}: tensor<1181184xf8E5M2>, %{{.*}}: tensor<1572864xf32>)

// INT: arith.sitofp
// CHECK: arith.mulf
// CHECK-NEXT: arith.addf
// CHECK-NEXT: linalg.yield
