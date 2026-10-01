// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t f16 -g 1 -m 1024 -k 769 -n 1000 --dynamic-dims m,n | FileCheck %s --check-prefix=MN
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t f32 -g 3 -m 64 -k 32 -n 16 --dynamic-dims g,m,n,k | FileCheck %s --check-prefix=ALL
// RUN: not rocmlir-gen --arch gfx1100 --operation gemm -t f16 -m 64 -k 32 -n 16 --dynamic-dims m,x 2>&1 | FileCheck %s --check-prefix=BAD-NAME
// RUN: not rocmlir-gen --arch gfx950 --operation gemm -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --dynamic-dims m 2>&1 | FileCheck %s --check-prefix=SCALED
// RUN: not rocmlir-gen --arch gfx1100 --operation conv -t f16 -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dynamic-dims n 2>&1 | FileCheck %s --check-prefix=NOT-GEMM

// MN: func.func @rock_gemm(%{{.*}}: tensor<1x?x769xf16>, %{{.*}}: tensor<1x769x?xf16>, %{{.*}}: tensor<1x?x?xf16>) -> tensor<1x?x?xf16>
// MN: rock.gemm {{.*}} : tensor<1x?x769xf16> * tensor<1x769x?xf16> -> tensor<1x?x?xf16>

// ALL: func.func @rock_gemm(%{{.*}}: tensor<?x?x?xf32>, %{{.*}}: tensor<?x?x?xf32>, %{{.*}}: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

// BAD-NAME: invalid --dynamic-dims entry 'x'; expected one of: g,m,n,k

// SCALED: --dynamic-dims is not supported for scaled gemm

// NOT-GEMM: --dynamic-dims is only supported for gemm
