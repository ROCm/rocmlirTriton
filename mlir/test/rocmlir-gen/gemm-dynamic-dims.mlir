// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t f16 -g 1 -m 1024 -k 769 -n 1000 --dynamic-dims m,n | FileCheck %s --check-prefix=MN
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t f32 -g 3 -m 64 -k 32 -n 16 --dynamic-dims g,m,n,k | FileCheck %s --check-prefix=ALL
// RUN: not rocmlir-gen --arch gfx1100 --operation gemm -t f16 -m 64 -k 32 -n 16 --dynamic-dims m,x 2>&1 | FileCheck %s --check-prefix=BAD-NAME
// RUN: not rocmlir-gen --arch gfx950 --operation gemm -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --dynamic-dims m 2>&1 | FileCheck %s --check-prefix=SCALED
// RUN: not rocmlir-gen --arch gfx1100 --operation gemm_gemm -t f16 -m 64 -k 32 -n 32 -gemmO 32 --dynamic-dims m 2>&1 | FileCheck %s --check-prefix=OTHER-OP

// MN: func.func @rock_gemm(%{{.*}}: tensor<1x?x769xf16>, %{{.*}}: tensor<1x769x?xf16>, %{{.*}}: tensor<1x?x?xf16>) -> tensor<1x?x?xf16>
// MN: rock.gemm {{.*}} : tensor<1x?x769xf16> * tensor<1x769x?xf16> -> tensor<1x?x?xf16>

// ALL: func.func @rock_gemm(%{{.*}}: tensor<?x?x?xf32>, %{{.*}}: tensor<?x?x?xf32>, %{{.*}}: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

// BAD-NAME: invalid --dynamic-dims entry 'x'; expected one of: g,m,n,k

// SCALED: --dynamic-dims is not supported for scaled gemm

// OTHER-OP: --dynamic-dims is only supported for gemm, conv, conv_bwd_data and attention
