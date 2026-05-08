// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv | FileCheck %s --check-prefix=GEMM-SCALED
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv -quantBlockSize 16 | FileCheck %s --check-prefix=GEMM-SCALED-16
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm --transScaleA --transScaleB -pv | FileCheck %s --check-prefix=GEMM-SCALED-BOTHTRANS
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm --transScaleA -pv | FileCheck %s --check-prefix=GEMM-SCALED-TRANSA
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm --transScaleB -pv | FileCheck %s --check-prefix=GEMM-SCALED-TRANSB
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv -scale_a_dtype f32 -scale_b_dtype f32 | FileCheck %s --check-prefix=GEMM-SCALED-F32
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv -scale_a_dtype f32 -scale_b_dtype f32 -quantBlockSize 16 | FileCheck %s --check-prefix=GEMM-SCALED-F32-16

// GEMM-SCALED: func.func @rock_gemm
// GEMM-SCALED-SAME: (%[[ARG0:.*]]: tensor<4096xf4E2M1FN>, %[[ARG1:.*]]: tensor<4096xf4E2M1FN>, %[[ARG2:.*]]: tensor<128xf8E8M0FNU>, %[[ARG3:.*]]: tensor<128xf8E8M0FNU>, %[[ARG4:.*]]: tensor<256xf32>)
// GEMM-SCALED: %[[A_EXPAND:.*]] = rock.transform %[[ARG0]]
// GEMM-SCALED-SAME: tensor<4096xf4E2M1FN> to tensor<1x16x256xf4E2M1FN>
// GEMM-SCALED: %[[B_EXPAND:.*]] = rock.transform %[[ARG1]]
// GEMM-SCALED-SAME: tensor<4096xf4E2M1FN> to tensor<1x256x16xf4E2M1FN>
// GEMM-SCALED: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG2]]
// GEMM-SCALED-SAME: tensor<128xf8E8M0FNU> to tensor<1x16x8xf8E8M0FNU>
// GEMM-SCALED: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-SAME: tensor<128xf8E8M0FNU> to tensor<1x16x8xf8E8M0FNU>
// GEMM-SCALED: %[[SCALEA:.*]] = rock.transform %[[SCALEA_EXPAND]]
// GEMM-SCALED-SAME: tensor<1x16x8xf8E8M0FNU> to tensor<1x16x8xf8E8M0FNU>
// GEMM-SCALED: %[[SCALEB:.*]] = rock.transform %[[SCALEB_EXPAND]]
// GEMM-SCALED-SAME: tensor<1x16x8xf8E8M0FNU> to tensor<1x16x8xf8E8M0FNU>
// GEMM-SCALED: rock.gemm %{{.*}} scaled by %[[SCALEA]] * %{{.*}} scaled by %[[SCALEB]] {quantBlockSize = 32 : i64}
// GEMM-SCALED-SAME: tensor<1x16x256xf4E2M1FN> scaled by tensor<1x16x8xf8E8M0FNU> * tensor<1x256x16xf4E2M1FN> scaled by tensor<1x16x8xf8E8M0FNU> -> tensor<1x16x16xf32>

// GEMM-SCALED: func.func @host_naive_gemm
// GEMM-SCALED-SAME: (%[[A:.*]]: tensor<4096xf4E2M1FN>, %[[B:.*]]: tensor<4096xf4E2M1FN>, %[[SCALEA:.*]]: tensor<128xf8E8M0FNU>, %[[SCALEB:.*]]: tensor<128xf8E8M0FNU>, %[[C:.*]]: tensor<256xf32>)
// GEMM-SCALED: linalg.generic {{.*}} ins(%[[A]] : tensor<4096xf4E2M1FN>)
// GEMM-SCALED:   arith.extf %{{.*}} : f4E2M1FN to f32
// GEMM-SCALED: linalg.generic {{.*}} ins(%[[B]] : tensor<4096xf4E2M1FN>)
// GEMM-SCALED:   arith.extf %{{.*}} : f4E2M1FN to f32
// GEMM-SCALED: linalg.generic {{.*}} ins(%[[SCALEA]] : tensor<128xf8E8M0FNU>)
// GEMM-SCALED:   arith.extf %{{.*}} : f8E8M0FNU to f32
// GEMM-SCALED: linalg.generic {{.*}} ins(%[[SCALEB]] : tensor<128xf8E8M0FNU>)
// GEMM-SCALED:   arith.extf %{{.*}} : f8E8M0FNU to f32
// GEMM-SCALED: tensor.expand_shape {{.*}} : tensor<4096xf32> into tensor<1x16x256xf32>
// GEMM-SCALED: tensor.expand_shape {{.*}} : tensor<4096xf32> into tensor<1x256x16xf32>
// GEMM-SCALED: tensor.expand_shape {{.*}} : tensor<128xf32> into tensor<1x16x8xf32>
// GEMM-SCALED: tensor.expand_shape {{.*}} : tensor<128xf32> into tensor<1x16x8xf32>
// GEMM-SCALED: linalg.generic
// GEMM-SCALED-SAME: ins(%{{.*}} : tensor<1x16x256xf32>, tensor<1x256x16xf32>, tensor<1x16x8xf32>, tensor<1x16x8xf32>) outs(%{{.*}} : tensor<1x16x16xf32>)
// GEMM-SCALED: ^bb0(%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-NEXT: linalg.yield

// GEMM-SCALED-16: func.func @rock_gemm
// GEMM-SCALED-16-SAME: (%[[ARG0:.*]]: tensor<4096xf4E2M1FN>, %[[ARG1:.*]]: tensor<4096xf4E2M1FN>, %[[ARG2:.*]]: tensor<256xf8E8M0FNU>, %[[ARG3:.*]]: tensor<256xf8E8M0FNU>, %[[ARG4:.*]]: tensor<256xf32>)
// GEMM-SCALED-16: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG2]]
// GEMM-SCALED-16-SAME: tensor<256xf8E8M0FNU> to tensor<1x16x16xf8E8M0FNU>
// GEMM-SCALED-16: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-16-SAME: tensor<256xf8E8M0FNU> to tensor<1x16x16xf8E8M0FNU>
// GEMM-SCALED-16: %[[SCALEA:.*]] = rock.transform %[[SCALEA_EXPAND]]
// GEMM-SCALED-16-SAME: tensor<1x16x16xf8E8M0FNU> to tensor<1x16x16xf8E8M0FNU>
// GEMM-SCALED-16: %[[SCALEB:.*]] = rock.transform %[[SCALEB_EXPAND]]
// GEMM-SCALED-16-SAME: tensor<1x16x16xf8E8M0FNU> to tensor<1x16x16xf8E8M0FNU>
// GEMM-SCALED-16: rock.gemm %{{.*}} scaled by %[[SCALEA]] * %{{.*}} scaled by %[[SCALEB]] {quantBlockSize = 16 : i64}
// GEMM-SCALED-16-SAME: tensor<1x16x256xf4E2M1FN> scaled by tensor<1x16x16xf8E8M0FNU> * tensor<1x256x16xf4E2M1FN> scaled by tensor<1x16x16xf8E8M0FNU> -> tensor<1x16x16xf32>

// GEMM-SCALED-16: func.func @host_naive_gemm
// GEMM-SCALED-16-SAME: (%[[A:.*]]: tensor<4096xf4E2M1FN>, %[[B:.*]]: tensor<4096xf4E2M1FN>, %[[SCALEA:.*]]: tensor<256xf8E8M0FNU>, %[[SCALEB:.*]]: tensor<256xf8E8M0FNU>, %[[C:.*]]: tensor<256xf32>)
// GEMM-SCALED-16: tensor.expand_shape {{.*}} : tensor<256xf32> into tensor<1x16x16xf32>
// GEMM-SCALED-16: tensor.expand_shape {{.*}} : tensor<256xf32> into tensor<1x16x16xf32>
// GEMM-SCALED-16: linalg.generic
// GEMM-SCALED-16-SAME: ins(%{{.*}} : tensor<1x16x256xf32>, tensor<1x256x16xf32>, tensor<1x16x16xf32>, tensor<1x16x16xf32>) outs(%{{.*}} : tensor<1x16x16xf32>)
// GEMM-SCALED-16: ^bb0(%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-16-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-16-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-16-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-16-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-16-NEXT: linalg.yield

// GEMM-SCALED-BOTHTRANS: func.func @rock_gemm
// GEMM-SCALED-BOTHTRANS-SAME: (%[[ARG0:.*]]: tensor<4096xf4E2M1FN>, %[[ARG1:.*]]: tensor<4096xf4E2M1FN>, %[[ARG2:.*]]: tensor<128xf8E8M0FNU>, %[[ARG3:.*]]: tensor<128xf8E8M0FNU>, %[[ARG4:.*]]: tensor<256xf32>)
// GEMM-SCALED-BOTHTRANS: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG2]]
// GEMM-SCALED-BOTHTRANS-SAME: tensor<128xf8E8M0FNU> to tensor<1x8x16xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-BOTHTRANS-SAME: tensor<128xf8E8M0FNU> to tensor<1x8x16xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: %[[SCALEA:.*]] = rock.transform %[[SCALEA_EXPAND]]
// GEMM-SCALED-BOTHTRANS-SAME: tensor<1x8x16xf8E8M0FNU> to tensor<1x8x16xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: %[[SCALEB:.*]] = rock.transform %[[SCALEB_EXPAND]]
// GEMM-SCALED-BOTHTRANS-SAME: tensor<1x8x16xf8E8M0FNU> to tensor<1x8x16xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: rock.gemm %{{.*}} scaled by tr %[[SCALEA]] * %{{.*}} scaled by tr %[[SCALEB]] {quantBlockSize = 32 : i64}
// GEMM-SCALED-BOTHTRANS-SAME: tensor<1x16x256xf4E2M1FN> scaled by tensor<1x8x16xf8E8M0FNU> * tensor<1x256x16xf4E2M1FN> scaled by tensor<1x8x16xf8E8M0FNU> -> tensor<1x16x16xf32>

// GEMM-SCALED-BOTHTRANS: func.func @host_naive_gemm
// GEMM-SCALED-BOTHTRANS-SAME: (%[[A:.*]]: tensor<4096xf4E2M1FN>, %[[B:.*]]: tensor<4096xf4E2M1FN>, %[[SCALEA:.*]]: tensor<128xf8E8M0FNU>, %[[SCALEB:.*]]: tensor<128xf8E8M0FNU>, %[[C:.*]]: tensor<256xf32>)
// GEMM-SCALED-BOTHTRANS: tensor.expand_shape {{.*}} : tensor<128xf32> into tensor<1x8x16xf32>
// GEMM-SCALED-BOTHTRANS: tensor.expand_shape {{.*}} : tensor<128xf32> into tensor<1x8x16xf32>
// GEMM-SCALED-BOTHTRANS: linalg.generic
// GEMM-SCALED-BOTHTRANS-SAME: ins(%{{.*}} : tensor<1x16x256xf32>, tensor<1x256x16xf32>, tensor<1x8x16xf32>, tensor<1x8x16xf32>) outs(%{{.*}} : tensor<1x16x16xf32>)
// GEMM-SCALED-BOTHTRANS: ^bb0(%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-BOTHTRANS-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-BOTHTRANS-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-BOTHTRANS-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-BOTHTRANS-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-BOTHTRANS-NEXT: linalg.yield

// GEMM-SCALED-TRANSA: func.func @rock_gemm
// GEMM-SCALED-TRANSA-SAME: (%[[ARG0:.*]]: tensor<4096xf4E2M1FN>, %[[ARG1:.*]]: tensor<4096xf4E2M1FN>, %[[ARG2:.*]]: tensor<128xf8E8M0FNU>, %[[ARG3:.*]]: tensor<128xf8E8M0FNU>, %[[ARG4:.*]]: tensor<256xf32>)
// GEMM-SCALED-TRANSA: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG2]]
// GEMM-SCALED-TRANSA-SAME: tensor<128xf8E8M0FNU> to tensor<1x8x16xf8E8M0FNU>
// GEMM-SCALED-TRANSA: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-TRANSA-SAME: tensor<128xf8E8M0FNU> to tensor<1x16x8xf8E8M0FNU>
// GEMM-SCALED-TRANSA: %[[SCALEA:.*]] = rock.transform %[[SCALEA_EXPAND]]
// GEMM-SCALED-TRANSA-SAME: tensor<1x8x16xf8E8M0FNU> to tensor<1x8x16xf8E8M0FNU>
// GEMM-SCALED-TRANSA: %[[SCALEB:.*]] = rock.transform %[[SCALEB_EXPAND]]
// GEMM-SCALED-TRANSA-SAME: tensor<1x16x8xf8E8M0FNU> to tensor<1x16x8xf8E8M0FNU>
// GEMM-SCALED-TRANSA: rock.gemm %{{.*}} scaled by tr %[[SCALEA]] * %{{.*}} scaled by %[[SCALEB]] {quantBlockSize = 32 : i64}
// GEMM-SCALED-TRANSA-SAME: tensor<1x16x256xf4E2M1FN> scaled by tensor<1x8x16xf8E8M0FNU> * tensor<1x256x16xf4E2M1FN> scaled by tensor<1x16x8xf8E8M0FNU> -> tensor<1x16x16xf32>

// GEMM-SCALED-TRANSA: func.func @host_naive_gemm
// GEMM-SCALED-TRANSA-SAME: (%[[A:.*]]: tensor<4096xf4E2M1FN>, %[[B:.*]]: tensor<4096xf4E2M1FN>, %[[SCALEA:.*]]: tensor<128xf8E8M0FNU>, %[[SCALEB:.*]]: tensor<128xf8E8M0FNU>, %[[C:.*]]: tensor<256xf32>)
// GEMM-SCALED-TRANSA: tensor.expand_shape {{.*}} : tensor<128xf32> into tensor<1x8x16xf32>
// GEMM-SCALED-TRANSA: tensor.expand_shape {{.*}} : tensor<128xf32> into tensor<1x16x8xf32>
// GEMM-SCALED-TRANSA: linalg.generic
// GEMM-SCALED-TRANSA-SAME: ins(%{{.*}} : tensor<1x16x256xf32>, tensor<1x256x16xf32>, tensor<1x8x16xf32>, tensor<1x16x8xf32>) outs(%{{.*}} : tensor<1x16x16xf32>)
// GEMM-SCALED-TRANSA: ^bb0(%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-TRANSA-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-TRANSA-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-TRANSA-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-TRANSA-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-TRANSA-NEXT: linalg.yield

// GEMM-SCALED-TRANSB: func.func @rock_gemm
// GEMM-SCALED-TRANSB-SAME: (%[[ARG0:.*]]: tensor<4096xf4E2M1FN>, %[[ARG1:.*]]: tensor<4096xf4E2M1FN>, %[[ARG2:.*]]: tensor<128xf8E8M0FNU>, %[[ARG3:.*]]: tensor<128xf8E8M0FNU>, %[[ARG4:.*]]: tensor<256xf32>)
// GEMM-SCALED-TRANSB: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG2]]
// GEMM-SCALED-TRANSB-SAME: tensor<128xf8E8M0FNU> to tensor<1x16x8xf8E8M0FNU>
// GEMM-SCALED-TRANSB: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-TRANSB-SAME: tensor<128xf8E8M0FNU> to tensor<1x8x16xf8E8M0FNU>
// GEMM-SCALED-TRANSB: %[[SCALEA:.*]] = rock.transform %[[SCALEA_EXPAND]]
// GEMM-SCALED-TRANSB-SAME: tensor<1x16x8xf8E8M0FNU> to tensor<1x16x8xf8E8M0FNU>
// GEMM-SCALED-TRANSB: %[[SCALEB:.*]] = rock.transform %[[SCALEB_EXPAND]]
// GEMM-SCALED-TRANSB-SAME: tensor<1x8x16xf8E8M0FNU> to tensor<1x8x16xf8E8M0FNU>
// GEMM-SCALED-TRANSB: rock.gemm %{{.*}} scaled by %[[SCALEA]] * %{{.*}} scaled by tr %[[SCALEB]] {quantBlockSize = 32 : i64}
// GEMM-SCALED-TRANSB-SAME: tensor<1x16x256xf4E2M1FN> scaled by tensor<1x16x8xf8E8M0FNU> * tensor<1x256x16xf4E2M1FN> scaled by tensor<1x8x16xf8E8M0FNU> -> tensor<1x16x16xf32>

// GEMM-SCALED-TRANSB: func.func @host_naive_gemm
// GEMM-SCALED-TRANSB-SAME: (%[[A:.*]]: tensor<4096xf4E2M1FN>, %[[B:.*]]: tensor<4096xf4E2M1FN>, %[[SCALEA:.*]]: tensor<128xf8E8M0FNU>, %[[SCALEB:.*]]: tensor<128xf8E8M0FNU>, %[[C:.*]]: tensor<256xf32>)
// GEMM-SCALED-TRANSB: tensor.expand_shape {{.*}} : tensor<128xf32> into tensor<1x16x8xf32>
// GEMM-SCALED-TRANSB: tensor.expand_shape {{.*}} : tensor<128xf32> into tensor<1x8x16xf32>
// GEMM-SCALED-TRANSB: linalg.generic
// GEMM-SCALED-TRANSB-SAME: ins(%{{.*}} : tensor<1x16x256xf32>, tensor<1x256x16xf32>, tensor<1x16x8xf32>, tensor<1x8x16xf32>) outs(%{{.*}} : tensor<1x16x16xf32>)
// GEMM-SCALED-TRANSB: ^bb0(%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-TRANSB-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-TRANSB-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-TRANSB-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-TRANSB-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-TRANSB-NEXT: linalg.yield

// GEMM-SCALED-F32: func.func @rock_gemm
// GEMM-SCALED-F32-SAME: (%[[ARG0:.*]]: tensor<4096xf4E2M1FN>, %[[ARG1:.*]]: tensor<4096xf4E2M1FN>, %[[ARG2:.*]]: tensor<128xf32>, %[[ARG3:.*]]: tensor<128xf32>, %[[ARG4:.*]]: tensor<256xf32>)
// GEMM-SCALED-F32: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG2]]
// GEMM-SCALED-F32-SAME: tensor<128xf32> to tensor<1x16x8xf32>
// GEMM-SCALED-F32: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-F32-SAME: tensor<128xf32> to tensor<1x16x8xf32>
// GEMM-SCALED-F32: %[[SCALEA_F32:.*]] = rock.transform %[[SCALEA_EXPAND]]
// GEMM-SCALED-F32-SAME: tensor<1x16x8xf32> to tensor<1x16x8xf32>
// GEMM-SCALED-F32: %[[SCALEB_F32:.*]] = rock.transform %[[SCALEB_EXPAND]]
// GEMM-SCALED-F32-SAME: tensor<1x16x8xf32> to tensor<1x16x8xf32>
// GEMM-SCALED-F32: %[[SCALEA:.*]] = arith.truncf %[[SCALEA_F32]] : tensor<1x16x8xf32> to tensor<1x16x8xf8E8M0FNU>
// GEMM-SCALED-F32: %[[SCALEB:.*]] = arith.truncf %[[SCALEB_F32]] : tensor<1x16x8xf32> to tensor<1x16x8xf8E8M0FNU>
// GEMM-SCALED-F32: rock.gemm %{{.*}} scaled by %[[SCALEA]] * %{{.*}} scaled by %[[SCALEB]] {quantBlockSize = 32 : i64}
// GEMM-SCALED-F32-SAME: tensor<1x16x256xf4E2M1FN> scaled by tensor<1x16x8xf8E8M0FNU> * tensor<1x256x16xf4E2M1FN> scaled by tensor<1x16x8xf8E8M0FNU> -> tensor<1x16x16xf32>

// GEMM-SCALED-F32: func.func @host_naive_gemm
// GEMM-SCALED-F32-SAME: (%[[A:.*]]: tensor<4096xf4E2M1FN>, %[[B:.*]]: tensor<4096xf4E2M1FN>, %[[SCALEA:.*]]: tensor<128xf32>, %[[SCALEB:.*]]: tensor<128xf32>, %[[C:.*]]: tensor<256xf32>)
// GEMM-SCALED-F32: tensor.expand_shape %[[SCALEA]] {{.*}} : tensor<128xf32> into tensor<1x16x8xf32>
// GEMM-SCALED-F32: tensor.expand_shape %[[SCALEB]] {{.*}} : tensor<128xf32> into tensor<1x16x8xf32>
// GEMM-SCALED-F32: linalg.generic
// GEMM-SCALED-F32-SAME: ins(%{{.*}} : tensor<1x16x256xf32>, tensor<1x256x16xf32>, tensor<1x16x8xf32>, tensor<1x16x8xf32>) outs(%{{.*}} : tensor<1x16x16xf32>)
// GEMM-SCALED-F32: ^bb0(%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-F32-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-F32-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-F32-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-F32-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-F32-NEXT: linalg.yield

// GEMM-SCALED-F32-16: func.func @rock_gemm
// GEMM-SCALED-F32-16-SAME: (%[[ARG0:.*]]: tensor<4096xf4E2M1FN>, %[[ARG1:.*]]: tensor<4096xf4E2M1FN>, %[[ARG2:.*]]: tensor<256xf32>, %[[ARG3:.*]]: tensor<256xf32>, %[[ARG4:.*]]: tensor<256xf32>)
// GEMM-SCALED-F32-16: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG2]]
// GEMM-SCALED-F32-16-SAME: tensor<256xf32> to tensor<1x16x16xf32>
// GEMM-SCALED-F32-16: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-F32-16-SAME: tensor<256xf32> to tensor<1x16x16xf32>
// GEMM-SCALED-F32-16: %[[SCALEA_F32:.*]] = rock.transform %[[SCALEA_EXPAND]]
// GEMM-SCALED-F32-16-SAME: tensor<1x16x16xf32> to tensor<1x16x16xf32>
// GEMM-SCALED-F32-16: %[[SCALEB_F32:.*]] = rock.transform %[[SCALEB_EXPAND]]
// GEMM-SCALED-F32-16-SAME: tensor<1x16x16xf32> to tensor<1x16x16xf32>
// GEMM-SCALED-F32-16: %[[SCALEA:.*]] = arith.truncf %[[SCALEA_F32]] : tensor<1x16x16xf32> to tensor<1x16x16xf8E8M0FNU>
// GEMM-SCALED-F32-16: %[[SCALEB:.*]] = arith.truncf %[[SCALEB_F32]] : tensor<1x16x16xf32> to tensor<1x16x16xf8E8M0FNU>
// GEMM-SCALED-F32-16: rock.gemm %{{.*}} scaled by %[[SCALEA]] * %{{.*}} scaled by %[[SCALEB]] {quantBlockSize = 16 : i64}
// GEMM-SCALED-F32-16-SAME: tensor<1x16x256xf4E2M1FN> scaled by tensor<1x16x16xf8E8M0FNU> * tensor<1x256x16xf4E2M1FN> scaled by tensor<1x16x16xf8E8M0FNU> -> tensor<1x16x16xf32>

// GEMM-SCALED-F32-16: func.func @host_naive_gemm
// GEMM-SCALED-F32-16-SAME: (%[[A:.*]]: tensor<4096xf4E2M1FN>, %[[B:.*]]: tensor<4096xf4E2M1FN>, %[[SCALEA:.*]]: tensor<256xf32>, %[[SCALEB:.*]]: tensor<256xf32>, %[[C:.*]]: tensor<256xf32>)
// GEMM-SCALED-F32-16: tensor.expand_shape %[[SCALEA]] {{.*}} : tensor<256xf32> into tensor<1x16x16xf32>
// GEMM-SCALED-F32-16: tensor.expand_shape %[[SCALEB]] {{.*}} : tensor<256xf32> into tensor<1x16x16xf32>
// GEMM-SCALED-F32-16: linalg.generic
// GEMM-SCALED-F32-16-SAME: ins(%{{.*}} : tensor<1x16x256xf32>, tensor<1x256x16xf32>, tensor<1x16x16xf32>, tensor<1x16x16xf32>) outs(%{{.*}} : tensor<1x16x16xf32>)
// GEMM-SCALED-F32-16: ^bb0(%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-F32-16-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-F32-16-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-F32-16-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-F32-16-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-F32-16-NEXT: linalg.yield
