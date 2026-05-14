// Native scaled MFMA is only available on gfx950, so the whole directory is
// gated by the sibling lit.local.cfg.
//
// All RUN blocks use -pv, so expected stdout is "[1 1 1]".

// scaled gemm f8 (block-scaled): A,B f8E4M3FN with f8E8M0FNU scale tensors,
// 1x64x256x128, pinned perf_config
// RUN: rocmlir-gen --arch %arch --operation gemm -scaledGemm -quantBlockSize 32 -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E8M0FNU -t f8E4M3FN -out_datatype f32 -g 1 -m 64 -k 256 -n 128 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// scaled gemm f4 (block-scaled): A,B f4E2M1FN with f8E8M0FNU scale tensors,
// 1x64x256x128, pinned perf_config
// RUN: rocmlir-gen --arch %arch --operation gemm -scaledGemm -quantBlockSize 32 -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E8M0FNU -t f4E2M1FN -out_datatype f32 -g 1 -m 64 -k 256 -n 128 --perf_config=gemm:v1:64,64,64,1,1,4,32,1,2,0,0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// scaled gemm f4 padded: same as above but with non-power-of-2 M and N
// (1x66x256x130) to exercise the padding path with a tile_k of 0 in the
// perf_config (which forces the tuner to pick a tile_k).
// RUN: rocmlir-gen --arch %arch --operation gemm -scaledGemm -quantBlockSize 32 -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E8M0FNU -t f4E2M1FN -out_datatype f32 -g 1 -m 66 -k 256 -n 130 --perf_config=gemm:v1:64,64,64,1,1,4,0,1,2,0,0 -pv \
// RUN: | rocmlir-driver -c \
// RUN: | rocm-run \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
