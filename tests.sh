#!/bin/bash -uvx

# Detect GPU architecture from system
ARCH=$(rocminfo | grep -o 'gfx[0-9a-z]*' | head -1)
if [ -z "$ARCH" ]; then
    echo "Error: Could not detect GPU architecture. Is rocminfo available?"
    exit 1
fi
echo "Detected GPU architecture: $ARCH"

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f32 --arch $ARCH --num_cu 256 -g 1 -m 64 -k 256 -n 128 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f32 --arch $ARCH --num_cu 256 -g 1 -m 8 -k 128 -n 8 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f16 --arch $ARCH --num_cu 256 -g 1 -m 64 -k 256 -n 128 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f32 --arch $ARCH --num_cu 256 -g 1 -m 64 -k 256 -n 128 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f32 --arch $ARCH --num_cu 256 -g 1 -m 8 -k 128 -n 8 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f16 --arch $ARCH --num_cu 256 -g 1 -m 64 -k 256 -n 128 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# https://github.com/ROCm/rocmlirTriton/pull/12
build/bin/rocmlir-gen --operation gemm -t f32 -out_datatype f32 -transA=false -transB=false -g 1 -m 1000 -n 405 -k 1024 --arch $ARCH --perf_config=gemm:v1:16,32,32,2,1,4,32,1,2,0,0 -pv  | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so  --entry-point-result=void

build/bin/rocmlir-gen --operation gemm -g 3 -m 1024 -k 769 -n 1024 --transA=false -t fp8_fp8 --schedule_version  1 --arch $ARCH -pv | rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so  --entry-point-result=void

build/bin/rocmlir-gen -pv --operation conv -t f16 -out_datatype f32 --arch $ARCH --num_cu 304 --fil_layout k01c --in_layout nc01 --out_layout nk01 --batchsize 1 --in_channels 64 --in_h 32 --in_w 32 --out_channels 32 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 --kernel-repeats 1 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen --perf_config=gemm:v1:256,128,32,1,1,4,16,4,1,0,0 -g 3 -m 1024 -k 769 -n 1024 --transA=false -t f16  --operation gemm --arch $ARCH -pv   | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen --perf_config=gemm:v1:64,64,32,1,1,4,16,1,1,0,0 -g 1 -m 4 -k 128 -n 4 --transA=false -t i8  --operation gemm --arch $ARCH -pv   | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

cd build && LIT_FILTER=Dialect/Rock ninja check-rocmlir && LIT_FILTER=e2e ninja check-rocmlir
