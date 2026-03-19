#!/bin/bash -uvx

# Detect GPU architecture from system
ARCH=$(rocminfo | grep -o 'gfx[0-9a-z]*' | head -1)
if [ -z "$ARCH" ]; then
    echo "Error: Could not detect GPU architecture. Is rocminfo available?"
    exit 1
fi

# Get Compute Units from GPU section (after the gfx Name line, not the CPU)
NUM_CU=$(rocminfo | grep -A 30 "Name:.*gfx" | grep "Compute Unit" | head -1 | grep -o '[0-9]*')
if [ -z "$NUM_CU" ]; then
    echo "Warning: Could not detect number of compute units, defaulting to 64"
    NUM_CU=64
fi

cd build && ninja check-rocmlir-build-only ci-performance-scripts && cd ..

echo "Detected GPU architecture: $ARCH"

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f32 --arch $ARCH --num_cu 256 -g 1 -m 64 -k 256 -n 128 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f32 --arch $ARCH --num_cu 256 -g 1 -m 8 -k 128 -n 8 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f16 --arch $ARCH --num_cu 256 -g 1 -m 64 -k 256 -n 128 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f32 --arch $ARCH --num_cu 256 -g 1 -m 64 -k 256 -n 128 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f32 --arch $ARCH --num_cu 256 -g 1 -m 8 -k 128 -n 8 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f16 --arch $ARCH --num_cu 256 -g 1 -m 64 -k 256 -n 128 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen --operation gemm -t f32 -out_datatype f32 -transA=false -transB=false -g 1 -m 1000 -n 405 -k 1024 --arch $ARCH --perf_config=gemm:v1:16,32,32,2,1,4,32,1,2,0,0 -pv  | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so  --entry-point-result=void

build/bin/rocmlir-gen --operation gemm -g 3 -m 1024 -k 769 -n 1024 --transA=false -t fp8_fp8 --schedule_version  1 --arch $ARCH -pv | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so  --entry-point-result=void

build/bin/rocmlir-gen -pv --operation conv -t f16 -out_datatype f32 --arch $ARCH --num_cu 304 --fil_layout k01c --in_layout nc01 --out_layout nk01 --batchsize 1 --in_channels 64 --in_h 32 --in_w 32 --out_channels 32 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 --kernel-repeats 1 --perf_config=gemm:v1:64,64,64,1,1,4,16,1,2,0,0 | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# Backwards weight convolution
build/bin/rocmlir-gen --operation conv_bwd_weight -t f32 -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -groupsize=1 -batchsize=64 -in_channels=64 -out_channels=64 -in_h=4 -in_w=4 -fil_h=2 -fil_w=2 -dilation_h=1 -dilation_w=1 -conv_stride_h=2 -conv_stride_w=2 -padding_h_l=2 -padding_h_r=1 -padding_w_l=2 -padding_w_r=0 -pv --arch=$ARCH | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so --entry-point-result=void

# Backwards data convolution tests (requiring multiple gemms within one kernel)
build/bin/rocmlir-gen --operation conv_bwd_data -t f32 -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -groupsize=1 -batchsize=64 -in_channels=64 -out_channels=64 -in_h=4 -in_w=4 -fil_h=2 -fil_w=2 -dilation_h=1 -dilation_w=1 -conv_stride_h=2 -conv_stride_w=2 -padding_h_l=2 -padding_h_r=1 -padding_w_l=2 -padding_w_r=0 -pv --arch=$ARCH | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so --entry-point-result=void

build/bin/rocmlir-gen --perf_config=gemm:v1:256,128,32,1,1,4,16,4,1,0,0 -g 3 -m 1024 -k 769 -n 1024 --transA=false -t f32  --operation gemm --arch $ARCH -pv   | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

build/bin/rocmlir-gen --perf_config=gemm:v1:64,64,32,1,1,4,16,1,1,0,0 -g 1 -m 4 -k 128 -n 4 --transA=false -t i8  --operation gemm --arch $ARCH -pv   | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# NOTE: This bug only occurs on gfx950, that's why we hardcode the arch.
build/bin/rocmlir-gen -operation gemm -t f16 -out_datatype f32 --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 -g 1 -m 4096 -k 11008 -n 4096 -transA=False -transB=False --perf_config="gemm:v1:16,16,16,1,1,1,32,1,2,0,0" --arch="gfx950" | timeout 5 build/bin/rocmlir-driver -c &> /dev/null
if [ $? -ne 0 ]; then
  echo "Error: Gemm test failed on gfx950"
  exit 1
fi

# Tuning driver
build/bin/rocmlir-gen -operation gemm -t f16 -out_datatype f32 --arch $ARCH --num_cu $NUM_CU -g 1 -m 1024 -k 1024 -n 1024 -transA=False -transB=False --perf_config= | build/bin/rocmlir-tuning-driver --tuning-space=quick --num-iterations=10 --warmup-iterations=1 --sleep-us=100 --use-median --show-all-measurements=false

# gemm+gemm

build/bin/rocmlir-gen -pv --arch $ARCH --operation gemm_gemm -t f32 -m 64 -n 64 -k 64 -gemmO 64 -g 1 | build/bin/rocmlir-driver --host-pipeline=highlevel | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# conv+gemm

build/bin/rocmlir-gen --operation conv_gemm -t f16 --arch $ARCH --fil_layout k01c --in_layout nc01 --out_layout nk01 --batchsize 2 --in_channels 256 --in_h 32 --in_w 32 --out_channels 128 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 -gemmO 16 -pv | build/bin/rocmlir-driver --host-pipeline=highlevel | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# basic attention

build/bin/rocmlir-gen -rand 1 -pv --arch $ARCH --operation attention -t f16 -seq_len_q 64 -seq_len_k 64 -head_dim_qk 64 -head_dim_v 64 -g 1 | build/bin/rocmlir-driver --host-pipeline=highlevel | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# GQA

build/bin/rocmlir-gen -num_heads_q 4 -num_heads_kv 2 -rand 1  -pv --arch $ARCH --operation attention -t f16 -seq_len_q 32 -seq_len_k 32 -head_dim_qk 32 -head_dim_v 32 -g 1 | build/bin/rocmlir-driver --host-pipeline=highlevel | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# causal

build/bin/rocmlir-gen -rand 1 --causal -pv --arch $ARCH --operation attention -t f16 -seq_len_q 64 -seq_len_k 64 -head_dim_qk 64 -head_dim_v 64 -g 1 | build/bin/rocmlir-driver --host-pipeline=highlevel | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# causal + GQA

build/bin/rocmlir-gen --causal -num_heads_q 4 -num_heads_kv 2 -rand 1  -pv --arch $ARCH --operation attention -t f16 -seq_len_q 32 -seq_len_k 32 -head_dim_qk 32 -head_dim_v 32 -g 1 | build/bin/rocmlir-driver --host-pipeline=highlevel | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion test

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion: constant zero

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_constant_zero_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion: constant nonzero

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_constant_nonzero_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion: math.exp input (non-zero-preserving: exp(0) = 1)

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_exp_input_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion: subtract constant output (non-zero-preserving: 0 - 1.0 = -1.0)

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_sub_const_output_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion: exp(x + 1.0) chained input (multiple non-zero-preserving ops)

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_exp_add_const_input_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion: type change

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_typechange_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion: cascaded type change in output

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_typechange_cascaded_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion: add two tensors, use result in both input and output fusion

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_add_tensors_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion: shared tensor for input and output fusion

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_shared_tensor_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion: interleaved output

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_interleaved_output_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion: self output

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_self_output_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion: reduction

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_reduce_output_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion test with split-k

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_splitk_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# fusion conv test

sed -e "s/gfx1100/$ARCH/g" -e "s/rock.num_cu = 96/rock.num_cu = $NUM_CU/g" fusion_conv_with_host.mlir | build/bin/rocmlir-driver -c | external/triton/llvm-project/build/bin/mlir-runner   --shared-libs=external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,build/lib/libconv-validation-wrappers.so,external/triton/llvm-project/build/lib/libmlir_runner_utils.so,external/triton/llvm-project/build/lib/libmlir_c_runner_utils.so   --entry-point-result=void

# TODO(rocmlirTriton): Intentionally disabled E2E tests for now since CPU validation is extremely slow.
# We will re-enable them once we have an optimized CPU validation code.

cd build && \
 LIT_FILTER=fusion/fusability ninja check-rocmlir && \
 LIT_FILTER=fusion/pr-e2e/ ninja check-rocmlir && \
 LIT_FILTER=fusion/nightly-misc-e2e/ ninja check-rocmlir && \
 LIT_FILTER=fusion/resnet50-e2e/ ninja check-rocmlir && \
 LIT_FILTER=Dialect/Rock ninja check-rocmlir && \
 LIT_FILTER=rocmlir-gen ninja check-rocmlir && \
 LIT_FILTER=Conversion ninja check-rocmlir && \
 LIT_FILTER=rocmlir-driver ninja check-rocmlir && \
 LIT_FILTER=capi ninja check-rocmlir && \
 LIT_FILTER=rocmlir-tuning-driver ninja check-rocmlir
