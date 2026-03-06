#!/bin/bash
# Test script to verify CPU validation produces correct results

set -e

# Detect GPU architecture from system
ARCH=$(rocminfo | grep -o 'gfx[0-9a-z]*' | head -1)
if [ -z "$ARCH" ]; then
    echo "Error: Could not detect GPU architecture. Is rocminfo available?"
    exit 1
fi

# Get Compute Units from GPU section
NUM_CU=$(rocminfo | grep -A 30 "Name:.*gfx" | grep "Compute Unit" | head -1 | grep -o '[0-9]*')
if [ -z "$NUM_CU" ]; then
    echo "Warning: Could not detect number of compute units, defaulting to 64"
    NUM_CU=64
fi

echo "Detected GPU architecture: $ARCH with $NUM_CU CUs"

# Branch name (required argument)
if [ -z "$1" ]; then
    echo "Usage: $0 <branch_name> [--rand]"
    echo "Example: $0 develop"
    echo "         $0 feature_branch --rand"
    exit 1
fi
BRANCH="$1"

# Random mode (optional, default off)
RAND_OPTS=""
if [ "$2" == "--rand" ]; then
    RAND_OPTS="-rand 1"
    echo "Random mode enabled (-rand 1)"
fi

echo "Running tests for branch: '${BRANCH}'"

# Common paths
ROCMLIR_GEN="./build/bin/rocmlir-gen"
ROCMLIR_DRIVER="./build/bin/rocmlir-driver"
MLIR_RUNNER="./external/triton/llvm-project/build/bin/mlir-runner"
SHARED_LIBS="./external/triton/llvm-project/build/lib/libmlir_rocm_runtime.so,./build/lib/libconv-validation-wrappers.so,./external//triton/llvm-project/build/lib/libmlir_runner_utils.so,./external//triton/llvm-project/build/lib/libmlir_c_runner_utils.so"

# Datatypes to test
DATATYPES=("f32" "f16" "bf16" "i8" "f8E5M2")

# Output directory
OUTPUT_DIR="prefill_results"
mkdir -p "$OUTPUT_DIR"

run_test() {
    local name="$1"
    local cmd="$2"
    echo "Running: $name..."
    # Filter out memory addresses (base@ = 0x...) which change between runs
    eval "$cmd" 2>&1 | sed 's/base@ = 0x[0-9a-f]*/base@ = ADDR/g' > "${OUTPUT_DIR}/${name}_${BRANCH}.txt"
    echo "  -> ${OUTPUT_DIR}/${name}_${BRANCH}.txt"
}

for DTYPE in "${DATATYPES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing datatype: $DTYPE"
    echo "=========================================="

    # GEMM Tests
    echo ""
    echo "  GEMM Tests"
    echo "  ----------"

    run_test "gemm_${DTYPE}" \
        "$ROCMLIR_GEN -pv -print-validation-results -print-inputs $RAND_OPTS --arch $ARCH --num_cu $NUM_CU --operation gemm -t $DTYPE -g 1 -m 64 -n 64 -k 64 | $ROCMLIR_DRIVER -c | $MLIR_RUNNER --shared-libs=$SHARED_LIBS --entry-point-result=void"

    # Skip gemm_splitk for i8 (SplitK not supported with i8)
    if [ "$DTYPE" != "i8" ]; then
        run_test "gemm_splitk_${DTYPE}" \
            "$ROCMLIR_GEN -pv -print-validation-results -print-inputs $RAND_OPTS --arch $ARCH --num_cu $NUM_CU --operation gemm -t $DTYPE -g 1 -m 64 -n 64 -k 256 --perf_config="gemm:v1:64,64,64,1,1,4,16,3,2,0,0" | $ROCMLIR_DRIVER -c | $MLIR_RUNNER --shared-libs=$SHARED_LIBS --entry-point-result=void"
    fi

    # Conv Tests
    echo ""
    echo "  Conv Tests"
    echo "  ----------"

    run_test "conv_fwd_${DTYPE}" \
        "$ROCMLIR_GEN -pv -print-validation-results -print-inputs $RAND_OPTS --arch $ARCH --num_cu $NUM_CU --operation conv -t $DTYPE -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=2 -in_channels=32 -out_channels=32 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 | $ROCMLIR_DRIVER -c | $MLIR_RUNNER --shared-libs=$SHARED_LIBS --entry-point-result=void"

    # Skip conv_bwd_weight for i8, f16, and f8E5M2
    if [ "$DTYPE" != "i8" ] && [ "$DTYPE" != "f16" ] && [ "$DTYPE" != "f8E5M2" ]; then
        run_test "conv_bwd_weight_${DTYPE}" \
            "$ROCMLIR_GEN -pv -print-validation-results -print-inputs $RAND_OPTS --arch $ARCH --num_cu $NUM_CU --operation conv_bwd_weight -t $DTYPE -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -groupsize=1 -batchsize=64 -in_channels=64 -out_channels=64 -in_h=4 -in_w=4 -fil_h=2 -fil_w=2 -dilation_h=1 -dilation_w=1 -conv_stride_h=2 -conv_stride_w=2 -padding_h_l=2 -padding_h_r=1 -padding_w_l=2 -padding_w_r=0 | $ROCMLIR_DRIVER -c | $MLIR_RUNNER --shared-libs=$SHARED_LIBS --entry-point-result=void"
    fi

    # Skip conv_bwd_data for i8 and f8E5M2
    if [ "$DTYPE" != "i8" ] && [ "$DTYPE" != "f8E5M2" ]; then
        run_test "conv_bwd_data_${DTYPE}" \
            "$ROCMLIR_GEN -pv -print-validation-results -print-inputs $RAND_OPTS --arch $ARCH --num_cu $NUM_CU --operation conv_bwd_data -t $DTYPE -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -groupsize=1 -batchsize=64 -in_channels=64 -out_channels=64 -in_h=4 -in_w=4 -fil_h=2 -fil_w=2 -dilation_h=1 -dilation_w=1 -conv_stride_h=2 -conv_stride_w=2 -padding_h_l=2 -padding_h_r=1 -padding_w_l=2 -padding_w_r=0 | $ROCMLIR_DRIVER -c | $MLIR_RUNNER --shared-libs=$SHARED_LIBS --entry-point-result=void"
    fi

done
