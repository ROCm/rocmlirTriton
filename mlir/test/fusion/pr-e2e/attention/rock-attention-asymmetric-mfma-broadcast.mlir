// RUN: rocmlir-gen -operation attention -t f32 --arch %arch -g 1 -seq_len_q 348 -seq_len_k 122 -num_heads_q 16 -num_heads_kv 4 -head_dim_qk 251 -head_dim_v 4 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=False -transV=True -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:64,32,32,2,1,4,0,1,1,0,0 -pv -relDiff_threshold 1e-4 \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c --mlir-print-ir-after=tritonamdgpu-accelerate-matmul -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefix=IR

// IR-LABEL: IR Dump After {{.*}}tritonamdgpu-accelerate-matmul
// IR: #[[ASYM_MFMA:mma[0-9]*]] = #ttg.amd_mfma<{{.*}}instrShape = [64, 4, 16]{{.*}}>
// IR: tensor<64x32xf32, #ttg.dot_op<{opIdx = 0, parent = #[[ASYM_MFMA]], kWidth = 1}>>
// IR: tensor<32x4xf32, #ttg.dot_op<{opIdx = 1, parent = #[[ASYM_MFMA]], kWidth = 1}>>

// RUN: rocmlir-gen -operation attention -t f32 --arch %arch -g 1 -seq_len_q 348 -seq_len_k 122 -num_heads_q 16 -num_heads_kv 4 -head_dim_qk 251 -head_dim_v 4 -with-attn-scale=False -with-attn-bias=True -transQ=True -transK=False -transV=True -transO=False -causal=False -return_lse=False -split_kv=1 --perf_config=attn:v1:64,32,32,2,1,4,0,1,1,0,0 -pv -relDiff_threshold 1e-4 \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void \
// RUN: | FileCheck %s
// CHECK: [1 1 1]
