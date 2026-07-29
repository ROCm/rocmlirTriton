// RUN: rocmlir-gen -operation attention -t f16 --arch %arch -g 1 -seq_len_q 1158 -seq_len_k 68 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 176 -head_dim_v 88 -with-attn-scale=False -with-attn-bias=False -transQ=True -transK=False -transV=False -transO=False -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:64,256,16,2,1,4,16,1,1,0,1 -pv -pr -pvr \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c --mlir-print-ir-after=tritonamdgpu-accelerate-matmul -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefix=KPACK

// KPACK-LABEL: IR Dump After {{.*}}tritonamdgpu-accelerate-matmul
// KPACK: #[[WMMA:mma[0-9]*]] = #ttg.amd_wmma<{{.*}}isTranspose = true{{.*}}>
// KPACK: tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #[[WMMA]], kWidth = {{[0-9]+}}}>>
// KPACK: tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[WMMA]], kWidth = {{[0-9]+}}}>>

// RUN: rocmlir-gen -operation attention -t f16 --arch %arch -g 1 -seq_len_q 1158 -seq_len_k 68 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 176 -head_dim_v 88 -with-attn-scale=False -with-attn-bias=False -transQ=True -transK=False -transV=False -transO=False -causal=True -return_lse=False -split_kv=1 --perf_config=attn:v1:64,256,16,2,1,4,16,1,1,0,1 -pv -pr -pvr \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void \
// RUN: | FileCheck %s
// CHECK: [1 1 1]
