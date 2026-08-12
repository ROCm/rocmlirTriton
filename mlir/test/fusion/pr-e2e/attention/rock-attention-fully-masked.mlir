// RUN: rocmlir-gen --arch %arch --operation attention -current_seq_len=2 -sliding_window_size=1 --causal -return_lse -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void \
// RUN: | FileCheck %s

// A causal sliding window can leave a query with no eligible keys. Both the
// GPU kernel and the CPU reference define that row's contribution as zero.

// CHECK: [1 1 1]
