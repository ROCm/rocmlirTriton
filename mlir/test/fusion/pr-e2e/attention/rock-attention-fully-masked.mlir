// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: rocmlir-gen --arch %arch --operation attention -last_valid_kv_index=2 -sliding_window_look_back=1 --causal -return_lse -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv -pr -pvr \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void \
// RUN: | FileCheck %s --check-prefix=DIRECT
// RUN: rocmlir-gen --arch %arch --operation attention -last_valid_kv_index=2 -sliding_window_look_back=1 --causal -return_lse -seq_len_q 4 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv -pr -pvr \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void \
// RUN: | FileCheck %s --check-prefix=MIXED-DIRECT
// RUN: rocmlir-gen --arch %arch --operation attention -last_valid_kv_index=2 -sliding_window_look_back=1 --causal -return_lse -seq_len_q 4 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f16 -rand 1 -rand_type float -pv -pr -pvr \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void \
// RUN: | FileCheck %s --check-prefix=F16-DIRECT
// RUN: rocmlir-gen --arch %arch --operation attention -last_valid_kv_index=3 -sliding_window_look_back=1 --causal --prefix_offset=1 -return_lse -seq_len_q 4 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv -pr -pvr \
// RUN: | rocmlir-driver --host-pipeline=highlevel \
// RUN: | rocmlir-driver -c \
// RUN: | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void \
// RUN: | FileCheck %s --check-prefix=PREFIX-DIRECT

// A causal sliding window can leave a query with no eligible keys. Both the
// GPU kernel and the CPU reference define that row's contribution as zero.
// The direct path also returns the fully masked row's -inf LSE. The mixed
// cases use q=0 as a fully masked row and later rows as valid rows.

// DIRECT: [1 1 1]
// DIRECT-NEXT: [1 1 1]
// DIRECT: [-inf]
// DIRECT: [0,  0,  0,  0
// DIRECT: [-inf]
// DIRECT: [0,  0,  0,  0
// MIXED-DIRECT: [1 1 1]
// MIXED-DIRECT-NEXT: [1 1 1]
// MIXED-DIRECT: [-inf,  {{[-+0-9.e]+}},  {{[-+0-9.e]+}},  {{[-+0-9.e]+}}]
// MIXED-DIRECT: {{^\[0,  0,  0,  0.*[1-9]}}
// F16-DIRECT: [1 1 1]
// F16-DIRECT-NEXT: [1 1 1]
// F16-DIRECT: [-inf,  {{[-+0-9.e]+}},  {{[-+0-9.e]+}},  {{[-+0-9.e]+}}]
// F16-DIRECT: {{^\[0,  0,  0,  0.*[1-9]}}
// PREFIX-DIRECT: [1 1 1]
// PREFIX-DIRECT-NEXT: [1 1 1]
// PREFIX-DIRECT: [-inf,  {{[-+0-9.e]+}},  {{[-+0-9.e]+}},  {{[-+0-9.e]+}}]
// PREFIX-DIRECT: {{^\[0,  0,  0,  0.*[1-9]}}
