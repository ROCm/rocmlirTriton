// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// A causal sliding window can leave a query with no eligible keys. Both the
// GPU kernel and the CPU reference define that row's contribution as zero.
// The direct path also returns the fully masked row's -inf LSE; the split-KV
// path exercises host recombination of fully masked partial results.

// Fully masked f32 row (seq_len_q = 1): output is zero and LSE is -inf.
// RUN: rocmlir-gen --arch %arch --operation attention -last_valid_kv_index=2 -sliding_window_look_back=1 --causal -return_lse -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv -pr -pvr | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=DIRECT
// DIRECT: [1 1 1]
// DIRECT-NEXT: [1 1 1]
// DIRECT: [-inf]
// DIRECT: [0,  0,  0,  0
// DIRECT: [-inf]
// DIRECT: [0,  0,  0,  0

// Same fully masked f32 row through split-KV host recombination.
// RUN: rocmlir-gen --arch %arch --operation attention -last_valid_kv_index=2 -sliding_window_look_back=1 --causal -return_lse -split_kv=8 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv -pr -pvr | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=SPLITKV
// SPLITKV: [1 1 1]
// SPLITKV: [0,  0,  0,  0
// SPLITKV: [0,  0,  0,  0

// Mixed f32 rows: q=0 is fully masked; later queries keep valid keys.
// RUN: rocmlir-gen --arch %arch --operation attention -last_valid_kv_index=2 -sliding_window_look_back=1 --causal -return_lse -seq_len_q 4 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv -pr -pvr | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=MIXED-DIRECT
// MIXED-DIRECT: [1 1 1]
// MIXED-DIRECT-NEXT: [1 1 1]
// MIXED-DIRECT: [-inf,  {{[-+0-9.e]+}},  {{[-+0-9.e]+}},  {{[-+0-9.e]+}}]
// MIXED-DIRECT: {{^\[0,  0,  0,  0.*[1-9]}}

// Same mixed f32 rows through split-KV host recombination.
// RUN: rocmlir-gen --arch %arch --operation attention -last_valid_kv_index=2 -sliding_window_look_back=1 --causal -return_lse -split_kv=8 -seq_len_q 4 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv -pr -pvr | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=MIXED-SPLIT
// MIXED-SPLIT: [1 1 1]
// MIXED-SPLIT: {{^\[0,  0,  0,  0.*[1-9]}}

// Same mixed-row case in f16, including the finite-sentinel empty-row guard.
// RUN: rocmlir-gen --arch %arch --operation attention -last_valid_kv_index=2 -sliding_window_look_back=1 --causal -return_lse -seq_len_q 4 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f16 -rand 1 -rand_type float -pv -pr -pvr | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=F16-DIRECT
// F16-DIRECT: [1 1 1]
// F16-DIRECT-NEXT: [1 1 1]
// F16-DIRECT: [-inf,  {{[-+0-9.e]+}},  {{[-+0-9.e]+}},  {{[-+0-9.e]+}}]
// F16-DIRECT: {{^\[0,  0,  0,  0.*[1-9]}}

// Same mixed f16 rows through split-KV host recombination.
// RUN: rocmlir-gen --arch %arch --operation attention -last_valid_kv_index=2 -sliding_window_look_back=1 --causal -return_lse -split_kv=8 -seq_len_q 4 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f16 -rand 1 -rand_type float -pv -pr -pvr | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=F16-SPLIT
// F16-SPLIT: [1 1 1]
// F16-SPLIT: {{^\[0,  0,  0,  0.*[1-9]}}

// Causal split-KV spanning several query and key tiles, so host recombination
// also exercises per-row validity counts beyond the first query block.
// RUN: rocmlir-gen --arch %arch --operation attention --causal -return_lse -split_kv=8 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=MULTIBLOCK-SPLIT
// MULTIBLOCK-SPLIT: [1 1 1]

// Prefix-causal mixed rows: prefix_offset plus the sliding window fully masks q=0.
// RUN: rocmlir-gen --arch %arch --operation attention -last_valid_kv_index=3 -sliding_window_look_back=1 --causal --prefix_offset=1 -return_lse -seq_len_q 4 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv -pr -pvr | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=PREFIX-DIRECT
// PREFIX-DIRECT: [1 1 1]
// PREFIX-DIRECT-NEXT: [1 1 1]
// PREFIX-DIRECT: [-inf,  {{[-+0-9.e]+}},  {{[-+0-9.e]+}},  {{[-+0-9.e]+}}]
// PREFIX-DIRECT: {{^\[0,  0,  0,  0.*[1-9]}}
