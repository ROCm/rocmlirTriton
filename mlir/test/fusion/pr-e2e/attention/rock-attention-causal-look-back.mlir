// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// End-to-end validation of the banded causal mask, where query m attends to
// keys [max(0, m - L), m].
//
// Unlike the sliding window, the band never leaves a query with no eligible
// keys: key m is always in range. What it does produce is a fully masked
// *tile*. The N-loop lower bound is derived from the block's first query row,
// so the block's last rows can find that first processed tile entirely below
// their own `m - L` bound. That row's partial softmax is empty, which is what
// the finite-sentinel / clamped-denominator guard exists for. L smaller than
// the M tile plus seq_len_q past the first M block is what triggers it.

// Narrow band, several M blocks: exercises both the skipped N-blocks and the
// fully masked first tile.
// RUN: rocmlir-gen --arch %arch --operation attention --causal -causal_look_back=8 -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=NARROW
// NARROW: [1 1 1]

// Same shape in f16, where the finite sentinel is a much smaller magnitude
// than f32's and an unguarded empty row would be most visible.
// RUN: rocmlir-gen --arch %arch --operation attention --causal -causal_look_back=8 -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f16 -rand 1 -rand_type float -pv | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=F16-NARROW
// F16-NARROW: [1 1 1]

// Ragged L and ragged sequence lengths: neither edge of the band lands on a
// tile boundary, so both edge tiles are trimmed by the per-element mask.
// RUN: rocmlir-gen --arch %arch --operation attention --causal -causal_look_back=17 -seq_len_q 130 -seq_len_k 130 -head_dim_qk 40 -head_dim_v 40 -t f32 -rand 1 -rand_type float -pv | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=RAGGED
// RAGGED: [1 1 1]

// L = 1, the narrowest band: every query attends to keys {m - 1, m} only.
// RUN: rocmlir-gen --arch %arch --operation attention --causal -causal_look_back=1 -seq_len_q 128 -seq_len_k 128 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=MINIMAL
// MINIMAL: [1 1 1]

// With LSE returned: the band's LSE is finite on every row, since no row is
// ever fully masked. The clamped denominator must not leak into it.
// RUN: rocmlir-gen --arch %arch --operation attention --causal -causal_look_back=8 -return_lse -seq_len_q 128 -seq_len_k 128 -head_dim_qk 32 -head_dim_v 32 -t f32 -rand 1 -rand_type float -pv -pr -pvr | rocmlir-driver --host-pipeline=highlevel | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=LSE
// LSE: [1 1 1]
// LSE-NEXT: [1 1 1]
// LSE-NOT: -inf
