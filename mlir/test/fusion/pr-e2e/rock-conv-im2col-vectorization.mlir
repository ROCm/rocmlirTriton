// E2E correctness for the im2col vectorization-hint path: this forward conv has
// a 4-divisible output width on f32, so getMaxVectorization proves contiguity 4
// and the input global load is widened to a 128-bit buffer load (see the
// companion assembly test im2col_vectorization_hint_assembly.mlir). Verify the
// widened kernel still computes the correct result against the CPU reference.

// RUN: rocmlir-gen -pv -print-verify-results=summary --operation conv -t f32 --arch %arch --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 1 --in_channels 4 --in_h 70 --in_w 70 --out_channels 64 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --groupsize 1 | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]
