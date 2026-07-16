// Stream-K (v5 perf-config: streamKMultiple=1, splitKFactor=1) on a bwd-data
// conv. The remainder wave re-splits K and atomic_adds partials into the
// zero-prefilled output; the clone verifier confirms the result matches the
// plain decomposition (stream-K may also fall back to data-parallel depending
// on the grid vs. num_cu, in which case the result is trivially correct).

// RUN: rocmlir-gen -pv -print-verify-results=summary --operation conv_bwd_data -t f32 -out_dtype f16 -fil_dtype f16 --arch %arch --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 1 --in_channels 192 --in_h 64 --in_w 64 --out_channels 384 --fil_h 4 --fil_w 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --groupsize 1 --perf_config="gemm:v5:64,64,16,1,1,4,16,1,1,0,0,1,-1,-1,-1,-1,-1,-1" | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void| FileCheck %s

// CHECK: [1 1 1]
