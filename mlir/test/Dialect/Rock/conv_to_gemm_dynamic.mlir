// RUN: rocmlir-opt -rock-conv-to-gemm -mlir-print-local-scope -split-input-file -verify-diagnostics %s | FileCheck %s

// Forward conv with dynamic N, C, Hi and Wi (and so Ho and Wo). The input's
// Embed takes Ho/Wo from the output argument, and the shared dimensions are
// assumed equal.
// CHECK-LABEL: func.func @conv_fwd_dynamic
// CHECK-SAME: (%[[FIL:.*]]: tensor<1x32x3x3x?xf16>, %[[IN:.*]]: tensor<?x1x?x?x?xf16>, %[[OUT:.*]]: tensor<?x1x?x?x32xf16>)
// CHECK: %[[FC_IDX:.*]] = tensor.dim %[[FIL]], %{{.*}}
// CHECK: %[[FC:.*]] = arith.index_cast %[[FC_IDX]]
// CHECK: %[[IC_IDX:.*]] = tensor.dim %[[IN]], %{{.*}}
// CHECK: %[[IC:.*]] = arith.index_cast %[[IC_IDX]]
// CHECK: %[[C_EQ:.*]] = arith.cmpi eq, %[[FC]], %[[IC]]
// CHECK: llvm.intr.assume %[[C_EQ]]
// CHECK: %[[IN_N_IDX:.*]] = tensor.dim %[[IN]], %{{.*}}
// CHECK: %[[IN_N:.*]] = arith.index_cast %[[IN_N_IDX]]
// CHECK: %[[OUT_N_IDX:.*]] = tensor.dim %[[OUT]], %{{.*}}
// CHECK: %[[OUT_N:.*]] = arith.index_cast %[[OUT_N_IDX]]
// CHECK: %[[N_EQ:.*]] = arith.cmpi eq, %[[IN_N]], %[[OUT_N]]
// CHECK: llvm.intr.assume %[[N_EQ]]
// CHECK-NOT: llvm.intr.assume
// CHECK: rock.transform %{{.*}} by {{.*}}Merge{s0, s1, s2} ["gemmN"] at [2] -> ["no", "0o", "1o"] at [0, 2, 3]>] symbols = [arg(2, 0), arg(2, 2), arg(2, 3)] bounds = [1, 32, (s0 * s1) * s2]
// CHECK: rock.transform %[[FIL]] by {{.*}}Merge{3, 3, s0} ["gemmK"] at [1] -> ["0", "1", "c"] at [2, 3, 4]>{{.*}}symbols = [arg(0, 4)] bounds = [1, s0 * 9, 32]
// CHECK: %[[PAD:.*]] = rock.transform %[[IN]] by {{.*}}Pad{1, 1, 1, 1}{{.*}}symbols = [arg(1, 0), arg(1, 2), arg(1, 3), arg(1, 4)] bounds = [s0, 1, s1 + 2, s2 + 2, s3]
// CHECK: %[[EMB:.*]] = rock.transform %[[PAD]] by {{.*}}<Embed{1, 2} ["0", "0o"] at [2, 3] -> ["0ipad"] at [2]>, <Embed{1, 2} ["1", "1o"] at [4, 5] -> ["1ipad"] at [3]>] symbols = [arg(1, 0), arg(1, 2), arg(1, 3), arg(1, 4), arg(2, 2), arg(2, 3)] bounds = [s0, 1, 3, s4, 3, s5, s3]
// CHECK: %[[GEMM_IN:.*]] = rock.transform %[[EMB]] by {{.*}}Merge{s0, s2, s3} ["gemmN"]
// CHECK: rock.gemm tr %{{.*}} * %[[GEMM_IN]] : tensor<1x?x32xf16> * tensor<1x?x?xf16> -> tensor<1x32x?xf16>
func.func @conv_fwd_dynamic(%filter: tensor<1x32x3x3x?xf16>, %input: tensor<?x1x?x?x?xf16>, %output: tensor<?x1x?x?x32xf16>) -> tensor<?x1x?x?x32xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel} {
  %0 = rock.conv(%filter, %input) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "gi", "0i", "1i", "ci"], output_layout = ["no", "go", "0o", "1o", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [2 : index, 2 : index]} : tensor<1x32x3x3x?xf16>, tensor<?x1x?x?x?xf16> -> tensor<?x1x?x?x32xf16>
  %1 = rock.store %0 to %output by set : tensor<?x1x?x?x32xf16> -> tensor<?x1x?x?x32xf16> to tensor<?x1x?x?x32xf16>
  return %1 : tensor<?x1x?x?x32xf16>
}

// -----

// Backward data with dynamic N, C and K: the spatial dims stay static, so only
// the passthrough and merge sizes are symbolic.
// CHECK-LABEL: func.func @conv_bwd_data_dynamic
// CHECK: llvm.intr.assume
// CHECK: rock.gemm tr {{.*}} : tensor<1x?x?xf32> * tensor<1x?x?xf32> -> tensor<1x?x?xf32>
func.func @conv_bwd_data_dynamic(%filter: tensor<1x?x3x3x?xf32>, %gradient: tensor<?x1x14x14x?xf32>, %input: tensor<?x1x14x14x?xf32>) -> tensor<?x1x14x14x?xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel} {
  %0 = rock.conv_bwd_data(%filter, %gradient) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "gi", "0i", "1i", "ci"], output_layout = ["no", "go", "0o", "1o", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : tensor<1x?x3x3x?xf32>, tensor<?x1x14x14x?xf32> -> tensor<?x1x14x14x?xf32>
  %1 = rock.store %0 to %input by set : tensor<?x1x14x14x?xf32> -> tensor<?x1x14x14x?xf32> to tensor<?x1x14x14x?xf32>
  return %1 : tensor<?x1x14x14x?xf32>
}

// -----

func.func @conv_fwd_dynamic_filter_spatial(%filter: tensor<1x32x?x3x16xf16>, %input: tensor<4x1x14x14x16xf16>, %output: tensor<4x1x14x14x32xf16>) -> tensor<4x1x14x14x32xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel} {
  // expected-error @+2 {{dynamic filter spatial dimensions are not supported}}
  // expected-error @+1 {{failed to legalize operation 'rock.conv'}}
  %0 = rock.conv(%filter, %input) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "gi", "0i", "1i", "ci"], output_layout = ["no", "go", "0o", "1o", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : tensor<1x32x?x3x16xf16>, tensor<4x1x14x14x16xf16> -> tensor<4x1x14x14x32xf16>
  %1 = rock.store %0 to %output by set : tensor<4x1x14x14x32xf16> -> tensor<4x1x14x14x32xf16> to tensor<4x1x14x14x32xf16>
  return %1 : tensor<4x1x14x14x32xf16>
}

// -----

func.func @conv_bwd_data_dynamic_spatial(%filter: tensor<1x32x3x3x16xf32>, %gradient: tensor<4x1x?x14x32xf32>, %input: tensor<4x1x?x14x16xf32>) -> tensor<4x1x?x14x16xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel} {
  // expected-error @+2 {{dynamic spatial dimensions are not supported for backward-data convolutions}}
  // expected-error @+1 {{failed to legalize operation 'rock.conv_bwd_data'}}
  %0 = rock.conv_bwd_data(%filter, %gradient) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "gi", "0i", "1i", "ci"], output_layout = ["no", "go", "0o", "1o", "ko"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : tensor<1x32x3x3x16xf32>, tensor<4x1x?x14x32xf32> -> tensor<4x1x?x14x16xf32>
  %1 = rock.store %0 to %input by set : tensor<4x1x?x14x16xf32> -> tensor<4x1x?x14x16xf32> to tensor<4x1x?x14x16xf32>
  return %1 : tensor<4x1x?x14x16xf32>
}
