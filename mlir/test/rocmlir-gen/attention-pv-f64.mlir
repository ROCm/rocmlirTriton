// Verify that -pv-f64 promotes the host CPU attention reference's interior
// to f64 for non-quantized attention (f32, f16), and that it errors out
// when used with quantized i8 attention or with non-attention kernels.

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv -pv-f64 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_F32
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f16 -pv -pv-f64 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_F16

// `-pv-f64` implies both `pv-f64` and `-pv-strict`. Check if they produce the same IR.
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f16 -pv-f64 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_F16
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f16 -pv-f64 -pv-strict | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_F16

// `-pv-f64` must reject i8 (quantized) attention. The f64 promotion does
// not apply to the i8 reference path, so rocmlir-gen errors out.
// RUN: not rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -seq_len_q 384 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 -t i8 -pv-f64 2>&1 | FileCheck %s --check-prefix=CHECK_I8

// f32 attention with -pv-f64

// CHECK_F32-LABEL: func.func @host_naive_attention
// CHECK_F32-SAME:  %[[outArg:[^:]*]]: memref<32768xf32>)
// CHECK_F32:       %[[qIn:.*]] = tosa.reshape {{.*}} -> tensor<1x1024x32xf32>
// CHECK_F32:       %[[qF64:.*]] = tosa.cast %[[qIn]] : (tensor<1x1024x32xf32>) -> tensor<1x1024x32xf64>
// CHECK_F32:       %[[kIn:.*]] = tosa.reshape {{.*}} -> tensor<1x32x1024xf32>
// CHECK_F32:       %[[kF64:.*]] = tosa.cast %[[kIn]] : (tensor<1x32x1024xf32>) -> tensor<1x32x1024xf64>
// CHECK_F32:       %[[vIn:.*]] = tosa.reshape {{.*}} -> tensor<1x1024x32xf32>
// CHECK_F32:       %[[vF64:.*]] = tosa.cast %[[vIn]] : (tensor<1x1024x32xf32>) -> tensor<1x1024x32xf64>
// CHECK_F32:       %[[qk:.*]] = tosa.matmul %[[qF64]], %[[kF64]], %{{.*}}, %{{.*}} {acc_type = f64} : (tensor<1x1024x32xf64>, tensor<1x32x1024xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<1x1024x1024xf64>
// CHECK_F32-DAG:   %[[qkMaxs:.*]] = tosa.reduce_max {{.*}} -> tensor<1x1024x1xf64>
// CHECK_F32-DAG:   %[[normQk:.*]]  = tosa.sub {{.*}} -> tensor<1x1024x1024xf64>
// CHECK_F32-DAG:   %[[exps:.*]]    = tosa.exp {{.*}} -> tensor<1x1024x1024xf64>
// CHECK_F32-DAG:   %[[expsSums:.*]] = tosa.reduce_sum {{.*}} -> tensor<1x1024x1xf64>
// CHECK_F32-DAG:   %[[invSums:.*]] = tosa.reciprocal {{.*}} -> tensor<1x1024x1xf64>
// CHECK_F32-DAG:   %[[softmax:.*]] = tosa.mul {{.*}} -> tensor<1x1024x1024xf64>
// CHECK_F32:       %[[result:.*]] = tosa.matmul %{{.*}}, %[[vF64]], %{{.*}}, %{{.*}} {acc_type = f64} : (tensor<1x1024x1024xf64>, tensor<1x1024x32xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<1x1024x32xf64>
// CHECK_F32:       %[[resultF32:.*]] = tosa.cast %[[result]] : (tensor<1x1024x32xf64>) -> tensor<1x1024x32xf32>
// CHECK_F32:       memref.copy %{{.*}}, %[[outArg]]
// CHECK_F32:       return

// f16 attention with -pv-f64

// CHECK_F16-LABEL: func.func @host_naive_attention
// CHECK_F16-SAME:  %[[outArg:[^:]*]]: memref<8192xf16>)
// CHECK_F16:       %[[qIn:.*]] = tosa.reshape {{.*}} -> tensor<1x256x32xf16>
// CHECK_F16:       %[[qF64:.*]] = tosa.cast %[[qIn]] : (tensor<1x256x32xf16>) -> tensor<1x256x32xf64>
// CHECK_F16:       %[[kIn:.*]] = tosa.reshape {{.*}} -> tensor<1x32x256xf16>
// CHECK_F16:       %[[kF64:.*]] = tosa.cast %[[kIn]] : (tensor<1x32x256xf16>) -> tensor<1x32x256xf64>
// CHECK_F16:       %[[vIn:.*]] = tosa.reshape {{.*}} -> tensor<1x256x32xf16>
// CHECK_F16:       %[[vF64:.*]] = tosa.cast %[[vIn]] : (tensor<1x256x32xf16>) -> tensor<1x256x32xf64>
// CHECK_F16:       %[[qk:.*]] = tosa.matmul %[[qF64]], %[[kF64]], %{{.*}}, %{{.*}} {acc_type = f64} : (tensor<1x256x32xf64>, tensor<1x32x256xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<1x256x256xf64>
// CHECK_F16-DAG:   tosa.reduce_max {{.*}} -> tensor<1x256x1xf64>
// CHECK_F16-DAG:   tosa.exp {{.*}} -> tensor<1x256x256xf64>
// CHECK_F16-DAG:   tosa.reduce_sum {{.*}} -> tensor<1x256x1xf64>
// CHECK_F16-DAG:   tosa.reciprocal {{.*}} -> tensor<1x256x1xf64>
// CHECK_F16:       %[[result:.*]] = tosa.matmul %{{.*}}, %[[vF64]], %{{.*}}, %{{.*}} {acc_type = f64} : (tensor<1x256x256xf64>, tensor<1x256x32xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<1x256x32xf64>
// CHECK_F16:       %[[resultF16:.*]] = tosa.cast %[[result]] : (tensor<1x256x32xf64>) -> tensor<1x256x32xf16>
// CHECK_F16:       memref.copy %{{.*}}, %[[outArg]]
// CHECK_F16:       return

// i8 attention with -pv-f64 must error out (no IR is produced).

// CHECK_I8: -pv-f64 is not supported for i8 (quantized) attention
